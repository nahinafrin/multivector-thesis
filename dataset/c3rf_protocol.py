"""
c3rf_protocol.py  —  STRICT TUNE-ON-DEV / MEASURE-ONCE-ON-TEST HARNESS
======================================================================

Purpose
-------
The earlier C3RF tuning (step_03c) used `--limit 500` of dataset_all.clean.jsonl,
which contained the ENTIRE pool of every rare class. Because train/val/test are
stratified random splits of that same pool, rare-class rows in test.jsonl were
seen during tuning. The reported F1 therefore measured overfitting, not
generalisation.

This harness restructures evaluation around the existing splits with
file-system-enforced discipline:

  Step 1  verify    : confirm val/test exist and don't overlap
  Step 2  tune      : run sweep + select operating points on val.jsonl ONLY
  Step 3  freeze    : write the chosen config to ./c3rf_frozen_config.json
  Step 4  test      : evaluate all systems on test.jsonl exactly ONCE
                       (refuses to run if test_results.json already exists,
                        unless --force is passed, which logs a warning)
  Step 5  report    : print measured numbers from the frozen artefacts

Hard guarantees enforced by the script:
  * tune never touches test.jsonl
  * test never runs without a frozen_config.json on disk
  * test refuses to overwrite a prior test_results.json silently
  * The C3RF logic itself is NOT modified -- only the data routing.

Usage:
    python c3rf_protocol.py verify
    python c3rf_protocol.py tune   --dev  ./merged_output/val.jsonl
    python c3rf_protocol.py freeze --review-as-red  --lg-soft-escalate-w 0.65 ...
    python c3rf_protocol.py test   --test ./merged_output/test.jsonl
    python c3rf_protocol.py report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from typing import Optional

from pipeline_common import read_jsonl
import step_03_injection_detection as s3
import step_03c_fusion_gate as c3rf


# --------------------------------------------------------------------------- #
# Paths used as the file-system enforcement layer                             #
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
DEV_CACHE        = os.path.join(HERE, "c3rf_dev_scores.jsonl")
TEST_CACHE       = os.path.join(HERE, "c3rf_test_scores.jsonl")
FROZEN_CONFIG    = os.path.join(HERE, "c3rf_frozen_config.json")
DEV_RESULTS      = os.path.join(HERE, "c3rf_dev_results.json")
TEST_RESULTS     = os.path.join(HERE, "c3rf_test_results.json")

DEFAULT_DEV  = os.path.join(HERE, "merged_output", "val.jsonl")
DEFAULT_TEST = os.path.join(HERE, "merged_output", "test.jsonl")


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _load_hashes(path: str) -> set[str]:
    return {_hash_prompt(r.get("prompt", "")) for r in read_jsonl(path)}


def _score_dataset(path: str, cache_path: str, limit: Optional[int],
                   threshold: float, base_url: str,
                   note: str = "") -> list[c3rf._Sample]:
    """Compute (s_I, s_L, category) for every row in `path`; cache to JSONL.

    Re-running with the same cache is idempotent: rows already cached (by
    prompt hash) are skipped, only new rows are scored. This lets us:
      * stop & resume a long Llama-Guard run
      * run dev tuning many times at zero cost after the first scoring pass
    """
    cfg = c3rf.FusionConfig()
    cache: dict[str, dict] = {}
    if os.path.exists(cache_path):
        for r in read_jsonl(cache_path):
            cache[r["hash"]] = r
        print(f"  loaded {len(cache)} cached scores from {os.path.basename(cache_path)}")

    rows = list(read_jsonl(path))
    if limit is not None:
        rows = rows[:limit]
    print(f"  scoring {len(rows)} rows from {os.path.basename(path)} {note}")

    out: list[c3rf._Sample] = []
    new_writes = 0
    t0 = time.time()
    for i, row in enumerate(rows):
        prompt = row.get("prompt", "")
        h = _hash_prompt(prompt)
        if h in cache:
            rec = cache[h]
        else:
            s_I, inj_detail = c3rf._injection_score(prompt, threshold=threshold)
            s_L, cat, _raw = c3rf._llamaguard_soft(prompt, cfg, base_url=base_url)
            inj_red = any(not d["valid"] for d in inj_detail.values()) or s_I >= threshold
            lg_unsafe = s_L >= 0.5
            rec = {
                "hash": h,
                "s_I": float(s_I),
                "s_L": float(s_L),
                "category": str(cat),
                "inj_red": bool(inj_red),
                "lg_unsafe": bool(lg_unsafe),
            }
            cache[h] = rec
            with open(cache_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            new_writes += 1
            if (i + 1) % 25 == 0:
                rate = (i + 1) / max(time.time() - t0, 1e-6)
                print(f"  ...scored {i+1}/{len(rows)}  ({rate:.2f} rows/s)")
        out.append(c3rf._Sample(
            gt_unsafe=str(row.get("safety", "")).lower() == "unsafe",
            attack=row.get("attack_type", "unknown"),
            s_I=rec["s_I"], s_L=rec["s_L"], category=rec["category"],
            rho=float(row.get("context_risk", 0.0) or 0.0),
            inj_red=rec["inj_red"], lg_unsafe=rec["lg_unsafe"]))
    print(f"  done: {len(out)} samples ({new_writes} new, {len(out)-new_writes} cached)")
    return out


def _score_at(samples: list[c3rf._Sample], cfg: c3rf.FusionConfig,
              treat_review_as_red: bool) -> dict:
    return c3rf._score_at(samples, cfg, treat_review_as_red)


def _summary_table(label: str, rep: dict) -> str:
    return (f"{label:18} P={rep['precision']:.3f}  R={rep['recall']:.3f}  "
            f"F1={rep['f1']:.3f}  FP={rep['fp']:<4} FN={rep['fn']:<4} "
            f"TP={rep['tp']:<4} TN={rep['tn']:<4} n={rep['n']}")


# --------------------------------------------------------------------------- #
# Step 1 — verify                                                             #
# --------------------------------------------------------------------------- #
def cmd_verify(args) -> int:
    print("\n=== STEP 1 — VERIFY SPLITS ===")
    for tag, path in [("dev (val.jsonl)", args.dev), ("test (test.jsonl)", args.test)]:
        if not os.path.exists(path):
            print(f"  MISSING: {tag} -> {path}")
            return 2
    dev_hashes  = _load_hashes(args.dev)
    test_hashes = _load_hashes(args.test)
    overlap = dev_hashes & test_hashes
    print(f"  dev  rows : {len(dev_hashes)}   ({args.dev})")
    print(f"  test rows : {len(test_hashes)}  ({args.test})")
    print(f"  overlap   : {len(overlap)}  (must be 0)")
    if overlap:
        print("  FAIL: dev and test share prompts. Re-split with a fixed seed.")
        return 3
    print("  OK: dev and test are disjoint.")
    # also assert that any prior cache, if present, was generated from these
    if os.path.exists(TEST_CACHE):
        print(f"  NOTE: a prior {os.path.basename(TEST_CACHE)} exists; will be reused if --reuse-cache")
    if os.path.exists(TEST_RESULTS):
        print(f"  WARNING: {os.path.basename(TEST_RESULTS)} already exists -- the test set has been touched once before.")
        print("           Re-running `test` will refuse without --force.")
    return 0


# --------------------------------------------------------------------------- #
# Step 2 — tune on dev                                                        #
# --------------------------------------------------------------------------- #
def cmd_tune(args) -> int:
    print("\n=== STEP 2 — TUNE ON DEVELOPMENT SET ONLY ===")
    print(f"  dev = {args.dev}    (test.jsonl is NOT loaded in this step)")
    if args.dev.endswith("test.jsonl"):
        print("  REFUSED: tune cannot run on test.jsonl.")
        return 2

    samples = _score_dataset(args.dev, DEV_CACHE, args.limit, args.threshold, args.base_url, note="(DEV)")

    # Operating-point sweeps. We sweep both threshold pairs and the two key
    # override gates (lg_soft_escalate_w, lg_hard_block_s) because the earlier
    # diagnostic showed thresholds alone are dormant when overrides dominate.
    print("\n  Sweeping operating points on DEV...")
    candidates = []
    for review_as_red in (False, True):
        for lg_se_w in (0.55, 0.60, 0.65, 0.70, 0.75):
            for lg_hb_s in (0.80, 0.85, 0.90):
                for lg_hb_w in (0.70, 0.80):
                    for tau_r in (0.30, 0.35, 0.40):
                        for tau_b in (0.55, 0.60, 0.70):
                            if tau_b <= tau_r:
                                continue
                            cfg = c3rf.FusionConfig(
                                tau_review=tau_r, tau_block=tau_b,
                                lg_soft_escalate_w=lg_se_w,
                                lg_hard_block_s=lg_hb_s,
                                lg_hard_block_w=lg_hb_w,
                            )
                            res = _score_at(samples, cfg, review_as_red)["fusion"]
                            candidates.append({
                                "review_as_red": review_as_red,
                                "tau_review": tau_r, "tau_block": tau_b,
                                "lg_soft_escalate_w": lg_se_w,
                                "lg_hard_block_s": lg_hb_s,
                                "lg_hard_block_w": lg_hb_w,
                                **res,
                            })
    # Pick three operating points to report
    def pick(label, key, filt=lambda r: True):
        pool = [c for c in candidates if filt(c)]
        pool.sort(key=key, reverse=True)
        return (label, pool[0]) if pool else (label, None)

    by_f1   = pick("Max F1 (no FP budget)",   lambda r: r["f1"])
    by_f1_b = pick("Max F1 (FP <= 60)",        lambda r: r["f1"], filt=lambda r: r["fp"] <= 60)
    by_recl = pick("Max recall (P >= 0.70)",  lambda r: r["recall"], filt=lambda r: r["precision"] >= 0.70)
    by_prec = pick("Max precision (R >= 0.60)",lambda r: r["precision"], filt=lambda r: r["recall"] >= 0.60)

    chosen_ops = [by_f1, by_f1_b, by_recl, by_prec]
    print("\n  Selected DEV operating points:")
    header = f"  {'label':28} review-as-red  tau_r tau_b  lg_se_w  lg_hb_s lg_hb_w   P     R     F1   FP  FN"
    print(header)
    for lab, op in chosen_ops:
        if op is None:
            print(f"  {lab:28} -- no candidate met filter --")
            continue
        print(f"  {lab:28} {str(op['review_as_red']):>5}        "
              f"{op['tau_review']:.2f} {op['tau_block']:.2f}   "
              f"{op['lg_soft_escalate_w']:.2f}    "
              f"{op['lg_hard_block_s']:.2f}   {op['lg_hard_block_w']:.2f}   "
              f"{op['precision']:.3f} {op['recall']:.3f} {op['f1']:.3f}  "
              f"{op['fp']:>3} {op['fn']:>3}")

    # Save dev results so the user can review before freezing
    dev_dump = {
        "dev_path": args.dev,
        "n_samples": len(samples),
        "selected_operating_points": [{"label": lab, "op": op} for lab, op in chosen_ops],
        "all_candidates": candidates,
    }
    with open(DEV_RESULTS, "w", encoding="utf-8") as f:
        json.dump(dev_dump, f, indent=2)
    print(f"\n  wrote {DEV_RESULTS}  ({len(candidates)} candidate ops)")
    print("\n  Next: run `python c3rf_protocol.py freeze <flags>` to commit a config.")
    return 0


# --------------------------------------------------------------------------- #
# Step 3 — freeze                                                             #
# --------------------------------------------------------------------------- #
def cmd_freeze(args) -> int:
    print("\n=== STEP 3 — FREEZE CONFIG ===")
    if os.path.exists(FROZEN_CONFIG) and not args.force:
        print(f"  REFUSED: {os.path.basename(FROZEN_CONFIG)} already exists.")
        print("  Pass --force to overwrite (only acceptable BEFORE test is run).")
        if os.path.exists(TEST_RESULTS):
            print(f"  AND {os.path.basename(TEST_RESULTS)} also exists -- overwriting frozen config")
            print("  after test has been run invalidates the test set. STOP.")
            return 2
        return 2

    cfg = c3rf.FusionConfig(
        alpha=args.alpha, beta=args.beta, gamma=args.gamma,
        delta=args.delta, eta=args.eta,
        tau_review=args.tau_review, tau_block=args.tau_block,
        inj_hard_block=args.inj_hard_block, inj_soft_escalate=args.inj_soft_escalate,
        severe_lg_block=args.severe_lg_block,
        lg_hard_block_s=args.lg_hard_block_s, lg_hard_block_w=args.lg_hard_block_w,
        lg_soft_escalate_s=args.lg_soft_escalate_s, lg_soft_escalate_w=args.lg_soft_escalate_w,
    )
    frozen = {
        "config": asdict(cfg),
        "treat_review_as_red": bool(args.review_as_red),
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "frozen_from_dev": os.path.basename(args.dev) if args.dev else "val.jsonl",
        "notes": args.notes or "",
    }
    with open(FROZEN_CONFIG, "w", encoding="utf-8") as f:
        json.dump(frozen, f, indent=2)
    print(f"  wrote {FROZEN_CONFIG}")
    print(json.dumps(frozen, indent=2))
    return 0


# --------------------------------------------------------------------------- #
# Step 4 — measure once on test                                               #
# --------------------------------------------------------------------------- #
def cmd_test(args) -> int:
    print("\n=== STEP 4 — MEASURE ONCE ON HELD-OUT TEST SET ===")
    if not os.path.exists(FROZEN_CONFIG):
        print(f"  REFUSED: no {os.path.basename(FROZEN_CONFIG)} on disk.")
        print("  Run `freeze` first.")
        return 2
    if os.path.exists(TEST_RESULTS) and not args.force:
        print(f"  REFUSED: {os.path.basename(TEST_RESULTS)} already exists.")
        print("  The held-out test set has already been measured once with the frozen config.")
        print("  Re-running invalidates the test set. Use --force only if you accept that.")
        return 2

    with open(FROZEN_CONFIG, "r", encoding="utf-8") as f:
        frozen = json.load(f)
    cfg = c3rf.FusionConfig(**frozen["config"])
    treat_review_as_red = frozen["treat_review_as_red"]
    print(f"  frozen config from {frozen['frozen_at']}  (dev={frozen['frozen_from_dev']})")
    print(f"  treat_review_as_red = {treat_review_as_red}")

    samples = _score_dataset(args.test, TEST_CACHE, args.limit, args.threshold,
                             args.base_url, note="(TEST)")

    # Score the same samples through all 4 systems for the comparison table.
    inj = s3.EvalCounts()
    org = s3.EvalCounts()
    fus = s3.EvalCounts()
    fus_block_only = s3.EvalCounts()      # frozen config but REVIEW NOT counted
    per_attack: dict[str, dict[str, s3.EvalCounts]] = {}

    for s in samples:
        inj.add(s.gt_unsafe, s.inj_red)
        org.add(s.gt_unsafe, s.inj_red or s.lg_unsafe)
        r = c3rf.fuse(s.s_I, s.s_L, s.category, s.rho, cfg)
        is_review = r.decision == "REVIEW"
        is_block  = r.decision == "BLOCK"
        red_combined = is_block or (treat_review_as_red and is_review)
        fus.add(s.gt_unsafe, red_combined)
        fus_block_only.add(s.gt_unsafe, is_block)
        d = per_attack.setdefault(s.attack, {
            "inj": s3.EvalCounts(), "or": s3.EvalCounts(),
            "fus": s3.EvalCounts(), "fus_bo": s3.EvalCounts(),
        })
        d["inj"].add(s.gt_unsafe, s.inj_red)
        d["or"].add(s.gt_unsafe, s.inj_red or s.lg_unsafe)
        d["fus"].add(s.gt_unsafe, red_combined)
        d["fus_bo"].add(s.gt_unsafe, is_block)

    # Decision-source audit: count how many BLOCKs came from overrides vs R>=tau_block
    override_counts: dict[str, int] = {}
    r_threshold_blocks = 0
    for s in samples:
        r = c3rf.fuse(s.s_I, s.s_L, s.category, s.rho, cfg)
        if r.override:
            override_counts[r.override] = override_counts.get(r.override, 0) + 1
        elif r.decision == "BLOCK":
            r_threshold_blocks += 1

    result = {
        "test_path": args.test,
        "n_samples": len(samples),
        "frozen_config": frozen,
        "measured_once_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "systems": {
            "injection_only": inj.report(),
            "or_gate":         org.report(),
            "c3rf_block_only": fus_block_only.report(),
            "c3rf_frozen":     fus.report(),
        },
        "per_attack": {
            atk: {k: v.report() for k, v in d.items()}
            for atk, d in sorted(per_attack.items())
        },
        "decision_source_audit": {
            "override_counts": override_counts,
            "r_threshold_blocks": r_threshold_blocks,
            "note": ("R-score threshold blocks vs override-driven blocks. "
                     "If r_threshold_blocks is 0, all BLOCKs came from override rules "
                     "and the continuous R score was dormant on this test set."),
        },
    }
    with open(TEST_RESULTS, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n  wrote {TEST_RESULTS}")
    cmd_report(args)
    return 0


# --------------------------------------------------------------------------- #
# Step 5 — report                                                             #
# --------------------------------------------------------------------------- #
def cmd_report(args) -> int:
    print("\n=== STEP 5 — REPORT (measured numbers only) ===")
    if not os.path.exists(TEST_RESULTS):
        print(f"  no {os.path.basename(TEST_RESULTS)} on disk -- run `test` first.")
        return 2
    with open(TEST_RESULTS, "r", encoding="utf-8") as f:
        r = json.load(f)
    print(f"  test set used exactly once on {r['measured_once_at']}")
    print(f"  parameters frozen at {r['frozen_config']['frozen_at']}  "
          f"(from {r['frozen_config']['frozen_from_dev']})")
    print(f"  n_samples = {r['n_samples']}\n")
    print("  HELD-OUT TEST METRICS (all systems on identical data):")
    for label, key in [("injection-only:", "injection_only"),
                       ("OR-gate       :", "or_gate"),
                       ("C3RF block-only:", "c3rf_block_only"),
                       ("C3RF frozen   :", "c3rf_frozen")]:
        print("  " + _summary_table(label, r["systems"][key]))

    print("\n  PER-ATTACK RECALL ON TEST SET (C3RF frozen):")
    for atk, d in r["per_attack"].items():
        rep = d["fus"]
        print(f"    {atk:30} n={rep['n']:<4} "
              f"P={rep['precision']:.3f} R={rep['recall']:.3f} F1={rep['f1']:.3f} "
              f"TP={rep['tp']:<3} FP={rep['fp']:<3} FN={rep['fn']:<3}")

    audit = r["decision_source_audit"]
    print("\n  DECISION-SOURCE AUDIT:")
    for k, v in sorted(audit["override_counts"].items(), key=lambda x: -x[1]):
        print(f"    override [{k}] : {v}")
    print(f"    R>=tau_block  blocks   : {audit['r_threshold_blocks']}")
    if audit["r_threshold_blocks"] == 0:
        print("    >>> R-score was DORMANT on the test set. All BLOCKs came from overrides.")
        print("        The system should be described as override-driven with a dormant")
        print("        continuous score, NOT as active continuous fusion.")
    else:
        print(f"    The continuous score contributed {audit['r_threshold_blocks']} BLOCKs.")
    return 0


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("verify")
    sp.add_argument("--dev",  default=DEFAULT_DEV)
    sp.add_argument("--test", default=DEFAULT_TEST)

    sp = sub.add_parser("tune")
    sp.add_argument("--dev",  default=DEFAULT_DEV)
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--threshold", type=float, default=0.5)
    sp.add_argument("--base-url", default="http://localhost:11434")

    sp = sub.add_parser("freeze")
    sp.add_argument("--dev",  default=DEFAULT_DEV)
    sp.add_argument("--review-as-red", action="store_true")
    sp.add_argument("--alpha", type=float, default=c3rf.FusionConfig.alpha)
    sp.add_argument("--beta",  type=float, default=c3rf.FusionConfig.beta)
    sp.add_argument("--gamma", type=float, default=c3rf.FusionConfig.gamma)
    sp.add_argument("--delta", type=float, default=c3rf.FusionConfig.delta)
    sp.add_argument("--eta",   type=float, default=c3rf.FusionConfig.eta)
    sp.add_argument("--tau-review",        type=float, default=c3rf.FusionConfig.tau_review)
    sp.add_argument("--tau-block",         type=float, default=c3rf.FusionConfig.tau_block)
    sp.add_argument("--inj-hard-block",    type=float, default=c3rf.FusionConfig.inj_hard_block)
    sp.add_argument("--inj-soft-escalate", type=float, default=c3rf.FusionConfig.inj_soft_escalate)
    sp.add_argument("--severe-lg-block",   type=float, default=c3rf.FusionConfig.severe_lg_block)
    sp.add_argument("--lg-hard-block-s",   type=float, default=c3rf.FusionConfig.lg_hard_block_s)
    sp.add_argument("--lg-hard-block-w",   type=float, default=c3rf.FusionConfig.lg_hard_block_w)
    sp.add_argument("--lg-soft-escalate-s",type=float, default=c3rf.FusionConfig.lg_soft_escalate_s)
    sp.add_argument("--lg-soft-escalate-w",type=float, default=c3rf.FusionConfig.lg_soft_escalate_w)
    sp.add_argument("--notes", default="")
    sp.add_argument("--force", action="store_true",
                    help="overwrite frozen config (forbidden if test already run)")

    sp = sub.add_parser("test")
    sp.add_argument("--test", default=DEFAULT_TEST)
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--threshold", type=float, default=0.5)
    sp.add_argument("--base-url", default="http://localhost:11434")
    sp.add_argument("--force", action="store_true",
                    help="overwrite prior test result (re-uses the test set; logs a warning)")

    sub.add_parser("report")

    args = ap.parse_args()
    return {"verify": cmd_verify, "tune": cmd_tune, "freeze": cmd_freeze,
            "test": cmd_test, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
