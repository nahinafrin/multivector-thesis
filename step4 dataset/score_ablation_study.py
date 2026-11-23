#!/usr/bin/env python3
"""
score_ablation_study.py -- table of ASR and benign cost per mitigation configuration.

Reads every configuration found in mitigation_results/ablation200/ (current code), pairs it with the
current-code undefended arm in mitigation_results/followup200/ (see BASELINES), and lists
earlier planted200 runs (older code) separately as cross-checks.
Attack ASR uses score_attack_success.attack_succeeded; benign correctness uses
score_benign_cost.answered_correctly. Wilson 95% CIs, exact two-sided McNemar vs the
undefended arm. Partial results are fine; missing runs are listed as "not run".

USAGE:  python score_ablation_study.py            -> prints markdown, writes ablation_table.json
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from score_attack_success import _row_verdict


def attack_succeeded(row):
    # Same definition as score_mitigation_ab.py: a row counts as a success only if it was
    # NOT neutralized (blocked rows are neutralized even if a canary flag is set).
    return not _row_verdict(row)["attack_neutralized"]


from score_benign_cost import answered_correctly

MR = HERE / "mitigation_results"
AB = MR / "ablation200"

CONFIGS = [  # (label, description, attack path, benign path)
    ("full", "all seven layers", AB / "on_full.jsonl", AB / "benign_on.jsonl"),
    ("noprompt", "all minus guarded prompt", AB / "on_noprompt.jsonl", AB / "benign_on_noprompt.jsonl"),
    ("norefuse", "all minus refuse-on-detection", AB / "on_norefuse.jsonl", AB / "benign_on_norefuse.jsonl"),
    ("nogrounding", "all minus grounding gate", AB / "on_nogrounding.jsonl", AB / "benign_on_nogrounding.jsonl"),
    ("nosanitize", "all minus sanitization", AB / "on_nosanitize.jsonl", AB / "benign_on_nosanitize.jsonl"),
    ("notighten", "all minus tightened rerank", AB / "on_notighten.jsonl", AB / "benign_on_notighten.jsonl"),
    ("nopresidio", "all minus Presidio masking", AB / "on_nopresidio.jsonl", AB / "benign_on_nopresidio.jsonl"),
    ("nodlp", "all minus DLP", AB / "on_nodlp.jsonl", AB / "benign_on_nodlp.jsonl"),
    ("only_refuse", "refuse-on-detection only", AB / "on_only_refuse.jsonl", AB / "benign_on_only_refuse.jsonl"),
    ("only_grounding", "grounding gate only", AB / "on_only_grounding.jsonl", AB / "benign_on_only_grounding.jsonl"),
    ("only_sanitize", "sanitization only", AB / "on_only_sanitize.jsonl", AB / "benign_on_only_sanitize.jsonl"),
    ("only_tighten", "tightened rerank only", AB / "on_only_tighten.jsonl", AB / "benign_on_only_tighten.jsonl"),
    ("only_prompt", "guarded prompt only", AB / "on_only_prompt.jsonl", AB / "benign_on_only_prompt.jsonl"),
    ("only_presidio", "Presidio masking only", AB / "on_only_presidio.jsonl", AB / "benign_on_only_presidio.jsonl"),
    ("only_dlp", "DLP only", AB / "on_only_dlp.jsonl", AB / "benign_on_only_dlp.jsonl"),
]
# Cross-checks: runs made BEFORE the Sep-22 pipeline/scorer fixes and before the presidio layer.
OLD = [
    ("old: full (6 layers)", MR / "planted200" / "on_full.jsonl", MR / "planted200_recheck" / "benign_on.jsonl"),
    ("old: norefuse", MR / "planted200" / "on_norefuse.jsonl", None),
    ("old: groundonly", MR / "planted200" / "on_groundonly.jsonl", None),
    ("old: groundplus", MR / "planted200" / "on_groundplus.jsonl", MR / "planted200_groundplus_recheck" / "benign_on_groundplus.jsonl"),
    ("old: refuseonly", MR / "planted200" / "on_refuseonly.jsonl", MR / "planted200" / "benign_on_refuseonly.jsonl"),
]


def rows(p):
    if not p.is_file():
        return None
    return {str(r.get("index")): r for r in (json.loads(l) for l in open(p, encoding="utf-8", errors="replace") if l.strip())}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * p, 100 * max(0, c - h), 100 * min(1, c + h))


def exact_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def ci(k, n):
    p, lo, hi = wilson(k, n)
    return f"{p:.1f}% [{lo:.1f}, {hi:.1f}]"


# Canonical undefended arm for every paired comparison in the paper (Table VII, Sec. V-E):
# followup200/off.jsonl and benign_off.jsonl, re-run on the CURRENT code. ablation200/off.jsonl
# is the July undefended arm from older code (61/200 succeed vs 59/200); it is kept only for
# the historical cross-check (--baseline july) and must not be paired with current-code arms.
BASELINES = {"current": (MR / "followup200" / "off.jsonl", MR / "followup200" / "benign_off.jsonl"),
             "july": (AB / "off.jsonl", AB / "benign_off.jsonl")}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", choices=sorted(BASELINES), default="current",
                    help="undefended arm to pair against (default: current = followup200, as in the paper)")
    args = ap.parse_args()
    pa_off, pb_off = BASELINES[args.baseline]
    off_a, off_b = rows(pa_off), rows(pb_off)
    print(f"[baseline] {args.baseline}: {pa_off.relative_to(HERE)}, {pb_off.relative_to(HERE)}")
    n_a = len(off_a); asr_off = sum(attack_succeeded(r) for r in off_a.values())
    n_b = len(off_b); ok_off = sum(answered_correctly(r) for r in off_b.values())
    print(f"Undefended: ASR {asr_off}/{n_a} = {ci(asr_off, n_a)};  benign correct {ok_off}/{n_b} = {ci(ok_off, n_b)}\n")
    print("| Configuration | ASR (95% CI) | fixed / broke | exact p | Benign blocked (95% CI) | Benign correct (95% CI) |")
    print("|---|---|---|---|---|---|")
    out = {}
    for label, desc, pa, pb in CONFIGS:
        A, B = rows(pa), rows(pb)
        if A is None and B is None:
            print(f"| {desc} (`{label}`) | not run | | | not run | not run |"); continue
        cell_a = cell_fb = cell_p = "not run"
        if A is not None:
            ids = sorted(set(off_a) & set(A)); n = len(ids)
            so = [attack_succeeded(off_a[i]) for i in ids]; sn = [attack_succeeded(A[i]) for i in ids]
            fixed = sum(a and not b for a, b in zip(so, sn)); broke = sum(b and not a for a, b in zip(so, sn))
            cell_a = f"{ci(sum(sn), n)} (n={n})"; cell_fb = f"{fixed} / {broke}"; cell_p = f"{exact_p(broke, fixed):.3g}"
            out[label] = dict(n=n, asr=sum(sn), fixed=fixed, broke=broke,
                              p=exact_p(broke, fixed))
        cell_blk = cell_ok = "not run"
        if B is not None:
            ids = sorted(set(off_b) & set(B)); n = len(ids)
            blk = sum(bool(B[i].get("blocked")) for i in ids); ok = sum(answered_correctly(B[i]) for i in ids)
            cell_blk = ci(blk, n); cell_ok = ci(ok, n)
            out.setdefault(label, {}).update(benign_n=n, benign_blocked=blk, benign_correct=ok)
        print(f"| {desc} (`{label}`) | {cell_a} | {cell_fb} | {cell_p} | {cell_blk} | {cell_ok} |")
    print("\nCross-check: earlier runs (older pipeline code; no Presidio layer; do not compare their rows to the table above as a controlled contrast)")
    print("| Earlier run | ASR (95% CI) | fixed / broke | exact p | Benign blocked | Benign correct |")
    print("|---|---|---|---|---|---|")
    for desc, pa, pb in OLD:
        A, B = rows(pa), (rows(pb) if pb else None)
        if A is None:
            print(f"| {desc} | not found | | | | |"); continue
        ids = sorted(set(off_a) & set(A)); n = len(ids)
        so = [attack_succeeded(off_a[i]) for i in ids]; sn = [attack_succeeded(A[i]) for i in ids]
        fixed = sum(a and not b for a, b in zip(so, sn)); broke = sum(b and not a for a, b in zip(so, sn))
        if B is not None:
            bi = sorted(set(off_b) & set(B)); blk = sum(bool(B[i].get("blocked")) for i in bi); ok = sum(answered_correctly(B[i]) for i in bi)
            cb, cc = ci(blk, len(bi)), ci(ok, len(bi))
        else:
            cb = cc = "n/a"
        print(f"| {desc} | {ci(sum(sn), n)} (n={n}) | {fixed} / {broke} | {exact_p(broke, fixed):.3g} | {cb} | {cc} |")
    out["_baseline"] = {"name": args.baseline, "attack": str(pa_off.relative_to(HERE)),
                        "benign": str(pb_off.relative_to(HERE)), "undefended_asr": asr_off, "n": n_a}
    name = "ablation_table.json" if args.baseline == "current" else f"ablation_table_{args.baseline}.json"
    (HERE / name).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\nRead leave-one-out rows against `full`: a layer matters if removing it RAISES ASR (or lowers benign blocking at no ASR cost).")
    print("Read add-one-in rows against the undefended arm. Single seed; treat differences inside the CIs as noise.")


if __name__ == "__main__":
    main()
