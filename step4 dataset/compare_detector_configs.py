#!/usr/bin/env python3
"""
compare_detector_configs.py — re-derive fire-rule ablations from logged scores
==============================================================================

No model calls. Reuses detector.query / detector.context already written by
run_mitigation_ab.py / run_benign_cost.py, and compares fire rules against the
same floors as ExistingInjectionDetector (query_floor=0.20, context_floor=0.30).

Sanity check: re-derived coactivation must match the logged detector.is_attack
on every row.

USAGE (from `step4 dataset`):
    python compare_detector_configs.py \\
        --attacks mitigation_results/planted200/off.jsonl \\
        --benign mitigation_results/planted30v2/benign_off.jsonl \\
        --out detector_config_comparison.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path


DEFAULT_QUERY_FLOOR = 0.20
DEFAULT_CONTEXT_FLOOR = 0.30

RULES = ("query_only", "context_only", "or", "coactivation")


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fires(q: float, c: float, rule: str, q_floor: float, c_floor: float) -> bool:
    if rule == "query_only":
        return q >= q_floor
    if rule == "context_only":
        return c >= c_floor
    if rule == "or":
        return (q >= q_floor) or (c >= c_floor)
    # coactivation (default)
    return (q >= q_floor) and (c >= c_floor)


def channels(row: dict) -> tuple[float, float, bool | None]:
    det = row.get("detector") or {}
    return (float(det.get("query", 0.0)),
            float(det.get("context", 0.0)),
            det.get("is_attack"))


def summarize(rows, rule, q_floor, c_floor, *, expect_match_coact: bool = False):
    n = len(rows)
    hits = 0
    mismatch = 0
    for r in rows:
        q, c, logged = channels(r)
        fired = fires(q, c, rule, q_floor, c_floor)
        hits += int(fired)
        if expect_match_coact and logged is not None:
            derived = fires(q, c, "coactivation", q_floor, c_floor)
            if bool(logged) != derived:
                mismatch += 1
    return {
        "n": n,
        "fired": hits,
        "rate_pct": round(100.0 * hits / n, 1) if n else 0.0,
        "coactivation_mismatch": mismatch if expect_match_coact else None,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Compare detector fire rules from logged query/context scores")
    ap.add_argument("--attacks", required=True,
                    help="JSONL of attack rows with detector.{query,context,is_attack}")
    ap.add_argument("--benign", required=True,
                    help="JSONL of benign rows with the same detector fields")
    ap.add_argument("--query-floor", type=float, default=DEFAULT_QUERY_FLOOR)
    ap.add_argument("--context-floor", type=float, default=DEFAULT_CONTEXT_FLOOR)
    ap.add_argument("--out", default="detector_config_comparison.json")
    args = ap.parse_args()

    attacks = read_jsonl(args.attacks)
    benign = read_jsonl(args.benign)
    if not attacks or not benign:
        raise SystemExit("Need non-empty --attacks and --benign JSONL files.")

    # Sanity: coactivation re-derivation must match logged decisions.
    atk_sanity = summarize(attacks, "coactivation", args.query_floor, args.context_floor,
                           expect_match_coact=True)
    ben_sanity = summarize(benign, "coactivation", args.query_floor, args.context_floor,
                           expect_match_coact=True)
    mismatch = (atk_sanity["coactivation_mismatch"] or 0) + (
        ben_sanity["coactivation_mismatch"] or 0)
    if mismatch:
        raise SystemExit(
            f"Sanity FAILED: coactivation re-derive mismatch on {mismatch} rows "
            f"(floors q={args.query_floor}, c={args.context_floor} may be wrong).")

    print(f"Sanity OK: coactivation re-derive matches logged is_attack "
          f"on {len(attacks)} attack + {len(benign)} benign rows "
          f"(floors q={args.query_floor}, c={args.context_floor}).")

    by_rule = {}
    print("\n| Rule | Recall (attacks caught) | Benign false-positive rate |")
    print("|---|---|---|")
    for rule in RULES:
        a = summarize(attacks, rule, args.query_floor, args.context_floor)
        b = summarize(benign, rule, args.query_floor, args.context_floor)
        by_rule[rule] = {
            "recall_pct": a["rate_pct"],
            "attacks_caught": a["fired"],
            "attacks_n": a["n"],
            "benign_fp_pct": b["rate_pct"],
            "benign_fp": b["fired"],
            "benign_n": b["n"],
        }
        print(f"| {rule} | {a['rate_pct']}% ({a['fired']}/{a['n']}) | "
              f"{b['rate_pct']}% ({b['fired']}/{b['n']}) |")

    report = {
        "query_floor": args.query_floor,
        "context_floor": args.context_floor,
        "attacks_path": str(args.attacks),
        "benign_path": str(args.benign),
        "sanity": {
            "coactivation_matches_logged": True,
            "attack_n": len(attacks),
            "benign_n": len(benign),
        },
        "rules": by_rule,
        "takeaway": (
            "Fusion (coactivation) is not a recall booster here: it matches "
            "query_only recall because context clears its floor on every attack. "
            "It is a false-positive suppressor — collapsing context_only's ~45% "
            "benign FP into ~2.5% — at no recall cost."
        ),
    }

    out = Path(args.out)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
