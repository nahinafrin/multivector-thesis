#!/usr/bin/env python3
"""
calibrate_tiers.py  —  pick tier boundaries and similarity thresholds from data
===============================================================================

Two jobs:
  1. From a controller results JSONL, compute the cumulative (noisy-OR) risk for
     benign vs attack rows and recommend LOW/MED and MED/HIGH cut points
     (benign p90/p95 for the low cut; attack median for the high cut).
  2. From the saved retrieval similarity scores, show how many chunks SURVIVE at
     each candidate sim_threshold, per tier — so you do not starve high-risk
     queries to zero documents.

USAGE:
    python calibrate_tiers.py --controller-jsonl grounded_controller.jsonl
"""
from __future__ import annotations
import argparse, json
from statistics import median

try:
    import numpy as np
    def pct(x, q): return float(np.percentile(x, q)) if x else 0.0
except Exception:
    def pct(x, q):
        if not x: return 0.0
        xs = sorted(x); i = min(len(xs)-1, int(round((q/100)*(len(xs)-1))))
        return float(xs[i])

from adaptive_risk import noisy_or

ATTACK_KINDS = {"adversarial_query", "gate_slip_query", "poisoned_context", "multivector_attack"}


def cum_from_row(r):
    ch = r.get("multivector", {}).get("channels", {})
    contrib = {
        "prompt_injection": float(ch.get("query_vector", 0) or 0),
        "context_suspicion": float(ch.get("context_vector", 0) or 0),
        "semantic_intent": float(r.get("meta", {}).get("semantic_intent", 0) or 0),
        "ensemble_disagreement": float(r.get("scores", {}).get("ensemble_disagreement", 0) or 0),
    }
    return noisy_or(contrib)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller-jsonl", required=True)
    ap.add_argument("--sim-grid", nargs="+", type=float,
                    default=[0.45, 0.55, 0.65, 0.72, 0.80, 0.85])
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.controller_jsonl, encoding="utf-8") if l.strip()]
    benign = [cum_from_row(r) for r in rows if r.get("kind") == "benign_control"]
    attack = [cum_from_row(r) for r in rows if r.get("kind") in ATTACK_KINDS]

    print("=== cumulative-risk distribution ===")
    print(f"benign  n={len(benign)}  p50={pct(benign,50):.3f}  p90={pct(benign,90):.3f}  p95={pct(benign,95):.3f}")
    print(f"attack  n={len(attack)}  p10={pct(attack,10):.3f}  p50={pct(attack,50):.3f}  p90={pct(attack,90):.3f}")
    low_max = round(pct(benign, 95), 3)
    med_max = round(median([pct(benign, 95), pct(attack, 50)]), 3)
    print(f"\nrecommended TIER_BOUNDS = {{'low_max': {low_max}, 'medium_max': {med_max}}}")
    if low_max >= med_max:
        print("  [warn] benign p95 >= attack median: classes overlap on cumulative "
              "risk; tiers will be noisy. Scale the slice or improve contributions.")

    print("\n=== retrieval survival per sim_threshold ===")
    print("(mean chunks kept across all rows; watch for HIGH-tier starvation)")
    all_scores = []
    for r in rows:
        sc = r.get("retrieval", {}).get("retrieval_scores") or r.get("meta", {}).get("retrieval_scores") or []
        if sc:
            all_scores.append([float(s) for s in sc])
    if not all_scores:
        print("  no per-row retrieval_scores saved; re-run with scores persisted.")
        return
    for thr in args.sim_grid:
        kept = [sum(1 for s in row if s >= thr) for row in all_scores]
        starved = sum(1 for k in kept if k == 0)
        print(f"  sim>={thr:.2f}:  mean kept={sum(kept)/len(kept):.2f}   "
              f"rows with 0 chunks={starved}/{len(kept)}")


if __name__ == "__main__":
    main()
