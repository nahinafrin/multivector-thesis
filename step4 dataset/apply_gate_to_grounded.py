"""
apply_gate_to_grounded.py — fold the front-end gate into grounded.jsonl.

Joins gate_scores.jsonl (from `run_steps_1_3_over_rag.py`) onto grounded.jsonl
by `index`, stamps the upstream `gate` field on every row, and RECOMPUTES the
Step 10 dynamic threshold + passed flag using the real `risk_score` instead
of the 0.0 placeholder that produced the original numbers.

This is a re-scoring pass only — the lex/model faithfulness components don't
depend on risk, so we don't have to re-run the slow bge-reranker. The
threshold and the pass/fail decision are the only fields that move.

Empirical FP-absorption note (rag-mini, 918 rows):
    The toxicity scanner produced 6 false positives on benign Wikipedia
    queries due to context-free lexical matching (triggers were words
    like "damn", "garbage", "bitches" in their biological/idiomatic
    senses) — the same FP/recall tension documented in the Step 3
    C3RF analysis. Critically, the adaptive cascade absorbed 5 of those
    6 without downstream impact; only 1 row in 918 (~0.1%) flipped from
    PASS to FAIL after risk was applied. This is the empirical evidence
    for the cascade's resilience to upstream false positives.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from step_10_grounding_judge import compute_grounding_threshold


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply Step 1-3 gate to grounded.jsonl.")
    ap.add_argument("--grounded", default="./grounded.jsonl")
    ap.add_argument("--gate", default="./gate_scores.jsonl")
    ap.add_argument("--out", default=None,
                    help="Default: overwrite --grounded after .bak backup.")
    args = ap.parse_args()

    grounded = Path(args.grounded)
    gate_path = Path(args.gate)

    # Load gate scores into an index -> dict map.
    gate_by_idx: dict[int, dict] = {}
    with open(gate_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            gate_by_idx[int(rec["index"])] = rec["gate"]
    print(f"[apply] loaded gate scores for {len(gate_by_idx)} rows")

    # Read grounded.
    rows = [json.loads(l) for l in open(grounded, encoding="utf-8") if l.strip()]

    backup: Path | None = None
    if args.out:
        dst = Path(args.out)
    else:
        backup = grounded.with_suffix(grounded.suffix + ".bak")
        shutil.copyfile(grounded, backup)
        dst = grounded

    threshold_dist_before: Counter[float] = Counter()
    threshold_dist_after: Counter[float] = Counter()
    pass_before = 0
    pass_after = 0
    moved_pass_to_fail = 0
    moved_fail_to_pass = 0
    risk_bands = Counter()

    with open(dst, "w", encoding="utf-8") as fout:
        for row in rows:
            idx = int(row.get("index", -1))
            gate = gate_by_idx.get(idx, {})
            row["gate"] = gate

            g = row.get("grounding", {})
            lex = float(g.get("lexical_score", 0.0))
            mdl = float(g.get("model_score", 0.0))
            faith = float(g.get("faithfulness", max(lex, mdl)))
            disagreement = float(row.get("disagreement", 0.0))

            old_thr = float(g.get("threshold_used", 0.70))
            old_pass = bool(g.get("passed", False))
            threshold_dist_before[round(old_thr, 2)] += 1
            pass_before += int(old_pass)

            risk = float(gate.get("risk_score", 0.0))
            if risk < 0.2: risk_bands["<0.2"] += 1
            elif risk < 0.4: risk_bands["0.2-0.4"] += 1
            elif risk < 0.6: risk_bands["0.4-0.6"] += 1
            elif risk < 0.8: risk_bands["0.6-0.8"] += 1
            else: risk_bands[">=0.8"] += 1

            new_thr = compute_grounding_threshold(risk, disagreement)
            new_pass = bool(faith >= new_thr)
            threshold_dist_after[round(new_thr, 2)] += 1
            pass_after += int(new_pass)

            if old_pass and not new_pass:
                moved_pass_to_fail += 1
            elif not old_pass and new_pass:
                moved_fail_to_pass += 1

            g["threshold_used"] = new_thr
            g["passed"] = new_pass
            g["risk_in"] = risk
            g["disagreement_in"] = disagreement
            row["grounding"] = g

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n[apply] rewrote {len(rows)} rows -> {dst}")
    if backup:
        print(f"[apply] backup at {backup}")

    print("\nRisk-score distribution (gate.risk_score):")
    for band in ["<0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", ">=0.8"]:
        print(f"  {band:8s} : {risk_bands[band]}")

    print("\nThreshold distribution BEFORE (risk_in was 0.0 everywhere):")
    for thr in sorted(threshold_dist_before):
        print(f"  {thr:.2f} : {threshold_dist_before[thr]}")
    print("Threshold distribution AFTER (with real gate.risk_score):")
    for thr in sorted(threshold_dist_after):
        print(f"  {thr:.2f} : {threshold_dist_after[thr]}")

    n = len(rows)
    print(f"\nPass-rate BEFORE: {pass_before}/{n} = {pass_before/n:.3f}")
    print(f"Pass-rate AFTER : {pass_after}/{n} = {pass_after/n:.3f}")
    print(f"  rows that flipped PASS -> FAIL (threshold raised under risk): "
          f"{moved_pass_to_fail}")
    print(f"  rows that flipped FAIL -> PASS: "
          f"{moved_fail_to_pass}")


if __name__ == "__main__":
    main()
