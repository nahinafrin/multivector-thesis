"""
recalibrate_semantic_gate.py

Properly recalibrates step_03d's NLI backend using a real, sizeable
distribution -- same percentile-based methodology already used (and
validated) for the multivector detector's channel floors earlier in this
project, applied here instead of trusting a 4-example demo.

Benign set: real QA questions from rag-mini-wikipedia via
kb_rag_mini_wikipedia.load_qa() -- genuinely diverse, not synthetic.

Attack set: your own existing adversarial_query rows from
grounded_controller.jsonl -- already-vetted jailbreak/injection-framed
content, not newly invented examples.

Usage:
    python recalibrate_semantic_gate.py
"""
from __future__ import annotations

import json

from step_03d_semantic_intent_gate import _score_nli, REVIEW_AT, BLOCK_AT
from kb_rag_mini_wikipedia import load_qa


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main() -> None:
    print("[1/3] loading benign questions (rag-mini-wikipedia QA set)...")
    qa = load_qa(limit=300)
    benign_questions = [q["question"] for q in qa]

    print("[2/3] loading attack questions (your own adversarial_query rows)...")
    rows = [json.loads(l) for l in open("grounded_controller.jsonl", encoding="utf-8")]
    attack_questions = [r["question"] for r in rows if r.get("kind") == "adversarial_query"]
    print(f"  benign n={len(benign_questions)}   attack n={len(attack_questions)}")

    print("[3/3] scoring all questions through the NLI backend (no Ollama needed)...")
    benign_scores = [_score_nli(q)[0] for q in benign_questions]
    attack_scores = [_score_nli(q)[0] for q in attack_questions]

    b_p95 = pct(benign_scores, 95)
    b_p99 = pct(benign_scores, 99)
    a_p10 = pct(attack_scores, 10)

    print(f"\nbenign intent scores: p50={pct(benign_scores,50):.3f}  "
          f"p95={b_p95:.3f}  p99={b_p99:.3f}  max={max(benign_scores):.3f}")
    print(f"attack intent scores:  min={min(attack_scores):.3f}  "
          f"p10={a_p10:.3f}  p50={pct(attack_scores,50):.3f}")

    separation = a_p10 - b_p95
    print(f"\nseparation (attack_p10 - benign_p95): {separation:.3f}  "
          f"{'SEPARATED -- recalibration viable' if separation > 0 else 'STILL OVERLAPPING'}")

    old_flagged = sum(1 for s in benign_scores if s >= REVIEW_AT)
    print(f"\n(reference: {old_flagged}/{len(benign_scores)} of these SAME benign "
          f"questions would be flagged under the CURRENT thresholds "
          f"REVIEW_AT={REVIEW_AT}, BLOCK_AT={BLOCK_AT})")

    if separation > 0:
        new_review_at = round(b_p95 + separation * 0.3, 3)
        new_block_at = round(b_p95 + separation * 0.7, 3)
        print(f"\nProposed new thresholds (from n={len(benign_questions)+len(attack_questions)} "
              f"real examples, not 4):")
        print(f"  REVIEW_AT = {new_review_at}")
        print(f"  BLOCK_AT  = {new_block_at}")
    else:
        print("\nNo threshold separates these distributions cleanly even at this "
              "larger sample size. This is a STRONGER, more defensible negative "
              "finding than the original 4-example demo failure -- worth reporting "
              "as-is. Recommend keeping the gate disabled for now and either citing "
              "this larger-scale calibration attempt as evidence, or trying "
              "backend='llm' as a follow-up instead of tuning the NLI backend further.")


if __name__ == "__main__":
    main()
