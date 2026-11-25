"""
step_07_context_ranking.py  —  STEP 7: Context Ranking (ADAPTIVE)
=================================================================

CHANGES vs the original:
  RISK-ADAPTIVE min_score. A chunk must now clear a relevance floor that
  tightens under high input risk:
      input_risk > 0.60  ->  min_score 0.20 -> 0.40
  At least one chunk is always retained so Step 8/9 is never stranded. The
  original kept a fixed top_n and never read the risk score, so the rerank
  layer was not part of the adaptive cascade at all.

Persists rerank_min_score to meta so the JSONL shows the adaptive value per row.

Install: pip install sentence-transformers
Run standalone:
    python step_07_context_ranking.py --demo
"""

from __future__ import annotations

import argparse
import math

from pipeline_common import PipelineState

_RERANKER = None
_RERANKER_NAME = "BAAI/bge-reranker-v2-m3"

BASE_MIN_SCORE = 0.20
STRICT_MIN_SCORE = 0.40
RISK_TIGHTEN_AT = 0.60


def _get_reranker(name: str = _RERANKER_NAME):
    global _RERANKER
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder
        _RERANKER = CrossEncoder(name)
    return _RERANKER


def _sigmoid(x: float) -> float:
    if x < -60:
        return 0.0
    if x > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def rerank(query: str, chunks: list[str], min_score: float
           ) -> list[tuple[str, float]]:
    if not chunks:
        return []
    model = _get_reranker()
    raw = model.predict([(query, c) for c in chunks])
    scored = sorted(((_sigmoid(float(s)), c) for s, c in zip(raw, chunks)),
                    key=lambda t: t[0], reverse=True)
    kept = [(s, c) for s, c in scored if s >= min_score]
    if not kept:                       # never strand the generator
        kept = [scored[0]]
    return [(c, s) for s, c in kept]


def run(state: PipelineState, top_n: int | None = None) -> PipelineState:
    if state.blocked or not state.context:
        return state

    risk = state.input_risk()
    pinned = state.meta.get("tier_rerank_min_score")
    min_score = float(pinned) if pinned is not None else (
        STRICT_MIN_SCORE if risk > RISK_TIGHTEN_AT else BASE_MIN_SCORE)

    ranked = rerank(state.prompt, state.context, min_score)
    if top_n:
        ranked = ranked[:top_n]

    state.ranked_context = [t for t, _ in ranked]
    state.meta["rerank_scores"] = [round(s, 4) for _, s in ranked]
    state.meta["rerank_min_score"] = min_score
    state.log("step_07_context_ranking",
              input_risk=round(risk, 4), min_score=min_score,
              in_chunks=len(state.context), out_chunks=len(ranked))
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 7: Context Ranking (adaptive)")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        for risk in (0.04, 0.73):
            st = PipelineState(prompt="how do I enable two-factor authentication")
            st.scores["fusion_risk"] = risk
            st.context = [
                "Our refund policy allows returns within 30 days.",
                "Two-factor authentication is enabled under Settings > Security.",
                "Business hours are 9am-5pm Monday to Friday.",
                "Enable 2FA by scanning the QR code in your Security settings.",
            ]
            st = run(st)
            print(f"\nrisk={risk} -> min_score={st.meta['rerank_min_score']} "
                  f"kept={len(st.ranked_context)}")
            for t, s in zip(st.ranked_context, st.meta["rerank_scores"]):
                print(f"  {s:.3f}  {t[:60]}")
    else:
        print("Use --demo, or call run() in the orchestrator.")


if __name__ == "__main__":
    main()
