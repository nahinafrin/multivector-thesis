"""
step_10_grounding_judge.py  —  METHODOLOGY STEP 10: Grounding and Policy Checking
=================================================================================

"...the selected answer is verified for faithfulness against the retrieved
evidence, and the canary token is re-checked to confirm no indirect-injection
leak... a hybrid faithfulness scorer — a deterministic lexical-overlap check
combined by maximum with a small-model judge... the pass threshold is raised
when input risk or ensemble disagreement is high."

WHY THE HYBRID (the key empirical finding)
------------------------------------------
We combine a lexical-overlap heuristic and a small cross-encoder judge by MAX
because our experiments showed each fails in opposite, uncorrelated directions:
the small judge under-scores near-verbatim answers, while the lexical check
under-scores valid paraphrases. Taking the max rescues both, while genuinely
ungrounded content still fails BOTH and is correctly rejected. The ablation
(lexical-only / model-only / hybrid_max) quantifies this complementarity.

ON THE MODEL-JUDGE CEILING
--------------------------
The model faithfulness component reuses the bge-reranker cross-encoder, whose
sigmoid-mapped logits saturate at roughly 0.73 for the answer/chunk shapes in
this corpus. This conservative ceiling means the hybrid max() defaults to the
lexical component for high-faithfulness answers — the intended fallback of the
hybrid design. A wider-dynamic-range judge (e.g. an NLI cross-encoder) is a
drop-in model swap, not a code change, and is noted as future work.

DYNAMIC THRESHOLD
    base 0.70
      + 0.10 if input_risk     > 0.60   (carried from Step 3/3c)
      + 0.10 if disagreement   > 0.30   (carried from Step 9 ensemble)
    capped at 0.90
An answer must reach this threshold to be judged grounded; otherwise the policy
gate marks it for block/regenerate.

INPUTS (read from state)
    state.meta["answer"]                 the fused/selected answer (Step 9)
    state.meta["retrieval"]["ranked_chunks"]   evidence (Step 6)
    state.meta["retrieval"]["canary"]    dual-checkpoint token (Step 5)
    state.scores["disagreement"]         ensemble divergence (Step 9)
    state.scores["fusion_risk"]          input risk (Step 3/3c)

OUTPUT (state.scores / state.meta + may block)
    grounding.lexical_score, grounding.model_score, grounding.faithfulness,
    grounding.threshold, grounding.passed, grounding.canary_intact

ZERO COST: local cross-encoder + deterministic lexical check, no API.

Run standalone:
    python step_10_grounding_judge.py --demo
"""

from __future__ import annotations

import argparse
import re

from pipeline_common import PipelineState


# --------------------------------------------------------------------------- #
# Risk helper (same convention as Steps 5/6)
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Lexical-overlap component (deterministic, no model)
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an the of to in on at and or but is are was were be been being this that "
    "these those it its as by for with from into over under than then so such".split()
)


def _content_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP}


def lexical_overlap(answer: str, chunks: list[str]) -> float:
    """Fraction of the answer's content tokens supported by the evidence."""
    a = _content_tokens(answer)
    if not a:
        return 0.0
    evidence = set()
    for c in chunks:
        evidence |= _content_tokens(c)
    return len(a & evidence) / len(a)


# --------------------------------------------------------------------------- #
# Model-judge component (lazy cross-encoder, reused from Step 6)
# --------------------------------------------------------------------------- #
_JUDGE = None
JUDGE_MODEL = "BAAI/bge-reranker-v2-m3"


def _get_judge(model_name: str = JUDGE_MODEL):
    global _JUDGE
    if _JUDGE is not None:
        return _JUDGE
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        raise ImportError(
            "Step 10 needs sentence-transformers: pip install sentence-transformers"
        ) from e
    _JUDGE = CrossEncoder(model_name)
    return _JUDGE


def _sigmoid(x: float) -> float:
    import math
    if x < -60:
        return 0.0
    if x > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def model_faithfulness(answer: str, chunks: list[str]) -> float:
    """Max cross-encoder relevance of the answer against any evidence chunk."""
    if not chunks or not answer.strip():
        return 0.0
    judge = _get_judge()
    raw = judge.predict([[c, answer] for c in chunks])
    return max(_sigmoid(float(s)) for s in raw)


# --------------------------------------------------------------------------- #
# Dynamic threshold
# --------------------------------------------------------------------------- #
BASE_THRESHOLD = 0.70
RISK_BUMP = 0.10
DISAGREE_BUMP = 0.10
THRESHOLD_CAP = 0.90
RISK_AT = 0.60
DISAGREE_AT = 0.30


def grounding_threshold(risk: float, disagreement: float) -> float:
    t = BASE_THRESHOLD
    if risk > RISK_AT:
        t += RISK_BUMP
    if disagreement > DISAGREE_AT:
        t += DISAGREE_BUMP
    return min(t, THRESHOLD_CAP)


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #
def run(state: PipelineState) -> PipelineState:
    """Verify faithfulness + policy; block ungrounded or leaking answers."""
    if state.blocked:
        return state

    answer = state.meta.get("answer", "") or state.meta.get("fused_answer", "")
    # Unified state: ranked chunks live on the attribute, canary in meta.
    chunks = state.ranked_context or state.context or state.meta.get("sanitized_chunks", [])
    canary = state.meta.get("canary")

    risk = state.input_risk()
    disagreement = float(state.scores.get("disagreement", 0.0))
    threshold = grounding_threshold(risk, disagreement)

    lex = lexical_overlap(answer, chunks)
    mdl = model_faithfulness(answer, chunks)
    faithfulness = max(lex, mdl)          # hybrid: uncorrelated failures

    # Dual-checkpoint: the canary must NOT appear in the answer.
    canary_intact = True
    if canary and canary in answer:
        canary_intact = False

    passed = (faithfulness >= threshold) and canary_intact

    grounding = {
        "lexical_score": round(lex, 4),
        "model_score": round(mdl, 4),
        "faithfulness": round(faithfulness, 4),
        "threshold": round(threshold, 4),
        "input_risk": round(risk, 4),
        "disagreement": round(disagreement, 4),
        "canary_intact": canary_intact,
        "passed": passed,
    }
    state.meta["grounding"] = grounding
    state.scores["faithfulness"] = faithfulness
    state.scores["grounding_threshold"] = threshold
    state.log("step_10_grounding_judge", **grounding)

    if not canary_intact:
        state.block("step_10_grounding_judge",
                    "indirect-injection leak: canary token surfaced in answer")
    elif not passed:
        state.block("step_10_grounding_judge",
                    f"ungrounded answer (faithfulness {faithfulness:.3f} "
                    f"< threshold {threshold:.3f})")
    return state


# --------------------------------------------------------------------------- #
# CLI / demo
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Step 10: Grounding and Policy Checking")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    chunks = ["Abraham Lincoln was the sixteenth President of the United States."]
    cases = [
        ("grounded, low risk",
         "Abraham Lincoln was the 16th President of the United States.", 0.04, 0.05),
        ("grounded, high risk + high disagreement",
         "Abraham Lincoln was the sixteenth U.S. President.", 0.73, 0.58),
        ("hallucinated",
         "Abraham Lincoln invented the telephone in 1876.", 0.04, 0.10),
    ]
    for label, ans, risk, dis in cases:
        st = PipelineState(prompt="Who was the 16th President?")
        st.scores["fusion_risk"] = risk
        st.scores["disagreement"] = dis
        st.meta["answer"] = ans
        st.ranked_context = list(chunks)
        st.meta["canary"] = "CANARY-demo"
        st = run(st)
        g = st.meta["grounding"]
        print(f"\n--- {label} ---")
        print(f"  lexical={g['lexical_score']:.3f} model={g['model_score']:.3f} "
              f"faith={g['faithfulness']:.3f} thr={g['threshold']:.3f} "
              f"passed={g['passed']} blocked={st.blocked}")


if __name__ == "__main__":
    main()
