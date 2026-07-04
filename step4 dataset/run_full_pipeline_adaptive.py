"""
run_full_pipeline_adaptive.py  —  adaptive orchestrator (wraps the existing steps)
==================================================================================

A thin variant of run_full_pipeline.process() that threads an AdaptivePolicy through
the pipeline so each stage reads its parameters from the active tier. It REUSES every
existing step module unchanged except for the small, clearly-marked hooks documented
in INTEGRATION.md. Keep the original process() intact for the byte-for-byte baseline;
this is the adaptive path used by run_three_way.py.

The function signature mirrors process() so the rest of the harness is unaffected.
"""
from __future__ import annotations

import step_01_user_input as s1
import step_02_normalization as s2
import step_03c_fusion_gate as s3c
import step_03d_semantic_intent_gate as s3d
import step_04_query_embedding as s4
import step_05_vector_search as s5
import step_06_context_sanitization as s6
import step_07_context_ranking as s7
import step_08_augmented_prompt as s8
import step_10_grounding_judge as s10
import step_13_safe_response as s13

from pipeline_common import PipelineState
from adaptive_risk import AdaptivePolicy
from adaptive_verification import verified_generate
from adaptive_prompt import profile_for

try:
    from run_full_pipeline import ensure_index
except Exception:                      # pragma: no cover
    def ensure_index(index_path: str = "./kb_wiki") -> None:  # type: ignore
        pass


def process_adaptive(question: str, *, mode: str = "adaptive",
                     poison_chunk: str | None = None,
                     query_suffix: str | None = None,
                     base_url: str = "http://localhost:11434") -> PipelineState:
    """Run one query through the adaptive pipeline in the given security `mode`.

    mode: "adaptive" | "static" | "none"  (see AdaptivePolicy).
    poison_chunk / query_suffix: optional injected attack material, mirroring the
    harness hooks the existing _segment() uses for the poisoned-context cohort.
    """
    if query_suffix:
        question = f"{question} {query_suffix}"
    st = PipelineState(prompt=question, raw_prompt=question)
    policy = AdaptivePolicy(mode=mode)

    # --- input analysis (always runs) -------------------------------------- #
    st = s1.run(st)
    st = s2.run(st)
    st = s3c.run(st)
    if st.blocked:
        return s13.run(st)
    st = s3d.run(st, backend="nli")
    if st.blocked:
        return s13.run(st)

    # --- first risk assessment from input-side signals --------------------- #
    cfg = policy.assess(st)            # selects tier from cumulative (input) risk

    # --- retrieval (adaptive top_k + similarity threshold) ----------------- #
    st = s4.run(st)
    st = s5.run(st, k=cfg.top_k)
    # similarity-threshold filter (Step 5 returns retrieval_scores in meta)
    _apply_similarity_threshold(st, cfg.sim_threshold)
    st.meta["active_top_k"] = cfg.top_k
    st.meta["active_sim_threshold"] = cfg.sim_threshold

    # optional poisoned-context injection (harness hook, same as _segment) ---
    if poison_chunk:
        st.context = list(st.context) + [poison_chunk]

    # --- sanitization (adaptive strictness) -------------------------------- #
    # Step 6 reads input_risk(); we additionally pin the tier's strictness so the
    # policy is the single source of truth. See patch in INTEGRATION.md.
    st.meta["tier_sanitization_strictness"] = cfg.sanitization_strictness
    st = s6.run(st)
    st.meta["active_sanitization"] = st.meta.get("sanitization_strictness", cfg.sanitization_strictness)

    # --- ranking (adaptive rerank floor) ----------------------------------- #
    st.meta["tier_rerank_min_score"] = cfg.rerank_min_score
    st = s7.run(st, top_n=cfg.top_k)

    # --- prompt construction (adaptive profile) ---------------------------- #
    st.meta["tier_prompt_profile"] = profile_for(cfg)
    st = s8.run(st)

    # --- generation + verification (adaptive: 1 vs ensemble, policy check) -- #
    st = verified_generate(st, cfg, base_url=base_url)

    # --- forward accrual: fold LATE signals (grounding/disagreement) -------- #
    cfg = policy.assess(st)            # may push tier UP (monotonic)

    # --- grounding feedback loop ------------------------------------------- #
    grounded = bool(st.meta.get("grounding", {}).get("passed", True))
    if not grounded and mode == "adaptive" and policy.tier == "high":
        # Late mismatch raised us to HIGH: one guarded re-pass under strict prompt
        # + ensemble before deciding. This is the risk-propagation feedback loop.
        st.meta["feedback_repass"] = True
        st.meta["tier_prompt_profile"] = profile_for(policy.config)
        st = s8.run(st)
        st = verified_generate(st, policy.config, base_url=base_url)
        grounded = bool(st.meta.get("grounding", {}).get("passed", True))

    # --- final decision ---------------------------------------------------- #
    if not grounded and policy.tier == "high":
        st.block("adaptive_policy", "ungrounded answer under high-risk policy")
    return s13.run(st)


def _apply_similarity_threshold(st, threshold: float) -> None:
    """Drop retrieved chunks below the tier's cosine-similarity floor.

    Step 5 stores parallel lists: st.context (texts) and meta['retrieval_scores'].
    Records how many survived so a starved high-risk query is visible in the audit
    (the benign-cost signal the thesis must report).
    """
    scores = st.meta.get("retrieval_scores") or []
    if not scores or len(scores) != len(st.context):
        return
    kept_texts, kept_scores = [], []
    for text, sc in zip(st.context, scores):
        if float(sc) >= threshold:
            kept_texts.append(text)
            kept_scores.append(sc)
    st.meta["retrieval_kept"] = len(kept_texts)
    st.meta["retrieval_dropped_by_threshold"] = len(st.context) - len(kept_texts)
    # Never leave context empty if everything was filtered: keep top-1 so the row
    # can still abstain honestly rather than crash. (Records the starve event.)
    if kept_texts:
        st.context = kept_texts
        st.meta["retrieval_scores"] = kept_scores
    else:
        st.meta["retrieval_starved"] = True
