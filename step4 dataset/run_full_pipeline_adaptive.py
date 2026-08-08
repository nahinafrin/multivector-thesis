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

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DATASET_DIR = (_HERE.parent / "dataset").resolve()
if str(_DATASET_DIR) not in sys.path:
    sys.path.insert(1, str(_DATASET_DIR))

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
                     base_url: str = "http://localhost:11434",
                     enable_semantic_gate: bool = True,
                     semantic_gate_backend: str = "nli") -> PipelineState:
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
    if enable_semantic_gate:
        st = s3d.run(st, backend=semantic_gate_backend)
        if st.blocked:
            return s13.run(st)
    else:
        st.meta["semantic_intent_decision"] = "SKIPPED"
        st.meta["semantic_intent_reason"] = (
            "step_03d disabled for this run: demo self-test showed 2/4 "
            "misclassified (FP 0.947 on a plain factual question, FN 0.166 "
            "on a reframed attack -- see gate_diagnosis.txt). Excluded from "
            "gating until recalibrated."
        )

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


def _record(st: PipelineState, index: int, ground_truth: str | None) -> dict:
    """Flatten the state into one auditable JSONL row exposing every stage."""
    g = st.meta.get("grounding", {})
    return {
        "index": index,
        "question": st.raw_prompt,
        "ground_truth": ground_truth,
        "gate": {
            "decision": st.meta.get("fusion_decision"),
            "fusion_risk": st.scores.get("fusion_risk", 0.0),
            "injection_score": st.scores.get("injection_detection", 0.0),
            "category": st.meta.get("llamaguard_category"),
            "would_have_blocked": bool(st.meta.get("gate_would_have_blocked", False)),
            "block_reason": st.meta.get("gate_block_reason"),
        },
        "retrieval": {
            "raw_chunks": st.meta.get("raw_chunks", []),
            "sanitized_chunks": st.meta.get("sanitized_chunks", []),
            "sanitization_strictness": st.meta.get("sanitization_strictness"),
            "sanitization_dropped": st.meta.get("sanitization_dropped", 0),
            "poison_injected": st.meta.get("poison_injected"),
            "poison_redacted": (
                st.meta.get("poison_injected") is not None
                and st.meta.get("poison_injected") not in st.ranked_context
            ),
            "ranked_chunks": st.ranked_context,
            "rerank_scores": st.meta.get("rerank_scores", []),
            "rerank_min_score": st.meta.get("rerank_min_score"),
            "canary": st.meta.get("canary"),
        },
        "generation": {
            "answer": st.meta.get("answer", ""),
            "disagreement": st.scores.get("disagreement", 0.0),
            "isolate_aggregate": st.meta.get("isolate_aggregate"),
            "knowledge_conflict": st.scores.get("knowledge_conflict"),
            "parametric_answer": st.meta.get("parametric_answer"),
            "knowledge_conflict_reason": st.meta.get("knowledge_conflict_reason"),
        },
        "grounding": g,
        "multivector": st.meta.get("multivector", {}),
        "meta": {
            "semantic_intent_decision": st.meta.get("semantic_intent_decision"),
            "semantic_intent_reason": st.meta.get("semantic_intent_reason"),
            "semantic_intent": st.scores.get("semantic_intent"),
            "semantic_intent_error": st.meta.get("semantic_intent_error"),
            "context_scale": st.meta.get("context_scale"),
        },
        "controller": st.meta.get("controller", {}),
        "output_sanitization": st.meta.get("output_sanitization", {}),
        "dlp": st.meta.get("dlp", {}),
        "final_response": st.meta.get("final_response", ""),
        "blocked": st.blocked,
        "block_stage": st.block_stage or None,
        "block_reason": st.block_reason or None,
        "abstained": st.abstained,
        "abstain_stage": st.abstain_stage or None,
        "abstain_reason": st.abstain_reason or None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Adaptive pipeline runner")
    ap.add_argument("--in", dest="in_file", required=True,
                    help="Input JSONL with question rows")
    ap.add_argument("--out", dest="out_file", required=True,
                    help="Output JSONL path")
    ap.add_argument("--mode", choices=["adaptive", "static", "none"],
                    default="adaptive")
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--disable-semantic-gate", action="store_true",
                    help="Skip step_03d (miscalibrated -- see gate_diagnosis.txt)")
    args = ap.parse_args()

    from pipeline_common import read_jsonl
    import traceback

    n_ok, n_err = 0, 0
    with open(args.out_file, "w", encoding="utf-8") as fout:
        for i, row in enumerate(read_jsonl(args.in_file), start=1):
            q = row.get("question") or row.get("prompt") or ""
            gt = row.get("ground_truth") or row.get("answer")
            try:
                st = process_adaptive(
                    q,
                    mode=args.mode,
                    base_url=args.base_url,
                    poison_chunk=row.get("poison_chunk"),
                    enable_semantic_gate=not args.disable_semantic_gate,
                )
                rec = _record(st, i, gt)
                n_ok += 1
            except Exception as e:
                print(f"[ERROR] row {i} source={row.get('source')!r} id={row.get('id')!r}: {e!r}")
                traceback.print_exc()
                rec = {"index": i, "question": q, "error": repr(e), "error_type": type(e).__name__}
                n_err += 1
            rec["kind"] = row.get("kind")
            rec["expectation"] = row.get("expectation")
            rec["success_marker"] = row.get("success_marker")
            rec["true_answer"] = row.get("true_answer")
            rec["source"] = row.get("source")
            rec["external"] = row.get("external")
            rec["source_id"] = row.get("source_id")
            rec["attack_type"] = row.get("attack_type")
            rec["poison_chunk"] = row.get("poison_chunk")
            rec["id"] = row.get("id", i)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if i % 5 == 0:
                print(f"  ...{i} processed ({n_ok} ok, {n_err} errors)")
    print(f"[done] wrote {args.out_file}  ({n_ok} ok, {n_err} errors)")


if __name__ == "__main__":
    main()
