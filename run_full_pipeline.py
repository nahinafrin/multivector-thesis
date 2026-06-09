"""
run_full_pipeline.py  —  the single chain that makes the adaptive cascade CAUSAL
================================================================================

This fixes the root problem: previously the input gate (Steps 1-3) and the RAG
half (Steps 4-8) used different state objects and were never run together, so
the carried-forward risk score was always 0.0 and the "adaptive cascade" was
only ever a post-hoc audit reconstruction.

Here ONE unified PipelineState flows through the whole pipeline:

    Step 1  user input
    Step 2  normalization (+ length flag)
    Step 3c C3RF gate         -> sets state.scores["fusion_risk"]   <-- the source
            (BLOCK short-circuits straight to Step 13 refusal)
    Step 4  query embedding
    Step 5  vector search
    Step 6  context sanitization  (reads fusion_risk -> adaptive threshold + canary)
    Step 7  context ranking       (reads fusion_risk -> adaptive min_score)
    Step 8  augmented prompt
    Step 9  generator ensemble    (-> answer + disagreement)
    Step 10 grounding + policy    (reads fusion_risk + disagreement -> threshold;
                                   re-checks the canary)
    Step 11 output sanitization   (Presidio PII + Llama-Guard verdict)
    Step 12 DLP scanner           (regex + Luhn + EDM)
    Step 13 safe response         (formatted answer or fixed refusal)

Because the risk is computed at Step 3c and read at Steps 6/7/10 from the SAME
state, the cascade is now causal end-to-end, not audited after the fact.

This module is an orchestration skeleton: it imports the step modules and wires
them. It does not re-implement any step. Models are loaded lazily by each step,
so this runs wherever the individual steps run.

Usage:
    python run_full_pipeline.py --question "Who was the 16th US president?"
    python run_full_pipeline.py --qa-file data/question-answer/test.jsonl --out grounded.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The gate modules (Steps 1-3/3c) live in ../dataset. Keep this directory first
# on sys.path so those modules import this unified `pipeline_common.py`, not the
# older dataset-local state class.
_HERE = Path(__file__).resolve().parent
_DATASET_DIR = (_HERE.parent / "dataset").resolve()
if str(_DATASET_DIR) not in sys.path:
    sys.path.insert(1, str(_DATASET_DIR))

from pipeline_common import PipelineState

# Gate half
import step_01_user_input as s1
import step_02_normalization as s2
import step_03c_fusion_gate as s3c
# RAG half
import step_04_query_embedding as s4
import step_05_vector_search as s5
import step_06_context_sanitization as s6
import step_07_context_ranking as s7
import step_08_augmented_prompt as s8
# Generation + output rail (adapt these imports to your filenames)
import step_09_generator_llm as s9
import step_10_grounding_judge as s10
import step_11_output_sanitization as s11
import step_12_dlp_scanner as s12
import step_13_safe_response as s13

from risk_feedback_controller import ControllerConfig, RiskController
from multivector import input_channels, multivector_risk
from calibration import load_calibration

_INDEX_READY = False
# Per-channel score calibration for the multi-vector path. Identity (no-op) when
# calibration.json is absent, so behaviour is unchanged until a calibrator is fit.
_CALIBRATION = load_calibration()


def ensure_index(index_path: str = "./kb_wiki") -> None:
    """Load the persisted FAISS KB once before Step 5 searches."""
    global _INDEX_READY
    if _INDEX_READY:
        return
    import kb_rag_mini_wikipedia as kb
    if not kb.load_persisted_index(index_path):
        raise RuntimeError(f"Could not load persisted FAISS index at {index_path}")
    _INDEX_READY = True


def process(question: str, *, k: int = 5, top_n: int = 3,
            treat_review_as_red: bool = False,
            poison_chunk: str | None = None,
            index_path: str = "./kb_wiki",
            continue_blocked_for_audit: bool = False,
            use_controller: bool = True,
            controller_cfg: ControllerConfig | None = None) -> PipelineState:
    """Run one question through the whole pipeline on a single state.

    If `poison_chunk` is given, it is spliced into the retrieved context after
    Step 5 and before Step 6, simulating indirect injection in a retrieved
    document so the sanitization layer (and canary) are exercised.

    When ``use_controller`` is True (the default), the retrieval -> generation
    -> grounding segment (Steps 4-10) is run inside the closed-loop
    RiskController, which reads ensemble disagreement, grounding faithfulness,
    and canary integrity after each pass and decides whether to accept,
    escalate-and-re-verify, broaden retrieval and retry, or refuse. Setting
    ``use_controller=False`` preserves the original single-pass behaviour
    byte-for-byte — that's the A/B switch the spec calls for so the controller
    can be evaluated against the same adversarial slice with everything else
    held constant.
    """
    st = PipelineState(prompt=question, raw_prompt=question)

    # --- input gate: computes the risk that everything downstream reads ---
    st = s1.run(st)
    st = s2.run(st)
    st = s3c.run(st, treat_review_as_red=treat_review_as_red)
    if st.blocked:                       # gate refused -> straight to refusal
        if not continue_blocked_for_audit:
            return s13.run(st)
        # Audit-only mode: keep the gate verdict, but temporarily continue so
        # retrieval-layer adaptivity can be measured on genuinely high-risk
        # inputs. Do not use this as production serving behavior.
        st.meta["gate_would_have_blocked"] = True
        st.meta["gate_block_stage"] = st.block_stage
        st.meta["gate_block_reason"] = st.block_reason
        st.blocked = False
        st.block_stage = ""
        st.block_reason = ""

    ensure_index(index_path)

    # Steps 4-10 packaged as a re-runnable segment the controller can call
    # repeatedly with different k / top_n. Kept as a closure so it captures
    # `poison_chunk` without leaking it into the controller's API.
    def _segment(state: PipelineState, *, k: int, top_n: int) -> PipelineState:
        state = s4.run(state)
        state = s5.run(state, k=k)
        if poison_chunk:                 # indirect-injection test case
            state.context = list(state.context) + [poison_chunk]
            state.meta["poison_injected"] = poison_chunk
        state = s6.run(state)            # reads input_risk() -> adaptive sanitization
        # Multi-vector PREVENTION: fuse the (now-exposed) query and context
        # injection channels. If both are softly co-active, write the joint
        # risk into effective_risk so Step 7 (rerank) and Step 10 (grounding)
        # tighten on this segment WITHOUT us having to duplicate their adaptive
        # logic. Detection (controller escalation / hard refuse) is wired
        # separately in risk_feedback_controller.
        raw_channels = input_channels(state)
        channels = _CALIBRATION.apply_channels(raw_channels)
        mv = multivector_risk(channels)
        if not _CALIBRATION.is_identity():
            mv["raw_channels"] = {c: round(v, 4) for c, v in raw_channels.items()}
            mv["calibrated"] = True
        state.meta["multivector"] = mv
        if mv["is_multivector"]:
            state.scores["effective_risk"] = max(state.input_risk(),
                                                 float(mv["joint_risk"]))
        state = s7.run(state, top_n=top_n)  # reads input_risk() -> adaptive rerank
        state = s8.run(state)
        state = s9.run(state)            # sets meta["answer"] + scores["disagreement"]
        state = s10.run(state)           # grounding judge; reads input_risk() + disagreement
        return state

    def _reground(state: PipelineState) -> PipelineState:
        # Cheap re-verify path used by the controller's escalate branch:
        # rerun Step 10 ONLY at the now-higher input_risk() / threshold,
        # NOT a fresh 3-model generation.
        return s10.run(state)

    if use_controller:
        ctrl = RiskController(controller_cfg or ControllerConfig(),
                              k0=k, top_n0=top_n)
        # Fail-CLOSED: if the controller itself raises for any reason, force a
        # refusal at Step 13. We unconditionally call st.block(...) here — not
        # `if not st.blocked` — because an exception mid-controller can leave
        # the state half-built (e.g. an answer set by Step 9 but grounding never
        # written), and we must NOT let that fall through into the output rail
        # with state.blocked == False. The controller's own audit trace is
        # preserved in state.meta["controller"] if it managed to write any.
        try:
            st = ctrl.run(st, _segment, _reground)
        except Exception as e:           # pragma: no cover (defense-in-depth)
            st.meta["controller_error"] = f"{type(e).__name__}: {e}"
            st.block("risk_feedback_controller",
                     f"controller error: {type(e).__name__}: {e}")
    else:
        st = _segment(st, k=k, top_n=top_n)

    if st.blocked:
        return s13.run(st)

    # --- output rail (unchanged) ---
    st = s11.run(st)                     # PII + Llama-Guard output verdict
    if st.blocked:
        return s13.run(st)
    st = s12.run(st)                     # deterministic DLP net
    return s13.run(st)                   # format or refuse


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
        },
        "grounding": g,
        "multivector": st.meta.get("multivector", {}),
        "controller": st.meta.get("controller", {}),
        "output_sanitization": st.meta.get("output_sanitization", {}),
        "dlp": st.meta.get("dlp", {}),
        "final_response": st.meta.get("final_response", ""),
        "blocked": st.blocked,
        "block_stage": st.block_stage or None,
        "block_reason": st.block_reason or None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Full pipeline (gate + RAG + output rail)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--question")
    g.add_argument("--qa-file", help="JSONL of {question, answer/ground_truth} rows")
    g.add_argument("--slice", dest="slice_file",
                   help="adversarial slice JSONL from build_adversarial_slice.py")
    ap.add_argument("--out", default="grounded.jsonl")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--index", default="./kb_wiki")
    ap.add_argument("--review-as-red", action="store_true")
    ap.add_argument("--continue-blocked-for-audit", action="store_true",
                    help="Audit-only: run downstream stages even if Step 3c would block.")
    ap.add_argument("--no-controller", action="store_true",
                    help="Disable the closed-loop risk-feedback controller "
                         "(reproduces the original single-pass behaviour exactly; "
                         "used for A/B evaluation of the controller).")
    ap.add_argument("--resume", action="store_true",
                    help="Slice mode: skip ids already present in --out and append "
                         "the rest, so a run can recover after a crash.")
    ap.add_argument("--max-attempts", type=int, default=None,
                    help="Override ControllerConfig.max_attempts (default 3). "
                         "Ignored when --no-controller is set.")
    args = ap.parse_args()

    use_controller = not args.no_controller
    controller_cfg = (ControllerConfig(max_attempts=args.max_attempts)
                      if (use_controller and args.max_attempts is not None)
                      else None)

    def _run(q: str, poison: str | None = None) -> PipelineState:
        return process(q, k=args.k, top_n=args.top_n,
                       treat_review_as_red=args.review_as_red,
                       poison_chunk=poison,
                       index_path=args.index,
                       continue_blocked_for_audit=args.continue_blocked_for_audit,
                       use_controller=use_controller,
                       controller_cfg=controller_cfg)

    if args.question:
        st = _run(args.question)
        print(json.dumps(_record(st, 0, None), ensure_ascii=False, indent=2))
        return

    from pipeline_common import read_jsonl

    if args.slice_file:
        # Optional resume: the full slice is slow and the native model/faiss stack
        # can crash mid-run. With --resume we keep the rows already completed
        # (rewriting the file to drop any partial trailing line) and only process
        # the remaining ids, appending. Re-running after a crash then makes
        # forward progress instead of starting over.
        done_ids: set = set()
        if args.resume and Path(args.out).exists():
            clean: list[str] = []
            for line in open(args.out, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue                 # drop partial/corrupt trailing line
                done_ids.add(rec.get("index"))
                clean.append(json.dumps(rec, ensure_ascii=False))
            with open(args.out, "w", encoding="utf-8") as f:
                f.write("\n".join(clean) + ("\n" if clean else ""))
            print(f"[resume] {len(done_ids)} rows already done in {args.out}")

        mode = "a" if args.resume else "w"
        with open(args.out, mode, encoding="utf-8") as fout:
            for row in read_jsonl(args.slice_file):
                rid = row.get("id", 0)
                if args.resume and rid in done_ids:
                    continue
                st = _run(row.get("question", ""), poison=row.get("poison_chunk"))
                rec = _record(st, rid, row.get("ground_truth"))
                rec["kind"] = row.get("kind")
                rec["expectation"] = row.get("expectation")
                rec["success_marker"] = row.get("success_marker")
                rec["true_answer"] = row.get("true_answer")
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
        print(f"[done] wrote slice results -> {args.out}")
        return

    with open(args.out, "w", encoding="utf-8") as fout:
        for i, row in enumerate(read_jsonl(args.qa_file), start=1):
            q = row.get("question") or row.get("prompt") or ""
            gt = row.get("ground_truth") or row.get("answer")
            st = _run(q)
            fout.write(json.dumps(_record(st, i, gt), ensure_ascii=False) + "\n")
            fout.flush()
            if i % 25 == 0:
                print(f"  ...{i} processed")
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
