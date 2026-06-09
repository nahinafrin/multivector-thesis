# Methodology — Unified Pipeline State and the Production-vs-Audit Evaluation Design

## Why a unified state was required

The initial implementation of the pipeline grew in two halves, each with its own `PipelineState` class. The *input-gate half* (Steps 1–3, 3c) carried a state with the fields needed for filtering and decision logging (`flag`, `block_stage`, two-argument `block(stage, reason)`). The *RAG half* (Steps 4–8) carried a separate state with the fields needed for retrieval and prompting (`context`, `ranked_context`, `augmented_prompt`, one-argument `block(reason)`). Because they were different classes in different files, the input-gate's `fusion_risk` score had no path to reach the retrieval-layer steps that were supposed to read it; the RAG orchestrator instantiated a fresh state with `injection_detection = 0.0` hard-coded, and the methodology's defining feature — *"an adaptive risk score that flows through every stage so a suspicious input makes downstream checks stricter"* — was, in the original code, architectural but not causal. Any reported adaptivity was an after-the-fact audit reconstruction over a `risk = 0` ground truth, not a measured effect on system behavior.

## The unified `PipelineState`

The two state classes are replaced by a single `PipelineState` (`pipeline_common.py`) that is a strict superset of both. It exposes the same fields each half previously needed (`flag`, `context`, `ranked_context`, `augmented_prompt`, `meta`, `scores`, `trace`), plus a small set of additions that the rest of the pipeline now reads:

- `block(arg1, arg2=None)` accepts either of the two prior call shapes, so the existing step files keep working unchanged: the gate-half's `block(stage, reason)` and the RAG-half's `block(reason)` both populate the same `(blocked, block_stage, block_reason, flag)` quadruple and emit a trace entry. This kept the rewrite confined to the state class itself rather than rippling through every step.
- `input_risk()` is the single helper every adaptive step calls. It returns `scores["fusion_risk"]` if the C3RF gate ran, falls back to `scores["injection_detection"]` if only the bare injection scanner ran, and finally to `0.0` if neither has populated. Steps 6, 7, and 10 query this helper rather than reading scores directly so a future gate variant (different score key) can be substituted without touching the downstream code.
- `meta` carries the per-row audit fields the JSONL output exposes (`raw_chunks`, `sanitized_chunks`, `sanitization_strictness`, `sanitization_dropped`, `rerank_min_score`, `canary`, `answer`, `grounding`, `output_sanitization`, `dlp`, `final_response`). The audit JSONL is a flattened serialization of this state, not a separately maintained log, so the recorded fields are by construction the same as the ones the steps acted on.

With this state in place, a single orchestrator (`run_full_pipeline.py`) runs the full sequence Step 1 → Step 13 on one mutable state object. The risk computed at Step 3c is the same value that Steps 6, 7, and 10 read; there is no longer a gap that an audit had to reconstruct after the fact.

## Adaptive thresholds wired against `input_risk()`

Three downstream steps now condition on `state.input_risk()`:

- **Step 6 (context sanitization).** The LLM-Guard `PromptInjection` scanner threshold tightens from a base of `0.50` to `0.30` when `input_risk > 0.60`. The same step inserts the dual-checkpoint canary recorded in `meta["canary"]` so Step 10 can verify it never leaks into the answer.
- **Step 7 (context reranking).** The cross-encoder minimum-score floor tightens from `0.20` to `0.40` under the same condition. The step guarantees at least one chunk is retained so Step 8/9 is never stranded; the strictness change is logged so each row's strictness value is visible in the JSONL.
- **Step 10 (grounding judge).** The pass threshold begins at `0.70` and adds `+0.10` for `input_risk > 0.60` and a further `+0.10` for `disagreement > 0.30`, capped at `0.90`. The faithfulness score itself is a hybrid `max(lexical_overlap, model_cross_encoder)` with documented complementary failure modes; the dynamic threshold is a separate adaptive layer that the same risk signal drives.

## Production vs. audit-continue evaluation design

Running the unified orchestrator on adversarial inputs surfaces a subtle but important evaluation question. In production serving, the C3RF gate is *meant* to block high-risk inputs at Step 3c, short-circuiting the pipeline to the fixed refusal at Step 13. That is the correct serving behavior — but it also means that on high-risk inputs the very retrieval-layer steps whose adaptivity we want to measure (Steps 6/7/10) never run, because the gate refuses upstream of them. A naïve evaluation that uses only production runs therefore cannot directly observe the cascade firing on the inputs most likely to fire it.

The evaluation design separates the two concerns:

- **Production mode.** The orchestrator's default behavior, used in `slice_results.jsonl`. Step 3c's `BLOCK` (and optionally `REVIEW`) decisions short-circuit straight to Step 13. This is the serving artifact: every row's `final_response` is the response a user would actually receive, every block reason is the reason the user would have been refused.
- **Audit-continue mode.** Selected with `run_full_pipeline.py --continue-blocked-for-audit`, used in `slice_results_audit_continue.jsonl`. Step 3c's decision is recorded as before (`gate.decision`, `gate.would_have_blocked = True`), but the orchestrator then resets the blocked flag and allows the row to flow through Steps 4–10 so the cascade's retrieval-layer behavior is observable on the high-risk input. The audit row preserves the *original* gate verdict in `gate.would_have_blocked` and `gate.block_reason`, so no audit row can be mistaken for a row that would have been served downstream in production.

This split is the way the evaluation reconciles two things the methodology has to demonstrate simultaneously: that the gate genuinely refuses in production (production-mode artifact) and that the downstream cascade genuinely tightens on inputs the gate refuses (audit-continue artifact). The same slice is run twice, the same row IDs appear in both artifacts, and the `gate.would_have_blocked` field makes the distinction machine-checkable rather than a reader's responsibility to infer.

## Threats to validity this design closes

Two examiner-style objections the older two-state design could not answer are addressed:

1. *"How do you know `fusion_risk` actually drove a threshold change, rather than the threshold change being inferred from a reconstructed risk after the fact?"* — Because the same `PipelineState` object passes through Steps 3c and 6/7/10, and the strictness/min-score/threshold values written to `meta` are by construction the values the steps used in that single forward pass. The trace log records the `input_risk` each step read, alongside the threshold it produced.

2. *"If the gate already blocks high-risk inputs, isn't the downstream adaptivity claim untestable?"* — No: it is measured on the audit-continue artifact, which is clearly labelled as an evaluation surface (`gate.would_have_blocked = True` on affected rows) and not presented as a serving behavior. The production artifact, scored separately, demonstrates the gate's blocking behavior unchanged.

The combined design is what enables the results section to make a single, defensible claim: that the input-gate risk score *causally* tightens downstream retrieval and grounding thresholds, while production serving still refuses the same inputs at the gate.
