# INTEGRATION.md — wiring the adaptive framework into your existing repo

Everything here drops into `step4 dataset/`. Your original `run_full_pipeline.py`
`process()` stays **untouched** so it remains the byte-for-byte baseline; the new
`run_full_pipeline_adaptive.py` is the adaptive path. Work in this order.

---

## Step 0 — copy the new modules in

Copy these into `step4 dataset/`:

```
adaptive_risk.py              # noisy-OR cumulative score + tiers (core)
adaptive_verification.py      # conditional ensemble + policy check
adaptive_prompt.py            # tier-aware system prompts
trust_aware_retrieval.py      # similarity x source-trust
run_full_pipeline_adaptive.py # the adaptive orchestrator
run_three_way.py              # the experiment runner
score_three_way.py            # the metric table
```

Quick check — the core module runs standalone:

```bash
python adaptive_risk.py
# noisy-OR of {...} = 0.654  (additive would be 0.920)
# tier = high ...
```

---

## Step 1 — let Step 6 accept a tier-pinned strictness

Your `step_06_context_sanitization.py` already adapts on `input_risk()`. Add ONE
optional override so the policy can be the single source of truth. Find where it
picks the threshold (around the `RISK_TIGHTEN_AT` logic) and change:

```python
# BEFORE
threshold = STRICT_THRESHOLD if risk > RISK_TIGHTEN_AT else BASE_THRESHOLD
```
```python
# AFTER  — honour an explicit tier strictness if the orchestrator set one
pinned = state.meta.get("tier_sanitization_strictness")
if pinned is not None:
    threshold = float(pinned)
else:
    threshold = STRICT_THRESHOLD if risk > RISK_TIGHTEN_AT else BASE_THRESHOLD
state.meta["sanitization_strictness"] = threshold
```

This is backward-compatible: when `tier_sanitization_strictness` is absent (the
original `process()`), behaviour is unchanged.

---

## Step 2 — let Step 7 accept a tier-pinned rerank floor

Same pattern in `step_07_context_ranking.py`. Replace:

```python
# BEFORE
min_score = STRICT_MIN_SCORE if risk > RISK_TIGHTEN_AT else BASE_MIN_SCORE
```
```python
# AFTER
pinned = state.meta.get("tier_rerank_min_score")
min_score = float(pinned) if pinned is not None else (
    STRICT_MIN_SCORE if risk > RISK_TIGHTEN_AT else BASE_MIN_SCORE)
```

---

## Step 3 — give Step 8 the tier prompt profile

In `step_08_augmented_prompt.py`, where the system preamble is assembled, prefer
the profile the orchestrator selected:

```python
# AFTER  — use the tier profile when present, else the existing default text
preamble = state.meta.get("tier_prompt_profile") or DEFAULT_SYSTEM_PREAMBLE
```

(`DEFAULT_SYSTEM_PREAMBLE` = whatever string the step uses today.)

---

## Step 4 — add a single-generator helper to Step 9 (optional but recommended)

`adaptive_verification` will fall back to running one ensemble model if this is
missing, but a dedicated helper is cleaner and cheaper. Add to
`step_09_generator_llm.py`:

```python
def generate_single(augmented_prompt: str,
                    model: str = "llama3.2:3b",
                    base_url: str = "http://localhost:11434") -> str:
    """One-model generation for LOW/MEDIUM tiers (cost saving)."""
    out = generate_candidates(augmented_prompt, models={model: model}, base_url=base_url)
    return next(iter(out.values())) if out else ""
```

---

## Step 5 — calibrate the tier boundaries (do NOT ship the defaults blind)

The defaults in `adaptive_risk.TIER_BOUNDS` and the per-tier `sim_threshold`
values are starting points. Calibrate them against YOUR cumulative-risk and
similarity distributions before reporting results:

```bash
python calibrate_tiers.py --controller-jsonl grounded_controller.jsonl
```

`calibrate_tiers.py` (below) prints: benign vs attack cumulative-risk percentiles
(for the LOW/MED and MED/HIGH cuts), and the realised retrieval count per tier at
each candidate `sim_threshold` (so you don't starve high-risk queries to zero
documents — the failure mode flagged in the module docstring).

---

## Step 6 — run the experiment

```bash
# build a slice if you don't have one (reuse your existing builder)
python build_adversarial_slice.py \
  --attack-jsonl ./merged_output/dataset_all.clean.jsonl \
  --qa-jsonl data/question-answer/test.jsonl \
  --n-adversarial 40 --n-poisoned 40 --n-benign 80 \
  --n-multivector 40 --n-gate-slip 20 --out adversarial_slice.jsonl

# run all three modes on the SAME slice
python run_three_way.py --slice adversarial_slice.jsonl --out-prefix three_way

# get the metric table (security / quality / efficiency, with Wilson CIs)
python score_three_way.py --prefix three_way
```

---

## What you can claim vs. what you must caveat

Claimable from this design:
- a single cumulative (noisy-OR) risk score that accrues forward and selects a tier;
- risk-proportional retrieval, sanitization, prompt, and verification;
- a grounding feedback loop that re-tiers/regenerates/refuses on late mismatch;
- an efficiency result: compute units and latency per mode, side by side with ASR.

Must caveat (put these in the thesis, don't skip them):
- trust-aware retrieval is a no-op on single-source rag-mini — either build a
  multi-source corpus to isolate it, or present it as an un-evaluated component;
- adaptive prompt construction is a soft defense; grounding is the real backstop;
- conditional verification trades compute for coverage — report how many
  successful attacks slipped through the cheap LOW/MEDIUM path;
- sample sizes are small — every rate carries a Wilson CI and the static-vs-adaptive
  comparison should be read through those intervals, not as point differences.
