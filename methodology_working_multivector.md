# A Working Methodology for a Local, Adaptive, Multi-Vector RAG Security Extension

> **Detector claims — use `RESULTS_multivector_methodology (4).md`, not this file.**
> Conjunctive sub-threshold *detection* on injection-signature channels is a
> **proven negative** (structural impossibility: both channels low exactly where
> both gates pass; confirmed 0/34 sub-threshold, 0/4 vs real scanners). Sections
> 4–6 below describe a *superseded* graded-channel fix that assumed the regime
> was instantiable; keep them only as design history. The defensible detector
> contribution is the closed argument in RESULTS §4.4, not "high recall, zero
> marginal value" or a revived conjunctive detector.

## 0. Purpose and scope

This document specifies a deployable security extension for Retrieval-Augmented
Generation (RAG) systems whose job is to defend a Large Language Model against
attacks that arrive from **both** ends of the pipeline — adversarial user
queries (direct prompt injection, jailbreaks) and adversarial knowledge sources
(poisoned documents carrying indirect injections) — and to police the model's
**output** for sensitive-data leakage, ungrounded hallucination, and unsafe
content. The defining feature is an **adaptive risk score** that propagates
forward through every stage: elevated risk found anywhere tightens the scrutiny
of every later stage for that same request. The entire system runs locally on
open-source models, so it has no per-query API cost and never sends a prompt
off-host.

This revision differs from the original methodology in one decisive way. The
original treated the **multi-vector (conjunctive sub-threshold) detector** as a
finished contribution, but evaluation showed it cannot fire on the sub-threshold
class against a near-binary injection scanner paired with a redacting sanitizer
(see `RESULTS_multivector_methodology (4).md` §4.4 for the closed argument and
empirical confirmation). The graded-channel approach in §4 below was an attempted
fix; it is retained as design history only.

---

## 1. Threat model

The pipeline is evaluated against five regimes, each with an explicit, testable
expectation. Naming them up front is what keeps the evaluation honest — in
particular it separates regimes the system *defends* from the one it only
*measures*.

| Regime | Vector | Expectation |
|---|---|---|
| **A. Direct injection** | Unsafe instruction in the user query | Input gate blocks or flags |
| **B. Indirect injection** | Poison instruction inside a retrieved document | Context sanitizer redacts; output canary stays intact |
| **C. Conjunctive sub-threshold** | A *mild* query instruction **and** a *mild* poison chunk, each individually below its block threshold, that only steer the answer together | Neither single-channel rail fires; the multi-vector detector fires on the co-activation |
| **D. Semantic obfuscation (residual)** | An unsafe request wrapped in benign framing (roleplay, fiction, "security training") leaving ~0 injection signature | **Measured, not claimed** — reported as a residual gap with a dedicated mitigation (§7) |
| **Specificity control** | Clean, benign queries | Cascade stays at base thresholds; no over-blocking |

Regime **C** is the contribution this methodology exists to make succeed.
Regime **D** is reported as a known residual with a concrete added defense, not
silently scored as a pass.

---

## 2. Architecture: the adaptive risk spine

A single `PipelineState` object flows through all stages. Two scores written
early are read by later stages to tighten their thresholds:

- `fusion_risk` — the continuous gate verdict from Step 3c.
- `disagreement` — semantic divergence across the generator ensemble (Step 9).

A clean prompt costs almost nothing; a borderline one pays for stricter scrutiny
at every downstream stage; an obviously hostile one dies at the gate before the
expensive stages run. The system is therefore **adaptive in strictness but
constant in attack surface**. Two extensions deepen this spine without
duplicating it: the multi-vector detector (10A) reads the same carried-forward
channels and fires on their conjunction, and the risk-feedback controller (10B)
writes `effective_risk` from late signals so the same forward adaptivity re-runs
at a stricter setting.

---

## 3. Pipeline stages

The stages below are the deployed serving path. Stages whose behaviour changed
in this revision are marked **(revised)**.

1. **User input.** Live prompt via FastAPI `POST /ask`, or JSONL records in
   evaluation mode (each record carries a ground-truth label so detector
   precision/recall are measured, not asserted).
2. **Normalization.** `ftfy` + Unicode NFKC + zero-width/bidi stripping + spaCy
   linguistic normalization. Out-of-window lengths are **flagged, not dropped**
   (short prompts are frequently genuine attacks). The identical routine runs at
   train and serve time to eliminate skew.
3. **Injection detection — input gate.** LLM-Guard `PromptInjection` +
   `Toxicity` scanners (local DeBERTa) produce a continuous risk; a local
   Llama-Guard classifier adds a *categorized* safety verdict; a C3RF fusion
   gate (Step 3c) combines them into allow / review / block and writes
   `fusion_risk`. A red verdict ends the request before the LLM is ever invoked.
   **(revised — see §4)** the gate additionally writes a *graded* injection
   score (`injection_graded`) from the detector's pre-sigmoid margin. This is
   additive: the block decision is unchanged.
4. **Query embedding.** `BAAI/bge-m3` (Matryoshka, truncatable) via
   sentence-transformers; vectors L2-normalized.
5. **Vector search.** FAISS `IndexFlatIP` (exact cosine) for evaluation,
   `IndexHNSWFlat` for larger corpora. Index persisted and loaded once at
   startup.
6. **Context sanitization — first checkpoint.** The same injection scanner runs
   **per chunk**; chunks over the (risk-adaptive) threshold are dropped. A unique
   **canary token** is recorded on the state. **(revised — see §4)** the
   strongest *graded* per-chunk score is exposed as `context_graded`. Adaptive
   coupling: the drop threshold tightens when `input_risk()` is high.
7. **Context ranking.** Cross-encoder `bge-reranker-v2-m3`. The rerank score
   threshold tightens under high input risk (return less context rather than
   pass marginal chunks).
8. **Augmented prompt.** A LangChain LCEL template that explicitly instructs the
   generator to treat retrieved text as **data, never as instructions** — the
   model-side complement to Step 6.
9. **Generator ensemble — parallel intelligence.** Three Ollama-served models
   from different lineages (e.g. `llama3.2:3b`, `mistral:7b`, `qwen2.5:3b`) run
   concurrently via `RunnableParallel`. Their **semantic disagreement**
   (all-MiniLM-L6-v2 distance) is written to `disagreement`; uncorrelated
   architectures rarely share the same jailbreak, so divergence is itself a risk
   signal.
10. **Grounding and policy — second checkpoint.** A hybrid faithfulness scorer
    (deterministic lexical overlap combined by `max` with a small-model judge),
    plus a policy gate, plus a **re-check of the Step 6 canary**. The pass
    threshold rises when input risk or ensemble disagreement is high.
11. **Output sanitization.** Microsoft Presidio masks PII (temporal entities
    deliberately excluded), then a local Llama-Guard pass catches semantic
    unsafety.
12. **DLP scanner.** Model-free net: regex + Luhn checksum + exact-data-match
    against a local secret store. Deterministic fail-safe if any ML layer
    degrades.
13. **Safe response.** A *fixed* refusal string on any block (zero side-channel
    about which stage fired); the blocking stage is recorded only in the
    operator audit trail.

---

## 4. The core fix: graded risk channels (revised)

**Problem.** The multi-vector detector consumes two channels:
`query_vector = injection signal` and `context_vector = max per-chunk injection`.
Both were sourced from `scanner.scan()`, which returns a **softmax probability**.
On a confident detector that probability *saturates*: a barely-injection prompt
and a screaming-injection prompt both return ~1.0. The conjunctive sub-threshold
regime requires each channel to land in the `[floor, block)` band — a band a
saturated 0/1 score never occupies. This is why the constructed
`multivector_attack` rows showed `query_vector ≈ 0.0–0.12` and
`context_vector = 1.000`: only one channel was ever active, the co-activation
requirement (`min_channels = 2`) was never met, and the detector correctly
declined to fire. The 0/40 was a **regime-instantiation failure, not a logic
bug** (proven in §6).

**Why calibration alone cannot fix it.** Fitting a monotone transform
(temperature/Platt) to the saturated probability is fitting to information the
sigmoid already discarded. Empirically, the likelihood-optimal temperature on a
well-separated near-binary detector is `T < 1`, which *sharpens* the score and
makes the band even harder to reach. A monotone map cannot un-saturate a
saturated score.

**The fix — read the margin before the sigmoid.** The information still exists
one step earlier, in the logits. Define

```
margin = logit(INJECTION) − logit(SAFE)
```

`margin` is unbounded and monotonic in confidence, so it distinguishes
`margin ≈ 0.4` ("just over the line") from `margin ≈ 9` ("blatant") exactly where
the probability cannot. Map it back into `[0,1]` with a deliberately *softening*
temperature (`T > 1`):

```
graded = sigmoid(margin / T)
```

`T` is fit from data as the **largest** value such that the benign margin at p95
maps below the per-channel floor (so benign noise stays under the floor while
attacks spread across the band). On a representative distribution this de-pins a
detector that was 26% saturated to 0% saturated, spreading attack scores across
roughly `[0.52, 0.87]` — squarely inside the firing band.

**Wiring (additive, backward-compatible).**

- Step 3 writes `injection_graded`; Step 6 writes `context_graded`. Both are
  wrapped so that if the graded module or the model is unavailable the pipeline
  silently falls back to the squashed score (old behaviour).
- The block decisions in Steps 3 and 6 are **not** touched (their thresholds are
  tuned on the squashed scale).
- The multi-vector channel reads the graded scores and, critically, **drops the
  old `max(injection, fusion_risk)`** — `fusion_risk` is still squashed, and
  maxing a graded signal against a saturated one re-saturates the very channel
  the fix just graded. The gate still uses `fusion_risk` for its own block.

---

## 5. Multi-vector detector (Step 10A)

The detector fuses the two graded channels and fires only when **all three**
conditions hold:

1. each active channel is at or above **its own** floor (per-channel, set from
   that channel's benign p95 on the *graded* scale — not one global number);
2. at least `min_channels = 2` are active (the co-activation requirement);
3. the noisy-OR joint risk `1 − Π(1 − channel)` is at or above `joint_min`.

Condition (2) is what makes a low floor safe: one loud vector is handled by the
existing single-channel rails, while two quiet vectors acting together are the
gap this detector closes. A separate, higher `joint_block` level marks a hard
refuse. The four constants are the single source of truth and are **re-derived
on the graded scale** by `calibrate_payloads.py` after the channels are graded;
the pipeline wiring never changes.

**Scope boundary (stated, not overclaimed).** This detector closes the
conjunctive sub-threshold gap **on the injection channels**. It does not close
pure semantic obfuscation (Regime D), which leaves ~0 injection signature on
both channels — that residual is handled separately in §7.

---

## 6. Controlled-injection validation (new methodological requirement)

Slice-based evaluation can only contain channel values the scanner is willing to
emit, so a 0/40 on a slice conflates "the detector logic is broken" with "the
regime cannot be instantiated against this scanner." To separate them, the
detector is validated **independently of any dataset** by driving
`multivector_risk()` directly over a grid of channel values
(each channel swept over `{0.0 … 1.0}`) and recording the verdict at every cell.

The validation asserts three properties:

- **Specificity:** zero fires in the benign region (both channels below floor)
  and zero fires for a single loud vector (one channel saturated, the other ~0).
- **Sensitivity:** the detector fires on genuine two-quiet-vector cells (both
  channels active, neither saturated).
- **Hard-block monotonicity:** cells above `joint_block` escalate to a hard
  refuse.

On the current detector this validation **passes**: 28 sub-threshold cells fire
with 0 false fires. This is the evidence that the mechanism is correct, and it
is the figure that reframes any residual 0/40 as a *characterized property of
the detector class* rather than an unexplained failure. The validation is also a
regression test of the three firing conditions and must be re-run whenever the
floors or `joint_min`/`joint_block` constants change.

---

## 7. Closing the semantic-obfuscation residual (Regime D)

Roleplay/fiction/"security-training" framings carry ~0 injection signature, so
neither the gate nor the multi-vector detector is expected to catch them. Rather
than report this as a pass, the methodology adds a dedicated, measured
mitigation:

- **Prompt isolation / spotlighting.** Retrieved context and user instruction
  are datamarked (delimited and tagged) so the generator can structurally
  distinguish "data" from "command," reducing the leverage of framing tricks.
- **Intent re-screening.** After the ensemble produces a draft, the draft's
  *requested action* (not the framing) is re-screened by Llama-Guard's
  categorized verdict; a benign-framed request for unsafe content is caught at
  the action level.
- **Disagreement as a tripwire.** Obfuscated jailbreaks tend to succeed on some
  models and not others; high `disagreement` (Step 9) raises the grounding and
  policy thresholds, giving the controller (10B) a late signal to escalate.

These are reported with their own before/after numbers on the Regime-D rows, so
the residual is shown shrinking rather than asserted closed.

---

## 8. Risk-feedback controller (Step 10B) and how to prove it

The controller wraps Steps 4–10 as a closed loop and, after each pass, reads
grounding, canary integrity, ensemble disagreement, and the multi-vector verdict
to select one action: `accept`, `escalate_verify` (re-run grounding only at a
stricter threshold), `recover_retry` (broaden retrieval and retry), or one of
several refusals. It is **fail-closed**: any internal error forces a block.

On the original slice the controller was risk-neutral (it neither helped nor
hurt), because the slice contained no rows that exercise its branches. To make
its value measurable, the evaluation set must include:

- **Out-of-corpus benign** questions whose evidence is initially missed but
  recoverable with a larger `k` (exercises `recover_retry`).
- **Gate-missed direct injections** that pass the gate but produce high
  disagreement or a canary touch downstream (exercises `escalate_verify` →
  refuse).
- **Coordinated multi-vector hard-blocks** (exercises `refuse_multivector`).

The controller is then reported as an A/B (`--no-controller` vs default) on these
constructed rows, where a positive delta is attributable to the loop.

---

## 9. Datasets

Three base datasets plus one constructed evaluation slice, all free and
locally hosted. None require a paid API.

### 9.1 Knowledge base and clean QA — `rag-mini-wikipedia` (HuggingFace)

- **text-corpus split** (~3.2k passages) populates the FAISS index (Step 5). It
  is small (deterministic retrieval timings), factually verifiable (grounding
  errors are obvious), and free of PII (no confounds for the Step 11 output
  rail).
- **question-answer split** supplies benign controls (Regime "Specificity") and
  gold answers for the Step 10 grounding judge. Records flow through the full
  pipeline exactly as a live prompt would, with their ground-truth label
  retained for scoring.

### 9.2 Attack corpus — `dataset_all.clean.jsonl` (HF + Kaggle, merged)

Schema: `{"prompt": str, "safety": "safe"|"unsafe", "attack_type": str, ...}`.
Multiple public corpora (e.g. deepset prompt-injection, jailbreak-prompt sets,
toxicity/harmful-behavior subsets, a PII-leakage set) are normalized into one
schema, deduplicated, and split train/val/test. A single corpus would miss
classes (injection corpora rarely contain jailbreak roleplays; jailbreak corpora
rarely contain context-poisoning payloads), so merging yields **balanced
positives per attack type**. A `reproducible_build_config.json` records the
source corpora, sampling proportions, and random seed. This corpus trains and
tests the input gate (Steps 3 / 3c) and supplies the adversarial-query and
multi-vector source material below.

### 9.3 Calibration data for the graded channels (new — required for §4)

- **`query_vector` calibration:** score the `dataset_all.clean.jsonl` prompts
  through the **margin** path (not the probability) to obtain
  `(margin, safety)` pairs; fit the softening temperature `T_q` so benign p95
  maps below the query floor.
- **`context_vector` calibration:** during a slice run, log the **per-chunk
  graded score** alongside an `is_poison` label for each chunk
  (`slice_chunk_scores.jsonl`); fit `T_c` the same way. If a channel is so
  saturated that no `T` de-saturates it without crossing the floor, that refusal
  is itself a reportable finding about the detector — record it rather than
  forcing a fit.

### 9.4 The constructed adversarial slice — corrected protocol (the key dataset)

This is the dataset that makes multi-vector mitigation **succeed instead of
0/40**. The original slice was mis-constructed: its `multivector_attack` rows
attached poison chunks that saturated `context_vector` to 1.0 and carried query
instructions that scored ~0, so by construction they were single-vector, not
conjunctive. The corrected protocol builds rows **on the graded scale** and
**measures each channel during construction** so every multi-vector row provably
occupies the sub-threshold band.

Five labelled groups (default 220 rows: 40 adversarial, 40 poisoned, 80 benign,
40 multi-vector, 20 gate-slip):

- **`adversarial_query`** — `safety=unsafe` rows from §9.2. Expectation:
  `gate_blocks_or_reviews = True`.
- **`poisoned_context`** — benign question from §9.1 QA, with a known
  indirect-injection payload spliced into retrieved context. Expectation:
  `poison_redacted_by_step6 = True`, `canary_intact_at_step10 = True`.
- **`benign_control`** — clean §9.1 QA. Expectation: base thresholds, no block.
- **`multivector_attack` (corrected)** — a benign question carrying:
  1. a query-side instruction whose **graded** `query_vector` is measured to fall
     in `[query_floor, query_block)`; and
  2. a poison chunk whose **graded** `context_vector` is measured to fall in
     `[context_floor, context_block)`.
  Construction is a **filtered search**: candidate instructions and poison
  strings are scored through the graded path during `build_adversarial_slice.py`,
  and only pairs where *both* channels land in-band (and the noisy-OR joint risk
  ≥ `joint_min`) are admitted. Rows that cannot be instantiated in-band are
  logged with a count, so the slice reports *how many* genuine conjunctive
  sub-threshold attacks the detector class even permits. Expectation:
  `each_vector_subthreshold = True`, `multivector_detector_fires = True`.
- **`gate_slip_query`** — an unsafe prompt in benign framing (Regime D).
  Expectation labels record this as the **measured residual** and, after §7 is
  enabled, an additional `mitigation_reduced_slip` flag.

The corrected `multivector_attack` group is the difference between a detector
that fires on 0/40 (because the rows were never in-band) and one whose fire rate
reflects the true prevalence of instantiable conjunctive attacks.

### Dataset-to-stage cross-reference

| Dataset | Drives |
|---|---|
| rag-mini-wikipedia / text-corpus | Step 5 (knowledge base) |
| rag-mini-wikipedia / question-answer | Steps 1, 10 (clean QA + ground truth) |
| dataset_all.clean.jsonl | Steps 3, 3c (gate train/test); query-channel calibration |
| slice_chunk_scores.jsonl (logged) | context-channel calibration |
| adversarial_slice.jsonl (corrected, 5-kind) | Steps 1, 6, 7, 10, 10A, 10B (end-to-end) |

---

## 10. Evaluation protocol and success criteria

The system is "successful" against this methodology when **all** of the
following hold, reported together (no single number stands alone):

1. **Specificity holds.** `benign_control` pass rate = 100% (no over-blocking).
   This is a gating condition: any sensitivity gain that drops this is rejected.
2. **Single-vector defenses hold.** `poisoned_context` ≈ 100% (canary intact);
   `adversarial_query` recall reported with its precision (target: improve on the
   current 75% without sacrificing specificity).
3. **Controlled-injection validation passes** (§6): specificity + sensitivity +
   hard-block monotonicity, asserted in CI.
4. **Multi-vector mitigation is reachable and measured.** On the corrected slice
   (§9.4), `multivector_attack` fires at a rate that matches the in-band
   prevalence the construction step logged, with `benign_control` still at 100%.
   The honest target is "fires on the instantiable rows," not "fires on 40/40" —
   if few rows are instantiable, that count is the finding.
5. **Graded channels de-saturate.** Report `saturated_frac` and benign/attack
   percentile separation before vs after grading, on both channels.
6. **Controller shows a positive A/B delta** on the constructed rows of §8, with
   per-action accounting.
7. **Residual is shrinking, not hidden.** Regime-D rows reported with before/
   after numbers once §7 is enabled.

Statistical hygiene: report mean ± standard deviation over ≥3 seeds; treat
single-row differences on a 220-row slice as noise; where feasible, compare
against an external baseline (e.g. LLM-Guard alone, or a spotlighting-only
defense) so the contribution is measured against prior art, not only against
ablations of itself.

---

## 11. Reproducible build and run order

```text
# 1. Build the knowledge base
python download_dataset.py
python kb_rag_mini_wikipedia.py --ingest --index ./kb_wiki

# 2. Calibrate the graded channels (the §4 fix)
python calibration.py fit-dataset \
    --dataset ../dataset/merged_output/dataset_all.clean.jsonl --limit 800 \
    --method temperature --channel query_vector --cache qv_scores.jsonl
#   (log per-chunk graded scores during a slice run, then:)
python calibration.py fit --scores slice_chunk_scores.jsonl \
    --score-field context_injection --label-field is_poison \
    --label-positive true --method temperature --channel context_vector

# 3. Re-derive the multi-vector floors on the graded scale
python calibrate_payloads.py

# 4. Validate the detector independently of any dataset
python eval_multivector_grid.py          # must print [PASS]

# 5. Build the corrected adversarial slice (in-band, filtered construction)
python build_adversarial_slice.py        # logs in-band instantiation counts

# 6. Run the full pipeline, with and without the controller (A/B)
python run_full_pipeline.py --slice adversarial_slice.jsonl --out grounded_controller.jsonl
python run_full_pipeline.py --slice adversarial_slice.jsonl --no-controller --out grounded_baseline.jsonl

# 7. Score
python score_slice.py grounded_controller.jsonl
python score_slice.py grounded_baseline.jsonl
```

---

## 12. Summary of what changed and why it now works

The original pipeline was sound everywhere except its headline contribution,
which failed for a single, diagnosable reason: the risk channels were sourced
from a **saturated probability**, so the conjunctive sub-threshold regime was
unreachable and the detector — correctly — never fired. This methodology
(1) reads the **pre-sigmoid margin** so the channels arrive graded,
(2) re-derives the floors on the graded scale,
(3) proves the detector logic with a dataset-independent **controlled-injection
grid**, and
(4) rebuilds the adversarial slice so its multi-vector rows are **measured to be
in-band**. Together these turn "0/40 by construction" into a detector whose fire
rate reflects the real prevalence of instantiable conjunctive attacks — a
defensible, working result — while the specificity control, single-vector
defenses, residual reporting, and controller A/B keep the rest of the claim
honest.
