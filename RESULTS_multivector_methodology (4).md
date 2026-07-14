# Multi-Vector Defense for RAG: Results and Methodology

*Reconciling the project's claims with the measured artifacts. Every figure
below is computed from the frozen runs `grounded_controller.jsonl` and
`grounded_baseline.jsonl` (220 rows each) and the calibration artifact
`calibration.json`.*

## 1. Contribution, stated honestly

This work places a multi-stage security pipeline around a local RAG system and
evaluates it against a labelled adversarial slice covering direct injection,
indirect (context) injection, semantic obfuscation, and a designed conjunctive
"multi-vector" class. The defensible contribution is threefold:

1. **A defense-in-depth pipeline that neutralizes *injection-style* multi-vector
   attacks end to end** — context sanitization, a dual-checkpoint canary, prompt
   spotlighting, and (optionally) dual-LLM privilege separation — demonstrated by
   **zero payload leaks across all 220 rows in both injection runs**.

2. **A negative result with a proof**: conjunctive *sub-threshold detection*
   built on injection-signature channels cannot be realized against a
   near-binary injection classifier paired with a redacting sanitizer. We give
   a closed argument for why, and confirm it empirically against the real
   scanners.

3. **A scoped boundary on a distinct attack class**: signature-free
   *misinformation*-style multi-vector attacks (planted false premise, no
   injection signature) succeed ~50% and are **not** addressed by this pipeline —
   spotlighting adds nothing over redaction on them, no existing signal detects
   them, and a purpose-built parametric-consistency check (`knowledge_conflict`)
   we tested also fails on benign false positives (§4.8). They require a
   different defense class.

We explicitly do **not** claim a working sub-threshold coordination detector, nor
that the pipeline defends the misinformation class. Both are retracted/scoped on
the evidence below.

## 2. System under test

Every prompt is scanned before generation (C3RF fusion gate over a DeBERTa-v3
injection score and a Llama-Guard category score). Every retrieved chunk is
sanitized (Step 6, llm-guard) with risk-adaptive strictness. Generation runs a
three-model local ensemble (Ollama) whose disagreement is itself a risk signal.
The answer is checked for grounding/faithfulness, the canary is verified, and
the output is DLP-scrubbed. A closed-loop controller can re-retrieve or refuse.
No paid APIs.

## 3. Evaluation slice

220 labelled rows in five classes, each carrying an `expectation` block:
`benign_control` (80, specificity), `poisoned_context` (40, indirect
injection), `adversarial_query` (40, gate adaptivity), `multivector_attack`
(40, the conjunctive class), `gate_slip_query` (20, semantic-obfuscation
residual).

## 4. Results

### 4.1 Specificity (benign_control)

The **gate** stays at base thresholds on clean traffic: of 80 benign rows it
flags **0**. Downstream, however, the cascade does produce benign false
refusals. The controller run blocks 10 benign rows and the baseline blocks 14;
none originate at the gate. Of the controller's 10, the measured breakdown is
**6 on the closed-loop attack-signal branch** (refused as "ungrounded with
attack signal", driven by ensemble-disagreement), **3 ungrounded-recovery**
refusals, and **1 controller error** (a model out-of-memory crash, not a
security decision). So benign specificity is clean at the gate but imperfect
downstream: ~6/80 benign rows are false-positives on the disagreement-as-risk
signal. The closed-loop controller still reduces total benign blocks vs the
open-loop baseline (10 vs 14).

### 4.2 Indirect-injection defense (poisoned_context)

The headline defensive result. Across all 40 poisoned rows in both runs, **40/40
poison chunks are redacted at Step 6 and 40/40 canaries remain intact**; the
benign query is answered. Defense here is **sanitization plus canary**, not
detection — and it is total.

### 4.3 Gate adaptivity (adversarial_query)

The gate flags **29/40 adversarial queries in the controller run** (25 BLOCK +
4 REVIEW) and **30/40 in the baseline** (25 BLOCK + 5 REVIEW); every adversarial
row is ultimately blocked (40/40) once downstream tightening is applied. The
residual the gate alone allows (11 controller, 10 baseline) is caught by the
cascade, not the gate.

### 4.4 Conjunctive sub-threshold detection — **negative result**

This is the class the multi-vector detector was built for, and it does not work.

**Measured.** On the 40 `multivector_attack` rows, 6 are gate-blocked before the
detector runs, leaving 34 evaluated. The detector fires on **1/34**, and that
single fire is *invalid*: its channels are `query_vector = 0.7`,
`context_vector = 1.0` — both supra-threshold — so it is a hard single-channel
detection, not a sub-threshold coordination. The intended condition,
`each_vector_subthreshold`, holds on **0/34** rows. 28/40 attacks are accepted
and answered (with the poison redacted; see §4.6).

**Why calibration cannot rescue it.** `calibration.json` shows the
`query_vector` channel fell back to `identity` (Platt was requested and
rejected, `helped = false`): `saturated_frac_raw = 0.92`, class separation
`separation_raw = -0.30`, and **worse** after calibration
(`separation_calibrated = -0.39`). Negative separation means attack and benign
distributions overlap; no monotone transform produces a usable graded band. The
`context_vector` channel was never calibrated at all.

**Empirical confirmation against the real scanners.** A subsequent attempt to
instantiate genuine conjunctive payloads — mild query suffixes plus "subtle"
poison chunks — was run through the *real* Step 3 (DeBERTa) and Step 6
(llm-guard) scanners. Result: **0/4 valid fires.** Two candidate pairs fired
only because the real scanner scored both channels at 1.0 (hard single-channel,
the gate would block the query); the other two did not fire because Step 6
dropped the poison (`survived_step6 = False`) before any coordination signal
could see it. Controls behaved correctly (benign: no fire; blatant
single-channel poison: context channel only, no multi-vector fire).

**The closed argument (why no payload fixes this).** A valid conjunctive fire
requires the query to pass Step 3 (so its injection score is *low*) **and** the
chunk to survive Step 6 (so its injection score is *low*). Any coordination
signal derived from injection signature is therefore also low on exactly the
rows that pass both gates — so it cannot fire. The only way to have a high
coordination signal while both injection scores are low is to measure something
other than injection (semantic intent), which is outside the scope of an
injection-signature detector. Conversely, any payload directive enough to raise
coordination saturates the near-binary classifier, at which point either Step 3
blocks the query or Step 6 drops the chunk — a single-channel catch, not a
conjunctive one. **There is no surviving-and-dangerous middle band for this
detector class.** This is a property of the detector, not a tuning failure.

### 4.5 Semantic-obfuscation residual (gate_slip_query)

Measurement-only. Roleplay/novel-framing wrappers around unsafe prompts carry
~0 injection signature on both channels; **5/20 (25%)** slip past the gate in
both runs. This residual is reported honestly; neither the gate nor any
injection-based detector is expected to close it.

### 4.6 End-to-end neutralization — **the result that holds**

The metric that matters is whether an attack's payload reached the user, not
whether a detector fired. Verified independently in each of the two 220-row
runs: **0 canary leaks**, and **74/74 injected poison chunks redacted in each
run** (40 in `poisoned_context` + 34 in `multivector_attack`), with the canary
intact — these are per-run figures, not a pooled 440-row denominator. Every
attack-class row (140 across the four attack kinds, per run) is neutralized — by
blocking (adversarial, gate-slip) or by redaction-plus-grounding (poisoned,
multi-vector). Neutralization succeeds *despite* detection failing on the
multi-vector class, which is the whole point of defense-in-depth.

*Caveat:* the frozen slice predates per-row `success_marker` instrumentation, so
this is measured on the canary signal. The end-to-end scorer
(`score_attack_success.py`) reads markers when present and falls back to the
canary otherwise, flagging marker-unavailable rows; a re-run on a marker-tagged
slice would tighten "neutralized" from "canary intact" to "canary intact AND
attacker's target string absent from the answer."

### 4.7 Closed-loop controller vs. baseline — utility, not security

The controller adds no measurable *security*: net multi-vector catch is ~0 in
both runs (§4.4). Where it differs is utility. It blocks **88** rows vs the
baseline's **101**; the 13-row gap is closed-loop *recovery* — on rows the
baseline refused as ungrounded, the controller's extra retrieval attempts
recovered a grounded answer (benign delivered 70 vs 66; poisoned 34 vs 28). This
costs extra ensemble generations. The comparison is also scorer-dependent and
should be reported as such: under the original gate-match scorer
(`slice_scoring_*.json`, prior version) the runs score 182 vs 183; under the
prevention-aware scorer (`slice_scoring_*.json`, current) 169 vs 170. Both
splits agree the two pipelines are *security-equivalent*; they differ only in
false-refusal cost. The honest claim is therefore "same security at lower
false-refusal cost via recovery," not "additional prevention."

### 4.8 Semantic multi-vector (misinformation) class

The conjunctive-injection negative (§4.4) raised the question of a *signature-free*
attack: a benign-looking query plus a planted-premise chunk that reads as neutral
reference text and asserts a false fact, carrying ~0 injection signature so it
passes Step 3 and survives Step 6. We built an 8-row slice of this class
(`build_semantic_slice.py`), every row admitted only after the *real* DeBERTa and
llm-guard scanners confirmed it passes the gate and is not redacted, and ran it
through the pipeline twice: once with the passive Step 8 and once with the
spotlighting Step 8 (§5). End-to-end success = the planted false value appears in
the answer and the true value does not (`score_attack_success.py`); rows that
assert the false value while also stating the true one were resolved by manual
reading.

**Neutralization (n = 8 per run):**

| Run | Attack succeeded | Neutralized | Rate |
|---|---|---|---|
| Passive Step 8 | 4/8 | 4/8 (2 clean answers + 2 corrections) | 50% |
| Spotlight Step 8 | 4/8 | 4/8 (1 blocked + 3 corrections) | 50% |

Spotlighting produced **no net change** on this class: 4/8 attacks succeeded
either way. It only reshuffled *how* the neutralized rows were saved (the passive
run gave 2 clean answers and 2 corrections; the spotlight run blocked 1 and
corrected 3), but the attack-success count is identical. This is the predicted
result — spotlighting strips *instruction*-authority, and a planted false fact is
not an instruction, so it passes straight through.

A necessary softener for any "50% neutralized" claim: most neutralized rows are
not clean defenses. The model typically *surfaces* the false value before
correcting it ("the sources say 87 °C, however the accepted value is 100 °C"),
which in a product is a degraded answer, not a clean refusal. Only ~2 rows (both
in the passive run) produced a genuinely uncontaminated answer.

**Separation — negative on both runs.** Using a real benign baseline
(`--benign-file grounded_controller.jsonl`), neither existing signal yields a
usable threshold: disagreement AUC 0.47 (passive) / 0.48 (spotlight), gaps
strongly negative; faithfulness AUC 0.62 (passive), gap −0.10. One number not to
overclaim: spotlight **faithfulness AUC rose to 0.86**, but the percentile gap is
−0.0225 (negative) at n = 8 — the attack's least-faithful tail still overlaps
benign, so there is no clean threshold. It is suggestive that datamarking/encoding
slightly depresses answer-faithfulness on attacks, but at n = 8 with a negative
gap it is **not** a usable detector and must be reported as "AUC rose to 0.86 with
no positive separation gap at n = 8," not as a positive result.

**Caveats.** n = 8 is a small slice; both the 50% rate and the AUCs are
directional, not robust point estimates, and the benign baseline is cross-run
(passive/injection benign vs semantic attack) — though the distribution overlap
is wide enough that a matched baseline is very unlikely to change the verdict.
Expanding the slice would tighten the numbers but is unlikely to move the
qualitative conclusion.

**Tested defense — parametric-consistency (`knowledge_conflict`): negative.** We
implemented a post-generation check (Step 9b) comparing the grounded answer
against the model's own parametric knowledge: a context-free answer is generated
and an LLM judge rates how strongly the grounded answer contradicts a *confident*
parametric claim (0–1), with an uncertainty guard forcing 0 when the parametric
answer is vague (so long-tail gap-filling is neither penalized nor, by the same
construction, detectable). Backend: `OllamaKnowledgeConflict` (llama3.2:3b
context-free answer + qwen2.5:3b judge); attack and benign were scored through the
*same* backend, so — unlike the cross-run baseline above — this comparison is
matched. It does not separate: attack (n = 8) p10/p50/p90 = 0.6/1.0/1.0; benign
(n = 80) p05/p50/p95 = 0.0/1.0/1.0; AUC 0.54 (≈ chance); gap
(attack_p10 − benign_p95) = −0.4 (positive required). The disqualifier is benign
saturation (median 1.0), and the cause is the free-form small-model judge, not the
parametric prior — it scores paraphrase, added specificity, and
broader-but-consistent answers as contradictions (e.g. "Indonesia is a republic,"
"Lee Kuan Yew PM 1959–1990," both agreements scored 1.0). Benign p05 = 0.0 shows
the agreement path works for a minority, but the judge over-fires on the rest;
because the failure is on the n = 80 benign side, it does not depend on the small
attack sample.

**Conclusion.** Misinformation-style multi-vector attacks succeed ~50% regardless
of spotlighting, and no answer-comparison signal separates them — faithfulness,
ensemble-disagreement, and a purpose-built parametric-consistency check
(`knowledge_conflict`) are all measured non-separators. The last of these is
itself a form of "verify the claim against trusted knowledge" using the model's
own prior as the trusted source, and it failed on benign false positives — so the
remaining direction is *external* provenance / source-trust weighting, not any
signal derived from the model's own outputs or priors. This class stands as a
measured, unsolved gap for this pipeline.

### Claim-vs-measured summary

| Claim | Intended | Measured | Status |
|---|---|---|---|
| Benign specificity | no over-blocking | 0/80 gate flags; 6/80 downstream security-branch false-refusals | holds at gate, imperfect downstream |
| Indirect-injection defense | redact + canary | 40/40 redacted, 40/40 canary | holds |
| Gate adaptivity | gate flags attacks | 29/40 (ctrl) / 30/40 (base) flagged, 40/40 caught by cascade | holds |
| Conjunctive sub-threshold detection | detector fires on co-activation | 1/34 fire (invalid); 0/34 sub-threshold; 0/4 vs real scanners | **does not hold (proven negative)** |
| Semantic-obfuscation closure | n/a (residual) | 25% slip, reported | measured residual |
| End-to-end neutralization | no payload reaches user | 0/220 leaks; 74/74 poison redacted | holds |
| Closed-loop adds prevention | extra catches | security-equivalent; recovery only | reframed to utility |
| Semantic (misinformation) neutralization | false fact doesn't reach user | 4/8 succeed, passive and spotlight alike (n=8) | **partial; spotlighting adds nothing** |
| Semantic-class detection | existing signal flags it | faithfulness/disagreement AUC ≤ 0.62 usable; 0.86 has negative gap | **does not hold** |
| Semantic-class tested defense (`knowledge_conflict`) | flag poison via parametric conflict | AUC 0.54, gap −0.4, benign median 1.0 (matched baseline) | **does not hold (judge over-fires)** |
| Fire-rule fusion (coactivation) | combine channels to catch more | same recall as query_only (40%); benign FP 45%→2.5% vs context_only | **holds as FP suppressor, not recall booster** |

### 4.9 Pre-generation filtering stack (mitigation A/B on planted attacks)

Separate from the 220-row conjunctive slice (§4.4), a **planted multi-vector
attack** harness measures whether turning on pre-generation mitigations (sanitize +
retrieval tightening + guarded prompts) reduces attack success *without* hard
refuse-on-detection.

**Matched n=200 (planted200, OFF ASR 31.0%).** Pre-gen stack without refuse
(`norefuse`) cuts ASR by only **2.0 pts** (31.0% → 29.0%, McNemar p=0.64, ns)
while benign false positives rise from **0% → 43%** (p<0.001;
`benign_cost_norefuse.json`). Full stack including refuse (`full`) cuts ASR by
**12.0 pts** (31.0% → 19.0%, p=0.002). The isolation arm `refuseonly` (refuse
on detection alone, no sanitize / tighten / guarded prompt) lands at the
**same 19.0%** as `full` (`report_refuseonly.json`) — so nothing beyond refuse
is buying ASR reduction. Adjacent ablations on the same OFF arm: `groundonly`
−0.5 pts; `groundplus` (grounding + refuse, no pre-gen filters) −12.5 pts —
refuse dominates, not sanitization/tightening/prompts.

The n=30 pilot (`planted30v2/report_norefuse.json`, 0.0-point drop) is superseded;
do not mix n=200 benign cost with n=30 ASR.

**Do not claim** a working conjunctive detector from this harness; the detector
is held fixed as an existing component and the contribution is mitigation layers.

### 4.10 Detector fire-rule comparison — fusion as FP suppressor

**Naming note — this is a different "coactivation" from §4.4.** §4.4's
co-activation is `multivector.py`'s sub-threshold condition: both channels
*below* their block floor yet jointly risky (proven structurally impossible
to fire validly). The fire rule below is `ExistingInjectionDetector`'s
`"coactivation"` option: both channels *above* their normal block floor — an
ordinary two-factor AND-gate, unrelated to sub-threshold detection. Same word,
opposite condition, two different modules; §4.4's negative result and this
section's positive one are both correct and do not conflict.

A separate question from mitigation ablations: *why fuse channels at all?* The
OFF-arm JSONLs already log raw per-channel scores (`detector.query`,
`detector.context`) for every row, so the live floors (`query_floor=0.20`,
`context_floor=0.30`) can be re-applied offline under alternate fire rules with
no model calls (`compare_detector_configs.py`). Sanity: re-derived coactivation
matches the pipeline's logged `is_attack` on all 200 attack + 200 benign rows.

| Rule | Recall (attacks caught) | Benign false-positive rate |
|---|---|---|
| query_only | 40.0% (80/200) | 3.0% (6/200) |
| context_only | **100.0%** (200/200) | **45.0%** (90/200) |
| or | 100.0% (200/200) | 45.5% (91/200) |
| coactivation — both-above-floor AND-gate (fusion, live default) | 40.0% (80/200) | **2.5%** (5/200) |

**Headline: fusion is not a recall booster here — it is a false-positive
suppressor that costs nothing in recall.** `context_only` / `or` "catch" every
attack, but context also fires on 45% of completely benign traffic: near-always-
on against this planted type (assertive, directive-sounding "fact-check"
strings in retrieved chunks), so high recall is not discrimination. Coactivation
matches `query_only` recall (80/200 both) because context already clears its
floor on every attack row; requiring *both* channels collapses benign FP from
45% (context alone) to 2.5% — slightly better than query alone's 3.0%. Context
alone looks like a perfect detector and is close to useless; coactivation is
what makes the context channel usable. Artifact:
`step4 dataset/detector_config_comparison.json`.

## 5. The defenses that hold

**Spotlighting (Step 8).** Retrieved context is wrapped in unforgeable
`<untrusted_document>` delimiters and, with rising risk, datamarked (sentinel
interleaving) or base64-encoded, under a system message that grants instruction
authority only to the system message and the user-question block. This strips
the context channel of instruction authority, so splitting a payload across
query and context buys the attacker nothing — independent of whether any
detector fires. (`step_08_augmented_prompt.py`.)

**Dual-LLM privilege separation.** A quarantined model reads raw context and may
emit only structured facts; the privileged model answers from those facts and
never sees raw context. An injected instruction is structurally demoted from
"command" to "one possibly-garbled fact," with a security argument rather than a
best-effort heuristic. (`quarantine_llm.py`.) This is the recommended hardening
for high-assurance deployments.

**Measured marginal value (updated).** The frozen injection runs were produced
*without* these two layers — they used the original Step 8 (a passive
instruction) — so the 0-leak neutralization in §4.6 was achieved by Step 6
redaction plus the canary alone, and on that slice the layers are
*unmeasurable* (every injection-flavored chunk is redacted before reaching Step
8, so nothing survives for them to act on). They are exercised only by payloads
that survive redaction — the signature-free semantic class — and there they have
now been measured directly (§4.8): **spotlighting's marginal value over redaction
on the misinformation class is zero** (4/8 attacks succeed with or without it).
This is consistent with the design boundary: spotlighting and the quarantine
extractor remove *instruction*-authority from context, which neutralizes
indirect-injection payloads, but a planted false *fact* is not an instruction, so
neither layer is the right defense for it. Spotlighting/quarantine remain sound,
recommended hardening for the injection threat they target; they are simply not a
defense for the semantic class.

## 6. Scope and limitations

- The conjunctive-detection negative result is specific to **injection-signature
  channels against a near-binary classifier with a redacting sanitizer**.
- The signature-free *semantic* (misinformation) multi-vector class has now been
  measured directly (§4.8) rather than left as a follow-up: it succeeds ~50%
  regardless of spotlighting; neither faithfulness nor ensemble-disagreement
  separates it; and a purpose-built parametric-consistency check
  (`knowledge_conflict`) we tested also fails on benign false positives. The
  separation results for faithfulness/disagreement are from a small slice (n = 8)
  and a cross-run benign baseline (the `knowledge_conflict` baseline is matched),
  so the rates are directional; a larger matched-baseline slice would tighten
  them but is unlikely to change the conclusion. The class is unaddressed by this
  pipeline; the remaining direction is *external* provenance / source-trust
  weighting, not any signal derived from the model's own outputs or priors.
- End-to-end neutralization on the *injection* classes is measured on the canary;
  marker-level confirmation requires re-running a marker-tagged slice.
- Semantic obfuscation (25% slip, §4.5) is a separate open residual, not
  addressed by this pipeline.

## 7. Reproducibility

Artifacts: `grounded_controller.jsonl`, `grounded_baseline.jsonl`,
`calibration.json`. Numbers in §4 reproduce from those files with a stdlib
reader (per-class counts of `gate.decision`, `blocked`,
`multivector.is_multivector`, `retrieval.poison_redacted`,
`grounding.canary_intact`). End-to-end neutralization:

```bash
python score_attack_success.py --in grounded_controller.jsonl grounded_baseline.jsonl
```

Semantic class (§4.8), from the project tree with the venv active:

```bash
python build_semantic_slice.py --out semantic_slice.jsonl                 # real-scanner validated; must not print STUB
python run_full_pipeline.py --slice semantic_slice.jsonl --out grounded_semantic_passive.jsonl     # passive Step 8
#   (swap in spotlighting step_08_augmented_prompt.py, then:)
python run_full_pipeline.py --slice semantic_slice.jsonl --out grounded_semantic_spotlight.jsonl   # spotlight Step 8
python score_attack_success.py --in grounded_semantic_passive.jsonl grounded_semantic_spotlight.jsonl
python grounding_separation_probe.py --in grounded_semantic_spotlight.jsonl --attack-kind semantic_multivector --benign-file grounded_controller.jsonl
python review_ambiguous.py --in grounded_semantic_spotlight.jsonl         # resolve ambiguous rows by hand
```

Parametric-consistency check (§4.8, `knowledge_conflict`) — note the benign
baseline must be regenerated through the same backend. `--kc-backend ollama` is
the live CLI default and is spelled out below only to make the backend explicit
and avoid accidentally selecting `stub` (which is demo-only and no-ops on real
questions):

```bash
python run_full_pipeline.py --slice semantic_slice.jsonl   --out grounded_semantic_kc.jsonl    --kc-backend ollama
python run_full_pipeline.py --slice adversarial_slice.jsonl --out grounded_controller_kc.jsonl --kc-backend ollama --resume   # matched benign baseline
python grounding_separation_probe.py --in grounded_semantic_kc.jsonl --attack-kind semantic_multivector --benign-file grounded_controller_kc.jsonl --signal-field knowledge_conflict
```

## Appendix — disposition of the prototype code

- **Keep** (valid under this result): `step_08_augmented_prompt.py`
  (spotlighting), `quarantine_llm.py` (dual-LLM), `score_attack_success.py`
  (end-to-end metric).
- **Do not claim as detection**: the `coordination_vector` channel and
  `coordination_channel.py`. Its lexical signal overlaps the injection scanner
  it sits on, so it is redundant by construction (see §4.4) and validated 0/4
  against the real scanners. The `multivector` module may remain wired as
  defense-in-depth (it refuses on strong co-activation, 3 rows) but must not be
  presented as conjunctive sub-threshold detection.
- **Partially keep**: the `success_marker` instrumentation added to the slice
  builder is useful for end-to-end measurement; the conjunctive
  `multivector_attack` pairs must be relabelled honestly (they are not validly
  sub-threshold against the real scanners), or the class dropped.
