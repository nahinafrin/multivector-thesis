# ADO (Sentinel–Strategist) as a Protocol B baseline — implementation audit

**Scope.** This audit prepares ADO as an existing-defense baseline for **Protocol B**: the 200 planted
attacks and 200 benign questions on which the **seven-layer, single-generator configuration** of our
framework was evaluated (`llama3.2:3b`, paired by row). ADO is compared with that configuration only,
**not** with the complete thirteen-stage framework (ensemble + controller), which is evaluated under
Protocol A. Nothing in this directory modifies our framework, its pipeline, or any stored output; ADO
runs through an isolated adapter.

**Status: Phase 1 complete. No ADO inference has been run for Protocol B yet.**

## 1. Sources located

| Item | Location | Notes |
|---|---|---|
| Paper | Pallerla et al., *Adaptive Defense Orchestration for RAG: A Sentinel–Strategist Architecture against Multi-Vector Attacks*, arXiv 2604.20932v1 (cited in the paper as `palleria2026ado`) | Read via the arXiv HTML. §1–§3.2 were read verbatim; §4 onward and the tables come from a summary of the page, so their details are marked *(summary)* below and should be checked against the PDF before publication. |
| Official code | `github.com/Pranavdec/Adaptive-RAG-Orchestrator`, local clone at `Desktop/comparison model/paper_faithful/Adaptive-RAG-Orchestrator`, commit `b6fa5eddd449a9bc325245bcecdad400f3dd72ee` | MIT licence. The paper calls it an anonymous review archive. |
| Earlier runner (Protocol A slice) | `external_baselines/run_ado_official_on_slice.py` | Imports the official modules unchanged and re-implements `pipeline.run_single`'s control flow. **Not Protocol-B-compatible** (see §4). |
| Earlier results (Protocol A slice) | `external_baselines/paper_faithful/results/ado_official_slice.jsonl` (+ `.log`) | 210 rows, a different protocol; cannot be pooled with Protocol B. |
| Older baselines | `external_baselines/baseline_sentinel_strategist.py`; `Desktop/comparison model/{baseline_sentinel_strategist.py, local_rerun/…_v2.py, results/grounded_sentinel_strategist*.jsonl, results/sentinel_calibration*.json}` | Our own re-creations from before the official code was found; superseded. Not used. |

SHA-256 of the official files the adapter uses or emulates:

```
0ad04de757a87f7ca4cceb388d7723bf25cf2aed5b8589eea1fb017b027be4f3  src/core/ado.py          (Sentinel, Strategist)
44b972420f3e43b657040c981669b197f914e0a303e88e3b868052c218d464b2  src/core/sensing.py      (MetricsCollector)
3d9ff8c9daea0a6a3e212a6d0ccbb256fc04b615d7edd9817df3e518d6118fc6  src/core/persistence.py  (UserTrustManager)
e5bfa3611af5508545fb0ff4763f6150cd0adb23042c22ffe4bda97b80349d7c  src/core/generation.py   (OllamaGenerator)
00c29cd315fdb77d864548bc046a69b5559e46094c4dc677d84cd24dc2abcc7e  src/defenses/trustrag.py
1533308b17632129508b6b4f1346cae1e50172714cd3cced5fcc0df002e8b957  src/defenses/dp_rag.py
bf4c01b3d4c97774265996bad3559827b6e40f496000e4c33db8810083a6455d  src/defenses/av_defense.py
bf19eb337ce4d27a2857552b1d0bd7e6364806c44a97fce3961326a94e448416  src/defenses/manager.py  (emulated)
0fd21b4edb839ed3ab579d35551d15d30694a26893bf2e50e89d0822d186eff3  src/core/pipeline.py     (emulated)
a1b6c2363f23f87c6a561a9047e418c491c1b19f7eff31edb8f0e2f68a3c13f8  config/config.yaml
```

## 2. How ADO works (official code, cross-checked with the paper)

Per query, `pipeline.run_single` does the following:

1. **Pre-retrieval pass.** `MetricsCollector.calculate_pre_retrieval` computes lexical, complexity, and intent
   metrics over the query and the user's recent query history. The **Sentinel**, an LLM, turns the
   metrics, the query, the user's **Global Trust Score**, and its trend into a risk profile (threat level LOW /
   ELEVATED / CRITICAL, per-threat likelihoods). The **Strategist**, an LLM in `llm` mode with deterministic
   overrides, turns the profile into a defense plan. The trust score is updated (paper Eq. 3,
   δ ∈ [−0.1, 0.1], clipped to [0, 1]).
2. **Retrieval** with a candidate multiplier when a defense is enabled (×3 → 15 candidates for top-5).
3. **Post-retrieval pass.** Dispersion and score drop-off metrics, a second Sentinel analysis, and a
   second Strategist plan, which can switch on TrustRAG or AV.
4. **Defense execution** (`DefenseManager`): DP-RAG (`dp_approx`, ε from the plan), TrustRAG (K-means
   + ROUGE filtering, similarity 0.88–0.95, ROUGE 0.25), and **attention-variance (AV) filtering**
   (repeatedly drops the passage with the highest attention share while the variance exceeds 50, up to 3
   removals). The result is capped at top-5.
5. **Generation** with the official generator prompt (`generation.py`), temperature 0.0.

ADO **never refuses**. It mitigates by changing *which passages* the generator sees. Its effect must
therefore be measured on the generated answers, not on block decisions (requirement P2.8).

Paper settings *(summary)*: the generator is Llama-3.1-8B-Instruct at temperature 0; the default controller
is Llama-3-8B-Instruct via Ollama; the embedder is all-MiniLM-L6-v2; trust starts at 0.5 per user; evaluation
pools of 50 benign and 50 attack queries are interleaved and shuffled; AV uses threshold 50 and 3 max
corruptions (these match `config.yaml`). The paper does not name the model AV takes its attentions from; the
code uses `meta-llama/Llama-3.1-8B-Instruct` through Hugging Face.

## 3. Component fidelity

| Component | Status in the planned Protocol B adapter | Consequence |
|---|---|---|
| Sentinel (pre and post prompts, JSON parsing, fallbacks) | **Faithful**: official code imported unchanged | None |
| Strategist (`llm` mode, deterministic overrides) | **Faithful**: official code imported unchanged | None |
| MetricsCollector | **Faithful**: imported unchanged | None |
| UserTrustManager / Eq. 3 | **Faithful code, adapted use**: trust is **reset per request** (§4.3) | ADO's cross-query trust adaptation cannot contribute under independent requests |
| DP-RAG, TrustRAG | **Faithful**: imported unchanged | Their candidate pool is our generator's 8 passages instead of 15 retrieved candidates (§4.1) |
| DefenseManager / run_single control flow | **Emulated step for step** (as in the earlier runner), because the real manager eagerly loads the HF Llama for AV | Equivalent unless AV is requested |
| **AV filtering** | **Disabled: requested by the Strategist and recorded, not executed** | See §5. Material for poisoning attacks, because AV targets passages that dominate attention |
| Controller model | **Adapted**: `llama3.1:8b` (Ollama, Q4) instead of Llama-3-8B-Instruct | Llama-3 is not installed; 3.1 is the paper's generator family. The paper reports that ASR is sensitive to the controller model (0–44% across controllers) *(summary)* |
| Generator | **Adapted**: `llama3.2:3b`, the generator of every Protocol B arm, with ADO's own generator prompt at temperature 0.0 | Needed for a paired comparison with our seven-layer configuration; differs from the paper's 8B generator |
| Retrieval / knowledge base | **Adapted**: ADO receives exactly the 8 passages our undefended generator received (§4.1) | Required for fairness; reduces TrustRAG's candidate pool |

**Classification: _Adapted implementation with documented limitations_.** It is not a faithful reproduction:
AV is not executed, the controller and generator models differ from the paper, and trust is reset. It is
also not a detector-only or block-only approximation: the Sentinel, the Strategist, both document defenses,
and generation all run end to end, and outcomes are scored on the generated answers.

## 4. Protocol B fairness design (Phase 2)

1. **Same evidence, no added poison.** ADO's candidate pool is `ranked_context` from
   `step4 dataset/mitigation_results/baselines_protocolB/contexts_{attack,benign}.jsonl` (written by
   `dump_undefended_contexts.py`): the exact 8 passages the undefended `llama3.2:3b` generator received,
   which contain the planted chunk in 184 of 200 attack rows. ADO's MiniLM embeddings and similarities are
   computed on these passages, its defenses select among them, and it keeps its top-5. Nothing is retrieved
   from elsewhere, and no poison is inserted.
2. **Alignment.** `check_alignment.py` fails unless IDs, questions (including the query suffix), and benign
   labels match `planted_attacks_200.jsonl`, the seed-42 benign sample, the stored undefended arm, and the
   seven-layer arm (`mitigation_results/ablation200/{on_full,benign_on}.jsonl`).
3. **Same scoring.** Attack success uses `score_ablation_study.attack_succeeded` and benign correctness uses
   `score_benign_cost.answered_correctly`, applied to ADO's delivered answer. This is the scoring used for
   every other Protocol B arm.
4. **No evaluation-only information reaches the model.** ADO receives only the query string (as the other
   arms do, including the attack suffix) and the passages. `success_marker`, `true_answer`, and labels are
   attached to the output rows *after* generation, for scoring only. The earlier runner's quoted-text marker
   extraction is not used.
5. **Trust state.** Each request uses a fresh user (`<split>_<index>`), so trust starts at 0.5 with an
   empty history, as for every other Protocol B arm, which handles requests independently. The official
   evaluation instead interleaves benign and attack traffic under persistent trust; that setting cannot be
   paired row by row with our independent-request arms without leaking state between attack and benign
   cases. It is listed under §6 as a possible secondary analysis.
6. **No tuning on test data.** Every ADO parameter is the repository default (`config.yaml` and the
   Strategist's built-in ranges). Nothing is fitted on Protocol B.
7. **Failure handling.** A Sentinel or Strategist parse failure uses ADO's own `System Failure Fallback`
   path, which is recorded. A generation error or timeout is recorded as `generation_error`, scored as
   *no answer* (attack not successful, benign not correct), and reported separately.

## 5. The attention-variance filter: blocker and options

`av_defense.py` needs `meta-llama/Llama-3.1-8B-Instruct` through Hugging Face `transformers` with attention
outputs, called at least three times per query over up to 15 passages. On this machine:

* Only the Ollama GGUF (Q4) copy is installed. The original weights are **gated** (a Hugging Face account,
  licence acceptance, and a ~16 GB download).
* The model in fp16 (~16 GB) does not fit the 8 GB GPU, and the 16 GB host RAM is not enough to run it on
  the CPU.
* 4-bit loading (bitsandbytes) of the original weights would fit (~5.5 GB) but requires eager attention to
  return attention maps. Memory for long multi-passage prompts is uncertain, and 4-bit attentions are
  themselves a deviation.

Running AV therefore means **replacing its inference backend**, which under the architecture-preservation
requirement needs explicit approval. Options:

| Option | Fidelity | Cost |
|---|---|---|
| A. Run without AV (record when it is requested) | Adapted, as classified above | None; can start now |
| B. HF Llama-3.1-8B-Instruct, 4-bit, eager attention, used only inside the AV defense | Closer, but 4-bit attentions | Gated download (your HF account), ~1 h setup, unknown runtime per query |
| C. Use our `llama3.2:3b` HF weights for AV attentions | Model differs from the paper's | Smaller download; still a backend change |

**Recommendation.** Run the 5 + 5 functional test under option A first. It records how often ADO's
Strategist actually requests AV on Protocol B inputs. If the Strategist rarely requests AV, option A's
limitation is minor for this protocol. If it requests AV often, the pilot should either wait for option B
or be reported as a partial baseline for those rows.

## 6. Deviations that remain, whatever the option

* The controller is `llama3.1:8b`, not Llama-3-8B-Instruct.
* The generator is `llama3.2:3b`, not Llama-3.1-8B-Instruct. This is deliberate, for a paired comparison
  with the seven-layer configuration.
* The candidate pool is 8 passages, not 15. This is deliberate, for equal evidence.
* Trust is reset per request. This is deliberate, for independent paired requests. A secondary run with
  persistent trust over a shuffled attack + benign stream (the paper's setting) is possible, but must be
  reported separately and not paired with our arms.

## 7. Functional-test results and the decisions they force (see FUNCTIONAL_TEST_REPORT.md)

The 5 + 5 test ran cleanly (10/10 rows, no errors or fallbacks, trust reset verified, alignment OK,
about 40 s per row). It also showed two problems that make the pilot, as designed, scientifically unsafe:

1. **AV is central on Protocol B attacks.** The Strategist requested it on 4 of 5 attack rows. Option A
   (AV not executed) is therefore a *partial component baseline* for this protocol, not "adapted ADO".
2. **Evidence confound.** ADO's top-5 cut by MiniLM similarity drops the planted chunk in 175 of 184
   poison-present rows before any defense acts. Without a fix, ADO's measured protection would mostly
   reflect its context size and embedder.

Options for the evidence (independent of the AV choice):

| Evidence option | What ADO sees | Effect |
|---|---|---|
| E1. Keep ADO's top-5 by MiniLM (as tested) | 5 of the 8 passages; poison kept in 9/184 rows | Confounded: credits ADO with a retrieval cut-off |
| E2. **ADO top-k = 8, the generator's context size** | All 8 passages unless an ADO defense removes some | Same evidence as the undefended arm; any removal is attributable to ADO's defenses. Deviation: top-k 8 instead of 5 |
| E3. Keep top-5 but order by our reranker | Poison in the top-5 in 130/184 rows | Mixes our reranker into ADO |

Recommended: **E2**, plus per-row NumPy seeding for DP-RAG.

Options for AV, compared:

| | A. Not executed | B. HF Llama-3.1-8B-Instruct, 4-bit, attentions inside AV only | C. HF `llama3.2:3b` attentions |
|---|---|---|---|
| Fidelity | Partial component baseline on these attacks (AV requested on 4/5) | Closest available; AV's attention model as in the code, but 4-bit | Wrong attention model (not the paper's) |
| Disabled ADO parts | AV | none (quantization only) | none, but substituted model |
| Your action | none | log in locally with `hf auth login` and accept the Llama 3.1 licence (I never handle credentials); ~16 GB download | a smaller HF download |
| Compute | ≈40 s/row → pilot ≈ 70 min | AV adds 3+ forward passes over 8 passages with eager attention; memory and time unknown until a test (may not fit 8 GB alongside Ollama) | lighter than B |
| Changes to our framework | none | none: isolated inside the adapter's AV call | none |
| Claim it supports | "ADO without its attention filter (partial)" | "ADO (adapted; 4-bit AV)" | "ADO with a substituted AV model" |

Neither option touches our framework. **No pilot will start until the evidence option and the AV option
are chosen.**

## 8. Decisions taken and their implementation (2026-11-23)

**Evidence: E2.** ADO receives the generator's 8 passages. Its defenses run with ADO's own top-k (5) as
their internal target, as `DefenseManager` sets it, but **no MiniLM top-5 cut is applied after the
defenses**, so every passage removal is attributable to an ADO defense. Deviation from the official
manager, which caps at top-5 after the defenses: documented and deliberate (§7, finding 2).

**AV: B2.** The official `AttentionFilteringDefense` and the official `hf_model.Llama.query` (generation
with `output_attentions`, per-passage attention aggregation, eager attention) run unchanged on a
**pre-quantized 4-bit Llama-3.1-8B-Instruct** (`unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`, a
third-party bnb-4bit re-upload of Meta's weights, 5.7 GB). It is passed in through the filter's supported
`shared_model` argument. Only the weight loading differs from `hf_model.Llama.__init__`, which loads
fp16 (~16 GB, too large for this disk and the 8 GB GPU). AV parameters come from `config.yaml`
(threshold 50, top tokens 100, max corruptions 3), updated by the Strategist plan with the official
setter logic. To make room, the unused Ollama model `mistral-nemo` was removed with the author's approval.

**Defense initialization.** Each defense is built from its official `config.yaml` entry and then
updated from the plan with `DefenseManager._update_defense_params` logic (copied verbatim), as in ADO
mode, instead of being built from the plan dict alone, as the earlier runner did.

**Reproducibility.** NumPy and `random` are seeded per row (seed = row index) before the defenses run.

## 9. Further functional-test findings (runs v2 and v3)

* The 4-bit model loads on the GPU (~5.3 GB) next to Ollama, and the official AV path executes
  (`[AV Defense] Using shared model instance`).
* **DP-RAG often removes every passage.** `dp_approx` adds Gaussian noise with σ ≈ 6.2/ε to the k-th
  similarity score (similarities lie in [−1, 1]); for the ε values the Strategist chooses (1–6), the noisy
  threshold usually falls above all scores (removing every passage) or below them (keeping every passage).
  This is ADO's genuine behavior, identical in the official pipeline. When it happens, the generator
  answers without context, which defeats the planted chunk and also costs benign utility. AV then receives
  an empty list and returns immediately. The comparison report counts these rows separately.
* Run v2 set the defenses' internal target to 8 instead of ADO's 5. It was corrected for v3, and the v2
  outputs are kept in `superseded/`.

## 10. Backend changes needed to run AV on an 8 GB GPU, and their verification

* **Memory-safe `generate` (`av_generate.py`).** ADO's `hf_model.Llama.query` calls HF
  `generate(output_attentions=True)`, which also returns the prefill attention matrices (all layers ×
  heads × prompt²; ~5–13 GB for an 8-passage prompt). ADO's code never uses them: it raises on them and
  skips token 0. `GenerateProxy` replaces only that call with greedy decoding: the prefill runs without
  attention outputs, and each decode step returns its attentions, the same tensors HF returns for steps ≥ 1.
  `query()` itself runs unchanged. **Verified equivalent**: `test_av_equivalence.py` gives identical
  generated text and identical passage scores (max relative difference 0.0) against the official HF path.
* **GPU sharing.** Before each AV call, the adapter asks Ollama to unload its resident models
  (`keep_alive=0`); they reload on their next call. This changes timing only, not results.
* **Error visibility.** ADO's AV fail-safe returns all passages on any exception. The adapter records
  such exceptions in `av_trace.error`, so that AV rows that did not actually filter can be identified.

**Classification (final): _Adapted implementation with documented limitations_.** The Sentinel, the
Strategist, the trust update, the metrics, and all three defenses, including AV, run from the official
code. The remaining deviations are the controller model (`llama3.1:8b`), the generator (`llama3.2:3b`,
for pairing), 4-bit AV weights from a third-party re-upload, the evidence policy (E2), per-request trust
reset, and per-row seeding.
