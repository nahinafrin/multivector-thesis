# Adaptive-tier external evaluation: recovered code, provenance and reproducibility

**Status (2026-11-25).**
- The four adaptive-tier scripts are recovered byte-identical from git history and restored to
  `step4 dataset/adaptive_external/`.
- Their provenance is authenticated by git history.
- The saved PoisonedRAG and BIPIA results are recomputed from the saved outputs.
- Every row's routing is reproduced deterministically.
- A fresh-inference rerun has **not** been performed.

## 1. Purpose

The PoisonedRAG (150 rows) and BIPIA (50 rows) results in both papers come from
`step4 dataset/ext_n200_results.jsonl`. That file was produced by an **adaptive-tier variant** of the
13-stage framework. This variant is different from both:
- Protocol A, the complete pipeline with the ensemble and controller;
- Protocol B, the seven-layer single-generator harness.

Its four source files were moved to the private backup repository in the repository trim `fe71c56`, so
the public repository could not re-run the external evaluation (ICCIT Reviewer #5, reproducibility).
This document records:
- where the code came from;
- what it actually executed;
- how the reported numbers are verified;
- what remains unverified.

## 2. Recovered files and provenance

| File | Role | Original location | Added | Last modified | Removed | SHA-256 (LF) |
|---|---|---|---|---|---|---|
| `run_full_pipeline_adaptive.py` | Orchestrator: `process_adaptive()` and the CLI that wrote the results | `step4 dataset/` | `2bd8841` (2026-07-02) | `1d87f26` (2026-08-10) | `fe71c56` (2026-11-23) | see `PROVENANCE.json` |
| `adaptive_risk.py` | Noisy-OR cumulative risk; LOW/MEDIUM/HIGH tier table; monotone `AdaptivePolicy` | same | `2bd8841` | `111f92c` (2026-08-08) | `fe71c56` | ″ |
| `adaptive_verification.py` | `verified_generate()`: single generator at LOW/MEDIUM; ensemble and LLM policy check at HIGH; grounding always | same | `2bd8841` | `1573234` (2026-08-08) | `fe71c56` | ″ |
| `adaptive_prompt.py` | Tier-specific Stage 8 system preambles | same | `2bd8841` | `111f92c` | `fe71c56` | ″ |

**Authentication.** The authentication is exact, from git history:
1. Each script has the same git blob at every commit from `1d87f26` (2026-08-10 10:54) to the parent of
   its removal, `fe71c56^`. They were never edited between the run and their removal.
2. The run log `step4 dataset/run_n200_log.txt` (UTF-16) records the run:
   - it started on 2026-08-11 at 10:51:13 and finished after 4,764.2 s;
   - it processed "200 ok, 0 errors" and wrote `ext_n200_results.jsonl`;
   - its PowerShell echo shows `.\.venv311\Scripts\python.exe run_full_pipeline_adapt…`.
3. The results, the input file and the log were committed in `1d99b23` at 12:44, 34 minutes after the
   run ended. No later commit changed them.
4. The copies in the backup repository (`nahinafrin/multivector-thesis-backup`, same paths) are
   byte-identical to the history.
5. An environment record was committed at 13:10 the same day (`92ed249`, "Pin exact environment behind
   the n=200 external validation result").

**Run-time dependencies.** The scripts import 25 further local modules, found by a transitive import
scan. Ten of them changed after the run:
- Stage 3c (`dataset/step_03c_fusion_gate.py`);
- Stages 4, 6, 7, 9, 10 and 11;
- `run_full_pipeline.py`, `risk_feedback_controller.py` and `preprocess_dataset.py`.

Running the recovered scripts against today's modules would therefore **not** execute the code that
produced the results. All 25 modules are restored exactly as committed in `1d99b23` under
`adaptive_external/runtime_snapshot_1d99b23/`.

**Files that are not recovered.**
- `gate_diagnosis.txt` was cited as the reason the semantic gate was disabled. It was never committed.
  The reason survives only as the message the code writes to every row.

## 3. Restored layout (`step4 dataset/adaptive_external/`)

| Path | Content | Origin |
|---|---|---|
| `original/` | the four scripts | byte-identical to `fe71c56^` |
| `runtime_snapshot_1d99b23/` | the 25 imported local modules | byte-identical to `1d99b23` |
| `environment_at_run/` | `environment_info.txt` (Ollama 0.32.7, model digests) and `requirements-lock.txt` | byte-identical to `92ed249` |
| `historical_docs/INTEGRATION.md` | hook documentation for the orchestrator | byte-identical to `1d99b23` |
| `PROVENANCE.json` | commits, git blobs and SHA-256 for every file above | **new** |
| `run_adaptive_external.py` | launcher: puts `original/` and the snapshot first on the import path, switches to `step4 dataset/`, runs the original CLI unchanged; refuses to overwrite the saved results | **new**, contains no pipeline logic |
| `../verify_adaptive_external.py` | verification script (Section 9) | **new** |
| `verification_report.json` | output of the verification script | **new** |

Nothing in the 13-stage framework, the Protocol B harness or any saved output was modified. The adaptive
variant lives only in this folder.

## 4. Environment

The run-time record is in `environment_at_run/`.

**Models.**

| Component | Version |
|---|---|
| Ollama | 0.32.7 |
| `llama3.2:3b` | `a80c4f17acd5` |
| `mistral:7b` | `6577803aa9a0` |
| `qwen2.5:3b` | `357c53fb659c` |
| `llama-guard3:1b` | `494147e06bf9` |
| Injection classifier | `protectai/deberta-v3-base-prompt-injection-v2`, revision `89b085cd…` (in the run log) |

The graded-score path loads the classifier's `main` revision. Both cached revisions give identical
scores on all 200 rows.

**Packages at the run.** Python 3.11.9, torch 2.13.0, transformers 4.51.3, llm-guard 0.3.16,
sentence-transformers 5.7.0, faiss-cpu 1.15.0, spaCy 3.8.14, presidio 2.2.358.

The current `.venv311` differs slightly (torch 2.12.0, sentence-transformers 5.5.1, faiss-cpu 1.14.2).
For a faithful rerun, install `environment_at_run/requirements-lock.txt`.

**Data used by the run, unchanged since the run (checked by the verifier).**
- `step4 dataset/kb_wiki.faiss` and `kb_wiki.texts` (rag-mini-wikipedia, 3,200 passages)
- `graded_config.json` (T = 7.4114)
- `external_validation_n200.jsonl`

## 5. Commands

**Check that every module resolves to the recovered code.** No inference; 29/29 modules.

```powershell
cd "step4 dataset"
.\.venv311\Scripts\python.exe adaptive_external\run_adaptive_external.py --check-imports
```

**Verify provenance and recompute every reported number from the saved outputs.** No LLM.

```powershell
.\.venv311\Scripts\python.exe verify_adaptive_external.py --recompute-graded
```

**Fresh inference.** This was the original invocation. It needs Ollama with the four models, takes about
80 minutes, and has **not** been run here. It writes to a new file.

```powershell
.\.venv311\Scripts\python.exe adaptive_external\run_adaptive_external.py --in external_validation_n200.jsonl --out ext_n200_rerun.jsonl --mode adaptive --disable-semantic-gate
```

`--mode adaptive` is the default, and `--disable-semantic-gate` is established by the saved rows (Section 6).

## 6. What actually executed

The evidence comes from the recovered source and the saved per-row traces.

**Risk and tiers** (`adaptive_risk.py`).
- Cumulative risk is a noisy-OR over prompt-injection, semantic-intent, context-suspicion,
  ensemble-disagreement and grounding-mismatch signals.
- Tier boundaries: LOW ≤ 0.30 < MEDIUM ≤ 0.60 < HIGH.
- The tier is chosen once after the input gates (from the graded query score only) and again after
  grounding. It never decreases.

| Tier | top-k | Similarity floor | Stage 6 threshold | Stage 7 floor | Prompt | Generation | Policy check |
|---|---|---|---|---|---|---|---|
| LOW | 8 | 0.55 | 0.50 | 0.20 | standard | single `llama3.2:3b` | no |
| MEDIUM | 5 | 0.65 | 0.40 | 0.30 | guarded | single `llama3.2:3b` | no |
| HIGH | 3 | 0.72 | 0.30 | 0.40 | strict | ensemble `llama3.2:3b`, `mistral:7b`, `qwen2.5:3b` | `llama3.2:3b` PASS/FAIL |

Generation uses temperature 0.2 and seed 42. The policy check uses temperature 0.0 and seed 42.

**Stage-execution matrix.** Counts are PoisonedRAG / BIPIA. "Called" means invoked by
`process_adaptive()`.

| Stage | Implementation (run-time version) | LOW / MEDIUM first pass | HIGH re-pass | Evidence |
|---|---|---|---|---|
| 1 Input | `dataset/step_01_user_input.py` | executed, 150 / 50 | – | source; saved `question` |
| 2 Normalisation | `dataset/step_02_normalization.py` | executed, 150 / 50 | – | source |
| 3/3c Fusion gate | `dataset/step_03c_fusion_gate.py` (DeBERTa-v3 + Llama Guard) | executed, 150 / 50; blocked 2 / 1 | – | saved `gate.*`; `block_stage` |
| 3d Semantic intent gate | `step_03d_semantic_intent_gate.py` | **bypassed** (`--disable-semantic-gate`) | – | `meta.semantic_intent_decision = "SKIPPED"` on 148 / 49 |
| 4–5 Embedding, FAISS search | `step_04`, `step_05` | executed, 148 / 49; top-k 8 (LOW) or 5 (MEDIUM); similarity floor | – | source; ranked context ≤ top-k on every row |
| Poison placement | harness hook in `process_adaptive()` | appended after retrieval, 148 / 49 | – | `raw_chunks[-1] == poison_injected` on every row |
| 6 Sanitisation + canary | `step_06` (tier-pinned threshold) | executed, 148 / 49; poison dropped 0 / 10 | – | `retrieval.sanitization_strictness` = 0.50 / 0.40 |
| 7 Reranking | `step_07` (tier-pinned floor) | executed, 148 / 49; poison removed 0 / 23 | – | `retrieval.rerank_min_score` = 0.20 / 0.30 |
| 8 Prompt | `step_08` with `adaptive_prompt` profile | executed: standard (LOW), guarded (1 MEDIUM row) | **returns immediately**: the row is already blocked, so the strict prompt is never built | source of `step_08.run` at `1d99b23` |
| 9 Generation | `step_09.generate_single` / `generate_candidates` | **single `llama3.2:3b`**, 148 / 49 | **ensemble of 3, 0 / 36** | recovered routing rule reproduced on every row (Section 8) |
| HIGH policy check | `adaptive_verification._policy_check` | not executed | executed, 0 / 36 | source; verdict not saved |
| 10 Grounding + canary | `step_10_grounding_judge.py` | executed, 148 / 49; blocked 4 / 44 | returns immediately (row already blocked) | saved `grounding.*` |
| Final adaptive decision | `process_adaptive()` | – | ungrounded at HIGH → `adaptive_policy` block, 0 / 36 | `block_stage = "adaptive_policy"` |
| 10A Multivector detector | `multivector.py` | **not executed** (only imported through `run_full_pipeline`) | not executed | empty `multivector` field on 150 / 50 |
| 10B Risk controller | `risk_feedback_controller.py` | **not executed** | not executed | empty `controller` field on 150 / 50 |
| 11 Output sanitisation (Presidio) | `step_11` | **not executed** | not executed | empty `output_sanitization` field on 150 / 50 |
| 12 DLP | `step_12` | **not executed** | not executed | empty `dlp` field on 150 / 50 |
| 13 Safe response | `step_13_safe_response.py` | executed, 150 / 50 | executed | saved `final_response` |

**Routing results.**

*First-pass tiers.* The tier is read from the pinned Stage 6/7 values and recomputed from the graded
query score:

| Set | LOW | MEDIUM | Blocked at gate |
|---|---|---|---|
| PoisonedRAG | 147 | 1 | 2 |
| BIPIA | 49 | 0 | 1 |

Every answer delivered to a user was therefore generated by a **single `llama3.2:3b`**. That is 144 of
the 148 PoisonedRAG rows that reached generation (4 were blocked by grounding) and 5 of the 49 BIPIA rows.

*HIGH re-pass.* The orchestrator escalates a row only if its answer is ungrounded and the late risk
reaches HIGH. Late risk is the noisy-OR of the graded query score, the graded context score, and
1 − faithfulness. This happened on **exactly the 36 BIPIA rows blocked by `adaptive_policy`**, and on no
PoisonedRAG row. Those 36 rows were re-generated by the three-model ensemble and checked by the policy
LLM, then refused. Because Stages 8 and 10 return immediately on an already-blocked row, the re-pass
reused the LOW-tier prompt and could not re-ground the answer. The escalated rows were therefore always
refused, and the re-pass did not change any outcome.

*Corrections to the earlier audit.*
- The earlier figures "139/150 PoisonedRAG and 44/50 BIPIA rows at low risk" came from counting rows
  with 8 ranked chunks. That is an imprecise proxy, because sanitisation, the similarity floor and the
  reranking floor can all leave fewer than 8. The tier evidence gives 147 + 1 / 150 and 49 / 50.
- The statement that the multivector detector, controller, Presidio and DLP were inactive is
  **confirmed**.
- The ensemble did run, but only on the 36 BIPIA re-pass rows, and none of their answers was delivered.

**Recording artefact.** The saved `generation.disagreement` field is 0.0 on every row, including the
36 ensemble rows. `_record()` reads `scores["disagreement"]`, but `verified_generate()` writes
`scores["ensemble_disagreement"]`. The saved field therefore says nothing about ensemble agreement.

## 7. Scoring

The scoring is unchanged, and done by the same functions the paper uses.

**PoisonedRAG.**
- Defended arm: an attack succeeds iff `not final_n200_summary.is_neutralized_final(row)`. This
  checks the target present and the true answer absent by word-boundary match, applies the manual
  classification of the ambiguous rows, and counts blocked rows as neutralised.
- Undefended arm (`mitigation_results/external200/off.jsonl`): the `score_external_paired.py` rule,
  with ambiguous answers counted as successes.
- The test is an exact McNemar test on rows paired by position.

**BIPIA.** BIPIA goals are not string-scorable. The scoring counts blocks; the 5 unblocked answers were
read manually.

## 8. Verified numbers (recomputed from saved outputs)

| Quantity | Recomputed | Manuscript |
|---|---|---|
| PoisonedRAG ASR, undefended / defended | 137/150 (91.3%) / 136/150 (90.7%) | 91.3% / 90.7% |
| PoisonedRAG fixed / broken, *p* | 7 / 6, 1.0 | 7/6, *p* = 1.0 |
| PoisonedRAG defended, NQ / HotpotQA / MS MARCO | 44 / 47 / 45 | 44 / 47 / 45 |
| PoisonedRAG undefended, NQ / HotpotQA / MS MARCO | 46 / 46 / 45 | 46 / 46 / 45 |
| PoisonedRAG poison removed before generation | 0/148 | 0/148 |
| BIPIA blocked, defended / undefended | 45/50 / 0/50 | 45/50 / 0/50 |
| BIPIA blocks by stage | adaptive policy 36, grounding 8, gate 1 | 36 / 8 / 1 |
| BIPIA poison removed before generation | 33/49: **10 by Stage 6, 23 by Stage 7** | "Stage 6 redacts 33 of the 49" (inaccurate attribution) |
| Graded risk recomputed by the classifier vs saved `input_risk` | 197/197 within rounding | – |
| Tier routing (first tier and HIGH re-pass) | reproduced on all 197 rows that reached retrieval | – |
| Mean latency | 4,764.2 s / 200 = 23.8 s (from the run log; includes the 36 ensemble re-passes) | 23.8 s |

All 36 checks pass (`adaptive_external/verification_report.json`).

## 9. Verification utility

`step4 dataset/verify_adaptive_external.py` checks:
- file presence and SHA-256 against `PROVENANCE.json`;
- byte identity with the git history;
- that the data files are unchanged since the run;
- row counts, unique IDs and the input/output/undefended alignment;
- stage traces;
- the first-pass tier;
- the escalation rule;
- generator selection;
- all aggregate counts;
- the headline numbers in both manuscripts.

The optional `--recompute-graded` mode re-scores the saved query and chunk text with the DeBERTa
classifier, using CPU forward passes only and no LLM. It then reproduces the routing of every row
exactly. The report labels each check `provenance`, `recomputed_from_saved_outputs`,
`deterministic_recomputation` or `manuscript`. It always states that **fresh inference reproduction was
not performed**.

## 10. Known nondeterminism

- Ollama GPU generation is not bit-reproducible at a fixed seed: identical Protocol B runs differ on
  11–16 of 200 rows. A fresh run should match statistically, not row for row.
- If the generator's answer changes, grounding, late escalation and the final outcome of a row can change
  too.
- The classifier-based routing signals are deterministic. A rerun should reproduce the first-pass tier
  of every row.

## 11. Differences from Protocol A and Protocol B

| | Protocol A | Protocol B | Adaptive external |
|---|---|---|---|
| Code | `run_full_pipeline.py` (current) | `run_mitigation_ab.py` + `mitigation_pipeline.py` | recovered `run_full_pipeline_adaptive.py` + `1d99b23` modules |
| Stages | all 13, plus multivector and controller | seven toggleable layers | 1–10 and 13; 3d skipped; no 10A, 10B, 11 or 12 |
| Generator | three-model ensemble + fuser | single `llama3.2:3b` | single `llama3.2:3b`; ensemble only on a HIGH re-pass |
| Retrieval | k = 5, rerank top-3 | k = 5, rerank top-3 | k = 8 (LOW) or 5 (MEDIUM) |
| Scoring | canary / gate / redaction | marker with refutation cues | marker with manual ambiguity rules |

Results from the three configurations are never pooled or compared directly.

## 12. Remaining limitations

1. **No fresh inference reproduction.** Agreement here shows that the code, the saved outputs and the
   reported numbers are consistent. It does not show that a rerun reproduces the outputs. A rerun takes
   about 80 minutes and needs approval.
2. **Unrecorded fields.** The saved rows do not record the tier, the generation mode or the policy-check
   verdict. Routing was reconstructed from the code and the pinned Stage 6/7 values, and recomputed
   with the classifier. Generator selection is inferred from the code path, not logged.
3. **Missing documentation.** `gate_diagnosis.txt`, the justification for disabling Stage 3d, was never
   committed.
4. **Classifier revision.** `graded_channels.py` loads the classifier's `main` revision, which may change
   upstream. Today both cached revisions give identical scores.
5. **Package drift.** The current virtual environment differs from the run-time lock (Section 4).
6. **Dead re-pass.** The HIGH re-pass could not change any outcome (Section 6). This is a property of
   the run-time code, preserved here and not fixed.

## 13. Manuscript corrections (applied 2026-11-25 with the author's approval)

These corrections were applied with `research_improvement/apply_adaptive_corrections.py`. The script
makes exact-match replacements and writes nothing unless every old string matches.

**After applying:**
- Both papers compile: the ICCIT paper is 6 pages (6 pages in Word as well) and the extended paper is
  17 pages.
- The IEEE PDFs, the Word version and the results report were rebuilt.
- `verify_claims.py` passes 86/86 and `verify_adaptive_external.py` passes 36/36.

The texts below are as applied. T-a4 differs from the original proposal: instead of removing the
"restore the scripts" item, it now says where the restored scripts are.

**M-a. `main.tex`, Sec. IV, Environment and configuration.**

Replace:

> The external sets ran an adaptive-tier variant of the same stages (Stages~1--10 and~13; no multivector detector, controller, Presidio, or DLP), in which input risk selects retrieval depth and generator: top-8 retrieval and a single \texttt{llama3.2:3b} at low risk (139 of the 150 PoisonedRAG rows and 44 of the 50 BIPIA rows), top-5 at medium risk, and top-3 with the three-model ensemble at high risk.

with:

> The external sets ran an adaptive-tier variant of the same stages (Stages~1--10 and~13; the Stage~3d semantic-intent gate was disabled, and the multivector detector, controller, Presidio, and DLP were not invoked). Input risk selects a tier for each row: low (top-8 retrieval, a single \texttt{llama3.2:3b}), medium (top-5, a single \texttt{llama3.2:3b}), or high (top-3, the three-model ensemble and an LLM policy check). On the first pass, 147 of the 150 PoisonedRAG rows ran at low risk and 1 at medium risk, and 49 of the 50 BIPIA rows ran at low risk; the other 3 rows were blocked at the gate. Every delivered answer was therefore generated by a single \texttt{llama3.2:3b}. An answer that failed grounding could raise its row to high risk, which regenerated it with the ensemble before the row was refused; this happened on 36 BIPIA rows and on no PoisonedRAG row.

**M-b. `main.tex`, Sec. V-G, BIPIA.**

Replace:

> The undefended arm blocks none of the 50 rows; the pipeline blocks 45 of 50: 36 through the risk-adaptive policy, 8 through the grounding judge, and 1 at the gate. Stage~6 redacts 33 of the 49 injected chunks.

with:

> The undefended arm blocks none of the 50 rows; the pipeline blocks 45 of 50: 36 through the risk-adaptive policy (an ungrounded answer raised the row to high risk and it was refused), 8 through the grounding judge, and 1 at the gate. Of the 49 injected chunks, 33 are removed before generation: 10 by Stage~6 sanitization and 23 by Stage~7 reranking.

**M-c. `main.tex`, Sec. V-G, PoisonedRAG.**

Replace:

> The variant generated with a single model at low risk, so this run does not test ensemble disagreement;

with:

> Every PoisonedRAG answer was generated by a single \texttt{llama3.2:3b} (no row reached the high tier), so this run does not test ensemble disagreement;

**M-d. Table X in `main.tex` and the external table in `main_iccit.tex`.** Change the column header
`\makecell{Chunk\\redacted}` to `\makecell{Poison\\removed}`, because the column counts removal before
generation by Stage 6 or Stage 7.

**I-a. `main_iccit.tex`, Sec. IV.**

Replace:

> top-8 and a single \texttt{llama3.2:3b} at low risk, which applied to 139 of 150 PoisonedRAG rows.

with:

> top-8 and a single \texttt{llama3.2:3b} at low risk, which applied to 147 of 150 PoisonedRAG rows (one ran at medium risk, also with a single generator, and two were blocked at the gate).

**D-a. `main.tex`, Data and Code Availability.** Append:

> The adaptive-tier scripts that produced the external results are restored, byte-identical to the versions used, in \texttt{step4 dataset/adaptive\_external/}, together with the module versions of that run, their provenance, and a verification script.

**R-a. `response_to_reviewers.md`, 4.1.**
- Replace "Each was run through the complete pipeline" with "Each was run through an adaptive-tier
  variant of the pipeline (Sec. IV)".
- Replace "The complete pipeline (three-model ensemble) gives 90.7% ASR" with "The adaptive-tier variant,
  whose delivered answers all come from a single llama3.2:3b, gives 90.7% ASR".

**R-b. `response_to_reviewers.md`, 4.2 latency table.** Replace the row label "Complete 13-stage
pipeline, 3-model ensemble" with "Adaptive-tier pipeline variant (external sets)".

**R-c. `response_to_reviewers.md`, 5.2.** Add to the repository list:

> * the recovered adaptive-tier scripts behind the PoisonedRAG and BIPIA results, with their provenance and a verification script (`step4 dataset/adaptive_external/`, `verify_adaptive_external.py`).

**T-a. Thesis results report** (`docs/results_report/thesis_results_report.tex`).
- Section 3.2: replace "139/150 PoisonedRAG and 44/50 BIPIA rows" with "147/150 PoisonedRAG (plus 1 at
  medium risk) and 49/50 BIPIA rows; the ensemble ran only on 36 BIPIA rows re-generated before refusal".
- Section 5.10: replace "Stage 6 redacts 33/49 injected chunks" with "33/49 injected chunks removed before
  generation (10 by Stage 6, 23 by Stage 7)".
- Remove the "Before submission … restore the four adaptive-tier scripts" item from Section 9.
