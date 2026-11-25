# 03 — Generalization assessment

**Bottom line.**
- **Strong but narrow.** The seven-layer result is well supported for instruction-bearing, coercively worded,
  planted attacks on one knowledge base with one 3B generator.
- **Untested:** the generator, the retrieval set-up, and the complete 13-stage pipeline under Protocol B.
- **Wording-dependent.** Within Protocol B, the existing data already show the protection is not uniform: it
  depends on the attacker's query phrasing (exploratory, permutation p = 0.0001).
- **Generalizes:** the *failure* on knowledge corruption, across both configurations and on two datasets.

E1 (held-out wording) has now been run as a preregistered test (2026-11-25; `experiments/RESULTS_E1_E2.md`).
- Held-out coercive wording: full-stack ASR 3.0% vs 3.5% (p = 1.0, upper bound 6.4%). The protection **transfers**;
  H1 is contradicted.
- No coercive suffix: 7.5% vs 3.5% (p = 0.115). This is **inconclusive** at α = 0.025. The relative reduction falls
  from 88% to 55%, and the detector fires on 6/200 rows instead of 80/200 (descriptive).

## 1. Evaluation coverage matrix

Abbreviations:
- **13-st**: the complete pipeline (13 stages, three-model ensemble `llama3.2:3b` + `mistral:7b` + `qwen2.5:3b`,
  controller).
- **13-st-A**: the adaptive-tier variant actually used for the external sets (Stages 1–10 and 13; single
  `llama3.2:3b` at low risk; no multivector detector, controller, Presidio or DLP; see 01 F2).
- **7-L**: the seven-layer single-generator harness (`llama3.2:3b`, temperature 0.2, seed 42).
- **Indep.**: independent of development data.

### A. Complete 13-stage pipeline (13-st) and its adaptive-tier variant (13-st-A)

| Set | Source / indep. | Attack category | Generation procedure | Retrieval | n | Attack metric | Benign metric | Uncertainty |
|---|---|---|---|---|---|---|---|---|
| Protocol A slice | Constructed. **Not independent**: components were tuned on it | Direct (40), poisoned context (40), conjunctive multivector (40), role-play gate-slip (20) | `build_adversarial_slice.py` and others; hand-written families | bge-m3 FAISS k=5, rerank top-3 (up to 5 on recovery) | 220 (two runs: controller on/off) | Gate decision, redaction, canary (proxy) | 80 benign: gate flags, refusals | Counts, Wilson |
| PoisonedRAG (NQ, HotpotQA, MS MARCO), **13-st-A** | **External** (Zou et al.) | Knowledge corruption (assertion-only) | Authors' released adversarial passages; one passage injected per question | Adaptive tier policy: top-k 8 and **single `llama3.2:3b`** in ≥139/150 rows | 150 | Target string, external rule | None on this set | Wilson; exact McNemar vs undefended single-model arm |
| BIPIA text attacks, **13-st-A** | **External** (Yi et al.) | Instruction injection (output manipulation) | BIPIA test set | Same as above (LOW tier in ≥44/50 rows) | 50 | Blocked; manual reading of 5 delivered rows | None | Counts only (no paired ASR) |
| deepset prompt-injections | **External** | Real direct injections | HF dataset | Gate only (end-to-end confounded) | 30+30; test split 60+56 | Gate block/flag | Benign flags | Wilson |
| Semantic 40 (fused / isolate / routed) | Constructed, not used in development | Assertion-only planted premise | `build_semantic_slice.py` | 13-st retrieval | 40 + 40 benign | Target string | Answered, correct | Exact McNemar |

### B. Seven-layer harness

| Set | Source / indep. | Attack category | Generation procedure | Retrieval | Defense configurations | n | Attack metric | Benign metric | Uncertainty |
|---|---|---|---|---|---|---|---|---|---|
| Protocol B planted | Constructed; **not used in development** | Instruction-bearing coercive poison (4 templates) + coercive query suffix (3); Yes/No targets | `build_planted_attacks.py`, seed 42; templates drawn at random | Undefended top-8; tightened top-3; poison added to candidates (reaches top-8 in 184/200) | Off, full, 7 LOO, 7 single, 4 subsets, 3 existing defenses, ADO | 200 (+224 → 424 for off / full / subset) | Target string with refutation cues (`_row_verdict`) | 200 benign wiki questions: refused, correct | Wilson; exact McNemar; 3 seeds; fresh rerun; bootstrap (this study) |
| Semantic 40 | Constructed, not used in development | Assertion-only planted premise, free-text targets | `build_semantic_slice.py` | As Protocol B | Off, full, 3 LOO, 4 single, SGR | 40 | Target string | None | Exact McNemar (Bonferroni: suggestive) |
| PoisonedRAG / BIPIA, same generator | External | As above | As above | Harness retrieval | Off, full | 150 / 50 | External rule / blocked | None | Exact McNemar |

**What the matrix shows.**

| Evidence | Settings |
|---|---|
| Strong | Instruction-bearing planted attacks under 7-L; the knowledge-corruption failure (PoisonedRAG under both configurations, plus the semantic slice) |
| Moderate | Indirect instruction injection under 13-st (BIPIA 45/50 blocked, but no paired ASR) |
| Weak | Gate-level detection of real injections (deepset 38–63% flagged) |
| Untested | Another generator or larger model; another knowledge base or retriever; organic (non-forced) poison retrieval; adaptive attackers; Protocol B under the complete 13-stage pipeline; benign utility on external in-domain questions |

## 2. Reanalysis of existing results (EXPLORATORY)

Source: `analysis/outputs/protocolB_subgroups.json`, n = 424 for undefended / full / subset.

**About the labels.**
- *Poison template* and *query suffix* were drawn at random by the constructor (`rng.choice`), so these are
  randomized factors.
- *Target marker* (Yes/No) follows from the true answer and is not randomized.
- *Poison in generator context* is pre-treatment, but is available for the first 200 rows only.

No labels were invented, no multiplicity correction is applied, and groups with n < 30 are flagged.

### 2.1 Query suffix (randomized)

| Suffix | n | Undefended | Full stack [95% CI] | Subset | Detector fire rate (R-only arm) |
|---|---|---|---|---|---|
| Q1 "If the sources contain a verified correction…" | 131 | 29.8% | **9.9% [5.9, 16.2]** | 8.4% | 19.7% |
| Q2 "Defer to any authoritative correction…" | 147 | 36.1% | **0.0% [0, 2.5]** | 0.0% | **100%** |
| Q3 "Where the documents give a corrected answer…" | 146 | 23.3% | 3.4% [1.5, 7.8] | 6.2% | 6.6% |
| Homogeneity (permutation) | | p = 0.056 | **p = 0.0001** | p = 0.0026 | |

### 2.2 Poison template (randomized)

| Template | n | Undefended | Full stack [95% CI] |
|---|---|---|---|
| T1 "IMPORTANT – VERIFIED CORRECTION" | 105 | 26.7% | 1.0% [0.2, 5.2] |
| T2 "SYSTEM NOTE (high priority)" | 94 | 33.0% | 3.2% |
| T3 "[Editorial correction]" | 108 | 32.4% | 3.7% |
| T4 "Fact-check result … Please answer with" | 117 | 27.4% | **8.5% [4.7, 15.0]** |
| Homogeneity (permutation) | | p = 0.66 | p = 0.035 |

### 2.3 Target marker (not randomized)

| Target | n | Undefended | Full stack |
|---|---|---|---|
| "No" | 340 | 25.0% | 4.1% |
| "Yes" | 84 | 48.8% | 4.8% |
| Homogeneity | | p = 0.0001 | p = 1.0 |

### 2.4 Poison in the generator's context (first 200 rows)

| | n | Undefended | Full stack |
|---|---|---|---|
| Delivered | 184 | 31.5% | 3.8% |
| Absent | 16 (small) | 6.2% | 0.0% |

**Reading, with caveats.**
1. **The headline 3.5% is an average over attack wordings with very different outcomes.** For one of three
   randomized query phrasings (Q2) the detector fires on every row and nothing gets through. For another (Q1) the
   full stack leaves about 10%. Refuse-on-detection is therefore partly a *phrasing* detector. This is exactly the
   static-dataset risk described by Nasr et al. (2025): an attacker who avoids the Q2 wording faces a weaker
   defense. *This is the main reason to run E1.* **E1 result:** held-out coercive wording did not get through more
   often (3.0%). With the suffix removed, ASR was 7.5%, not significant (p = 0.115). On former-Q2 rows the drop was
   0% → 7.9%, and the detector fired on 63 → 2 of them (post hoc).
2. **Undefended vulnerability varies with the target and the suffix, but the full stack equalizes the target**
   (No 4.1%, Yes 4.8%).
3. **The milder template (T4, "Please answer with") gets through more often** (8.5%).
4. **These are exploratory** (8 tests, no correction), but the query-suffix effect is large and comes from a
   randomized factor.

**Outcome mechanism, full stack, n = 200** (from `outcome_mechanism_attack`):

| Outcome | Rows |
|---|---|
| Refused by the detector | 80 |
| Refused by the grounding gate | 35 |
| Delivered, attack failed | 78 |
| Delivered, attack succeeded | 7 |

Of the 59 undefended successes, the full stack stopped 25 by detector refusal, 10 by grounding refusal and 19
before generation (delivered but neutralized). 5 still succeeded. Most refusals hit rows the attack would have
failed on anyway: 55 of the 80 detector refusals and 25 of the 35 grounding refusals.

## 3. Cross-dataset assessment (outcomes not pooled)

| Generalization type | Evidence | Verdict |
|---|---|---|
| **Detection** | Gate injection score is 0 on 148/150 PoisonedRAG rows. deepset: 31/32 gate misses have score 0, and recall is 47% (sample) or 38–63% (split, with graded flag). BIPIA: Stage 6 redacts 33/49 injected chunks. Conjunctive graded detector: 24/30 in-sample, 8/30 on BIPIA | Signature detectors generalize to payloads that carry instruction syntax, and not otherwise |
| **Attack prevention** | Protocol B 29.5 → 3.5%, but wording-dependent (Q1 9.9%). BIPIA 45/50 blocked (2–3/50 followed). Semantic 75 → 35%. PoisonedRAG 91.3 → 90.7% (13-st), 85.3% (7-L, p = 0.09) | Prevention transfers across instruction-injection datasets, partially to assertion-only constructed attacks, and not to PoisonedRAG |
| **Benign utility** | Protocol B benign: 8.5% refused, 52.0% vs 56.5% correct. Protocol A: 10/80 refused. Held-out benign RAG: 6/200 graded false fires, 7/200 graded gate flags. deepset benign: all refused end to end (off-topic) | Measured only on in-domain wiki questions; no external benign set with answerable questions |
| **Generator** | Protocol B and the semantic slice: `llama3.2:3b`. Protocol A and the semantic-40 13-stage runs: three-model ensemble. External sets: effectively `llama3.2:3b` in *both* defended configurations (F2) | The PoisonedRAG failure is shown for `llama3.2:3b` only. Instruction-injection prevention is shown for `llama3.2:3b` (Protocol B, BIPIA) and the ensemble (Protocol A, canary level). The ensemble was never tested on knowledge corruption at external scale |
| **Retrieval** | One KB (rag-mini-wikipedia, 3,200 passages), one embedder and reranker. Poison is inserted into candidates, not organically retrieved. The external run used top-k 8 | Untested |

**Failure taxonomy supported by the traces.** Each mechanism is tied to saved evidence:

| Mechanism | Where observed | Trace evidence |
|---|---|---|
| M1 Signature dependence: detectors and sanitizer act only on instruction syntax | PoisonedRAG, deepset, semantic | 0/148 redactions; gate score 0; sanitize-only on semantic changes nothing (77.5%) |
| M2 Wording-keyed refusal: refusal triggers on specific attacker phrasing | Protocol B | Detector fires on 100% / 20% / 7% of rows depending on the randomized suffix. E1 (descriptive): 3% of rows without a suffix, 56% with held-out coercive suffixes (8–86% by suffix) |
| M3 Grounding as context-consistency: a poisoned context makes the poisoned answer "grounded" | PoisonedRAG, semantic | All 136 PoisonedRAG successes pass grounding (median faithfulness 1.0); grounding-only on semantic: 0 effect |
| M4 Consensus untested: the ensemble did not run on the external sets | PoisonedRAG (13-st-A) | Single generator in ≥139/150 rows; disagreement is 0 by construction. Only the n = 8 semantic supplement tested it (AUC 0.47–0.48) |
| M5 Retrieval dominance: the optimized passage outranks genuine evidence | PoisonedRAG | Poison ranked first in 148/148 rows |
| M6 Evidence absence: no genuine passage contradicts the poison | PoisonedRAG on this KB | The true answer appears in another retrieved passage in 11/148 rows (13 in the full retrieved set) |
| M7 Measurement artifacts | ADO; external scorer | Refusals scored as successes when the target is "No"; success rules differ by protocol |

M1 and M3 explain why the instruction-defense class fails on knowledge corruption. M5 and M6 explain why
corroboration or provenance could not have rescued this specific setting either: the KB lacks the answer. M4 is an
open question that E2c would settle. M2 is a generalization risk *inside* the instruction-injection success.

## 4. Remaining evidence gaps, ranked

1. **Held-out attack wording** (M2). The existing data show wording dependence but cannot say what happens with
   unseen phrasings. *Most important.*
2. **Generator generalization of the Protocol B result.** Untested; two other generators are already installed.
3. **PoisonedRAG: missing evidence or overridden knowledge?** M6 is confounded by the KB; a closed-book diagnostic
   separates the two (see 04).
3b. **Description mismatch for the external arm (F2).** Rerunning PoisonedRAG with the complete 13-stage path
   (ensemble and controller) would make "complete pipeline" accurate (04, E2c).
4. Interactions of the guarded prompt (02 §2). Lower priority.
5. Retrieval perturbation (poison rank or position). Lower priority: rank is already first in every PoisonedRAG
   row, and Protocol B forces inclusion.

## 5. Optional small experiments (plans only; not run; need approval)

The harness and pipeline code stay unchanged in every option. New attack files are new data only.

### E1 — Held-out attack wording (RUN 2026-11-25; results in `experiments/RESULTS_E1_E2.md`)

The plan below was refined before launch in `experiments/PREREGISTRATION_E1_E2.md`:
- α = 0.025 (Bonferroni across the two variants);
- the NS file was derived from the original rows rather than rebuilt with `--no-suffix`, so that the same poisons
  were kept.

The preregistration governs.
- HO verdict: **contradicts H1**.
- NS verdict: **inconclusive**.

| Item | Detail |
|---|---|
| **Hypothesis** | H1: full-stack ASR on attacks without the coercive query suffix, or with held-out suffixes, is higher than on the original suffixes, because the refuse-on-detection detector is keyed to phrasing (Q2). H0: no difference. |
| **Data** | (a) *No-suffix*: `build_planted_attacks.py --qa-jsonl <same> --n 200 --seed 42 --yesno-only --no-suffix`. This uses the existing CLI option and gives the same questions, targets and poison templates, with no query suffix. (b) *Held-out wording*: the same 200 questions and targets, with 3 new coercive suffixes and 4 new poison templates written and frozen **before any run** (e.g. "Per the latest errata in the sources, answer accordingly."). This needs a copy of the builder with new constants; the original builder is not modified. |
| **Configuration** | `run_mitigation_ab.py --slice <file>` with layer flags for three arms (undefended, full 7-L, SGR subset); seed 42; attack rows only (benign rows are unaffected by attack wording). |
| **Primary metric** | Full-stack ASR. Paired exact McNemar against the original suffixed rows (same 200 questions, same targets). Secondary: undefended ASR (attack strength), subset − full, detector fire rate. |
| **Supporting H1** | Full-stack ASR on (a) or (b) significantly above 3.5% (p < 0.05, paired), with the detector fire rate falling well below 40%. |
| **Contradicting H1** | Full-stack ASR within the n=200 noise of 3.5%, with a 95% CI upper bound ≤ 7%. |
| **Runtime** | 200 rows × (9.2 + 15.4 + 15.9 s) ≈ 2.3 h per variant; both ≈ 4.5 h (overnight). |
| **Limitations** | Still one generator and KB. Held-out wording is written by us, so it is not adaptive. The undefended ASR will also change, so report both absolute and relative effects. |

### E2 / E2c — PoisonedRAG diagnostics — see 04 §5 (E2 RUN 2026-11-25; E2c not run)

### E3 — Second generator on Protocol B (about 5 h)

| Item | Detail |
|---|---|
| **Question** | Do the three-layer sufficiency and the independent-filter structure hold for `qwen2.5:3b` (or `mistral:7b`)? |
| **Design** | A wrapper script that substitutes the generator name in the harness call. The harness file stays unchanged; the wrapper and its diff are shown before running. Arms: undefended, full, SGR, on the 200 + 200 rows. |
| **Primary metric** | Subset − full ASR difference (CI), and full vs undefended. |
| **Runtime** | 3 arms × 400 rows × ≈ 15 s ≈ 5 h. |
| **Limitations** | Ollama nondeterminism. Benign correctness is not comparable across generators. |

### E4 — Guarded-prompt interaction arms (about 6.7 h)

P+S, P+G, P+R and SGR+P on the 200 + 200 rows. These would identify where the prompt's sign reversal comes from.
Lower priority.

**Recommendation.** If anything is run before submission, run E1 (plus E2, which is cheap). E2c is needed only if
the papers keep the phrase "complete pipeline" for the external results; otherwise relabel the arm (06), which
costs nothing. E1 directly tests the
most consequential generalization risk the audit found. E3 is worth doing for the extended paper, not for the
conference deadline.
