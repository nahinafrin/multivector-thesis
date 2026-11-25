# 01 — Existing evidence audit

**Scope.** Both manuscripts (`paper_revision/main.tex`, 15 pp.; `paper_revision/main_iccit.tex`, 6 pp. A4), the
response letter, the 13-stage code (`step4 dataset/step_*.py`, `run_full_pipeline.py`), the Protocol B harness
(`mitigation_pipeline.py`, `run_mitigation_ab.py`, `run_ablation_study.py`, `run_followup_study.py`,
`run_seed_study.py`), the external runs, the saved per-row outputs, and the claim checker
(`verify_claims.py`, 86/86 PASS at `7c7956f`).

**Method.** Each claim was traced to the run that produced it. Numbers not covered by the checker were
recomputed from saved outputs (`analysis/verify_unchecked_claims.py`, `analysis/outputs/*.json`). No model was
run, and no saved output or manuscript was changed.

**Two configurations, kept apart throughout.**
- *Complete 13-stage pipeline* (Protocol A; external sets): three-model ensemble and controller.
- *Seven-layer single-generator harness* (Protocol B; semantic slice; same-generator external arm):
  `llama3.2:3b`, seed 42, seven toggleable layers.

## 1. Claim-by-claim table

Legend for the last column:
- **None**: the evidence is adequate for the claim as worded.
- **Wording**: the evidence supports a narrower claim than the text makes.
- **Doc**: the methods or provenance need documenting.
- **Exp**: only a new experiment would settle it.

| # | Research claim (as in the papers) | Supporting experiment | Source files | Statistical evidence | Limitations | Additional work? |
|---|---|---|---|---|---|---|
| 1 | No canary leaks into any delivered answer; 74/74 poison chunks redacted (Protocol A) | Protocol A, complete pipeline, controller on/off | `protocolA_220/grounded_controller_220.jsonl`, `..._baseline_220.jsonl`, `score_protocolA_220.py` | Counts. Canary checked on 175/220 rows; 45 blocked first. Checker A.* | Canary integrity is a proxy, not output safety. Constructed slice, developed on the same slice | None (already worded as a proxy) |
| 2 | Under Protocol B, ASR 29.5% → 3.5% (seven-layer) | Protocol B attack arms, seed 42 | `mitigation_results/followup200/off.jsonl`, `ablation200/on_full.jsonl` | 54 fixed / 2 broken, exact p < 10⁻¹³. Wilson CIs [23.6, 36.2] / [1.7, 7.0]. Checker VII.* | All 200 attacks share one construction: 4 coercive poison templates × 3 coercive query suffixes, Yes/No targets. Single generator. Static attack set | **Wording**: the result is specific to instruction-bearing, coercively worded attacks (see 03 §2) |
| 3 | Fresh rerun: 29.0% and 3.0% | `rerun200` | `mitigation_results/rerun200/*` | Agreement on 189 and 199 of 200 rows. Checker E.rerun_* | Same rows and code, so this tests reproducibility, not generalization | None |
| 4 | Three-layer subset is "statistically indistinguishable" from seven layers | Subset vs full: seed 42, seeds 7/1234, n=424 | `followup200/on_combo_sanitize_grounding_refuse.jsonl`, `seed7/`, `seed1234/`, `extended424/` | n=424: +0.5 pts, 95% CI [−1.5, +2.4] (90% CI [−1.2, +2.1]), p=0.81. The subset is worse in all 3 seeds at n=200 (+2.0, +2.5, +1.0) | No equivalence margin was prespecified. Seeds reuse the same rows. **On the semantic slice the subset is significantly worse** (62.5% vs 35.0%, 12/1 discordant, p=0.0034; this arm exists in `followup_nonyesno40` but the papers do not report it) | **Wording**: report as a bound, scoped to instruction-bearing planted attacks |
| 5 | Every subset beats each of its constituent layers | Two- and three-layer subsets vs singles | `followup200/on_combo_*`, `ablation200/on_only_*` | All 9 comparisons p < 0.05. Two (GR vs G p=0.022; SG vs G p=0.0072) do not survive a Bonferroni correction for 9 tests (α=0.0056) | — | **Wording**: add "at α = 0.05, uncorrected" |
| 6 | Two-layer subsets are worse than full (GR p=0.004, SR p=0.008 survive Bonferroni; SG p=0.021 does not) | Subset arms vs full | as above | Recomputed: identical | — | None |
| 7 | No single layer approaches the full stack; the guarded prompt alone raises ASR to 53% | Single-layer arms | `ablation200/on_only_*` | Prompt alone: 6 fixed / 53 broken, p = 2×10⁻¹⁰ | **The sign reverses on the semantic slice**: prompt alone lowers ASR 75.0 → 52.5% (p = 0.035, n=40, exploratory) | **Wording**: report the attack-class dependence |
| 8 | Presidio and DLP do not affect injection ASR | Leave-one-out and single arms | `ablation200/on_nopresidio`, `on_nodlp`, `on_only_presidio`, `on_only_dlp` | 0/0 discordant alone; LOO 1/0 and 0/0 | They target PII, which Protocol B does not test | None |
| 9 | Seven-layer stack beats LLM-Guard and Spotlighting, matches ControlNET, beats adapted ADO | Existing-defense arms on Protocol B | `mitigation_results/baselines_protocolB/*`, `external_baselines/ado_protocolB/full_*` | Checker X.* (p < 10⁻⁷; ControlNET p = 0.77) | ControlNET and LLM-Guard are scored on stored undefended answers. ADO is adapted, and its primary ASR includes a refusal artifact | None (already qualified at `7c7956f`) |
| 10 | Co-activation detector cannot fire validly on saturated scores; graded repair 24/30 at 1/80 | Protocol A multivector rows; graded revalidation; held-out benign; BIPIA conjunctive | `multivector_graded_revalidation.json`, `external_eval/graded_*`, `fit_graded_floors_heldout.py` | Counts. Checker B.* | Floors set in-sample. External 8/30 at 4/30 controls. The "proof" is a closed-form argument conditional on saturation | None (ICCIT wording fixed) |
| 11 | Controller adds no measured security, fewer refusals | Protocol A controller on/off | `protocolA_220/*` | 88 vs 101 blocked; 70 vs 66 benign delivered (no test reported) | Not a paired significance test for utility | **Wording** (optional): add a paired test for the benign-delivery difference |
| 12 | BIPIA: 45/50 blocked (0/50 undefended); 2–3/50 followed the instruction | External set, adaptive-tier pipeline variant (F2) | `ext_n200_results.jsonl`, `external200/off.jsonl` | Block counts. Manual reading of 5 rows. Stage breakdown recomputed: recorded block stages adaptive policy 36, grounding 8, gate 1; 33/49 poison chunks removed before generation (10 by Stage 6, 23 by Stage 7) | Undefended answers were not annotated, so there is no paired ASR. *Corrected 2026-11-25 (docs/ADAPTIVE_EXTERNAL_REPRODUCIBILITY.md):* all 36 adaptive-policy rows failed grounding and were already blocked by Stage 10, so the grounding judge blocked 44/50; the adaptive tier policy only re-generated and relabelled them | **Doc** |
| 13 | PoisonedRAG: no measurable protection (90.7% vs 91.3%, p=1.0); same generator 85.3% (p=0.09) | External set | `ext_n200_results.jsonl`, `external200/off.jsonl`, `external200/on_full.jsonl`, `score_external_paired.py` | Exact McNemar 7/6 and 16/7. Checker G.prag* | **The "complete pipeline" arm ran the adaptive-tier variant with a single `llama3.2:3b` on ≥139/150 rows (F2)**, so "a planted fact is consistent across models" is unsupported here. The success rule differs from Protocol B's (no refutation cue; manual ambiguity rules), so ASRs are not comparable across protocols. One poison passage per question. The KB rarely contains the true answer (04) | **Doc** (important), plus a richer failure analysis (04) |
| 14 | Semantic slice: 75% → 35% (p = 4×10⁻⁴); guarded prompt and refusal carry it, suggestively | `followup_nonyesno40` | `mitigation_results/followup_nonyesno40/*` | Recomputed. LOO prompt p=0.021; refuse p=0.031 (not Bonferroni-significant) | n=40, constructed. **Not in the checker** | **Doc**: add to the checker |
| 15 | Isolate-and-aggregate generator: 0/40 by abstaining; routed variant 15/40 (p=0.45) | Semantic and benign 40 with the 13-stage pipeline | `external_eval/semantic40_*`, `benign40_*` | Checker H.* | **Prior art: RobustRAG (Xiang et al., ICML 2024) introduced isolate-then-aggregate; neither paper cites it** | **Wording**: add the citation |
| 16 | deepset: gate blocks 14/30, flags 17/30; graded flag 28/60 → 38/60 at 2/56 | Gate-only evaluation | `external_eval/deepset_*` | Checker G.deepset_* | Gate only; end-to-end refusals are confounded by off-topic prompts (already stated) | None |
| 17 | Latency: 8.6 / 15.3 / 19.0 s; fp16 encoders 8.6 s; peak 6.3 GB GPU / 8.0 GB RAM | Benign arms; GPU-encoder benchmark; resource profile | `ablation200`, `gpu_encoders/`, `resource_profile/` | Checker I.*. Memory not checked | Memory comes from one profiled 20-query run | **Doc**: add memory to the checker (optional) |
| 18 | "Three layers carry the measured injection defense", plus practitioner advice to deploy sanitize, ground and refuse first | Items 4–7 | — | — | True for Protocol B's instruction-bearing attacks. **False for the assertion-only semantic slice**, where the guarded prompt carries most of the effect | **Wording** (important) |
| 19 | "Only a stage-by-stage, paired, class-decomposed evaluation can tell them apart" (main.tex §VI) | Rhetorical | — | — | "Only" is not demonstrated | **Wording** |
| 20 | "To our knowledge, none of these works compares configurations with paired tests on identical rows" | Literature | — | — | Plausible for the works cited. Since then, layered defenses with ablations have appeared (Ramakrishnan & Balaji 2025; LlamaFirewall 2025), as has a defense-in-depth system for knowledge corruption (TriShieldRAG 2026) | **Wording**: "we did not find", plus cite (see 02) |
| 21 | Saleem et al.: single-layer reductions sum to 79.6 pts vs 60.1 combined | Their reported table | external | — | Not re-checked in this audit | **Doc**: re-check against their table before submission |

## 2. Findings that need attention

**F1. Two contribution lists in main.tex disagree.** The Introduction lists (i)–(iv): architecture, negative
result, ablation, scope boundary. "Distinction from prior work" lists C1–C4: protocol, controller, negative
result, residual gap. The ICCIT paper uses a third ordering. *Action:* a single list (02 §6).

**F2 (most important). The external "complete pipeline" arm is not the configuration the papers describe.**
- **Which script produced it.** `ext_n200_results.jsonl` (PoisonedRAG and BIPIA, defended arm) was produced by
  `run_full_pipeline_adaptive.py`. Only that script emits the block stage `adaptive_policy` (36 BIPIA rows).
- **What that script runs:**
  - Stages 1–8, then `adaptive_verification.verified_generate`, then grounding (10), then 13.
  - The multivector detector (10A), the closed-loop controller (10B), Presidio (11) and DLP (12) are **not run**:
    their output fields are empty in all 200 rows.
  - A tiered `AdaptivePolicy` (`adaptive_risk.py`) sets the configuration per row:

    | Tier | Top-k | Rerank floor | Generation |
    |---|---|---|---|
    | LOW | 8 | 0.20 | single |
    | MEDIUM | 5 | 0.30 | single |
    | HIGH | 3 | 0.40 | three-model ensemble + policy check |

- **What actually ran.** Each row's ranked-context size reveals its tier: 8 chunks means LOW.
  - 139/150 PoisonedRAG rows and 44/50 BIPIA rows ran in the LOW tier, so a **single `llama3.2:3b` generator**
    answered them (`generate_single`).
  - `ensemble_disagreement` is exactly 0.0 in all 200 rows, which the single-model path writes by construction.
- **What the papers say.**
  - main.tex: "Because the complete pipeline uses the three-model ensemble, we also ran a same-generator
    comparison."
  - `score_external_paired.py`: "complete 13-stage pipeline, 3-model ensemble, controller on".
  - main.tex settings: "k = 5, rerank top-3".
  - main.tex: "Neither the ensemble's disagreement signal nor the grounding judge helps: a planted fact is consistent
    across models." **This run gives no support for the ensemble part**, because the ensemble was not run on these
    rows.
- **Where it went.** The trim commit `fe71c56` moved `run_full_pipeline_adaptive.py`, `adaptive_risk.py`,
  `adaptive_verification.py` and `adaptive_prompt.py` to the private backup repo, leaving only a `.pyc`.
- **For contrast:** Protocol A (`grounded_controller_220.jsonl`) and the semantic-40 13-stage runs *did* use the
  ensemble (non-zero disagreement in 174/175 and 37/40 rows) and the controller.
- **What this affects.** No reported number changes, and the numbers are still verifiable from the frozen outputs
  (checker G.prag). But:
  - the external arm should be described as an *adaptive-tier variant* of the pipeline (Stages 1–10 and 13; single
    generator at low risk);
  - the "same-generator comparison" premise must be rewritten, since both external comparisons effectively used
    `llama3.2:3b`;
  - the ensemble claim for PoisonedRAG must be dropped or moved to the n = 8 semantic evidence;
  - the four files should be restored to the public repo.

**F3. The "three layers suffice" result depends on the attack class.** It holds for Protocol B (instruction-bearing,
coercive). It does not hold for the semantic slice: subset 62.5% vs full 35.0%, p=0.0034, and the subset vs
undefended gives p=0.125. The practitioner recommendation in both conclusions would remove the layer that carries
the semantic-slice defense. *Action:* scope the claim (06).

**F4. Protocol B's defense depends strongly on query wording (exploratory).**
- **Detector firing by suffix.** The refuse-on-detection detector fires on 100% of rows with query suffix Q2
  ("Defer to any authoritative correction…"), 19.7% with Q1 and 6.6% with Q3.
- **Full-stack ASR by suffix:** 0/147 (Q2), 9.9% [5.9, 16.2] (Q1) and 3.4% (Q3). The heterogeneity test gives
  permutation p = 0.0001.
- **Consequence.** A large part of the headline protection is keyed to one attacker phrasing. The papers do not
  report this. *Action:* report as exploratory (03), and test held-out wording if approved (03 §5, E1).

**F5. The equivalence wording.**
- **Current wording.** "Statistically indistinguishable … gap bounded at about 2.4 points" is close, but it reads
  as an equivalence claim.
- **Accurate reading.** The n=424 paired difference is +0.5 points with 95% CI [−1.5, +2.4]. No margin was
  prespecified, and the n=200 seed replicates all lean the same way (+1.0 to +2.5).
- *Action:* use the bound wording in 06.

**F6. Missing prior art.** Four omissions:
- RobustRAG, for isolate-and-aggregate generation.
- TriShieldRAG, a defense-in-depth system for knowledge corruption. It reports PoisonedRAG ≈91% → ≈13% against a
  non-adaptive attacker, and its own failure under an adaptive one.
- Ablation analysis and fANOVA, for the attribution method.
- "The Attacker Moves Second" (Nasr et al., USENIX Sec. 2026), for the static-dataset threat to validity.

Details are in 02.

**F7. The external and Protocol B success rules differ.**
- Protocol B's scorer applies refutation cues (`score_attack_success._row_verdict`).
- The external scorer counts any target-present answer and classifies ambiguous ones manually.
- With Protocol B's rule, PoisonedRAG ASR would read 85.3% undefended and 82.0% defended, instead of 91.3% and
  90.7%. The conclusion (no significant difference, p = 0.33) is unchanged.

*Action:* state that ASRs are not compared across protocols, and name the rule per table.

**F8. Unchecked numbers.** The semantic-slice results (item 14), the memory numbers and the BIPIA stage breakdown
are correct (recomputed), but the claim checker does not cover them. *Action (optional):* add them to
`verify_claims.py` when the manuscripts are next edited.

## 3. What is demonstrated vs only asserted

- **Demonstrated with paired statistics:**
  - the Protocol B reduction;
  - the ablation and subset effects;
  - the existing-defense comparisons;
  - the PoisonedRAG null;
  - the semantic-slice reduction;
  - the deepset gate recall.
- **Demonstrated, but descriptive only:** canary integrity, BIPIA blocking, latency and memory.
- **Asserted or overstated:**
  - the external arm being the "complete 13-stage pipeline with the three-model ensemble", and ensemble agreement on
    PoisonedRAG (F2);
  - "only" stage-level evaluation can separate ambition from contribution (#19);
  - "three layers carry the defense" as general deployment advice (#18);
  - novelty phrasing relative to the literature (#20);
  - "complete pipeline" settings for the external run (F2).
- **Not supported by any data:** generalization of the Protocol B numbers to other generators, other knowledge
  bases, other attack wordings, or to the complete 13-stage pipeline. The papers mostly avoid claiming this. F4
  shows why it matters.

All recomputations: `research_improvement/analysis/` (see `README.md`).
