# Response to Reviewers

**Paper:** Attributing Security in a Local, Zero-Cost Defense-in-Depth Pipeline for Retrieval-Augmented Generation

We thank the reviewers for their careful and constructive comments. Several of them changed our conclusions, not just the wording. The main changes are:

* **Larger, independent evaluation.** We now report 920 scored rows across seven sets, up from 220. These include 150 PoisonedRAG rows, 50 BIPIA rows, and 60 deepset rows, none of which we constructed.
* **Ablation against simpler defenses.** A three-layer subset (sanitization, grounding, and refusal on detection) is statistically indistinguishable from the full stack. We now recommend that subset.
* **Narrower neutralization claim.** "100% neutralization" is gone. We separate canary integrity (0/220 leaks) from attacker content in delivered outputs, where attack success is 3.5%, not 0%.
* **Correction to the conjunctive-detector result.** The structural impossibility holds for *saturated* classifier scores only. Reading graded logit margins instead restores detection in-sample (24/30), but only modestly on external data (8/30), and benign false fires rise from 1/80 to 6/200 on held-out questions.
* **Explicit scope.** The title, abstract, and a new "Scope" paragraph in the introduction now state the paper's central boundary: the pipeline contains instruction-carrying injection but offers no measurable protection against knowledge corruption. We present the latter as a measured limit of this defense class, not as a threat we mitigate.
* **Semantic gap.** It is now measured on 40 rows (up from 8) and replicated on PoisonedRAG, where the defended pipeline is no better than undefended RAG (90.7% vs. 91.3% attack success, paired p = 1.0). We add a practical-implications discussion.
* **New sections and material.** We add measured latency, VRAM, and RAM; a full environment and version table; an explicit statistical-methods paragraph; and four redrawn vector figures.

Section and table numbers below refer to the revised manuscript.

---

## Reviewer #3

**3.1 — Limited novelty; no demonstrated improvement over simpler defenses.**
We agree that most components are known, and we now say so explicitly ("Distinction from prior work", Sec. II, and the new positioning table, Table I). Our contribution is the attribution protocol and its findings, not a new detector. To test directly whether the cascade beats simpler defenses, we added a current-code ablation (new Sec. V-E, Table VII, Figs. 2–3) on 200 attack and 200 benign rows. It covers seven leave-one-out arms, seven single-layer arms, and four subsets, each paired row by row with the same undefended arm.

* No single layer comes close to the full stack. The best single layers leave 15–18% ASR, against 3.5% for the full stack.
* The subset sanitize + grounding + refuse reaches 5.5% ASR against 3.5% for the full stack (exact McNemar p = 0.34), with 9.5% vs. 8.5% benign blocking. The two are statistically indistinguishable, so the paper now explicitly recommends this simpler configuration.
* We repeated the undefended, full-stack, and subset arms with two more generation seeds (7 and 1234) on the same rows. Undefended ASR is 30.3 ± 0.8%, full stack 4.2 ± 0.6%, and subset 6.0 ± 0.9% across the three replicates. The subset is not significantly different from the full stack in any replicate (p = 0.34, 0.27, 0.77), but it is worse in all three, by 1.8 ± 0.8 points, and the paper now states this small cost. To narrow the interval, we built 224 further planted attacks with the same constructor (all remaining yes/no questions in the bank; n = 424 in total) and ran the three arms on them. At n = 424 the subset reaches 4.7% against 4.2% for the full stack (p = 0.81; paired difference +0.5 points, 95% CI [−1.5, +2.4]), so the cost of dropping the other four layers is bounded at about 2.4 points (Sec. V-E, "Larger sample").

We also compared against three external defenses on the same 210 paired rows (Sec. V-E, "External baselines"):

* **LLM-Guard detector alone:** 50% attack recall.
* **Paper-faithful ControlNET:** same blocking rate as our pipeline (p = 0.85), but it removes the poison before generation on 33/70 context-attack rows, versus 67/70 for ours.
* **Official Sentinel–Strategist (ADO) code:** removes the poison on 25–44 of 70 rows, versus 67/70 for ours (p ≤ 1.6×10⁻⁶).

**3.2 — Small, artificial 220-row dataset.**
The evaluation now covers 920 rows across seven sets (Table III, Sec. IV):

* Protocol A: the original 220 rows.
* Protocol B: 200 planted attacks and 200 benign rows.
* A 40-row semantic slice.
* External data: PoisonedRAG (NQ, HotpotQA, MS MARCO; 150 rows), BIPIA (50), and deepset prompt-injections (60).

We justify the stratified slice as an attribution instrument, not a prevalence sample. We also report a simulated power analysis: at n = 200, an exact McNemar test detects a 10-point ASR difference with power 0.87–0.96.

**3.3 — 100% neutralization is insufficiently supported; it relies on canary integrity rather than verification of malicious output.**
We agree, and we removed the claim. Sec. V-C now reports two levels:

* **Level 1, canary integrity:** 0/220 leaks and 74/74 redactions. This shows that no canary leaked into a delivered answer and that every poisoned chunk was removed before generation, and nothing more. (Some blocked direct-attack rows are stopped after generation, so we no longer say that no injected instruction reached the generator.)
* **Level 2, target-string checks on delivered outputs:** 7/200 planted attacks still succeed under the full stack (3.5%, 95% CI [1.7, 7.0], against 29.5% undefended). On the semantic slice, 14/40 succeed; on PoisonedRAG, 136/150 succeed, against 137/150 for a paired undefended arm. BIPIA blocks 45/50. BIPIA goals are not string-scorable, so we read the 5 unblocked answers manually: 2 followed the injected instruction, 2 did not, and 1 is indeterminable, giving 2–3/50 (4–6%) defended attack success.

The abstract, claims table (Table X), and conclusion were rewritten to match.

**3.4 — The conjunctive detector fails, and semantic misinformation succeeds in 50% of cases.**

*Detector.* While re-examining this result we found that the closed-form impossibility depends on one premise: the scanner's outputs are saturated (near-binary). With graded DeBERTa logit margins as channels, the detector fires validly (both channels sub-threshold) on 24/30 multivector rows, with 1/80 false fires on benign rows (Sec. V-B, Table VI). We state two limits plainly. First, the floors were set on the same slice's benign rows, so 1/80 is an in-sample rate. We have now tested this out of sample on 200 benign rag-mini-wikipedia questions that appear in no evaluation slice: the detector fires on 6/200 (3.0%, 95% CI [1.4, 6.4]). The query floor transfers (benign 95th percentile 0.18), but the 0.30 context floor sits inside the benign range (95th percentile 0.59), so the detector relies on the query floor and joint minimum; raising the context floor to the held-out 95th percentile removes the false fires and the valid fires alike (1/200 and 0/30). We therefore present the graded repair as evidence that the impossibility is specific to saturated channels, not as a deployable detector. Second, on external BIPIA conjunctive data the detector fires on 8/30 attacks and on 4/30 benign controls. (An earlier draft reported 10/30, which came from the mitigation harness's looser co-activation rule; we now apply the pipeline's own rule to both attacks and controls.) We also explain why the original failure was not a security failure: every multivector row was neutralized anyway by Stage-6 redaction.

*Semantic.* The 50% rate came from n = 8 rows, whose 95% CI is [21.5, 78.5]. On the new 40-row slice, the full stack reduces ASR from 75.0% to 35.0% (p = 4×10⁻⁴). On PoisonedRAG, the defended pipeline is no better than a paired undefended arm (90.7% vs. 91.3%, p = 1.0). We now state this as the dominant open risk and add a practical-implications discussion (see 5.4).

---

## Reviewer #4

**4.1 — Validate on standardized benchmarks (e.g., BIPIA) or larger RAG benchmarks.**
We added BIPIA (50 rows), PoisonedRAG on NQ, HotpotQA, and MS MARCO (150 rows), and deepset prompt-injections (60 rows), all run through the complete pipeline and paired row by row with an undefended arm on the same rows and poison (new Sec. V-G, Table VIII). Results:

* **BIPIA:** 45/50 blocked (0/50 undefended) and 33/49 injected chunks redacted.
* **PoisonedRAG:** no measurable protection. Attack success is 91.3% undefended and 90.7% defended (7 rows fixed, 6 broken, exact McNemar p = 1.0). Because the complete pipeline uses a 3-model ensemble, we also ran a same-generator comparison (llama3.2:3b in both arms, seven defense layers): 91.3% → 85.3% (16 fixed, 7 broken, p = 0.09; 95% CI for the difference [−12.2, +0.2] points). Neither comparison shows a significant reduction. This is PoisonedRAG passages carry no instruction signature and a planted fact is "grounded" in the poisoned passage.
* **deepset:** the gate blocks 14/30 real injections (17/30 including review flags) with 0/30 benign false positives; on the full held-out test split it blocks 23/60 at 1/56 benign false positives. (An earlier draft reported 11/30 from a run whose output was not saved; we re-ran the evaluation with a committed script and per-row outputs.) 31 of the 32 injections the gate passes have a scanner score of exactly 0, so no threshold change can catch them. Without changing the architecture, we let the gate also flag a prompt when the graded DeBERTa margin that Stage 3 already computes exceeds a threshold chosen on the deepset *train* split. On the test split this raises flagged injections from 28/60 to 38/60 at 2/56 benign flags, and adds review flags on 7/200 held-out benign RAG questions. End to end the pipeline refuses all 30 sampled injections, but also all 30 benign deepset prompts (they are off-topic for the wiki corpus), so we do not count downstream refusal on this set as detection.

We report the weak results with the same prominence as the strong ones.

**4.2 — Latency and VRAM/RAM overhead per query.**
New Sec. V-I with Table IX reports median and 95th-percentile end-to-end latency for each configuration, taken from per-row timings on 200 rows per arm:

| Configuration | Median latency per benign query |
|---|---|
| Undefended | 8.6 s |
| Sanitize + refuse | 10.7 s |
| Full seven-layer stack | 15.3 s |
| Complete 13-stage pipeline with 3-model ensemble | 23.8 s (mean) |
| ADO baseline | 31.5 s |

Memory was measured with a ~1 Hz sampler during a 20-query profiled run of the complete pipeline: peak GPU memory 6.3 GB of 8 GB (median 2.2 GB), and host RAM for the Python process a median of 6.4 GB and a peak of 8.0 GB. The pipeline therefore fits an 8 GB laptop GPU with about 8 GB of free host RAM (Sec. V-I, "Memory"). We also identify the main cost, the CPU-bound grounding judge, and give a throughput estimate.

We have now also tested the optimization the first submission left open: moving the encoders (bge-m3, the shared bge-reranker of Stages 7 and 10, DeBERTa) to the GPU, with no change to any stage or threshold. In isolation the encoder stages fall from 4.0 s to 0.11 s per query (fp16). End to end, fp32 GPU encoders are counter-productive on an 8 GB card (median 13.3 → 22.6 s) because their 5.2 GB peak leaves too little room for the generators; in fp16 (3.0 GB peak) the full stack runs at a median of 8.6 s (95th percentile 10.1 s) against 13.3 s with CPU encoders in the same session, with identical decisions on all 50 benign questions. This required one fix: Stage 10 now shares Stage 7's reranker instead of loading a second copy of the same model (Sec. V-I, "Encoders on the GPU"; Table IX). CPU remains the default, so every reported number reproduces unchanged.

**4.3 — Readability of Figs. 1 and 4.**
All figures are now vector graphics (TikZ/pgfplots) with ≥ 8 pt labels. Fig. 1 is redrawn as a six-step workflow at full text width. Fig. 4 is redrawn as one labeled bar chart, with 95% Wilson CIs and n on every bar. Figs. 2 and 3 are regenerated from current-code data.

---

## Reviewer #5

**5.1 — Distinguish contributions from existing defenses.**
See 3.1. We added "Distinction from prior work" with contributions C1–C4, a "Defense systems for RAG" paragraph (ControlNET, ADO, layered frameworks), the positioning table (Table I), and head-to-head comparisons.

**5.2 — Environment, hardware, software versions, model configurations.**
We added a new "Environment and configuration" paragraph and Table V. They cover:

* **Hardware:** Ryzen 5 7640HS, 16 GB RAM, RTX 4060 Laptop with 8 GB.
* **Software:** OS, Python, Ollama, and exact library versions.
* **Models:** model IDs and quantization (Q4_K_M generators, Q8_0 Llama Guard).
* **Settings:** every threshold and fusion weight, k/top-n, temperatures, and seeds.
* **Protocols:** which protocol uses which generator configuration.

A pinned `requirements_lock_venv311.txt` is added to the repository. The repository now has a README that maps every table and figure to the script that produces it and the committed per-row outputs it reads, plus an MIT licence and a dataset-licence table. The paper adds "Data and Code Availability" and "Ethics Statement" sections.

**5.3 — Small dataset; justify representativeness; evaluate on independent test sets.**
See 3.2 and 4.1. Sec. IV now explains what the stratified slice represents (one OWASP LLM01 threat model per class) and what it does not represent (real-world prevalence). It also adds three independent external test sets.

**5.4 — Investigate the semantic vulnerability (4/8) and discuss practical implications.**
The slice grows to 40 rows, with layer-level attribution (Sec. V-H):

* Grounding cannot help, because a planted fact is "grounded" in the poisoned context.
* The guarded prompt and refusal account for the reduction, but neither survives Bonferroni correction at n = 40.
* We also ran the pipeline's isolate-and-aggregate generator (answer per chunk, accept only cross-chunk agreement) on the same 40 rows. ASR falls to 0/40 against 18/40 for the fused ensemble in the same run, but almost entirely by abstaining (34/40). On 40 benign questions it answers only 15 (fused: 37), so the paper reports it as a utility-prohibitive option, not a fix (Sec. V-H). A routed variant that uses the agreement check only when retrieved passages disagree keeps utility (36/40 benign answered) but reduces semantic ASR only from 18/40 to 15/40 (p = 0.45), which is not significant.
* PoisonedRAG replicates the gap externally.

A new "Practical security implications" paragraph covers who is exposed (any corpus an outsider can write to), the consequence (integrity loss through confident, cited false answers), and four deployment measures: provenance as a security boundary, corroboration, surfaced citations, and human review for high-stakes answers. We state clearly that these measures are not evaluated here.

**5.5 — Statistical analysis: McNemar interpretation, CIs, sample sizes.**
A new "Statistical analysis" paragraph (Sec. IV) explains:

* McNemar uses only discordant pairs b/c, under H₀ b ~ Bin(b+c, ½). We use the exact test because b+c is often below 25, and we print b/c next to every p.
* A small p says nothing about effect size, and a large p is not evidence of equivalence. This is why we call the three-layer subset "indistinguishable", not "equivalent".
* We use Wilson intervals because they keep coverage near 0 and 1.
* We state the n for every test.
* We apply Bonferroni correction to the leave-one-out (α = 0.007) and subset (α = 0.0125) families and flag results that do not survive it. As a result, two semantic-slice attributions are now labeled "suggestive".
* We define AUC and the percentile gap precisely.
* Run-to-run variance: the headline Protocol B arms are now replicated under three generation seeds and reported as mean ± SD with per-replicate tests (Sec. V-E, "Replicates"). Ollama GPU generation is not bit-reproducible at a fixed seed, so the paper treats these runs as independent replicates and says so in the threats to validity.

The earlier-code arms are kept and clearly labeled (Sec. V-F). We retract their "refusal is the only significant single layer" result, which the current-code ablation contradicts.

**5.6 — Computational overhead, memory, latency, scalability.**
See 4.2. We also added a "Scalability" paragraph covering sequential throughput (≈ 240 queries/hour/laptop), what scales it, and the untested effect of the HNSW index at larger corpus sizes.

**5.7 — Equations, figures, tables, proofreading.**

* **Equations:** Eqs. (1)–(3) now define every symbol (x_q, x_c, f_i, k, τ_J), and the firing rule is stated as an equation rather than in prose.
* **Proofreading:** the full text was proofread. Typographical errors fixed include "classifier,r", "breaks,k s", "queries,s", "q-0.7", "p ¡ .001", and the capitalized "Is at or above".
* **Abstract:** the inaccurate "on-directional, off-directional, and multimodal" description was removed; no multimodal attacks were evaluated.
* **Figures:** all four are redrawn as vector graphics.
* **Tables:** tables now state n and CIs consistently.

---

### Notes for the authors (remove before submission)
* Bibliographic details of ControlNET, BIPIA (KDD 2025), PoisonedRAG (USENIX Security 2025), ADO (first author Pallerla) and Saleem et al. (arXiv:2606.19660) were checked against their arXiv pages. The Saleem et al. related-work text and positioning-table row were corrected after reading their full paper: they do evaluate all seven layer subsets, but without significance tests or CIs, and their target model is GPT-4o via API.
* The original raster figures were not available as source. Figs. 1–4 are rebuilt from the data and need a visual check after compiling (e.g., on Overleaf).
