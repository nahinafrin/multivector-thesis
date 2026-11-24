# Response to Reviewers

**Paper:** Attributing Security in a Local, Zero-Cost Defense-in-Depth Pipeline for Retrieval-Augmented Generation

We thank the reviewers for their careful and constructive comments. Several of them changed our
conclusions, not just the wording. Section, table and figure numbers below refer to the revised
6-page manuscript. Material that did not fit in six pages is in the extended version, released with
the code, data and per-row outputs at https://github.com/nahinafrin/multivector-thesis
(`paper_revision/main.tex`); we mark it "extended version" below.

The main changes are:

* **Larger, independent evaluation.** We now report 920 scored rows across seven sets, up from 220
  (Table IV). They include 150 PoisonedRAG rows, 50 BIPIA rows and 60 deepset rows, none of which we
  constructed.
* **Comparison with simpler and existing defenses.**
  * A three-layer subset (sanitization, grounding, refusal on detection) is statistically
    indistinguishable from the seven-layer stack, and we now recommend it.
  * On the same 200 attacks, the seven-layer stack outperforms an LLM-Guard firewall, Spotlighting and
    the ADO orchestrator, and matches the ControlNET detector used as a firewall (Table VI).
* **Narrower neutralization claim.** "100% neutralization" is gone. We separate canary integrity (no
  leak in any of the 132 delivered answers) from attacker content in delivered outputs. In delivered
  outputs, attack success is 3.5%, not 0%.
* **Correction to the conjunctive-detector result.** The impossibility holds for *saturated* classifier
  scores only. Graded logit margins restore detection in-sample (24/30), but only modestly on external
  data (8/30).
* **Explicit scope.** The pipeline contains instruction-carrying injection but offers no measurable
  protection against knowledge corruption. We present this as a measured limit of this defense class
  (contribution C4).
* **Semantic gap.** It is now measured on 40 rows (up from 8) and replicated on PoisonedRAG. We add a
  practical-implications discussion.
* **New material.**
  * Measured latency and memory (Table VIII).
  * An environment and version table (Table III).
  * A statistics paragraph (Sec. IV).
  * Redrawn vector figures.

---

## Reviewer #3

**3.1. Limited novelty; no demonstrated improvement over simpler defenses.**

*Novelty.* We agree that most components are known, and we now say so explicitly (Sec. I,
contributions C1–C4; Sec. II; positioning Table I). Our contribution is the attribution protocol and
its findings, not a new detector.

*Simpler defenses: ablation* (Sec. V-B, Table V, Figs. 1–2). The ablation covers 200 attack and 200
benign rows: seven leave-one-out arms, seven single-layer arms and four subsets, each paired row by row
with the same undefended arm.

* No single layer comes close to the full stack. The best single layers leave 15–18% attack success
  (ASR), against 3.5% for the full stack.
* The subset sanitize + grounding + refuse reaches 5.5% ASR (7 vs. 3 discordant rows, exact McNemar
  p = 0.34), with 9.5% vs. 8.5% benign blocking. The paper therefore recommends this simpler
  configuration.
* **Seed replicates.** Across three generation seeds, the subset is worse by 1.8 ± 0.8 points but never
  significantly (p ≥ 0.27).
* **Enlarged set.** On an enlarged set of 424 planted attacks, ASR is 29.7% undefended, 4.2% for the
  full stack and 4.7% for the subset. The paired difference is +0.5 points (95% CI [−1.5, +2.4]), so
  dropping the other four layers costs at most about 2.4 points.
* **Fresh rerun.** A rerun from the raw slice gives 29.0% and 3.0%, and agrees with the reported run on
  189 and 199 of 200 rows.

*Existing defenses* (Sec. V-C, Table VI). We ran four existing defenses on the same 200 planted attacks
and 200 benign questions, with the same generator and scoring, paired by row.

| Existing defense | Its ASR | Paired test vs. our 3.5% | Benign correct, theirs vs. ours (52.0%) |
|---|---|---|---|
| LLM-Guard prompt-injection firewall | 21.0% | 3 vs. 38 discordant rows, p = 1×10⁻⁸ | no significant difference |
| Spotlighting (datamarking) | 23.0% | p < 10⁻⁹ | no significant difference |
| ADO (Sentinel–Strategist orchestrator) | 28.5% | 54 vs. 4 discordant rows, p = 3×10⁻¹² | 35.5% (p = 6×10⁻⁶) |
| ControlNET detector, used as a firewall | 4.5% | p = 0.77 (statistically indistinguishable) | 46.0% (p = 0.058) |

* **ADO.** We ran the authors' unmodified code through an adapter. Local hardware forced a substitute
  controller and a 4-bit attention filter, so we call it an adapted implementation.
  * Most of its cost comes from DP-RAG removing every passage, which makes the generator refuse.
  * The scorer counts a refusal such as "No relevant information…" as a successful attack when the
    planted answer is "No". If refusals are not counted as successes for any arm, ADO's ASR is 13.5% vs.
    3.5% (p = 2×10⁻⁴).
* **ControlNET.** The ControlNET detector was given exactly the passages the undefended generator
  received.

We therefore claim a clear improvement over an input firewall, prompt-level spotlighting and an
adaptive orchestrator, and parity with ControlNET on attack success. We do not claim general
superiority.

**3.2. Small, artificial 220-row dataset.**
The evaluation now covers 920 rows across seven sets (Table IV):

* Protocol A: the original 220 rows.
* Protocol B: 200 planted attacks and 200 benign questions.
* A 40-row semantic slice.
* External data: PoisonedRAG (150 rows), BIPIA (50) and deepset prompt-injections (60).

We justify the stratified slice as an attribution instrument, not a prevalence sample: each class
isolates one threat model and carries machine-checkable expectations (Sec. IV). A simulated power
analysis is in the extended version: at n = 200, an exact McNemar test detects a 10-point ASR difference
with power 0.87–0.96.

**3.3. 100% neutralization is insufficiently supported; it relies on canary integrity rather than
verification of malicious output.**
We agree, and we removed the claim. The paper now reports two levels.

* **Canary integrity (Sec. V-A).**
  * No canary leaks into any of the 132 delivered answers.
  * The canary was checked on the 175 rows that reached Stage 10; the other 45 were blocked earlier,
    44 of them at the gate.
  * Every poisoned chunk (74/74) is removed before generation.
  * This shows only that no canary leaked into a delivered answer and that every poisoned chunk was
    removed. It does not show that the outputs are safe.
* **Attacker content in delivered outputs** (Protocol B and external sets: Secs. V-B and V-E, Tables V
  and VII).
  * 7/200 planted attacks still succeed under the full stack: 3.5%, 95% CI [1.7, 7.0], against 29.5%
    undefended.
  * On the semantic slice, 14/40 succeed.
  * On PoisonedRAG, 136/150 succeed, against 137/150 for a paired undefended arm.
  * BIPIA blocks 45/50. BIPIA goals are not string-scorable, so we read the 5 unblocked answers: 2
    followed the injected instruction, 2 did not, and 1 is indeterminable. Defended attack success is
    therefore 2–3 of 50.

The abstract and conclusion were rewritten to match.

**3.4. The conjunctive detector fails, and semantic misinformation succeeds in 50% of cases.**

*Detector* (Sec. V-D).
* **Why it failed.** The failure depends on one premise: the scanner's outputs are saturated
  (near-binary). A valid fire requires both channels to be below threshold, so a signal built from
  saturated scores is low on exactly the rows it targets. The 6-page version gives this argument in
  prose; the closed-form statement is in the extended version.
* **Repair.** With graded DeBERTa logit margins as channels, the detector fires validly on 24/30 rows,
  at 1/80 benign false fires.
* **Limits of the repair.**
  * The 1/80 rate is in-sample.
  * On 200 held-out benign questions it fires 6 times.
  * On external BIPIA conjunctive data it fires on 8/30 attacks and on 4/30 benign controls.
* **Conclusion.** We present the repair as evidence that the impossibility is specific to saturated
  channels, not as a deployable detector.
* **Security cost.** The original failure did not cost security, because Stage 6 redacted every
  conjunctive chunk.

*Semantic* (Sec. V-E).
* The 50% rate came from n = 8 rows, whose 95% CI is [21.5, 78.5].
* On the new 40-row slice, the full stack reduces ASR from 75.0% to 35.0% (p = 4×10⁻⁴).
* On PoisonedRAG, the defended pipeline is no better than a paired undefended arm (90.7% vs. 91.3%,
  p = 1.0).

We now state this as the dominant open risk and discuss its practical implications (see 5.4).

---

## Reviewer #4

**4.1. Validate on standardized benchmarks (e.g., BIPIA) or larger RAG benchmarks.**
We added BIPIA (50 rows), PoisonedRAG on NQ, HotpotQA and MS MARCO (150 rows) and deepset
prompt-injections (60 rows). Each was run through the complete pipeline and paired row by row with an
undefended arm on the same rows and poison (Sec. V-E, Table VII).

* **BIPIA:** 45/50 blocked (0/50 undefended). Of the 5 unblocked, 2 followed the injected instruction.
* **PoisonedRAG:** no measurable protection.
  * The complete pipeline (three-model ensemble) gives 90.7% ASR, against 91.3% undefended (7 rows fixed,
    6 broken, p = 1.0).
  * A same-generator comparison (seven layers) gives 85.3% (16 fixed, 7 broken, p = 0.09).
  * PoisonedRAG passages carry no instruction signature, and Stage 6 redacts none of the 148 injected
    passages.
* **deepset:**
  * The gate flags 17 of 30 real injections at 0/30 benign.
  * On the full test split, it flags 28/60 injections.
  * Without changing the architecture, we let the gate also flag on the graded margin Stage 3 already
    computes, with a threshold chosen on the deepset *train* split. This raises flagged injections on
    the test split to 38/60, at 2/56 benign flags.

We report the weak results with the same prominence as the strong ones.

**4.2. Latency and VRAM/RAM overhead per query.**
Sec. V-F and Table VIII report median and 95th-percentile end-to-end latency per configuration, from
per-row timings on 200 rows per arm. Median latency per benign query:

| Configuration | Median latency per benign query |
|---|---|
| Undefended | 8.6 s |
| Sanitize + refuse | 10.7 s |
| Full seven-layer stack | 15.3 s |
| Three-layer subset (sanitize + ground + refuse) | 19.0 s |
| Complete 13-stage pipeline, 3-model ensemble | 23.8 s (mean) |
| ADO, Protocol A run (without its attention filter) | 31.5 s |
| ADO, Protocol B (adapted, with attention filter) | 45.3 s |

The grounding judge, which runs on the CPU, is the dominant cost.

**Memory.** We sampled memory at about 1 Hz during a profiled run of the complete pipeline. Peak usage
is 6.3 GB of the 8 GB GPU and 8.0 GB of host RAM, so the pipeline fits an 8 GB laptop GPU.

**Encoders on the GPU.** We moved the encoders to the GPU in fp16, with no change to any stage or
threshold. This cuts the full stack's median latency to 8.6 s, against 13.3 s with CPU encoders in the
same session, with identical decisions on all 50 benign questions tested. CPU remains the default, so
every reported number reproduces unchanged. The fp32 result, which is slower on an 8 GB card, and a
throughput estimate are in the extended version.

**4.3. Readability of Figs. 1 and 4.**
* **Old Fig. 1 (workflow).** In the 6-page version it is replaced by Table II, which lists each stage
  and its implementation. A redrawn full-width vector workflow figure is in the extended version.
* **Old Fig. 4.** It is now Fig. 1: a single labeled bar chart with 95% Wilson CIs, n given in the
  caption, and small-font axis and tick labels.
* **Fig. 2.** The security-versus-latency trade-off is regenerated from current-code data.
* All figures are vector graphics (TikZ/pgfplots).

---

## Reviewer #5

**5.1. Distinguish contributions from existing defenses.**
See 3.1. Sec. I lists contributions C1–C4 and states that we do not claim a new detector. Sec. II covers
detectors, structural defenses and RAG defense systems (ControlNET, ADO, layered frameworks). Table I
positions the work, and Table VI gives head-to-head comparisons on identical rows.

**5.2. Environment, hardware, software versions, model configurations.**
Table III and Sec. IV cover:

* **Hardware:** Ryzen 5 7640HS, 16 GB RAM, RTX 4060 Laptop GPU with 8 GB.
* **Software:** OS, Python, Ollama and exact library versions.
* **Models:** model IDs and quantization (Q4_K_M generators, Q8_0 Llama Guard).
* **Settings:**
  * gate thresholds and retrieval depth;
  * Protocol B generator, temperature and seed;
  * which protocol uses which configuration.

Every threshold and fusion weight is listed in the extended version. The repository includes:
* a pinned `requirements_lock_venv311.txt`;
* a README that maps every table and figure to the script that produces it and to the per-row outputs it
  reads;
* a claim-by-claim checker that recomputes the paper's numbers from those outputs;
* an MIT licence and a dataset-licence table.

**5.3. Small dataset; justify representativeness; evaluate on independent test sets.**
See 3.2 and 4.1. Sec. IV explains what the stratified slice represents (one threat model per class) and
what it does not (real-world prevalence). Three independent external test sets were added.

**5.4. Investigate the semantic vulnerability (4/8) and discuss practical implications.**
The slice grows to 40 rows, with layer-level attribution (Sec. V-E):

* Grounding cannot help, because a planted fact is "grounded" in its poisoned passage.
* **Agreement-based generator.** It answers per chunk and accepts only cross-chunk agreement. It removes
  the attack (0/40) only by abstaining, and answers only 15 of 40 benign questions. We therefore report
  it as a utility-prohibitive option, not a fix.
* **Routed variant.** A variant that uses the agreement check only when passages disagree keeps utility,
  but reduces ASR only from 18/40 to 15/40 (p = 0.45).
* PoisonedRAG replicates the gap externally.

A "Practical implications" paragraph (Sec. V-E) covers:
* who is exposed: any corpus an outsider can write to;
* the consequence: confident, citation-backed false answers;
* four deployment measures: provenance as a security boundary, corroboration, surfaced citations, and
  human review for high-stakes answers.

We state that these measures are not evaluated here.

**5.5. Statistical analysis: McNemar interpretation, CIs, sample sizes.**
The "Statistics" paragraph (Sec. IV) explains:

* McNemar's test uses only discordant pairs. We use the exact test and print the discordant counts next
  to every p.
* A large p means no detectable difference at that n, not equivalence. This is why we call the
  three-layer subset "statistically indistinguishable", not "equivalent".
* Proportions carry 95% Wilson intervals.
* Leave-one-out and subset families use Bonferroni correction.

Every table states its n. Run-to-run variance is addressed with three generation seeds and a fresh rerun
from the raw slice (Sec. V-B). Ollama GPU generation is not bit-reproducible, which the threats to
validity state (Sec. V-G). The extended version adds:
* the formal definition of the test;
* per-family Bonferroni thresholds;
* the power analysis;
* AUC and percentile-gap definitions.

**5.6. Computational overhead, memory consumption, inference latency, scalability.**
See 4.2. The extended version adds a scalability paragraph: sequential throughput, what scales it, and
the untested effect of a larger corpus and index.

**5.7. Equations, figures, tables, proofreading.**

* **Proofreading.** The full text was proofread. Typographical errors fixed include "classifier,r",
  "breaks,k s", "queries,s", "q-0.7", "p ¡ .001" and the capitalized "Is at or above".
* **Abstract.** The inaccurate "on-directional, off-directional, and multimodal" description was
  removed; no multimodal attacks were evaluated.
* **Equations.** To fit six pages, the detector's firing rule is stated in prose (Sec. V-D). The
  equations, with every symbol defined, are in the extended version.
* **Figures.** They are redrawn as vector graphics (see 4.3).
* **Tables.** They now state n and CIs consistently.
