# 05 — Independent review (written as a skeptical conference reviewer)

## Answers to the eight questions

**1. What new knowledge does the study produce?**

Not the defense. Its components are known, and the paper says so. The new knowledge is empirical and specific to
this setting:
- **(a) The three key layers act as approximately independent filters.** Observed vs predicted under independence
  is within ±1.3 points in every cell. This explains the diminishing returns, and explains why three layers capture
  nearly all of the seven-layer effect on instruction-bearing attacks.
- **(b) The minimal sufficient subset depends on the attack class.** On assertion-only attacks the three-layer
  subset is significantly worse than seven layers (+27.5 points, p = 0.0034, n = 40), because the guarded prompt
  carries that defense. Alone, the same prompt is strongly harmful on instruction-bearing attacks.
- **(c) The headline protection is wording-dependent.** Full-stack ASR ranges from 0% to 9.9% across three randomized
  query phrasings (exploratory).
- **(d) A stage-by-stage account of why instruction-signature defenses fail on PoisonedRAG.**
  - Rank 1 in every row.
  - Zero signature.
  - Fully "grounded".
  - Counter-evidence almost never present in this KB.
- **(e) The closed-form saturation argument for co-activation detection.**

Items (a)–(c) are not in the current manuscripts. They come from saved data.

**2. Is the attribution methodology distinct from prior component ablations?** Only incrementally.
- **Prior art.** Ablation of layered prompt-injection defenses exists (Saleem/Ahmed et al. 2026: all subsets;
  Ramakrishnan & Balaji 2025; LlamaFirewall). Factorial interaction analysis is textbook (fANOVA; ablation
  analysis).
- **What differs:** exact paired tests on identical rows, a measured noise floor, an independence null, and
  class-conditional repetition.
- **Verdict.** These are good practice, not a new method. A reviewer would accept "a careful, reproducible
  application", and would reject "a novel attribution methodology".

**3. Does it give insights a full-stack-vs-undefended comparison would miss?** Yes, and this is the strongest
argument for the paper:
- class-dependent sufficiency (1b);
- the guarded prompt's sign reversal;
- non-contributing layers (Presidio and DLP on injection, the tightened rerank);
- the independent-filter structure (1a);
- wording dependence (1c).

None of these is visible in "29.5% → 3.5%".

**4. Are the three-layer findings supported by the statistics?**
- *For instruction-bearing planted attacks:* yes, as a bound, not as equivalence. At n = 424 the cost of dropping
  four layers is +0.5 points, 95% CI [−1.5, +2.4]. The point estimate leans the same way (+1.0 to +2.5) in all three
  n = 200 seeds, so "up to about 2.5 points" is the honest reading. "Statistically indistinguishable" is technically
  true but invites an equivalence reading, and should be replaced.
- *As general deployment advice:* **no**. The semantic-slice subset arm contradicts it, and both conclusions
  currently recommend the subset without that qualification.

**5. Does the study clearly separate the 13-stage system from the seven-layer configuration?**
- Mostly yes in the ICCIT paper (after `7c7956f`).
- **But a third configuration is unacknowledged.** The external results labelled "complete pipeline" come from an
  adaptive-tier variant that, for nearly every PoisonedRAG and BIPIA row:
  - generated with a single `llama3.2:3b`;
  - ran neither the multivector detector, the controller, Presidio nor DLP.
- **This is the most serious issue found.** It changes no number. It does invalidate one sentence ("a planted fact
  is consistent across models", as supported by this run) and one premise ("because the complete pipeline uses the
  ensemble, we also ran a same-generator comparison"). It must be corrected before submission.

**6. Are external generalization claims appropriately restricted?**
- The knowledge-corruption failure is stated carefully.
- The instruction-injection success is stated for Protocol B and BIPIA, without saying that:
  - it rests on one generator and one KB;
  - it is concentrated on one attacker phrasing.
- Nasr et al. (USENIX Sec. 2026) makes this a predictable reviewer objection. The papers should say the attack set is
  static and non-adaptive, and report the wording heterogeneity.

**7. Is the PoisonedRAG negative result sufficiently investigated?** After 04, nearly.
- **Established:** where each stage fails.
- **Open:**
  - ~~whether context overrides known answers~~ answered by E2: yes, in about half of the defended successes;
  - whether the ensemble described in the papers behaves differently (E2c, about 1.2 h);
  - whether counter-evidence would change the outcome (E2b).
- **Gap in Related Work.** The paper should cite TriShieldRAG and RobustRAG. They show that defenses *designed* for
  knowledge corruption can reduce non-adaptive PoisonedRAG. Our negative result is about this defense class, not
  about the problem.

**8. Can the revised contribution statements survive a skeptical novelty review?** The C1–C4 in 02 §6 can,
*provided*:
- C2 is framed as an application of standard designs;
- C3 is scoped to the seven-layer configuration and the evaluated attack classes, with (c) and (d) marked
  exploratory;
- C4(b) does not claim an ensemble result.

The current C1 ("an attribution protocol … so a layer's contribution is measured rather than assumed") is fine in
substance, but sits beside an inconsistent second list in main.tex.

## Rating of each proposed contribution (02 §6)

| Contribution | Rating | Reason |
|---|---|---|
| C1 System (local 13-stage pipeline, forward-propagated risk) | **Supported but incremental** | Built and measured. Known components. The controller adds no measured security |
| C2 Evaluation protocol (paired factorial and leave-one-out, independence null, noise floor, class-conditional, claim checker) | **Supported but incremental** | Fully executed on saved data and reproducible from frozen outputs. Methodologically standard |
| C3(a) Independent-filter structure and Shapley shares | **Strongly supported** (within Protocol B) | Full 2³ factorial, bootstrap CIs, all independence residuals ≈ 0 |
| C3(b) Three layers within 2.4 points of seven (instruction-bearing) | **Strongly supported as a bound** | n = 424 CI plus three seeds. Not an equivalence test |
| C3(c) Class-dependent minimal subset; guarded-prompt sign reversal | **Supported, exploratory** | p = 0.0034 and p = 0.035 at n = 40; not prespecified |
| C3(d) Wording dependence | **Partly resolved by E1** | Protection transfers to held-out coercive wording (preregistered: 3.0% vs 3.5%, p = 1.0, upper bound 6.4%). Without a coercive suffix, ASR is 7.5% (p = 0.115, inconclusive); the relative reduction falls from 88% to 55% and the detector fires 80 → 6 rows (descriptive). A "weaker without coercive phrasing" claim remains unconfirmed |
| C4(c) PoisonedRAG: prior override vs evidence absence (E2) | **Supported, descriptive** | llama3.2:3b knows the answer closed-book in 54.4% [46.0, 62.5] of defended successes; knowing gives no protection |
| C4(a) Saturation argument plus graded repair | **Supported but incremental** | Argument holds under its premise; the repair transfers poorly (8/30) |
| C4(b) PoisonedRAG failure mechanism | **Strongly supported for signature, rank and grounding; not supported for ensemble consensus** | Traces are complete for Stages 3–10, but the ensemble did not run |

## What is genuinely improved by this work (no new inference)

1. A precise, honest replacement for "indistinguishable": a bound plus seed direction.
2. Three new, data-backed findings (independent filters, class-dependent sufficiency, wording dependence). They
   turn the attribution study from "here are ablation numbers" into "here is what attribution reveals that
   end-to-end numbers hide".
3. A mechanism-level PoisonedRAG account and a failure taxonomy.
4. A correctly positioned novelty claim, with the prior art the current papers miss.
5. Discovery of the external-arm description mismatch, which a careful reviewer re-running the code would find.

## What remains unresolved

- No adaptive attacker.
- One KB and essentially one generator for every attack-success result.
- Held-out wording was written by us (E1), not by an adaptive attacker. The no-suffix effect is inconclusive at
  n = 200.
- Prompt-layer interactions not identified (E4).
- No ensemble evidence on external knowledge corruption (E2c).
- Benign utility only on in-domain wiki questions.
- The four adaptive-pipeline source files are missing from the public repository.

**Recommendation to the authors:**
1. Apply the wording, scope and description corrections in 06. They need no experiments, and they remove the issues
   a reviewer would most likely catch.
2. E1 and E2 were run on 2026-11-25 (`experiments/RESULTS_E1_E2.md`). Run E2c (about 1.2 h) only if the "complete
   pipeline" label is kept.
3. Do not add a defense layer.
