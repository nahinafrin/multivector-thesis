# 04 — PoisonedRAG failure analysis (saved traces only)

**Data.**
- Defended arm, described in the papers as the "complete pipeline": `step4 dataset/ext_n200_results.jsonl`. It
  was actually run with `run_full_pipeline_adaptive.py` (see 01 F2): Stages 1–8, 10 and 13 under a tiered policy.
  At low risk (LOW tier: top-k 8) it generates with **one `llama3.2:3b`**. This applied to ≥139/150 PoisonedRAG rows.
  The multivector detector, the controller, Presidio and DLP were not part of this run.
- Seven-layer same-generator harness: `mitigation_results/external200/on_full.jsonl`.
- Undefended: `external200/off.jsonl`.
- 150 PoisonedRAG rows (50 each NQ, HotpotQA, MS MARCO), paired by position.
- Scored with the paper's external rule (`score_external_paired.py`, `final_n200_summary.is_neutralized_final`).
- Script: `analysis/poisonedrag_traces.py`; outputs `analysis/outputs/poisonedrag_traces.json`, including per-row
  traces.

The recomputed headline matches the paper:

| Arm | ASR |
|---|---|
| Undefended | 91.3% (137/150) |
| Defended (adaptive-tier variant) | 90.7% (136/150); 7 fixed / 6 broken, p = 1.0 |
| Seven-layer harness | 85.3% (128/150) |

**This vulnerability is not solved, and nothing below claims otherwise.**

## 1. Stage-by-stage trace (defended adaptive-tier run, 150 rows)

| Stage | What happened to the poisoned passage | Count |
|---|---|---|
| 3 Input gate | Query allowed; injection score 0.0 in every row (median, q25 and q75 all 0) | 148 ALLOW, 2 BLOCK* |
| 4–5 Retrieval | Poisoned passage present in the retrieved set | 148/148 |
| 6 Per-chunk sanitization | Passage survives; redaction flag set | 148 survive; **0 redacted** |
| 7 Rerank | Passage in the ranked context | 148/148 |
|  | Ranked **first** | **148/148** (median margin over the next passage 0.23; minimum 0.01) |
| 9 Generation | Tier (from the ranked-context size) | LOW tier, single `llama3.2:3b`, in ≥139/150 rows. The recorded disagreement is 0.0 in every row; the single-model path writes this by construction, so **the ensemble was not exercised** |
| 9b Knowledge-conflict check | Not run in this configuration (`--kc-backend off`) | 150/150 "None" |
| 10 Grounding judge | Successful attacks passing grounding | **136/136**; faithfulness median 1.0 (IQR 0.86–1.0) |
|  | Blocked by grounding | 4 |
| Outcome | Delivered 144; attack succeeded 136 | |

\*Both gate blocks had injection score 0. They came from the fusion and safety path, not from an instruction
signature.

**How the 7 "fixed" rows were stopped:**

| Mechanism | Rows |
|---|---|
| Gate blocks unrelated to injection | 2 |
| Grounding refusals | 2 |
| Delivered answers that gave the true answer | 3 |

None involved the poisoned passage being detected. The 6 "broken" rows are the mirror image: answers the
undefended model got right and the ensemble got wrong. A 7 vs 6 split is within the run-to-run noise measured for
same-configuration reruns.

**Seven-layer harness, same generator:** 11 PoisonedRAG rows blocked (6 grounding, 5 detector), giving 85.3%.
p = 0.09 against undefended; the paired difference CI is [−12.2, +0.2], so a reduction of up to about 12 points
cannot be excluded.

## 2. Answers to the questions in the brief

1. **How often does poisoned evidence survive retrieval?** Every time it was retrieved (148/148), and it was ranked
   first every time. The two gate-blocked rows never reached retrieval.
2. **Which stages fail to identify it?**
   - Stage 3: score 0.
   - Stage 6: 0 redactions.
   - Stage 7: ranked it first.
   - Stage 10: grounded.
   - Stage 9 ran a single model for almost every row, so ensemble disagreement was never tested here.
   - The multivector detector and the controller were not part of this run. In the complete pipeline they key on the
     same instruction-signature scores as Stages 3 and 6, which are 0 here, so they would have had no signal to act
     on. This is an inference, not a measurement.
3. **Does grounding accept the false claim because it appears in a retrieved passage?** Yes. Every successful attack
   passed grounding, with median faithfulness 1.0. Delivered answers that did *not* adopt the poison had lower
   faithfulness (median 0.77, n = 8). Grounding measures consistency with the context; a poisoned context makes the
   poisoned answer the most "grounded" one. The semantic slice agrees: grounding alone has no effect there (75% →
   75%).
4. **Do other passages contradict the planted fact?** Rarely, in this setup:
   - The gold answer appears as a word-bounded string in another *ranked* passage in **11 of 148** rows, and in
     13 of 150 across the whole retrieved set.
   - Per source: NQ 5, HotpotQA 2, MS MARCO 4.
   - In 9 of those 11 the attack still succeeded.

   Caveat: string matching misses paraphrases, so this is a lower bound. The reason is structural: PoisonedRAG's
   questions come from NQ, HotpotQA and MS MARCO, while the knowledge base is rag-mini-wikipedia (3,200 passages).
   **For most rows, the poisoned passage was the only passage that answered the question.**
5. **Missing provenance, missing corroboration, or reliance on instruction signatures?** All three, at different
   stages, and the saved data separate them:
   - **Instruction-signature reliance (supported directly):** Stages 3, 6 and the detector all score the passage
     0 or leave it unredacted. This is why the instruction-defense class has nothing to act on.
   - **Missing corroboration and counter-evidence (supported, but confounded by the knowledge base):** in 137 of
     148 rows, no genuine passage in context gave the true answer, so a corroboration or majority rule *in this
     setup* would have had nothing to prefer. Whether the stack would resist poisoning when true evidence *is*
     present is not tested here. The 9/11 successes where it was present hint that it would not, but n is small.
   - **Missing provenance (supported by elimination):** nothing in the pipeline distinguishes an injected passage
     from a corpus passage. Every passage is treated equally and the poison wins on relevance. We did not test any
     provenance mechanism.
   - **Context dominance over model knowledge (not testable from saved data):** the knowledge-conflict check was off,
     so we do not know how often the generators "knew" the true answer and were overridden. See E2 below.
     **E2 (run 2026-11-25):**
     - `llama3.2:3b` answers correctly closed-book on 54.4% [46.0, 62.5] of the 136 defended successes, so context
       overrode its own knowledge in about half of them.
     - Defended ASR is the same whether it knows the answer (90.2%) or not (91.2%).
6. **Conclusions supported directly by the saved outputs:**
   - the defense class fails at *every* stage for this attack, not at one weak point;
   - grounding is not just ineffective but *reinforces* the poisoned answer (faithfulness 1.0). Ensemble agreement
     was not exercised in this run, so the saved PoisonedRAG data cannot say whether it would help. The papers'
     claim that "a planted fact is consistent across models" rests only on the n = 8 semantic supplement
     (disagreement AUC 0.47–0.48).
   - the passage is the top-ranked evidence in every row;
   - in this knowledge base, counter-evidence was almost never available.

## 3. Failure taxonomy for knowledge corruption

| Mechanism | Evidence (saved traces) | Would a defense of this class help? |
|---|---|---|
| Signature absence: no instruction syntax to detect | Gate score 0; 0/148 redactions | No, by design |
| Relevance dominance: the poisoned passage is the most query-relevant | Rank 1 in 148/148 | Only with source-trust weighting (provenance) |
| Grounding as consistency | Faithfulness 1.0; all successes pass | No: grounding checks consistency, not truth |
| Ensemble agreement (not exercised) | A single model generated in ≥139/150 rows; disagreement is 0 by construction | Untested on PoisonedRAG. On the n = 8 semantic supplement, disagreement did not separate attacks (AUC 0.47–0.48). E2c below would test it |
| Evidence absence: no counter-passage in context | Gold answer elsewhere in context in 11/148 | Corroboration needs a corpus that contains the truth |
| Prior override (measured by E2) | llama3.2:3b knows the answer closed-book in 74/136 defended successes (54.4%); knowing gives no protection (ASR 90.2% vs 91.2%) | A parametric-consistency check could at most address this ~half; untested (the pipeline's KC stage was off) |
| Evidence absence, closed-book (measured by E2) | In the other ~46% the generator does not know the answer | Only external evidence or provenance |

The taxonomy explains the result more precisely than "PoisonedRAG has no instruction signature". It also says what
a successful defense would need to change: rank or provenance, and evidence availability. Grounding and signature
detection do not touch these. Whether model consensus could is untested here. This matches the published landscape:
- TriShieldRAG (2026) uses ingest screening, trust-weighted re-ranking and cross-model consensus. It reports
  PoisonedRAG ≈91% → ≈13% against a non-adaptive attacker. Its consensus uses heterogeneous models (including an API
  model) and adds re-retrieval on disagreement.
- It loses most of that against an adaptive attacker (62% vs 56% undefended).
- Our result is consistent with this: our stack has neither trust weighting nor ingest screening for factual text.

## 4. Scoring caveats found while tracing (do not change the paper's numbers)

- **The external rule differs from Protocol B's rule.** It has no refutation cues. With Protocol B's
  `attack_succeeded`, the same rows give 85.3% undefended vs 82.0% defended (p = 0.33). The conclusion is
  unchanged.
- **Percent-valued targets never match.** Word-boundary matching fails for targets that end in a non-word character
  (e.g. "73%"). Row 112, "approximately 73%", is scored as a failure in both arms. The rule is applied identically
  to both arms, so the paired comparison is unaffected, but absolute ASR is a slight undercount.

## 5. Diagnostic experiments

### E2 — Closed-book knowledge probe (RUN 2026-11-25; see `experiments/RESULTS_E1_E2.md` §3)

**Result:**
- Among the 136 defended successes, `llama3.2:3b` knows the answer closed-book in **54.4% [46.0, 62.5]**.
- Other models (descriptive): mistral:7b 69.1%, qwen2.5:3b 34.6%.
- Defended ASR when llama knows vs not: 90.2% vs 91.2% (Fisher p = 1.0).
- Both mechanisms, prior override and evidence absence, are present at roughly equal rates.

The design below is the original plan. The preregistration (`experiments/PREREGISTRATION_E1_E2.md`) governs, and it
used temperature 0.2 (the pipeline's own call), not 0.

| Item | Detail |
|---|---|
| **Question** | Does the attack override knowledge the generator has (context dominance), or exploit questions it cannot answer (evidence absence)? |
| **Design** | Ask `llama3.2:3b` (and the two other ensemble models) the 150 PoisonedRAG questions with **no context**: temperature 0, seed 42, the generator's own plain QA prompt. Score whether the closed-book answer contains the gold answer and not the target. No pipeline code is touched. |
| **Primary metric** | Among the 136 defended successes: the fraction where the closed-book answer was correct (prior overridden). |
| **Reading** | A high fraction means the defense lets context override correct parametric knowledge, so a parametric-consistency signal could in principle help (the pipeline's disabled KC stage is that signal). A low fraction means the attack mostly hits questions the models cannot answer; only external evidence or provenance can help. |
| **Statistics** | Wilson intervals. A paired McNemar between closed-book correctness and defended success. |
| **Runtime** | 150 × 3 models × ~2 s ≈ 15 min. |
| **Limitations** | Closed-book correctness is itself noisy, and string matching undercounts paraphrases. |

### E2b (optional) — Counter-evidence perturbation

Add the gold passage for each question next to the poison, keeping everything else fixed, and rerun the defended
pipeline. This separates evidence absence from relevance dominance.

It needs the source passages from the NQ, HotpotQA and MS MARCO corpora. I did not confirm they are on disk;
`external_datasets_raw_source/` should be checked first. Runtime ≈ 150 × 25 s ≈ 1 h.

### E2c (optional) — Ensemble on PoisonedRAG

Rerun the 150 PoisonedRAG rows through `run_full_pipeline.py`. That is the 13-stage path with the three-model
ensemble and controller that Protocol A used, so no code change is needed. This would establish whether the
complete pipeline, *as described in the papers*, behaves like the adaptive variant. It is also the only way to
support or retract the "consistent across models" claim on external data.
- Runtime ≈ 150 × 25–30 s ≈ 1.2 h.
- It is a *rerun of the described configuration*, not a new defense.

**Recommendation.**
- E2 is cheap and directly answers question 5 of the brief.
- E2c fixes a description mismatch, and is worth it if the papers keep calling the external arm the "complete
  pipeline".
- E2b is optional and depends on data availability.
