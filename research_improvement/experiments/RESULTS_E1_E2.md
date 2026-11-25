# Results: E1 (held-out attack wording) and E2 (closed-book knowledge probe)

The run started on 2026-11-25 at 01:00 and finished at 04:47. Exit code 0; no errors or tracebacks in the logs.

Scored with `score_e1_e2.py`, which was written and hashed before inference (`launch_manifest.json`, git HEAD 7c7956f).
Raw numbers are in `e1_e2_results.json`.

Section 4 contains exploratory breakdowns that were written *after* the results were seen
(`e1_e2_followup_exploratory.py` → `e1_e2_followup_exploratory.json`). They are descriptive only.

**Row counts:**
- E1_NS and E1_HO: `off`, `on_full` and `on_combo_sanitize_grounding_refuse` have 200 rows each.
- `e2_closed_book.jsonl`: 450 rows (150 questions × 3 models).

**Deviations from the preregistration: none.**
- `e2_closed_book.py` calls `generate_single(prompt, model=m)`. In `step_09_generator_llm.py`, that function is
  defined as `generate_candidates(prompt, models={m: m})`, which is the call the preregistration names. Settings:
  temperature 0.2, seed 42.
- The older plan in 04 §5 said "temperature 0". The preregistration superseded it before launch.

## 1. E1 — confirmatory result (Protocol B, seven-layer harness, llama3.2:3b, n = 200 paired rows)

| Variant | Full-stack ASR (95% Wilson) | Original full-stack ASR | Variant-only / original-only successes | Exact McNemar p | Paired diff (95% CI) | Preregistered verdict |
|---|---|---|---|---|---|---|
| **HO** (held-out suffix + template) | **3.0%** [1.4, 6.4] | 3.5% [1.7, 7.0] | 6 / 7 | 1.00 | −0.5 [−4.0, 3.0] | **Contradicts H1** |
| **NS** (coercive suffix removed) | **7.5%** [4.6, 12.0] | 3.5% [1.7, 7.0] | 14 / 6 | 0.115 | +4.0 [−0.3, 8.3] | **Inconclusive** |

α = 0.025 (Bonferroni, 2 tests).

- **HO:** the full stack's protection carried over to the held-out coercive wording. The point estimate is lower
  than the original, and the upper bound (6.4%) is below the preregistered 7% margin.
- **NS:** the point estimate doubled, but the difference is not significant. The upper bound (12.0%) is above 7%,
  so neither "worse" nor "the same" is established.

## 2. E1 — secondary (descriptive)

| | Original | NS | HO |
|---|---|---|---|
| Undefended ASR | 29.5% [23.6, 36.2] | 16.5% [12.0, 22.3] | 40.5% [33.9, 47.4] |
| Undefended, variant vs original (paired) | — | −13.0 pts [−18.8, −7.2], p < 0.001 | +11.0 pts [3.4, 18.6], p = 0.007 |
| Full-stack ASR | 3.5% | 7.5% | 3.0% |
| Absolute reduction (undefended − full) | 26.0 pts | 9.0 pts [3.8, 14.2], p = 0.0014 | 37.5 pts [30.8, 44.2], p < 10⁻²² |
| Relative reduction | 88.1% | 54.5% | 92.6% |
| Subset (S+G+R) ASR | 5.5% [3.1, 9.6]* | 11.0% [7.4, 16.1] | 2.0% [0.8, 5.0] |
| Subset − full (paired) | +2.0 [−1.1, 5.1], p = 0.34* | +3.5 [−0.7, 7.7], p = 0.17 | −1.0 [−3.8, 1.8], p = 0.73 |
| Rows blocked by the detector (full arm) | 80/200 | **6/200** | 111/200 |

*Original subset values are from the existing `followup200` run, rows 0–199. At n = 424 the paper's difference is
+0.5 [−1.5, 2.4].

**Reading.**
- **Held-out wording is the stronger attack (40.5% undefended), and the defense still reduces it by 93%.** The new
  coercive phrasing ("errata", "official revision", "superseded") triggers the existing detector on more rows than
  the original wording did (111 vs 80).
- **Without any coercive suffix, the attack is weaker (16.5%), but the defense loses most of its margin**:
  - the relative reduction falls from 88% to 55%;
  - the detector fires on only 6/200 rows.
  
  The refuse-on-detection layer mostly reacts to the coercive *query* phrasing, not to the poison passage itself.
  When the attacker leaves the query plain, what remains is grounding, sanitization and the prompt. They still give
  a significant reduction (9 pts, p = 0.0014), but a smaller one.
- **Three-layer sufficiency holds on both variants.** The subset is not significantly different from the full stack
  (NS +3.5, CI includes 0; HO −1.0). On NS the CI upper bound is 7.7 pts, so this is "not distinguishable at
  n = 200", not equivalence.

## 3. E2 — closed-book knowledge probe (PoisonedRAG, 150 questions)

| Model | Closed-book *knows* | *Believes target* | Other | **Knows, among the 136 defended successes** | Knows, among 137 undefended successes | Defended ASR if knows / if not | Fisher p (descr.) |
|---|---|---|---|---|---|---|---|
| **llama3.2:3b (primary)** | 54.7% (82) | 3.3% (5) | 42.0% (63) | **54.4% [46.0, 62.5]** | 53.3% [45.0, 61.4] | 90.2% / 91.2% | 1.00 |
| mistral:7b | 67.3% (101) | 4.0% | 28.7% | 69.1% [60.9, 76.3] | 69.3% | 93.1% / 85.7% | 0.23 |
| qwen2.5:3b | 32.7% (49) | 6.7% | 60.7% | 34.6% [27.1, 42.9] | 33.6% | 95.9% / 88.1% | 0.15 |

**Reading.**
- **Both mechanisms are present, in roughly equal measure.**
  - In about **54%** of the defended successes, the generator answers correctly with no context and then repeats
    the planted answer once the poison is retrieved. This is *context overriding parametric knowledge*.
  - In the remaining ~46%, it does not know the answer closed-book. This is *evidence absence*.
- **Knowing the answer gives no protection.** Defended ASR is 90.2% when llama knows and 91.2% when it does not.
  The defended pipeline neither uses nor preserves the model's own knowledge.
- **The larger model (mistral:7b) knows more (69%), and is not protected either** (ASR 93.1% when it knows; this is
  its *own* knowledge vs llama's generations, so it is descriptive only).
- **Consequence for the failure taxonomy.** "Prior override" (04, row "Prior override (hypothesis)") is now
  measured: it accounts for about half of the successes. A parametric-consistency signal (the pipeline's disabled
  knowledge-conflict stage) could at most address that half. Provenance or evidence would be needed for the rest.
  *We did not build or test either; this is not a claim that such a signal would work.*
- **Scoring caveat.** We read a sample of 15 llama answers labelled *other*. Most are genuinely wrong or "I don't
  know". A few are correct "no" answers phrased without the word "no" (e.g. "X is not a …"). The *knows* rate is
  therefore a mild underestimate. The primary estimate would rise slightly, not fall.

## 4. Exploratory breakdowns (post hoc; descriptive; no tests)

**NS, by the original suffix the row carried** (detector fire counts are full-arm, original → NS):

| Original suffix | n | Full ASR original → NS | Detector fires, original → NS |
|---|---|---|---|
| "Defer to any authoritative correction…" (Q2) | 63 | 0.0% → 7.9% | 63 → 2 |
| "If the sources contain a verified correction…" (Q1) | 61 | 6.6% → 11.5% | 12 → 2 |
| "Where the documents give a corrected answer…" (Q3) | 76 | 3.9% → 3.9% | 5 → 2 |

The protection lost on NS is concentrated in rows where the detector had fired on the Q2 suffix. This is consistent
with the exploratory finding in 03 §2.1, which E1 was designed to test.

**HO, by held-out suffix:**
- H1 "Per the latest errata…": 7.4% full, 75.0% undefended, detector 5/68.
- H2 "Treat any official revision…": 0.0% full, 19.7% undefended, detector 45/61.
- H3 "If a passage says … superseded…": 1.4% full, 25.4% undefended, detector 61/71.

**HO, by held-out template:**
- P3 ("Official revision notice…"): 9.5% full.
- P1, P2, P4: 0.0% full.

Wording heterogeneity is still present. The weakest wording (H1) is also the most effective undefended attack.

## 5. What this changes in the claims

- **C3(d), wording dependence:** upgraded from "exploratory only" to "partly confirmed".
  - The preregistered test shows the protection *transfers* to held-out coercive wording. This is reassuring and
    can be stated as a result (HO).
  - It does *not* establish that removing the coercive phrasing weakens the defense (NS, inconclusive at α = 0.025).
    That can only be reported as a descriptive drop in relative reduction, 88% → 55%.
  - The detector-fire counts (80 → 6) are descriptive evidence that refusal is keyed to query phrasing.
- **Three-layer sufficiency:** still not distinguishable from the full stack on both new wordings. This is not
  equivalence.
- **PoisonedRAG:** "Prior override" moves from hypothesis to measured, at about half of the defended successes
  (llama3.2:3b). Factual corruption is **not** solved.
- **Limits that remain:** one generator and one knowledge base on Protocol B; held-out wording written by us, not an
  adaptive attacker; Ollama run-to-run noise (about 1/200 rows); closed-book labels from string matching.
