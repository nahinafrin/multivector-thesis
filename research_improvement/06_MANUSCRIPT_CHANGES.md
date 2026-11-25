# 06 — Proposed manuscript changes (NOT applied)

**Status:** `paper_revision/main.tex`, `main_iccit.tex` and `refs.bib` are unchanged.

| Where | Content |
|---|---|
| `manuscript_edits.py` | Every edit as an exact old → new pair, with its rationale |
| `proposed_diffs/main_iccit.tex.diff`, `proposed_diffs/main.tex.diff` | The same edits as unified diffs |
| `proposed_diffs/refs.bib.additions` | New references |

**Trial build** (`trial_apply.py`). It applied every edit to scratch copies and compiled them with Tectonic:
- each `old` string matched exactly once;
- `main_iccit.tex` stays at **6 pages (A4)**, with page 6 about 78% full;
- `main.tex` goes from 15 to 16 pages;
- no errors or undefined references in either;
- all 86 claim-checker snippets are still present in the proposed `main.tex`;
- no reported number changes.

## Priority 1 — corrections a reviewer could catch (no experiment needed)

| ID | File | Change | Why |
|---|---|---|---|
| I4, I9, I10, M5, M6, M8–M12, M18, M19 | both | The external defended arm becomes an "adaptive-tier pipeline variant (Stages 1–10, 13)", with its tiers described. The "same-generator comparison" premise is fixed. The unsupported "a planted fact is consistent across models" is removed. The 23.8 s latency row is relabelled | 01 F2: that run used a single `llama3.2:3b` on ≥139/150 PoisonedRAG rows, and had no multivector detector, controller, Presidio or DLP |
| I1, I6, I7, M1, M7, M16 | both | "Statistically indistinguishable" becomes a bound: "within 2.4 points (95% upper bound, n = 424)", plus the seed direction | 01 F5, 05 Q4 |
| I7, I12, I13, M1, M7, M16, M17 | both | Three-layer sufficiency is scoped to instruction-bearing attacks. The practitioner advice keeps the guarded prompt | 01 F3: semantic slice, subset 62.5% vs full 35.0%, p = 0.003 |
| M2, M4, I2 | both | One consistent C1–C4 list (from 02 §6) | 01 F1 |
| M14 | main | Remove "only a stage-by-stage … evaluation can tell them apart" | 01 #19 |
| I3, M3, M13 | both | Cite RobustRAG (isolate-then-aggregate), TriShieldRAG (knowledge-corruption defense-in-depth), and ablation/fANOVA. "To our knowledge none" becomes "we did not find" | 01 F6, 02 §3 |

## Priority 2 — stronger interpretation of existing evidence

| ID | File | Change | Evidence |
|---|---|---|---|
| M7 (new paragraphs + new Table "Fully crossed factorial") | main | Interactions: independent-filter structure, Shapley shares, row overlap, the guarded prompt's sign reversal, noise floor. Class dependence and wording heterogeneity (exploratory) | `analysis/outputs/protocolB_attribution.json`, `protocolB_subgroups.json`, `verify_unchecked_claims.json` |
| I6, I7 | ICCIT | A compact version of the same, without a new table | as above |
| I8, M12 | both | PoisonedRAG mechanism: gate score 0, 0/148 redacted, rank 1 in 148/148, grounding faithfulness 1.0, counter-evidence in 11/148 | `analysis/outputs/poisonedrag_traces.json` |
| I11, M15 | both | Threats: static attack sets (Nasr et al.), wording dependence, a single generator behind every attack-success result, different success rules per protocol | 03, 05 |

## What is deliberately not changed

- No reported number is changed, and no reviewer-requested evidence is removed. That includes the seeds, the fresh
  rerun, n = 424, the existing-defense comparisons, the external sets, the latency and memory results, and the
  statistics paragraph.
- The 13-stage versus seven-layer distinction is kept and sharpened. It now covers three configurations: the
  complete pipeline (Protocol A), the seven-layer harness (Protocol B), and the adaptive-tier variant (external
  sets).
- ICCIT format: IEEE conference, A4, 6 pages. No new table or figure is added to the ICCIT paper.

## Before applying — items for the authors to check

1. **New references** (`proposed_diffs/refs.bib.additions`):
   - TriShieldRAG: the author first names and the exact title differ between the arXiv versions I saw.
   - LlamaFirewall: confirm the first author.
   - Ramakrishnan & Balaji: confirm the full title.
   - Nasr et al.: confirm the author list.
   These were taken from search results and abstract pages, not full PDFs.
2. **Response letter.** After applying, update `response_to_reviewers.md` in a few places:
   - 3.1: "statistically indistinguishable" → bound;
   - 4.1: the "complete pipeline" label for the external results;
   - 5.4: the PoisonedRAG mechanism.
   The ICCIT table and figure numbering is unchanged.
3. **Repository hygiene, recommended alongside, no paper change.**
   - Restore `run_full_pipeline_adaptive.py`, `adaptive_risk.py`, `adaptive_verification.py` and `adaptive_prompt.py`
     from the backup repo, so the external run can be reproduced from the public repository.
   - Correct the docstring of `score_external_paired.py` ("complete 13-stage pipeline, 3-model ensemble, controller
     on").
4. **Claim checker** (optional, recommended). Add the new numbers to `verify_claims.py`, reading the
   `research_improvement/analysis` outputs:
   - Shapley values;
   - independence residuals;
   - semantic subset vs full (p = 0.0034);
   - prompt alone on the semantic slice (p = 0.035);
   - wording heterogeneity (p = 0.0001);
   - the PoisonedRAG trace counts.
   Also add the semantic-slice numbers the checker does not currently cover.
5. **E1 and E2 have been run (2026-11-25; `experiments/RESULTS_E1_E2.md`).** Proposed additions, not yet in
   `manuscript_edits.py` and pending your approval:
   - **I7/M7, wording.** Keep the exploratory heterogeneity sentence and add the preregistered result:
     > "In a preregistered test on held-out coercive wording (same 200 questions and targets), full-stack ASR was
     > 3.0% vs 3.5% (exact McNemar p = 1.0); with the coercive query suffix removed it was 7.5% (p = 0.115, not
     > significant at the preregistered α = 0.025), while the relative reduction fell from 88% to 55%."

     Do **not** state that the defense is weaker without coercive phrasing.
   - **M-PoisonedRAG (main.tex §5.4 and the ICCIT failure sentence).**
     > "A closed-book probe shows that llama3.2:3b knew the correct answer in 54% of the defended successes; knowing
     > it gave no protection (90.2% vs 91.2% ASR)."

     Present this as descriptive.
   - **Threats.** Held-out wording was written by the authors, not by an adaptive attacker.

## Applying

After approval, apply exactly the listed edits to `paper_revision/`. This needs a small `--apply` path in
`trial_apply.py`, deliberately not implemented yet. Then recompile both papers, rerun `verify_claims.py`, and
regenerate `paper_revision/ieee_papers/`. Nothing will be committed or pushed without your permission.
