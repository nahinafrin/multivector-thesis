# docs/

Working notes written during the project. **The paper (`../paper_revision/main.tex`) is the
authoritative statement of methods and results**, and the root `README.md` maps every paper result to
the script and frozen output that produce it. Each file below carries a status note at the top.

| File | What it is |
|---|---|
| `THESIS_OVERVIEW.md`, `thesis_overview.pdf` | Thesis overview and plan |
| `methodology.txt` | Earlier methodology draft |
| `methodology_working_multivector.md` | Proposal for the graded-margin multivector fix, written before it was tested out of sample |
| `LATEST_METHODOLOGY_REFERENCE.txt` | Implementation-level working reference from the revision |
| `RESULTS_multivector_methodology.md` | Results write-up from the original submission |
| `UNIFIED_RESULTS.md` | Generated intermediate report (Protocol A vs. B) |
| `RECONCILIATION_NOTE.md` | Note reconciling result narratives during the revision |
| `ROW_1_4_8_EXTERNAL_VALIDATION_RUNBOOK.md` | Runbook for the external-validation runs |

Status of the graded multivector detector, as reported in the paper (Sec. V-B): it fails on saturated
scores (0/34 valid fires); graded logit margins recover 24/30 valid fires in-sample, but give 6/200
false fires on held-out benign questions and 8/30 on BIPIA, and no context floor separates attack
contexts from benign ones. It is not a validated working detector.
