# research_improvement — novelty and generalization assessment (local only, not committed)

## Reports

1. `01_EXISTING_EVIDENCE_AUDIT.md`: each claim traced to its evidence, plus the problems found (F1–F8).
2. `02_NOVELTY_ASSESSMENT.md`: a formal attribution protocol, layer contributions and interactions, the prior-art
   comparison, and proposed C1–C4.
3. `03_GENERALIZATION_ASSESSMENT.md`: a coverage matrix, exploratory subgroup analysis, cross-dataset
   assessment, failure taxonomy, and optional experiment plans E1–E4.
4. `04_POISONEDRAG_FAILURE_ANALYSIS.md`: a stage-by-stage trace analysis, plus diagnostics E2, E2b and E2c.
5. `05_INDEPENDENT_REVIEW.md`: a skeptical review with a rating for each contribution.
6. `06_MANUSCRIPT_CHANGES.md`: proposed edits. Not applied; see `manuscript_edits.py` and `proposed_diffs/`.

## Analyses

All read saved outputs only. No model inference, no change to any saved output.

Run with the project environment (`step4 dataset/.venv311`):

```
python analysis/protocolB_attribution.py    # factorial, Shapley, interactions, leave-one-out, subset bounds, noise floor
python analysis/protocolB_subgroups.py      # EXPLORATORY: randomized templates, target, poison delivery
python analysis/poisonedrag_traces.py       # PoisonedRAG / BIPIA stage traces (paper's external scoring rule)
python analysis/verify_unchecked_claims.py  # numbers verify_claims.py does not cover; semantic-slice subset arms
```

- Outputs go to `analysis/outputs/*.json`.
- Bootstrap: 5,000 row resamples, seed 20261125.
- The Protocol B attack and benign scorers are the paper's own (`score_ablation_study.attack_succeeded`,
  `score_benign_cost.answered_correctly`). External rows use `score_external_paired.py`'s rule.

## Trial of the proposed edits

```
python trial_apply.py <scratch_dir> <path/to/tectonic.exe>   # scratch copies only; GPU venv (PyMuPDF)
```

It applies `manuscript_edits.py` to copies, compiles them, writes `proposed_diffs/*.diff`, and reports page
counts (`proposed_diffs/trial_compile.txt`).

## Experiments (new inference; preregistered)

The only part of this folder that runs models: `experiments/`. Nothing under `step4 dataset/` is written.

- `PREREGISTRATION_E1_E2.md`: the plan, fixed before inference. `launch_manifest.json` holds the SHA-256 of the
  inputs and git HEAD.
- `e1_build_variants.py`: builds the attack files `e1_attacks_NS.jsonl` and `e1_attacks_HO.jsonl`.
- `e2_closed_book.py`: the E2 closed-book run.
- `run_e1_e2.sh`: the runner. Outputs go to `E1_NS/`, `E1_HO/` and `e2_closed_book.jsonl`.
- `score_e1_e2.py`: preregistered scoring, written to `e1_e2_results.json`.
- `e1_e2_followup_exploratory.py`: post hoc breakdowns, descriptive only.
- **`RESULTS_E1_E2.md`: results and their reading.**
