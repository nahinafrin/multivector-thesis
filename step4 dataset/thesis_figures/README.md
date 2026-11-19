# Thesis figures

Run the generator from the repository root to refresh these SVG figures:

```powershell
.\.venv311\Scripts\python.exe "step4 dataset\generate_thesis_figures.py"
```

The current-code follow-up figures are derived from the scored JSONL arms under
`../mitigation_results/followup200/` and
`../mitigation_results/followup_nonyesno40/`. The PII figure uses the extraction
and incidental reports under `../row8_mitigation_results/`. Error bars show
Wilson 95% confidence intervals.
The baseline-drift figure compares the earlier July undefended arm with the
fresh current-code arm; it is a reproducibility check, not a paired treatment
comparison.

## Current-code follow-up figures

- `followup_attack_success_all.svg` — attack success for the fresh baseline and all 19 configurations
- `followup_benign_blocking_all.svg` — benign block rate by configuration
- `followup_benign_correct_all.svg` — benign answer correctness by configuration
- `followup_security_utility_tradeoff.svg` — attack success versus observed benign blocking
- `followup_leave_one_out_attack.svg` — full stack and leave-one-out attack success
- `followup_leave_one_out_transitions.svg` — paired attack transitions for the leave-one-out arms
- `followup_single_layer_attack.svg` — single-layer attack success versus baseline
- `followup_combinations_attack.svg` — four combinations versus the full stack
- `followup_all_paired_attack_transitions.svg` — paired attack transitions for all configurations
- `followup_benign_block_transitions.svg` — paired benign-block transitions for all configurations
- `followup_combinations_vs_components.svg` — paired outcomes for combinations versus constituent layers
- `followup_baseline_drift.svg` — July versus current undefended attack and benign results
- `followup_nonyesno_attack_success.svg` — attack success on the 40-row free-text slice
- `followup_nonyesno_blocking.svg` — blocked rows on the 40-row free-text slice
- `followup_nonyesno_transitions.svg` — paired attack transitions versus its undefended baseline
- `followup_nonyesno_vs_full.svg` — paired attack transitions versus the full stack
- `followup_pii_leak_rates.svg` — extraction and incidental PII leakage rates
- `followup_confusion_matrix_full.svg` — full-stack attack-blocking versus benign-allowing outcomes
- `followup_confusion_matrices_all.svg` — the same operational confusion matrix for the fresh baseline and every scored configuration

The six earlier figures in this folder are retained as reference material.
They use older evaluation protocols, scorer comparisons, corpus data, or
architecture documentation; do not interpret them as results from the
current-code follow-up.

The repository does not contain epoch-level training measurements or
epoch-level training measurements for these latest evaluations, so training
curves are not inferred or fabricated. The confusion matrices are operational
outcome matrices: the true row is the test case type (attack or benign), and
the predicted column is whether the system blocked or allowed it. They measure
attack neutralization and benign false blocks, not a standalone classifier's
class predictions.
