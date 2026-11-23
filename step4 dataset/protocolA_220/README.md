# Protocol A, 220-row runs (paper Sec. V-A to V-D, Tables IV and X)

| File | What it is | Restored from |
|---|---|---|
| `adversarial_slice_220.jsonl` | The 220-row slice: 80 benign, 40 poisoned-context, 40 direct, 40 multivector, 20 gate-slip | git `27bf190` |
| `grounded_controller_220.jsonl` | Complete pipeline, controller enabled | git `27bf190` |
| `grounded_baseline_220.jsonl` | Complete pipeline, controller disabled | git `27bf190` |
| `protocolA_220_report.json` | Output of `../score_protocolA_220.py` | generated |

These are the runs behind every Protocol A number in the paper (0/220 canary leaks, 74/74 and
40/40 redactions, 0/80 benign gate flags, 29/40 and 30/40 direct injections flagged at the gate,
40/40 blocked by the cascade, 5/20 gate-slip rows passing the gate, 1 fire on 34 post-gate
multivector rows, and the controller-vs-baseline counts 88/101 blocked, 70/66 benign and 34/28
poisoned-context rows delivered).

The same file names in `step4 dataset/` were later overwritten by a rebuilt slice: 210 rows
(30 multivector) plus 12 `borderline_adaptive` rows. That rebuilt slice is what the graded-margin
experiment of Sec. V-B uses (`grounded_controller_graded.jsonl`); it is not the source of the
Protocol A headline numbers.

    python score_protocolA_220.py
