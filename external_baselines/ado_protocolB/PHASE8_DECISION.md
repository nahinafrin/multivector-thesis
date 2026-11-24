# Phase 8: should ADO run on the full Protocol B set (200 attacks + 200 benign)?

**Status: not started. It needs the user's approval.**

## Observed throughput (pilot, 100 rows)

| Split | Rows | Summed row latency | Median / mean per row |
|---|---|---|---|
| Attack | 50 | 1.38 h | 45 s / 99 s |
| Benign | 50 | 1.51 h | 44 s / 109 s |
| Total | 100 | 2.89 h | (wall clock ≈ 3 h with model loading) |

Mean latency is driven by AV: about 40 % of rows request it, and AV on 6–8 real passages takes
~150–300 s.

## Cost of the full run

- The 100 pilot rows are a subset of the 400. The runner is resumable, but the pilot rows used the
  same code and configuration, so they can be **reused**, and only the **300 remaining rows** need to run.
- Estimate: 300 × ~104 s ≈ **8.7 h**, about **9–10 h of wall clock** including loading. This is one
  overnight run on the RTX 4060 (8 GB). The GPU is fully occupied, and Ollama models are unloaded
  before each AV call.
- Disk: a few MB of output. No new models are needed (4-bit AV model and Ollama models already present;
  ~8.8 GB free).
- Monetary cost: none (local).
- A new preregistered selection file is needed: the remaining 150 + 150 rows, to be written by an
  extension of `make_selection.py` without redrawing the pilot.

## What the full run would and would not change

- The direction is already clear and significant at n = 50:
  - ADO has higher attack success than the seven-layer configuration under both the primary and the S1 metric (p = 3.8×10⁻⁶ and p = 0.016);
  - ADO has lower benign correctness (p = 0.021).
- The full run would:
  - put ADO on the same n = 200 + 200 as the other existing-defense baselines (Spotlighting, LLM-Guard, ControlNET), so it can appear in the same table;
  - tighten the CIs (±14 points → ±7 points);
  - give a stable estimate of the refusal artifact.

## Recommendation

**Run it, if ADO is to appear in the main existing-defense table.** For consistency with the other
baselines, that table needs n = 200 + 200. Before launching:
1. **Preregister S1** (refusals are not attack successes) as a secondary metric, together with the
   manual adjudication rule, so that for the 300 new rows it is no longer post-hoc.
2. Keep every setting identical to the pilot: no tuning on pilot results. The pilot showed nothing
   that needs fixing in the adapter.

If ADO appears only as a pilot-scale comparison in the text, the pilot is sufficient. Report it as
n = 50 + 50, with its CIs and the caveats in `README.md`.

## Outcome (approved by the user and run 2026-11-23 20:06 – 2026-11-24 05:56)

- The 300 remaining rows ran in 9 h 50 min of wall clock, in line with the estimate. There were no
  generation or AV errors, and one Sentinel fallback (attack row 42, kept as recorded).
- The merged 200 + 200 set passed the alignment check, and scoring was byte-identical on two runs.
- Results are in `README.md` and `full_comparison.md`. ADO's attack success was 28.5% primary and 13.5%
  under S1, against 3.5% for the seven-layer configuration. Benign correct was 35.5% vs. 52.0%. All
  paired tests have p ≤ 2×10⁻⁴.
