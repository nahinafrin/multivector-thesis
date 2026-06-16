"""
demo_calibration_effect.py — show, numerically, that calibrating the near-binary
channels moves them into the graded sub-threshold band the detector needs.

This uses the REAL fit functions in calibration.py (pure numpy, no models), on a
synthetic near-binary score distribution that mimics the deberta scanner's
behaviour described in the methodology: benign ~0.0, attack ~0.95-0.999, almost
nothing in between. It then writes a calibration.json so you can immediately run

    python eval_multivector_grid.py --calibration calibration.json

and watch the saturated (0.0 / 1.0) corner of the grid spread into the firing
region. On REAL data you replace the synthetic scores with a fit-dataset run
(see the printed commands at the end); the math is identical.
"""

from __future__ import annotations

import numpy as np

from calibration import (
    fit, Calibrator, CalibrationSet,
)

rng = np.random.default_rng(0)

# Synthetic near-binary scanner output: this is the pathology, not a strawman.
# 400 benign prompts cluster near 0; 400 attacks cluster near 1, all confident.
benign = np.clip(rng.beta(1.2, 60, size=400), 0, 1)          # mostly < 0.05
attack = np.clip(1.0 - rng.beta(1.2, 60, size=400), 0, 1)    # mostly > 0.95
scores = np.concatenate([benign, attack])
labels = np.concatenate([np.zeros(400), np.ones(400)])

print("=== synthetic near-binary scanner distribution ===")
print(f"  benign : p50={np.percentile(benign,50):.3f}  p95={np.percentile(benign,95):.3f}")
print(f"  attack : p10={np.percentile(attack,10):.3f}  p50={np.percentile(attack,50):.3f}")
sat = float(np.mean((scores < 0.01) | (scores > 0.99)))
print(f"  saturated_frac (stuck within 0.01 of 0/1): {sat:.3f}")

for method in ("temperature", "platt"):
    cal, diag = fit(scores, labels, method=method)
    print(f"\n=== fit method = {method} ===")
    print(f"  params: {cal.to_dict()}")
    r, c = diag["raw"], diag["calibrated"]
    print(f"  saturated_frac:  raw={r['saturated_frac']:.3f} -> cal={c['saturated_frac']:.3f}")
    print(f"  class separation (attack_p10 - benign_p95): "
          f"raw={diag['separation_raw']:.3f} -> cal={diag['separation_calibrated']:.3f}")
    print("  transform of representative raw scores:")
    for v in (0.0, 0.05, 0.5, 0.9, 0.97, 0.999, 1.0):
        print(f"     raw {v:>6.3f} -> cal {cal.transform(v):.3f}")

# Save the temperature calibrator for BOTH channels (it de-saturates without
# needing labels per-channel) so the grid eval can use it immediately.
cal_temp, _ = fit(scores, labels, method="temperature")
cs = CalibrationSet(
    channels={"query_vector": cal_temp, "context_vector": cal_temp},
    meta={"note": "DEMO temperature calibrator fit on synthetic near-binary scores; "
                  "replace with fit-dataset on real data before reporting."},
)
cs.save("calibration.json")
print("\n[saved] calibration.json (query_vector + context_vector = temperature)")

print("""
--- next steps on REAL data (identical math, real scanner) ---
  # 1. fit query_vector from the labelled attack set (scores prompts itself):
  python calibration.py fit-dataset \\
      --dataset ../dataset/merged_output/dataset_all.clean.jsonl --limit 800 \\
      --method temperature --channel query_vector --cache qv_scores.jsonl

  # 2. fit context_vector from per-chunk scores you log during a slice run
  #    (point --score-field at the recorded per-chunk context_injection score):
  python calibration.py fit --scores slice_chunk_scores.jsonl \\
      --score-field context_injection --label-field is_poison \\
      --label-positive true --method temperature --channel context_vector

  # 3. re-derive the per-channel floors on the CALIBRATED scale, then re-run:
  python calibrate_payloads.py        # writes new DEFAULT_SOFT_* / joint_min
  python run_full_pipeline.py --slice adversarial_slice.jsonl --out grounded_cal.jsonl
  python score_slice.py grounded_cal.jsonl

Note: calibration.py will REFUSE to save (falls back to identity) if a channel is
so saturated that no monotone transform de-saturates it. That refusal is itself
the finding — report it as "this detector class cannot express a sub-threshold
band on channel X", which is exactly what the grid eval characterizes.
""")
