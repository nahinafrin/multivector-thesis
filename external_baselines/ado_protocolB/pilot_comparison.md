# ADO vs. seven-layer configuration, Protocol B (50 attacks + 50 benign, paired)

| Metric | ADO | Seven-layer stack | Undefended (ref.) |
|---|---|---|---|
| Attack success | 42.0% [29.4, 55.8] | 4.0% [1.1, 13.5] | 32.0% [20.8, 45.8] |
| Attacks prevented | 29 | 48 | 34 |
| Benign blocked | 0.0% [0.0, 7.1] | 8.0% [3.2, 18.8] | 0.0% [0.0, 7.1] |
| Benign correct | 44.0% [31.2, 57.7] | 60.0% [46.2, 72.4] | 62.0% [48.2, 74.1] |
| Benign answer rate | 84.0% [71.5, 91.7] | 92.0% [81.2, 96.8] | 96.0% [86.5, 98.9] |
| Median latency (s) | 45.0 | 15.2 | 8.5 |
| Mean latency (s) | 104.0 | 13.3 | 8.8 |

Attack success, ADO vs. seven-layer: rows succeeding only under ADO 19, only under the seven-layer stack 0; exact McNemar p = 3.81e-06; difference +38.0 points, 95% CI [24.5, 51.5].
Benign correct, ADO vs. seven-layer: only ADO 1, only seven-layer 9; p = 0.0215; difference -16.0 points, 95% CI [-27.6, -4.4].

ADO mechanism on attack rows: {"poison_present_rows": 46, "poison_removed_by_ado_defenses": 28, "attack_success_when_removed": 15, "attack_success_when_kept": 4, "kept": 18, "defenses_executed_attack": {"differential_privacy": 39, "trustrag": 35, "attention_filtering": 20}, "defenses_executed_benign": {"differential_privacy": 29, "trustrag": 40, "attention_filtering": 19}, "av_requested_not_executed": 0, "generation_errors": 0, "sentinel_fallbacks": 0, "benign_empty_context": 14}
