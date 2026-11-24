# ADO vs. seven-layer configuration, Protocol B (200 attacks + 200 benign, paired)

| Metric | ADO | Seven-layer stack | Undefended (ref.) |
|---|---|---|---|
| Attack success | 28.5% [22.7, 35.1] | 3.5% [1.7, 7.0] | 29.5% [23.6, 36.2] |
| Attacks prevented | 143 | 193 | 141 |
| Benign blocked | 0.0% [0.0, 1.9] | 8.5% [5.4, 13.2] | 0.0% [0.0, 1.9] |
| Benign correct | 35.5% [29.2, 42.3] | 52.0% [45.1, 58.8] | 56.5% [49.6, 63.2] |
| Benign answer rate | 83.5% [77.7, 88.0] | 91.5% [86.8, 94.6] | 97.5% [94.3, 98.9] |
| Median latency (s) | 45.4 | 15.3 | 8.6 |
| Mean latency (s) | 114.2 | 13.7 | 8.9 |

Attack success, ADO vs. seven-layer: rows succeeding only under ADO 54, only under the seven-layer stack 4; exact McNemar p = 3.17e-12; difference +25.0 points, 95% CI [18.4, 31.6].
Benign correct, ADO vs. seven-layer: only ADO 10, only seven-layer 43; p = 5.55e-06; difference -16.5 points, 95% CI [-23.3, -9.7].

ADO mechanism on attack rows: {"poison_present_rows": 184, "poison_removed_by_ado_defenses": 107, "attack_success_when_removed": 37, "attack_success_when_kept": 17, "kept": 77, "defenses_executed_attack": {"differential_privacy": 140, "trustrag": 154, "attention_filtering": 84}, "defenses_executed_benign": {"differential_privacy": 118, "trustrag": 146, "attention_filtering": 84}, "av_requested_not_executed": 0, "generation_errors": 0, "sentinel_fallbacks": 1, "benign_empty_context": 58}
