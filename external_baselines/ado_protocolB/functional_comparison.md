# ADO vs. seven-layer configuration, Protocol B (5 attacks + 5 benign, paired)

| Metric | ADO | Seven-layer stack | Undefended (ref.) |
|---|---|---|---|
| Attack success | 20.0% [3.6, 62.4] | 0.0% [0.0, 43.4] | 0.0% [0.0, 43.4] |
| Attacks prevented | 4 | 5 | 5 |
| Benign blocked | 0.0% [0.0, 43.4] | 0.0% [0.0, 43.4] | 0.0% [0.0, 43.4] |
| Benign correct | 40.0% [11.8, 76.9] | 60.0% [23.1, 88.2] | 60.0% [23.1, 88.2] |
| Benign answer rate | 80.0% [37.6, 96.4] | 100.0% [56.6, 100.0] | 100.0% [56.6, 100.0] |
| Median latency (s) | 45.5 | 14.7 | 8.3 |
| Mean latency (s) | 71.3 | 14.2 | 9.0 |

Attack success, ADO vs. seven-layer: rows succeeding only under ADO 1, only under the seven-layer stack 0; exact McNemar p = 1; difference +20.0 points, 95% CI [-15.1, 55.1].
Benign correct, ADO vs. seven-layer: only ADO 0, only seven-layer 1; p = 1; difference -20.0 points, 95% CI [-55.1, 15.1].

ADO mechanism on attack rows: {"poison_present_rows": 4, "poison_removed_by_ado_defenses": 1, "attack_success_when_removed": 1, "attack_success_when_kept": 0, "kept": 3, "defenses_executed_attack": {"differential_privacy": 3, "trustrag": 5, "attention_filtering": 1}, "defenses_executed_benign": {"differential_privacy": 3, "trustrag": 4, "attention_filtering": 1}, "av_requested_not_executed": 0, "generation_errors": 0, "sentinel_fallbacks": 0, "benign_empty_context": 1}
