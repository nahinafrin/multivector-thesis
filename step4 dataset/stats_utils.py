"""stats_utils.py - one shared place for Wilson CIs and bootstrap CIs.

Every script currently computing its own confidence interval
(score_mitigation_ab.py, grounding_separation_probe.py, calibrate_thresholds.py)
should import from here instead of re-deriving the formula, so a reviewer only
has to check the interval math once.
"""
from __future__ import annotations

import math
import random


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. z=1.96 -> 95% CI.

    Preferred over the naive normal-approximation CI for small n or proportions
    near 0/1 (exactly the regime several of this project's slices are in).
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z ** 2 / n
    center = p + z ** 2 / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n)
    lo = (center - margin) / denom
    hi = (center + margin) / denom
    return (max(0.0, lo), min(1.0, hi))


def bootstrap_ci(values: list[float], stat_fn=None, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 42) -> tuple[float, float, float]:
    """Percentile bootstrap CI for an arbitrary statistic (default: mean).

    Use this for the AUC / faithfulness-gap numbers in
    grounding_separation_probe.py at n=8 - a Wilson interval doesn't apply to
    a continuous statistic like AUC, but a bootstrap CI does, and at n=8 it
    will (correctly) come back very wide, which is the honest picture.
    """
    if stat_fn is None:
        stat_fn = lambda xs: sum(xs) / len(xs)
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0)
    point = stat_fn(values)
    boots = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(stat_fn(sample))
    boots.sort()
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = int((1 - alpha / 2) * n_boot) - 1
    return (point, boots[max(0, lo_idx)], boots[min(n_boot - 1, hi_idx)])


def mcnemar_exact_p(b: int, c: int) -> float:
    """Exact McNemar test p-value for paired discordant counts b, c.

    Reimplemented here (matches the values already appearing in
    mitigation_summary.json) so every A/B report calls the SAME function.
    """
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) * (0.5 ** n) for i in range(0, k + 1)) * 2
    return min(1.0, p)
