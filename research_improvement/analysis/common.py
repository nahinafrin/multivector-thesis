"""Shared helpers for the research_improvement analyses. Read-only use of saved outputs."""
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S4 = ROOT / "step4 dataset"
MR = S4 / "mitigation_results"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(S4))
from score_ablation_study import attack_succeeded, exact_p  # noqa: E402  (the paper's own scorer)
from score_benign_cost import answered_correctly  # noqa: E402

SEED = 20261125
B = 5000


def load(p):
    """Rows keyed by integer index (some files store the index as a string)."""
    return {int(r["index"]): r for r in (json.loads(l) for l in open(p, encoding="utf-8") if l.strip())}


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * p, 1), round(100 * max(0, c - h), 1), round(100 * min(1, c + h), 1)]


def paired(x, y, ids):
    """x, y: dict id->bool (event, e.g. attack success). Returns discordants, exact McNemar p,
    difference (x - y) in points with a Wald 95% CI (as in the paper)."""
    n = len(ids)
    b = sum(x[i] and not y[i] for i in ids)
    c = sum(y[i] and not x[i] for i in ids)
    d = (b - c) / n
    se = math.sqrt(max((b + c) / n - d * d, 0.0) / n)
    return {"n": n, "only_x": b, "only_y": c, "p_exact": exact_p(b, c), "diff_pts": round(100 * d, 1),
            "ci95": [round(100 * (d - 1.96 * se), 1), round(100 * (d + 1.96 * se), 1)],
            "ci90": [round(100 * (d - 1.645 * se), 1), round(100 * (d + 1.645 * se), 1)]}


def bootstrap(stat, ids, b=B, seed=SEED):
    """Percentile CI of stat(sample_ids) under row resampling (rows are the paired unit)."""
    rng = random.Random(seed)
    ids = list(ids)
    vals = sorted(stat([ids[rng.randrange(len(ids))] for _ in ids]) for _ in range(b))
    return [round(vals[int(0.025 * b)], 1), round(vals[int(0.975 * b) - 1], 1)]


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT / name)
