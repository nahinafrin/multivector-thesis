#!/usr/bin/env python3
"""
This file bundles TWO small modules meant to live as SEPARATE files in the repo
— split them when you copy this in:

  step4 dataset/stats_utils.py          (the WilsonCI / bootstrap helpers)
  step4 dataset/scale_and_merge_slice.py (the driver that scales small slices)

WHY THESE EXIST
----------------
Several headline numbers in the project rest on very small n: the semantic
multi-vector slice is n=8, the first mitigation A/B is n=30, and the
methodology doc itself says results at these sizes are "directional, not
final" and that "expanding the slice would tighten the numbers." Nobody has
actually done that yet, and where CIs ARE reported (mitigation_summary.json)
they're computed ad hoc per script rather than through one shared, tested
utility — so there's a real risk of a Wilson CI being computed slightly
differently in two different scripts. stats_utils.py is the single place that
should happen from now on; scale_and_merge_slice.py is the driver that
actually grows n=8 -> a configurable target and re-merges with the existing
slice, deduplicated.
"""
from __future__ import annotations

# =========================================================================== #
# FILE 1/2 — step4 dataset/stats_utils.py
# =========================================================================== #
STATS_UTILS_PY = r'''
"""stats_utils.py — one shared place for Wilson CIs and bootstrap CIs.

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
    grounding_separation_probe.py at n=8 — a Wilson interval doesn't apply to
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
'''

# =========================================================================== #
# FILE 2/2 — step4 dataset/scale_and_merge_slice.py
# =========================================================================== #
SCALE_SLICE_PY = r'''
#!/usr/bin/env python3
"""scale_and_merge_slice.py — grow a small evaluation slice and merge it with
the existing one, deduplicated by prompt text.

Wraps whichever slice builder you already have (build_semantic_slice.py for
the n=8 misinformation slice, build_planted_attacks.py for the n=30 mitigation
slice) rather than reimplementing construction logic — this script's only job
is to call the builder at a larger --n, merge with what you already scored,
and dedup so you don't have to throw away the existing n=8/n=30 results.

USAGE
------
    # grow the semantic (misinformation) slice from 8 to 40 rows:
    python scale_and_merge_slice.py \
        --builder build_semantic_slice.py --builder-args "--out semantic_slice_v2.jsonl --n 40" \
        --existing semantic_slice.jsonl \
        --out semantic_slice_merged.jsonl

    # grow the planted-attack mitigation slice from 30 to 150:
    python scale_and_merge_slice.py \
        --builder build_planted_attacks.py \
        --builder-args "--qa-jsonl data/question-answer/test.jsonl --n 150 --yesno-only --out planted_attacks_v3.jsonl" \
        --existing planted_attacks_v2.jsonl \
        --out planted_attacks_merged.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path


def read_jsonl(path: str) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def dedup_key(row: dict) -> str:
    text = row.get("prompt") or row.get("question") or json.dumps(row, sort_keys=True)
    return re.sub(r"\s+", " ", str(text).strip().lower())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--builder", required=True, help="e.g. build_semantic_slice.py")
    ap.add_argument("--builder-args", required=True, help="quoted CLI args for the builder")
    ap.add_argument("--existing", required=True, help="the current, already-scored slice")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[run] python {args.builder} {args.builder_args}")
    result = subprocess.run([sys.executable, args.builder, *shlex.split(args.builder_args)])
    if result.returncode != 0:
        print("[error] builder failed — fix it before merging.", file=sys.stderr)
        sys.exit(result.returncode)

    # The builder's --out path is embedded in builder-args; extract it so we
    # know what to read back without re-parsing every builder's own argparse.
    new_out = None
    parts = shlex.split(args.builder_args)
    if "--out" in parts:
        new_out = parts[parts.index("--out") + 1]
    if not new_out:
        print("[error] could not find --out in --builder-args; pass it explicitly.", file=sys.stderr)
        sys.exit(1)

    existing_rows = read_jsonl(args.existing)
    new_rows = read_jsonl(new_out)
    seen = {dedup_key(r) for r in existing_rows}

    merged = list(existing_rows)
    added = 0
    for r in new_rows:
        k = dedup_key(r)
        if k in seen:
            continue
        seen.add(k)
        merged.append(r)
        added += 1

    with open(args.out, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[merge] existing={len(existing_rows)} new_from_builder={len(new_rows)} "
          f"added={added} total={len(merged)} -> {args.out}")
    print("\nNEXT STEP: re-run run_full_pipeline.py + score_attack_success.py + "
          "grounding_separation_probe.py on --out, and report the CIs via "
          "stats_utils.wilson_ci / bootstrap_ci instead of a bare point estimate.")


if __name__ == "__main__":
    main()
'''

if __name__ == "__main__":
    from pathlib import Path
    Path("stats_utils.py").write_text(STATS_UTILS_PY.strip() + "\n", encoding="utf-8")
    Path("scale_and_merge_slice.py").write_text(SCALE_SLICE_PY.strip() + "\n", encoding="utf-8")
    print("Wrote stats_utils.py and scale_and_merge_slice.py to the current directory.")
