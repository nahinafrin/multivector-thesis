#!/usr/bin/env python3
"""
count_mv_fire_rate.py
========================
Counts the multi-vector detector's fire rate on a grounded_*.jsonl output
from run_full_pipeline.py, restricted to kind=="multivector_attack" rows,
and reports a Wilson 95% CI (via stats_utils.wilson_ci, the same formula
used throughout this project's other addenda).

Reads state.meta["multivector"]["is_multivector"] per row -- only populated
when --defense-profile full (the default) was used for the run.

USAGE
------
    python count_mv_fire_rate.py --in grounded_mv130.jsonl
"""
from __future__ import annotations
import argparse
import json

from stats_utils import wilson_ci


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input", required=True)
    ap.add_argument("--kind", default="multivector_attack")
    args = ap.parse_args()

    total = 0
    fired = 0
    missing_field = 0
    for line in open(args.input, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("kind") != args.kind:
            continue
        total += 1
        mv = row.get("multivector")
        if not mv:
            missing_field += 1
            continue
        if mv.get("is_multivector"):
            fired += 1

    if total == 0:
        print(f"No rows with kind=={args.kind!r} found in {args.input}.")
        return

    lo, hi = wilson_ci(fired, total)
    print(f"kind={args.kind}  n={total}")
    print(f"fired: {fired}/{total} ({100*fired/total:.1f}%)")
    print(f"Wilson 95% CI: ({lo:.3f}, {hi:.3f})")
    if missing_field:
        print(f"\nWARNING: {missing_field}/{total} rows had no top-level `multivector` "
              f"field at all (defense_profile != 'full', or the row's pipeline "
              f"run short-circuited before Step 6) -- these are excluded from "
              f"the fire count above, not counted as non-fires. If this number "
              f"is large, re-check the run used the default/full defense profile.")


if __name__ == "__main__":
    main()
