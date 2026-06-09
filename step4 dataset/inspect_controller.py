"""
inspect_controller.py  —  what did the closed-loop controller actually do?
==========================================================================

The quick tally that separates "controller ran and decided X" from
"controller never ran (gate blocked first)". Use this BEFORE
``calibrate_thresholds.py`` whenever you want a one-glance answer to
"did the controller fire at all on this slice?".

It reports four things:

  1. How many rows the gate killed before the controller ever ran.
  2. The distribution of controller ``final_action`` over the rows that did
     reach it.
  3. How many extra 3-model ensemble generations the recover branch cost.
  4. A breakdown of (row kind x controller action) so you can see exactly
     which row types triggered which branches.

Usage
-----
    python inspect_controller.py grounded_controller.jsonl
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from pipeline_common import read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "jsonl",
        nargs="?", default="grounded_controller.jsonl",
        help="controller-on JSONL produced by run_full_pipeline.py",
    )
    args = ap.parse_args()

    rows = list(read_jsonl(args.jsonl))
    if not rows:
        print(f"no rows in {args.jsonl!r}", file=sys.stderr)
        sys.exit(2)

    actions: Counter[str] = Counter()
    by_kind: Counter[tuple[str, str]] = Counter()
    extra_gens = 0
    never_ran = 0

    for r in rows:
        kind = r.get("kind") or "(unknown)"
        ctrl = r.get("controller") or {}
        if not ctrl:
            # Gate blocked before the controller's segment ever ran.
            never_ran += 1
            by_kind[(kind, "gate_blocked_before_controller")] += 1
            continue
        action = ctrl.get("final_action") or "(no_action)"
        actions[action] += 1
        by_kind[(kind, action)] += 1
        # Each extra attempt past the first is one extra 3-model generation.
        extra_gens += max(0, int(ctrl.get("n_attempts", 1)) - 1)

    print(f"[inspect] {args.jsonl}  rows={len(rows)}")
    print(f"  controller never ran (gate-blocked): {never_ran}")
    print(f"  extra ensemble generations from recovery: {extra_gens}")

    print("\nfinal_action distribution (over rows that reached the controller):")
    if not actions:
        print("  (none — every row was gate-blocked before the controller ran)")
    else:
        for action, n in actions.most_common():
            print(f"  {action:<32s}  n={n}")

    print("\nbreakdown by (row kind x action):")
    for (kind, action), n in sorted(by_kind.items()):
        print(f"  {kind:<22s}  {action:<32s}  n={n}")


if __name__ == "__main__":
    main()
