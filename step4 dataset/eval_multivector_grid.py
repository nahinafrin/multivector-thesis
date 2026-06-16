"""
eval_multivector_grid.py — controlled-injection evaluation of the multi-vector
detector, isolated from the (near-binary) underlying scanners.

WHY THIS EXISTS
---------------
On the constructed adversarial slice the multi-vector detector fires on ~0 of
its target rows. Inspecting the rows shows WHY: query_vector sits ~0.0-0.12 and
context_vector saturates at 1.0, so only ONE channel is ever active and the
co-activation requirement (min_channels=2) is never met. That 0/40 therefore
conflates two very different things:

    (a) "the detector logic is broken"                         vs
    (b) "the conjunctive sub-threshold REGIME cannot be
         instantiated against a near-binary scanner"

You cannot tell (a) from (b) by looking at slice results, because the slice can
only contain channel values the scanner is willing to emit. This script removes
the scanner from the loop: it drives multivector_risk() directly over a GRID of
channel values and reports the detector's verdict at each cell. If the detector
fires correctly across the genuine sub-threshold band, then the mechanism is
sound and the 0/40 is explained by (b) — a characterized property of the
detector class, not a bug. That is a publishable negative result rather than an
unexplained zero.

It also doubles as a regression TEST of the three firing conditions:
    1. per-channel floor   2. co-activation (>= min_channels)   3. joint_min.

No models, no network, no GPU — pure arithmetic over the real multivector.py, so
it runs instantly and anywhere.

USAGE
-----
    python eval_multivector_grid.py
    python eval_multivector_grid.py --calibration calibration.json   # show the
                                                                      # de-saturated grid
    python eval_multivector_grid.py --json grid.json                 # machine-readable
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

from multivector import (
    input_channels,
    multivector_risk,
    DEFAULT_SOFT_PER_CHANNEL,
    DEFAULT_SOFT,
    DEFAULT_MIN_CHANNELS,
    DEFAULT_JOINT_MIN,
    DEFAULT_JOINT_BLOCK,
)

# The grid we sweep each channel over. Deliberately straddles the per-channel
# floors (0.15 / 0.20) and the hard-block joint level so every regime is covered:
# below-floor noise, the sub-threshold band, and saturated single-vector spikes.
AXIS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90, 1.0]

# "Sub-threshold" means a per-channel value that, on its own, a single-channel
# rail would NOT block. We treat anything below this as the band the detector
# is meant to own; anything at/above it is a loud single vector handled elsewhere.
SINGLE_CHANNEL_BLOCK = 0.85


def verdict_symbol(mv: dict) -> str:
    if mv["hard_block"]:
        return "X"   # hard block (coordinated, refuse outright)
    if mv["is_multivector"]:
        return "#"   # multi-vector detected (escalate / tighten)
    return "."       # no detection


def evaluate(channels_xy, *, calibration=None):
    """Return (rows, summary) for the full grid. channels_xy yields (q, c)."""
    rows = []
    fired_subthreshold = 0          # the cells that matter: both quiet, detector fires
    fired_total = 0
    false_fire_benign = 0           # both below floor but detector fired (should be 0)
    false_fire_single = 0           # one loud, other ~0, detector fired (should be 0)

    for q_raw, c_raw in channels_xy:
        raw = {"query_vector": q_raw, "context_vector": c_raw}
        chans = calibration.apply_channels(raw) if calibration else dict(raw)
        mv = multivector_risk(chans)

        q, c = chans["query_vector"], chans["context_vector"]
        both_subthreshold = (q < SINGLE_CHANNEL_BLOCK) and (c < SINGLE_CHANNEL_BLOCK)
        below_floor = (q < DEFAULT_SOFT_PER_CHANNEL["query_vector"]) and \
                      (c < DEFAULT_SOFT_PER_CHANNEL["context_vector"])
        single_loud = ((q >= SINGLE_CHANNEL_BLOCK) and (c < DEFAULT_SOFT_PER_CHANNEL["context_vector"])) or \
                      ((c >= SINGLE_CHANNEL_BLOCK) and (q < DEFAULT_SOFT_PER_CHANNEL["query_vector"]))

        if mv["is_multivector"]:
            fired_total += 1
            if both_subthreshold:
                fired_subthreshold += 1
            if below_floor:
                false_fire_benign += 1
            if single_loud:
                false_fire_single += 1

        rows.append({
            "query_raw": round(q_raw, 4), "context_raw": round(c_raw, 4),
            "query_cal": round(q, 4), "context_cal": round(c, 4),
            "joint_risk": mv["joint_risk"], "n_active": mv["n_active"],
            "is_multivector": mv["is_multivector"], "hard_block": mv["hard_block"],
            "both_subthreshold": both_subthreshold,
        })

    summary = {
        "min_channels": DEFAULT_MIN_CHANNELS,
        "joint_min": DEFAULT_JOINT_MIN,
        "joint_block": DEFAULT_JOINT_BLOCK,
        "floors": DEFAULT_SOFT_PER_CHANNEL,
        "single_channel_block": SINGLE_CHANNEL_BLOCK,
        "cells": len(rows),
        "fired_total": fired_total,
        "fired_subthreshold_band": fired_subthreshold,
        "false_fire_benign": false_fire_benign,
        "false_fire_single_loud": false_fire_single,
    }
    return rows, summary


def print_grid(calibration=None, title="RAW (no calibration)"):
    print(f"\n=== multi-vector verdict grid — {title} ===")
    print("rows = context_vector (poison channel), cols = query_vector (instruction channel)")
    print("  legend:  '.' none   '#' multi-vector fires   'X' hard block\n")
    header = "  ctx\\qry " + "".join(f"{q:>6.2f}" for q in AXIS)
    print(header)
    for c in reversed(AXIS):                         # high context at top
        cells = []
        for q in AXIS:
            raw = {"query_vector": q, "context_vector": c}
            chans = calibration.apply_channels(raw) if calibration else raw
            cells.append(f"{verdict_symbol(multivector_risk(chans)):>6}")
        print(f"  {c:>6.2f}  " + "".join(cells))


def main():
    ap = argparse.ArgumentParser(description="Controlled-injection grid eval of the multivector detector")
    ap.add_argument("--calibration", default=None,
                    help="optional calibration.json — shows the de-saturated grid")
    ap.add_argument("--json", default=None, help="write machine-readable results here")
    args = ap.parse_args()

    cal = None
    if args.calibration:
        from calibration import load_calibration
        cal = load_calibration(args.calibration)
        if cal.is_identity():
            print(f"[warn] {args.calibration} is identity — calibrated grid == raw grid")

    grid_xy = [(q, c) for c in AXIS for q in AXIS]

    # 1. Specificity proof on the explicit corner cases (no calibration needed).
    state = SimpleNamespace(scores={"injection_detection": 0.0, "fusion_risk": 0.10,
                                    "context_injection": 1.0})
    print("=== input_channels() composition check ===")
    print("  query_vector = max(injection_detection, fusion_risk); context_vector = context_injection")
    print(" ", input_channels(state),
          "->", "fires" if multivector_risk(input_channels(state))["is_multivector"] else "no fire",
          "(saturated single context vector must NOT fire — co-activation guard)")

    # 2. The raw grid: this is the regime the live scanner can actually produce.
    print_grid(None, "RAW (what the near-binary scanner emits)")
    rows_raw, summ_raw = evaluate(grid_xy, calibration=None)
    print("\n  summary(raw):", json.dumps(summ_raw, separators=(",", ":")))

    # 3. The detector firing where it should: genuine two-quiet-vector cells.
    print("\n  cells in the SUB-THRESHOLD band where the detector fires (both channels"
          f" < {SINGLE_CHANNEL_BLOCK}):")
    hits = [r for r in rows_raw if r["is_multivector"] and r["both_subthreshold"]]
    if hits:
        for r in hits:
            print(f"    q={r['query_raw']:.2f}  c={r['context_raw']:.2f}  "
                  f"joint={r['joint_risk']:.3f}  n_active={r['n_active']}  "
                  f"{'HARD' if r['hard_block'] else 'mv'}")
    else:
        print("    (none)")

    # 4. Optional calibrated grid — the FIX: a saturated (0.0, 1.0) scanner output
    #    is mapped into the graded band, populating the firing region the raw
    #    scanner can never reach.
    out = {"raw": {"summary": summ_raw, "rows": rows_raw}}
    if cal and not cal.is_identity():
        print_grid(cal, f"CALIBRATED via {args.calibration}")
        rows_cal, summ_cal = evaluate(grid_xy, calibration=cal)
        print("\n  summary(calibrated):", json.dumps(summ_cal, separators=(",", ":")))
        out["calibrated"] = {"summary": summ_cal, "rows": rows_cal}

    # 5. Assertions — this script is also a regression test of the three gates.
    assert summ_raw["false_fire_benign"] == 0, "benign below-floor cells must not fire"
    assert summ_raw["false_fire_single_loud"] == 0, "single loud vector must not fire (co-activation)"
    assert summ_raw["fired_subthreshold_band"] > 0, \
        "detector must fire on genuine two-quiet-vector cells, else logic is broken"
    print("\n[PASS] specificity holds (no benign/single-vector false fires) AND the "
          "detector fires on genuine conjunctive sub-threshold cells.")
    print("       => the 0/40 on the slice is REGIME-LIMITED (the scanner can't emit "
          "in-band scores), not a logic bug.")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\n[saved] {args.json}")


if __name__ == "__main__":
    main()
