"""
verify_channels.py — Phase 4 gate: are accepted multivector rows actually populated?

Prints query_vector and context_vector for every multivector_attack row that the
controller accepted. If you mostly see 0.0 0.0, the slice is not testing
co-activation — go back to calibrate_payloads.py and rebuild the slice.

Usage:
    python verify_channels.py grounded_controller.jsonl
"""

from __future__ import annotations

import argparse
import sys

from pipeline_common import read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", nargs="?", default="grounded_controller.jsonl")
    args = ap.parse_args()

    n_accept = n_zero = 0
    for r in read_jsonl(args.jsonl):
        if r.get("kind") != "multivector_attack":
            continue
        if (r.get("controller") or {}).get("final_action") != "accept":
            continue
        ch = (r.get("multivector") or {}).get("channels", {})
        qv = round(float(ch.get("query_vector", 0) or 0), 3)
        cv = round(float(ch.get("context_vector", 0) or 0), 3)
        n_accept += 1
        if qv == 0.0 and cv == 0.0:
            n_zero += 1
        print(f"  query_vector={qv}  context_vector={cv}  id={r.get('id')}")

    if n_accept == 0:
        print("no accepted multivector_attack rows (good if all were caught)")
        return

    print(f"\n[verify] accepted multivector rows: {n_accept}  "
          f"both channels zero: {n_zero}")
    if n_zero > n_accept // 2:
        print("STOP: most accepted rows have 0.0 channels — payloads are sub-floor.")
        print("Re-run calibrate_payloads.py, paste winners, rebuild slice.")
        sys.exit(1)
    print("GO: channels look populated on most accepted rows.")


if __name__ == "__main__":
    main()
