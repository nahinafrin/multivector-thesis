#!/usr/bin/env python3
"""
inspect_row1_injection_only.py -- zero-rerun diagnostic for the Row 1
"normalization LOWERS recall on fullwidth/zero_width" finding.

WHY: row1_mitigation_ab_report.json already contains, for each of the three
arms (raw_baseline, obfuscated_off_mitigation, obfuscated_on_mitigation), not
just the fused "fusion" numbers the console summary printed, but also
"injection_only" (the deberta-v3 injection classifier ALONE, no Llama-Guard)
and "or_gate" (injection OR Llama-Guard). Comparing injection_only across the
three arms -- for free, from data you already have on disk -- tells us WHERE
the OFF>ON recall gap comes from:

  - if injection_only shows the same OFF>ON drop  -> the deberta injection
    classifier itself is keying on the raw zero-width/fullwidth byte noise
    (most likely: those obfuscation tricks appear as "attack" examples in
    whatever adversarial/jailbreak data the classifier was fine-tuned on, so
    it learned the unicode pattern as a shortcut for the label, not the
    underlying semantic instruction).
  - if injection_only is flat/similar OFF vs ON but the FUSED number still
    drops -> Llama-Guard's soft score s_L (or the fusion's agreement bonus
    gamma, which rewards the two detectors agreeing) is doing the work
    instead/also.

No pipeline re-run needed -- this only reads the JSON already written by
row1_obfuscation_mitigation_ab.py.

USAGE:
    python inspect_row1_injection_only.py row1_mitigation_results\\row1_mitigation_ab_report.json
"""
from __future__ import annotations
import json
import sys


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python inspect_row1_injection_only.py <report.json>")
    with open(sys.argv[1], encoding="utf-8") as f:
        report = json.load(f)

    arms = [
        ("raw_baseline", report["raw_baseline"]),
        ("obfuscated_off_mitigation", report["obfuscated_off_mitigation"]),
        ("obfuscated_on_mitigation", report["obfuscated_on_mitigation"]),
    ]

    print(f"{'arm':40} {'inj_only R':>11} {'inj_only P':>11} {'or_gate R':>10} {'fusion R':>9}")
    for name, rep in arms:
        io = rep["injection_only"]
        og = rep["or_gate"]
        fu = rep["fusion"]
        print(f"{name:40} {io['recall']:>11.3f} {io['precision']:>11.3f} "
              f"{og['recall']:>10.3f} {fu['recall']:>9.3f}")

    print(
        "\nRead this as: if 'inj_only R' drops from obfuscated_off -> "
        "obfuscated_on by roughly the same amount the fused recall drops "
        "(0.900 -> 0.567 in the run you pasted), the injection classifier "
        "itself is the one reacting to raw unicode noise, and normalization "
        "is correctly stripping a signal that was never real semantic "
        "detection. If inj_only stays flat instead, look at or_gate R next "
        "to isolate Llama-Guard's contribution."
    )


if __name__ == "__main__":
    main()
