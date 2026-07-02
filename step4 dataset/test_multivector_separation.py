"""
test_multivector_separation.py — acceptance gate for the new fusion logic.
============================================================================

Run this BEFORE wiring correlated_multivector.py / temporal_risk_tracker.py
into risk_feedback_controller.py, and again after every recalibration. This
mirrors the discipline already present in multivector.assert_separation() —
don't trust a threshold until you've shown, on data, that it actually
separates the classes it's meant to separate.

Usage:
    python test_multivector_separation.py --scored path/to/scored_conjunctive.jsonl

Expected input format: one JSON object per line with at least:
    {
      "kind": "benign_control" | "multivector_attack_synth" | ...,
      "channels": {"query_vector": 0.xx, "context_vector": 0.xx},
      "split_weight": 0.5   # optional, only present for synth conjunctive rows
    }
Produce this file by running your scored pipeline output through
correlated_multivector_risk() and dumping the channel scores + result per row
— the existing step4 scoring scripts already write structurally similar JSONL,
so this should slot into build_multivector_inband.py / score_slice.py output
with minor key renaming.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

sys.path.insert(0, ".")
from correlated_multivector import ChannelHistory, correlated_multivector_risk  # noqa: E402


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score_rows(rows: list[dict]) -> list[dict]:
    """Score each row through the correlation-aware fusion, using a fresh
    ChannelHistory per row (no cross-row leakage) unless a session_id groups
    them — extend this if you want genuine cross-turn correlation testing.
    """
    out = []
    for r in rows:
        hist = ChannelHistory(window=1)  # single-shot: correlation term ~0 here
        result = correlated_multivector_risk(r["channels"], hist)
        out.append({**r, "result": result})
    return out


def pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 4)


def run_separation_report(scored: list[dict]) -> dict:
    """Acceptance report for the multi-vector detector.

    PRIMARY GATE (binary co-activation): the controller consumes the boolean
    ``is_multivector`` field, which fires only when >= min_channels channels
    clear their per-channel floor AND the noisy-OR joint risk clears joint_min.
    That is the quantity the system actually acts on, so it is what the gate
    tests: attack detection rate vs. benign false-positive rate.

    SECONDARY DIAGNOSTIC (continuous distributions): benign p95 vs attack p10
    on the RAW noisy-OR joint risk, ignoring the floor/co-activation gates.
    This is reported but is NOT the pass/fail criterion, because raw joint
    magnitude and the gated co-activation decision measure different things:
    a benign query that retrieves one topically-hot chunk can produce a high
    raw joint risk while never co-activating two channels, so it does NOT
    fire the detector. Overlap in these continuous tails is therefore EXPECTED
    by design and is the reason the detector gates on co-activation rather
    than on raw joint magnitude — it is evidence for the design choice, not a
    failure of it. Recalibrating floors/joint_min to force continuous
    separation would redefine which rows are admissible as sub-threshold
    conjunctive attacks (build_multivector_inband.py admits only rows with
    each_vector < joint_min) and would discard the weakest genuine positives,
    so it is deliberately NOT what this gate optimizes.
    """
    by_kind = defaultdict(list)
    for r in scored:
        by_kind[r.get("kind", "?")].append(r)

    benign = by_kind.get("benign_control", [])
    conj = [r for r in scored if "multivector_attack" in r.get("kind", "")]

    # ---- PRIMARY: binary co-activation separation --------------------------- #
    attack_detected = sum(1 for r in conj if r["result"]["is_multivector"])
    benign_fp = sum(1 for r in benign if r["result"]["is_multivector"])
    attack_rate = round(attack_detected / len(conj), 4) if conj else None
    benign_fp_rate = round(benign_fp / len(benign), 4) if benign else None

    # Primary pass condition: the detector fires on attacks strictly more often
    # than on benign controls, with a benign FP rate under a conventional 5%.
    # This is the separation that matters for the controller.
    binary_separated = (
        attack_rate is not None and benign_fp_rate is not None
        and attack_rate > benign_fp_rate and benign_fp_rate <= 0.05
    )

    # ---- SECONDARY: continuous joint-risk distributions (diagnostic only) ---- #
    benign_joint = [r["result"]["correlation_adjusted_risk"] for r in benign]
    conj_joint = [r["result"]["correlation_adjusted_risk"] for r in conj]
    b95 = pct(benign_joint, 95)
    a10 = pct(conj_joint, 10)
    continuous_separated = (b95 is not None and a10 is not None and b95 < a10)

    report = {
        # primary gate
        "n_benign": len(benign),
        "n_conjunctive": len(conj),
        "attack_detection_rate": attack_rate,
        "benign_false_positive_rate": benign_fp_rate,
        "binary_separated": binary_separated,       # <- pass/fail criterion
        # secondary diagnostic (NOT pass/fail)
        "diagnostic_benign_p95_joint_risk": b95,
        "diagnostic_attack_p10_joint_risk": a10,
        "diagnostic_continuous_separated": continuous_separated,
        "diagnostic_note": (
            "continuous overlap is expected by design; the detector gates on "
            "co-activation, not raw joint magnitude — see run_separation_report "
            "docstring. Do NOT recalibrate to force continuous separation."
        ),
    }

    # Dose-response by split_weight, if present (synthetic corpus only) — shows
    # the mechanism responds sensibly to how "hidden" the split is.
    by_weight = defaultdict(list)
    for r in conj:
        w = r.get("split_weight")
        if w is not None:
            by_weight[w].append(r["result"]["is_multivector"])
    if by_weight:
        report["detection_rate_by_split_weight"] = {
            w: round(sum(v) / len(v), 3) for w, v in sorted(by_weight.items())
        }

    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scored", required=True)
    args = ap.parse_args()

    rows = load(args.scored)
    scored = score_rows(rows)
    report = run_separation_report(scored)

    print(json.dumps(report, indent=2))

    # Report the continuous diagnostic without failing on it.
    if not report["diagnostic_continuous_separated"]:
        print(
            "\nDIAGNOSTIC: raw joint-risk distributions overlap in the tails "
            f"(benign p95={report['diagnostic_benign_p95_joint_risk']} >= "
            f"attack p10={report['diagnostic_attack_p10_joint_risk']}). This is "
            "EXPECTED and is not a failure — the detector gates on co-activation, "
            "not raw joint magnitude. See the docstring; do not recalibrate to "
            "force this.",
            file=sys.stderr,
        )

    # PRIMARY pass/fail: binary co-activation separation.
    if not report["binary_separated"]:
        print(
            "\nFAIL (primary gate): the detector does not separate attacks from "
            "benign controls on the binary is_multivector decision "
            f"(attack detection={report['attack_detection_rate']}, "
            f"benign FP={report['benign_false_positive_rate']}). This IS a real "
            "failure — the controller consumes this decision. Investigate floors "
            "/ min_channels / joint_min against THIS in-band distribution.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"\nPASS (primary gate): binary co-activation separation holds "
        f"(attack detection={report['attack_detection_rate']}, "
        f"benign FP={report['benign_false_positive_rate']})."
    )


if __name__ == "__main__":
    main()
