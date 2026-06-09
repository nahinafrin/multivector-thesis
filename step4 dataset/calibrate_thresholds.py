"""
calibrate_thresholds.py  —  honest threshold calibration from a real run
========================================================================

This is the rewrite called for in the multi-vector spec, Part E. It replaces
the earlier disagreement-only version that overstated what disagreement could
do as an attack signal.

What it does differently
------------------------
  1. SPLITS gate-blocked rows from gate-reaching rows. Disagreement is
     reported ONLY over rows that actually reached generation, because rows
     the gate killed never had a chance to disagree.
  2. CALIBRATES the multi-vector ``joint_risk`` — the actual sharp attack
     signal on this corpus — by comparing benign-control vs multivector_attack
     distributions, and explicitly warns if they overlap (which would mean
     either ``soft`` is set too high or Step 6's ``context_injection`` isn't
     getting populated).
  3. REPORTS the gate-slip gap as a known residual, not as something to
     calibrate. Roleplay/novel-framing attacks have ~0 injection signal in
     both channels, so they pass the gate; that is the limitation the
     writeup should state honestly.
  4. CAPS ``recover_min_faithfulness`` below the base grounding threshold,
     so a quirky tail in benign-ungrounded faithfulness can't silently
     disable recovery by suggesting a value at or above the threshold.

Usage
-----
    # 1. Build a slice that includes multi-vector and gate-slip rows:
    python build_adversarial_slice.py \\
        --attack-jsonl ../dataset/merged_output/dataset_all.clean.jsonl \\
        --qa-jsonl data/question-answer/test.jsonl \\
        --n-multivector 40 --n-gate-slip 20 \\
        --out adversarial_slice.jsonl

    # 2. Run with the controller on:
    python run_full_pipeline.py --slice adversarial_slice.jsonl \\
        --out grounded_controller.jsonl

    # 3. Calibrate:
    python calibrate_thresholds.py grounded_controller.jsonl

The script never writes anywhere — you decide whether to plug the suggested
numbers into ``ControllerConfig`` / ``multivector_risk`` yourself.
"""

from __future__ import annotations

import argparse
import statistics as stats
import sys

from pipeline_common import read_jsonl

# Must match step_10's BASE_THRESHOLD so the recover-min cap stays meaningful.
BASE_GROUNDING_THRESHOLD = 0.70


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _percentile(values: list[float], p: float) -> float:
    """Plain-Python percentile (no numpy dependency). p in [0, 1]."""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    idx = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return s[idx]


def _reached_generation(row: dict) -> bool:
    """True iff the row's generation block carries an answer (i.e. Steps 4-10
    actually ran, regardless of whether grounding ultimately passed)."""
    return bool((row.get("generation") or {}).get("answer"))


def _joint_risk(row: dict) -> float:
    return float((row.get("multivector") or {}).get("joint_risk", 0.0) or 0.0)


def _disagreement(row: dict) -> float:
    return float((row.get("generation") or {}).get("disagreement", 0.0) or 0.0)


def _faithfulness(row: dict) -> float:
    return float((row.get("grounding") or {}).get("faithfulness", 0.0) or 0.0)


def _gate_allowed(row: dict) -> bool:
    return (row.get("gate") or {}).get("decision") == "ALLOW"


# --------------------------------------------------------------------------- #
# Report sections
# --------------------------------------------------------------------------- #
def report_multivector(benign: list[dict], multivector: list[dict]) -> None:
    """The real attack signal on this corpus: multi-vector joint_risk."""
    print("\n=== Multi-vector joint_risk (the calibrated attack signal) ===")
    if not benign or not multivector:
        print("  insufficient rows: "
              f"benign_control n={len(benign)}, "
              f"multivector_attack n={len(multivector)}")
        return

    b_j = [_joint_risk(r) for r in benign]
    m_j = [_joint_risk(r) for r in multivector]
    b_mean, b_p95 = stats.mean(b_j), _percentile(b_j, 0.95)
    m_mean, m_p10 = stats.mean(m_j), _percentile(m_j, 0.10)

    print(f"  benign_control     n={len(b_j):3d}  "
          f"mean={b_mean:.3f}  p95={b_p95:.3f}")
    print(f"  multivector_attack n={len(m_j):3d}  "
          f"mean={m_mean:.3f}  p10={m_p10:.3f}")

    if b_p95 < m_p10:
        midpoint = round((b_p95 + m_p10) / 2, 3)
        print(f"  -> detector separates the classes; suggested "
              f"DEFAULT_JOINT_MIN midpoint = {midpoint}")
        print("     (also run: python calibrate_payloads.py <this.jsonl> "
              "for per-channel floors)")
    else:
        print("  -> WARNING: distributions overlap.")
        print("     Fix payloads first (python calibrate_payloads.py), rebuild")
        print("     slice, re-run — do NOT paste joint_min until separated.")
        print("     Verify Step 6 populates context_injection; run verify_channels.py.")


def report_disagreement(benign: list[dict], attack_kinds: list[dict]) -> None:
    """Disagreement is reported only over rows that reached generation. Gate-
    blocked rows never had a chance to disagree, so including them would
    confound the distributions."""
    print("\n=== Disagreement (gen-reaching rows only — honest framing) ===")
    b_d = [_disagreement(r) for r in benign if _reached_generation(r)]
    a_d = [_disagreement(r) for r in attack_kinds if _reached_generation(r)]
    if not b_d or not a_d:
        print(f"  insufficient gen-reaching rows: "
              f"benign n={len(b_d)}, attack n={len(a_d)}")
        return
    b_p95 = _percentile(b_d, 0.95)
    a_p10 = _percentile(a_d, 0.10)
    print(f"  benign (gen-reaching) n={len(b_d):3d}  "
          f"mean={stats.mean(b_d):.3f}  p95={b_p95:.3f}")
    print(f"  attack (gen-reaching) n={len(a_d):3d}  "
          f"mean={stats.mean(a_d):.3f}  p10={a_p10:.3f}")
    if b_p95 < a_p10:
        print(f"  -> disagreement separates here; midpoint "
              f"{round((b_p95 + a_p10) / 2, 3)} is a candidate "
              f"disagreement_attack threshold.")
    else:
        print("  -> disagreement does NOT separate. The multi-vector "
              "signal above is the one to rely on.")


def report_recover_min(benign: list[dict]) -> None:
    """recover_min_faithfulness from benign rows the grounding judge rejected,
    capped below the base grounding threshold so it can't silently disable
    recovery."""
    print("\n=== recover_min_faithfulness (capped below base threshold) ===")
    rejected = [_faithfulness(r) for r in benign
                if not (r.get("grounding") or {}).get("passed", True)]
    if not rejected:
        print(f"  no benign-control rows were ungrounded (base threshold "
              f"{BASE_GROUNDING_THRESHOLD}); leave the default in place.")
        return
    raw = _percentile(rejected, 0.90)
    capped = round(min(raw, BASE_GROUNDING_THRESHOLD - 0.05), 3)
    print(f"  benign-ungrounded faithfulness n={len(rejected)}  "
          f"p90={raw:.3f}")
    print(f"  suggested ControllerConfig.recover_min_faithfulness = {capped}  "
          f"(capped below base grounding threshold {BASE_GROUNDING_THRESHOLD})")
    if len(rejected) < 20:
        print(f"  WARNING: only {len(rejected)} samples — treat as "
              "indicative, not final. Rebuild the slice with a larger "
              "n-benign before locking the number in.")


def report_gate_slip_gap(gate_slip: list[dict]) -> None:
    """The residual the multi-vector detector cannot close: pure semantic
    obfuscation. We report it; we do not claim to fix it."""
    print("\n=== Gate-slip gap (residual; reported, not solved) ===")
    if not gate_slip:
        print("  no gate_slip_query rows in this run; rebuild the slice "
              "with --n-gate-slip to measure this gap.")
        return
    slipped = sum(1 for r in gate_slip if _gate_allowed(r))
    pct = 100.0 * slipped / len(gate_slip)
    print(f"  obfuscated attacks that passed the gate: "
          f"{slipped}/{len(gate_slip)} ({pct:.0f}%)")
    print("  -> the injection detector cannot catch semantic-obfuscation")
    print("     attacks (roleplay/novel/security-training framing).")
    print("     Closing this gap requires a semantic signal, not an")
    print("     injection signal — future work.")


def report_controller_actions(rows: list[dict]) -> None:
    """A small bonus: which controller branches actually fired on this run.
    Pairs with inspect_controller.py for the full action tally."""
    print("\n=== Controller final_action distribution (kind x action) ===")
    from collections import Counter
    tally: Counter[tuple[str, str]] = Counter()
    for r in rows:
        kind = r.get("kind") or "(unknown)"
        ctrl = r.get("controller") or {}
        if not ctrl:
            tally[(kind, "gate_blocked_before_controller")] += 1
            continue
        tally[(kind, ctrl.get("final_action") or "(no_action)")] += 1
    for (kind, action), n in sorted(tally.items()):
        print(f"  {kind:<22s}  {action:<32s}  n={n}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "jsonl",
        nargs="?", default="grounded_controller.jsonl",
        help="controller-on JSONL produced by run_full_pipeline.py "
             "(default: grounded_controller.jsonl)",
    )
    args = ap.parse_args()

    rows = list(read_jsonl(args.jsonl))
    if not rows:
        print(f"no rows in {args.jsonl!r}", file=sys.stderr)
        sys.exit(2)

    benign = [r for r in rows if r.get("kind") == "benign_control"]
    multivector = [r for r in rows if r.get("kind") == "multivector_attack"]
    gate_slip = [r for r in rows if r.get("kind") == "gate_slip_query"]
    attack_kinds = [r for r in rows if r.get("kind") in (
        "adversarial_query", "multivector_attack",
        "poisoned_context", "gate_slip_query",
    )]

    print(f"[calibrate] {args.jsonl}  rows={len(rows)}")
    print(f"  kinds: benign_control={len(benign)}, "
          f"multivector_attack={len(multivector)}, "
          f"gate_slip_query={len(gate_slip)}, "
          f"all_attack_kinds={len(attack_kinds)}")

    report_multivector(benign, multivector)
    report_disagreement(benign, attack_kinds)
    report_recover_min(benign)
    report_gate_slip_gap(gate_slip)
    report_controller_actions(rows)


if __name__ == "__main__":
    main()
