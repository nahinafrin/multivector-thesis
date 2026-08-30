#!/usr/bin/env python3
"""
revalidate_multivector_graded.py
=================================
TARGET LOCATION IN REPO:  step4 dataset/revalidate_multivector_graded.py
(same directory as multivector.py, graded_channels.py, calibrate_payloads.py)

WHY THIS SCRIPT EXISTS
-----------------------
graded_channels.py already reads the DeBERTa logit margin instead of the
saturated softmax probability, and step_03_injection_detection.py /
step_06_context_sanitization.py already write the graded scores into
state.scores["injection_graded"] / state.scores["context_graded"]. multivector.py
already prefers those graded channels over the squashed ones. In other words:
the FIX described in methodology_working_multivector.md is already wired into
the codebase.

What has NOT been done yet is closing the loop: re-running the real 220-row
adversarial_slice.jsonl through the CURRENT pipeline (so injection_graded /
context_graded are actually populated on every row), then checking whether the
multivector detector's fire rate on the `multivector_attack` rows actually moves
off the old 0/34 (or 1/34-invalid) result once it is reading graded channels
instead of squashed ones. Until that comparison exists, "the fix works" is only
validated on eval_multivector_grid.py's synthetic sweep, not on the project's
own attack slice — which is exactly the gap flagged in the analysis report.

This script does that comparison. It does NOT re-run the LLM ensemble (Step 9)
or re-download models: it only needs a grounded_*.jsonl that has already been
produced by `python run_full_pipeline.py --slice adversarial_slice.jsonl
--out grounded_controller_graded.jsonl` on the *current* code. If that file is
missing the graded fields, it tells you exactly that instead of guessing.

USAGE
-----
    # 1. Regenerate the slice run on current code (writes injection_graded /
    #    context_graded because step_03 / step_06 already populate them):
    python run_full_pipeline.py --slice adversarial_slice.jsonl \
        --out grounded_controller_graded.jsonl

    # 2. Compare old (squashed) vs new (graded) fire rate on the SAME rows:
    python revalidate_multivector_graded.py \
        --old grounded_controller.jsonl \
        --new grounded_controller_graded.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from multivector import multivector_risk, DEFAULT_SOFT_PER_CHANNEL, DEFAULT_JOINT_MIN


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def channels_from_row(row: dict, prefer_graded: bool) -> dict[str, float] | None:
    """Reconstruct the two multivector channels from a scored row's audit trail.

    grounded_*.jsonl rows have scores nested under gate/ and retrieval/ sections
    (see pipeline_common.py PipelineState -> dict serialization in run_full_pipeline.py's
    _record() function). We read it defensively because older runs may not have
    the graded fields at all.
    """
    gate = row.get("gate", {})
    retrieval = row.get("retrieval", {})
    
    if prefer_graded:
        query = gate.get("injection_graded")
        ctx = retrieval.get("context_graded")
        if query is None or ctx is None:
            return None  # this row genuinely has no graded signal — don't fake it
    else:
        query = gate.get("injection_score", 0.0)
        ctx = retrieval.get("context_injection", 0.0)
    return {"query_vector": float(query or 0.0), "context_vector": float(ctx or 0.0)}


def fire_report(rows: list[dict], prefer_graded: bool, label: str) -> dict:
    mv_rows = [r for r in rows if r.get("kind") == "multivector_attack"]
    gate_blocked = [r for r in mv_rows if r.get("blocked") and r.get("block_stage") in
                    ("step_03_injection_detection", "step_03c_fusion_gate")]
    evaluated = [r for r in mv_rows if r not in gate_blocked]

    fires, subthreshold_fires, missing = 0, 0, 0
    for r in evaluated:
        ch = channels_from_row(r, prefer_graded)
        if ch is None:
            missing += 1
            continue
        result = multivector_risk(ch)
        if result["is_multivector"]:
            fires += 1
            both_subthreshold = all(
                v < 0.9 for v in result["channels"].values()
            )
            if both_subthreshold:
                subthreshold_fires += 1

    print(f"\n=== {label} ===")
    print(f"  multivector_attack rows total   : {len(mv_rows)}")
    print(f"  gate-blocked before detector ran : {len(gate_blocked)}")
    print(f"  evaluated by detector            : {len(evaluated)}")
    print(f"  missing graded fields (skipped)  : {missing}")
    print(f"  detector fired                   : {fires}/{len(evaluated) - missing}")
    print(f"  ...of which genuinely sub-threshold (both channels < 0.9): "
          f"{subthreshold_fires}")
    return {
        "label": label, "total": len(mv_rows), "gate_blocked": len(gate_blocked),
        "evaluated": len(evaluated) - missing, "fired": fires,
        "subthreshold_fired": subthreshold_fires, "missing_graded": missing,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old", required=True, help="grounded_controller.jsonl (frozen, squashed scores)")
    ap.add_argument("--new", required=True, help="grounded_controller_graded.jsonl (rerun on current code)")
    ap.add_argument("--out", default="multivector_graded_revalidation.json")
    args = ap.parse_args()

    old_rows = read_jsonl(args.old)
    new_rows = read_jsonl(args.new)

    print(f"[config] DEFAULT_SOFT_PER_CHANNEL={DEFAULT_SOFT_PER_CHANNEL}  "
          f"DEFAULT_JOINT_MIN={DEFAULT_JOINT_MIN}")

    before = fire_report(old_rows, prefer_graded=False, label="BEFORE (squashed scores, frozen run)")
    after = fire_report(new_rows, prefer_graded=True, label="AFTER (graded margin, current code)")

    verdict = "IMPROVED" if after["fired"] > before["fired"] else (
        "UNCHANGED" if after["fired"] == before["fired"] else "REGRESSED")
    print(f"\n[verdict] fire rate {before['fired']} -> {after['fired']}  ({verdict})")
    if after["missing_graded"] > 0:
        print("[warn] some rows in --new are missing injection_graded/context_graded — "
              "confirm run_full_pipeline.py was actually run on the current codebase, "
              "not a stale grounded_*.jsonl copied from an earlier run.")

    Path(args.out).write_text(json.dumps({"before": before, "after": after,
                                          "verdict": verdict}, indent=2))
    print(f"[saved] {args.out}")
    print("\nNEXT STEP if fire rate improved but is still low: re-run "
          "calibrate_payloads.py against the *graded* channel distribution in "
          "--new (not the old squashed one) to re-derive DEFAULT_SOFT_PER_CHANNEL "
          "and DEFAULT_JOINT_MIN, then re-run this script once more before "
          "claiming the fix in the writeup.")


if __name__ == "__main__":
    main()
