#!/usr/bin/env python3
"""
run_baseline_comparison.py
=============================
TARGET LOCATION IN REPO:  step4 dataset/run_baseline_comparison.py
(driver script; also requires a small argparse addition to
run_full_pipeline.py — see the PATCH block below)

WHY THIS SCRIPT EXISTS
-----------------------
Every comparison in the repo so far is an ablation of the project's OWN
pipeline against itself (controller vs. no-controller, hybrid vs.
lexical-only grounding, etc.) — there is no comparison against an external
baseline (a plain LLM-Guard-only pipeline, or a spotlighting-only pipeline
with none of the later stages). methodology_working_multivector.md §10
explicitly calls for this ("compare against an external baseline ... so the
contribution is measured against prior art, not only against ablations of
itself") but it hasn't been run. This is that missing comparison.

THREE PROFILES
---------------
  llmguard_only   Step 3 gate + Step 6 per-chunk sanitization only. No
                  spotlighting (Step 8 uses a plain prompt), no multivector
                  detector, no controller. This is the closest thing to
                  "off-the-shelf LLM-Guard defense" and is the right external
                  reference point since LLM-Guard is the actual off-the-shelf
                  library this project builds on.
  spotlight_only  Adds Step 8 spotlighting/datamarking on top of the above,
                  still no multivector detector, no controller. Isolates
                  spotlighting's own contribution (the analog of the
                  Microsoft spotlighting paper's own reported setting).
  full            The project's complete pipeline, unchanged.

REQUIRED PATCH to run_full_pipeline.py
----------------------------------------
Add one argparse flag and gate the relevant stages on it. Find the argparse
block (near the existing --no-controller / --no-multivector-signal flags) and
add:

    ap.add_argument("--defense-profile",
                    choices=["full", "llmguard_only", "spotlight_only"],
                    default="full",
                    help="external-baseline profile for run_baseline_comparison.py")

Then in the request-processing function (wherever Step 8 is invoked and
wherever the multivector detector / controller are invoked), branch on it:

    if args.defense_profile == "llmguard_only":
        state = step_08_plain_prompt(state)       # existing plain prompt path,
                                                    # i.e. whatever step_08 did
                                                    # BEFORE spotlighting was added
    else:
        state = step_08_augmented_prompt.run(state)   # spotlighting (current default)

    if args.defense_profile in ("llmguard_only", "spotlight_only"):
        # skip Step 10A / 10B entirely — no multivector detector, no controller
        state = step_10_grounding_judge.run(state)
    else:
        state = risk_feedback_controller.run(state)   # full pipeline, unchanged

This keeps `full` byte-for-byte identical to today's default (the same
guarantee INTEGRATION.md already makes for run_full_pipeline_adaptive.py vs.
run_full_pipeline.py), so existing frozen results are unaffected.

USAGE (once the patch above is in place)
------------------------------------------
    python run_baseline_comparison.py --slice adversarial_slice.jsonl
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROFILES = ["llmguard_only", "spotlight_only", "full"]


def run_profile(slice_file: str, profile: str, out_dir: Path) -> Path:
    out_path = out_dir / f"grounded_{profile}.jsonl"
    cmd = [sys.executable, "run_full_pipeline.py", "--slice", slice_file,
          "--defense-profile", profile, "--out", str(out_path)]
    print(f"[run] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"run_full_pipeline.py failed for profile={profile}")
    return out_path


def score_profile(grounded_path: Path, out_dir: Path, profile: str) -> dict:
    """Reuse the project's OWN scorers rather than reinventing metrics, so the
    comparison is apples-to-apples with every other number in the repo."""
    asr_out = out_dir / f"asr_{profile}.json"
    subprocess.run([sys.executable, "score_attack_success.py", "--in", str(grounded_path)],
                   check=False)
    slice_out = out_dir / f"slice_scoring_{profile}.json"
    subprocess.run([sys.executable, "score_slice.py", str(grounded_path)], check=False)
    # score_attack_success.py / score_slice.py print their own summaries and
    # (per their existing behavior) write a *_scoring.json / report next to the
    # input — adjust these two filenames to match whatever each script's --out
    # convention actually is in your checkout before trusting this block.
    return {"grounded": str(grounded_path)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slice", default="adversarial_slice.jsonl")
    ap.add_argument("--out-dir", default="baseline_comparison")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for profile in PROFILES:
        grounded = run_profile(args.slice, profile, out_dir)
        results[profile] = score_profile(grounded, out_dir, profile)

    summary_path = out_dir / "baseline_comparison_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[done] wrote per-profile grounded_*.jsonl + summary -> {summary_path}")
    print("\nBuild the comparison TABLE by hand from the three grounded_*.jsonl "
          "files using the SAME per-kind breakdown already used in "
          "RESULTS_all_steps.txt (poison_redacted, canary_intact, gate ALLOW/"
          "BLOCK/REVIEW counts) — that gives you, for the first time, a row "
          "like:\n"
          "  llmguard_only : 40/40 redacted? X/40 canary intact? ...\n"
          "  spotlight_only: ...\n"
          "  full          : 40/40 redacted, 40/40 canary intact (already known)\n"
          "which is the external-baseline comparison the methodology doc calls for.")


if __name__ == "__main__":
    main()
