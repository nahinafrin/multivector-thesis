#!/usr/bin/env python3
"""
run_baseline_comparison.py
=============================
Runs the pipeline under three defense profiles (llmguard_only, spotlight_only,
full) against the same slice, so the project's own contribution is measured
against an external baseline rather than only ablations of itself.

USAGE
------
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
    subprocess.run([sys.executable, "score_attack_success.py", "--in", str(grounded_path)],
                   check=False)
    subprocess.run([sys.executable, "score_slice.py", str(grounded_path)], check=False)
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


if __name__ == "__main__":
    main()