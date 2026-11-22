#!/usr/bin/env python3
"""
ensemble_generalization_check.py
===================================
Runs the SAME slice through both ensembles (small: llama3.2:3b/mistral:7b/
qwen2.5:3b vs. large: llama3.2:3b/mistral:7b/llama3.1:8b) and reports whether
the AUC / separation-gap numbers from grounding_separation_probe.py move in a
materially different direction with the larger model swapped in.

USAGE
------
    python ensemble_generalization_check.py \
        --slice semantic_slice.jsonl \
        --small-out grounded_semantic_small_ensemble.jsonl \
        --large-out grounded_semantic_large_ensemble.jsonl
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def run_pipeline(slice_file: str, out_file: str, ensemble_env: str) -> None:
    env = dict(os.environ)
    env["RAG_ENSEMBLE"] = ensemble_env
    cmd = [sys.executable, "run_full_pipeline.py", "--slice", slice_file, "--out", out_file]
    print(f"[run RAG_ENSEMBLE={ensemble_env}] {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)


def compare(small_out: str, large_out: str) -> None:
    cmd_small = [sys.executable, "grounding_separation_probe.py", "--in", small_out,
                 "--attack-kind", "semantic_multivector", "--benign-file", "grounded_controller.jsonl"]
    cmd_large = [sys.executable, "grounding_separation_probe.py", "--in", large_out,
                 "--attack-kind", "semantic_multivector", "--benign-file", "grounded_controller.jsonl"]
    print("\n=== small ensemble (llama3.2:3b / mistral:7b / qwen2.5:3b) ===")
    subprocess.run(cmd_small, check=False)
    print("\n=== large ensemble (llama3.2:3b / mistral:7b / llama3.1:8b) ===")
    subprocess.run(cmd_large, check=False)
    print("\nCompare the two AUC / separation-gap blocks above by eye, or add a "
          "--out json flag to grounding_separation_probe.py and diff the files. A "
          "materially different AUC/gap under the larger model is evidence the "
          "current results are stack-specific; a similar result strengthens the "
          "claim that the finding generalizes beyond three small local models.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slice", required=True)
    ap.add_argument("--small-out", required=True)
    ap.add_argument("--large-out", required=True)
    ap.add_argument("--skip-run", action="store_true",
                    help="skip regeneration, just compare existing --small-out/--large-out")
    args = ap.parse_args()

    if not args.skip_run:
        run_pipeline(args.slice, args.small_out, "small")
        run_pipeline(args.slice, args.large_out, "large")
    compare(args.small_out, args.large_out)


if __name__ == "__main__":
    main()
