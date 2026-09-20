#!/usr/bin/env python3
"""
ensemble_generalization_check.py
===================================
TARGET LOCATION IN REPO:  step4 dataset/ensemble_generalization_check.py
(plus a small addition to step_09_generator_llm.py — see PATCH below)

WHY THIS SCRIPT EXISTS
-----------------------
The generation ensemble is three small local models (llama3.2:3b, mistral:7b,
qwen2.5:3b). Every disagreement- and faithfulness-based signal in the project
(the Step 9 disagreement score, the semantic-class AUC numbers in §4.8) is
measured only on this 3B–7B stack. It's unknown whether those signals hold, get
stronger, or wash out with a larger model in the mix — which matters because a
production deployment would likely use at least one larger model, and a
reviewer will ask whether the results are an artifact of small-model
disagreement rather than a real security signal.

PATCH to step_09_generator_llm.py
-----------------------------------
The file already keys its ensemble as a dict, e.g.:

    ENSEMBLE = {
        "llama3.2:3b": "logic",
        "mistral:7b":  "styling",
        "qwen2.5:3b":  "generalism",
    }

Add an alternate config alongside it (do not remove the original — this must
stay an opt-in comparison, not a replacement of the frozen results):

    ENSEMBLE_WITH_LARGER_MODEL = {
        "llama3.2:3b": "logic",
        "mistral:7b":  "styling",
        "llama3.1:8b": "generalism",   # swaps out qwen2.5:3b for a larger model
    }

And thread an `ensemble: dict[str, str] | None = None` kwarg through
`generate_candidates(...)` (falling back to `ENSEMBLE` when None) so callers
can select which stack runs without duplicating the function.

Pull the model first:  ollama pull llama3.1:8b

THIS SCRIPT
------------
Runs the SAME slice (or a subset of it — start with just the 8-row semantic
slice and the 40-row multivector slice, since those are what disagreement/
faithfulness are supposed to separate) through both ensembles and reports
whether the AUC / separation-gap numbers from grounding_separation_probe.py
move in a materially different direction with the larger model swapped in.

USAGE
------
    python ensemble_generalization_check.py \
        --slice semantic_slice.jsonl \
        --small-out grounded_semantic_small_ensemble.jsonl \
        --large-out grounded_semantic_large_ensemble.jsonl
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_pipeline(slice_file: str, out_file: str, ensemble_env: str) -> None:
    # run_full_pipeline.py doesn't currently take an --ensemble flag; the
    # simplest non-invasive way to select the alternate ENSEMBLE_WITH_LARGER_MODEL
    # dict without a second argparse flag is an env var step_09 checks at import
    # time. Add this to step_09_generator_llm.py right after the ENSEMBLE dicts:
    #
    #     import os
    #     ACTIVE_ENSEMBLE = (ENSEMBLE_WITH_LARGER_MODEL
    #                        if os.environ.get("RAG_ENSEMBLE") == "large"
    #                        else ENSEMBLE)
    #
    # and use ACTIVE_ENSEMBLE wherever ENSEMBLE is currently referenced.
    import os
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
    print("\nCompare the two AUC / separation-gap blocks above by eye, or pipe "
          "grounding_separation_probe.py's --out json (add one if it doesn't "
          "already exist) into a diff. A materially different AUC/gap under the "
          "larger model is evidence the current results are stack-specific and "
          "should be reported as such; a similar result strengthens the claim "
          "that the finding generalizes beyond three small local models.")


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
