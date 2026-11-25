"""Launcher for the RECOVERED adaptive-tier external pipeline.

NEW FILE (written 2026-11-25 for reproducibility). It is NOT part of the recovered code and contains no pipeline
logic. It only arranges the import path so that the recovered scripts run against the module versions that existed
when the saved external results were produced, then executes the recovered orchestrator unchanged:

  * original/                 the four recovered scripts, byte-identical to git blobs at fe71c56^ (unchanged since
                              1d87f26, 2026-08-10, i.e. before the 2026-08-11 external run)
  * runtime_snapshot_1d99b23/ every local module those scripts import (transitively), exactly as committed in
                              1d99b23 (the commit that added the saved results, 34 minutes after the run ended)

Why a launcher: six imported modules (Stages 3c, 4, 6, 7, 9, 10 and run_full_pipeline.py) changed after the run.
Running the recovered scripts against today's step modules would NOT execute the code that produced the results.

Usage (from anywhere; the working directory is switched to `step4 dataset/`, where the run executed, because the
recovered code reads ./kb_wiki.faiss, ./kb_wiki.texts and ./graded_config.json relative to it):

    python run_adaptive_external.py --check-imports
        Import every module and report where it was loaded from. No model inference.

    python run_adaptive_external.py --in external_validation_n200.jsonl --out <new_file>.jsonl \
        --mode adaptive --disable-semantic-gate
        FRESH INFERENCE (needs Ollama with llama3.2:3b, mistral:7b, qwen2.5:3b; ~80 min). Arguments are passed
        through to the recovered run_full_pipeline_adaptive.main(). Never point --out at ext_n200_results.jsonl.
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE / "original"
SNAPSHOT = HERE / "runtime_snapshot_1d99b23"
RUN_DIR = HERE.parent                     # step4 dataset/
PROTECTED = {"ext_n200_results.jsonl"}


def _arrange_path() -> None:
    # Drop any entry that would resolve to today's step modules, then put the recovered code first.
    today = {str(RUN_DIR.resolve()), str((RUN_DIR.parent / "dataset").resolve()), str(HERE)}
    sys.path[:] = [p for p in sys.path if p and str(Path(p).resolve()) not in today]
    sys.path[0:0] = [str(ORIGINAL), str(SNAPSHOT)]
    os.chdir(RUN_DIR)


def check_imports() -> int:
    _arrange_path()
    import importlib
    names = ["run_full_pipeline_adaptive", "adaptive_risk", "adaptive_verification", "adaptive_prompt"] + sorted(
        p.stem for p in SNAPSHOT.glob("*.py"))
    bad = 0
    for n in names:
        try:
            m = importlib.import_module(n)
            origin = Path(m.__file__).resolve()
            ok = origin.parent in (ORIGINAL.resolve(), SNAPSHOT.resolve())
            bad += not ok
            print(f"{'OK  ' if ok else 'BAD '} {n:34s} {origin.relative_to(RUN_DIR.parent)}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL {n:34s} {type(e).__name__}: {e}")
    print(f"{len(names) - bad}/{len(names)} modules imported from the recovered code")
    return 1 if bad else 0


def main() -> int:
    if sys.argv[1:] == ["--check-imports"]:
        return check_imports()
    if "--out" in sys.argv and Path(sys.argv[sys.argv.index("--out") + 1]).name in PROTECTED:
        raise SystemExit("refusing to overwrite the saved results file; choose a new --out path")
    _arrange_path()
    sys.argv[0] = str(ORIGINAL / "run_full_pipeline_adaptive.py")
    runpy.run_path(sys.argv[0], run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
