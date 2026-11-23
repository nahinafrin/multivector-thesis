#!/usr/bin/env python3
"""
run_improvement_queue.py -- the GPU runs behind the follow-up experiments (items 2-4, 6).

Runs sequentially (one GPU), resumable: a job whose output already has the expected row count
is skipped. Progress goes to external_eval/improvement_queue.log and per-job logs.

  1. deepset train split through the gate        -> external_eval/deepset_gate_train_rows.jsonl   (item 2: threshold refit)
  2. held-out benign graded channels             -> external_eval/graded_heldout_benign.jsonl    (item 3: floor refit)
  3. deepset 30+30 through the complete pipeline -> external_eval/deepset_e2e.jsonl              (item 2: end-to-end)
  4. 40-row semantic slice, complete pipeline, fused vs isolate-and-aggregate generator
                                                 -> external_eval/semantic40_{fused,isolate}.jsonl (item 4)
  5. 224 extra planted attacks: undefended, full stack, sanitize+grounding+refuse
                                                 -> mitigation_results/extended424/                (item 6)

    .venv311\\Scripts\\python.exe run_improvement_queue.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EE = HERE / "external_eval"
EXT = HERE / "mitigation_results" / "extended424"
LOG = EE / "improvement_queue.log"
TRAIN = HERE.parent / "row1_obfuscation_prompt_injection" / "deepset_prompt_injections_train.jsonl"
COMBO = "combo_sanitize_grounding_refuse"


def jobs(py: str):
    yield ("deepset_train_gate", [py, "run_deepset_gate_eval.py", "--data", str(TRAIN), "--tag", "deepset_gate_train"],
           EE / "deepset_gate_train_rows.jsonl", 546)
    yield ("graded_heldout", [py, "collect_graded_heldout_benign.py", "--n", "200"],
           EE / "graded_heldout_benign.jsonl", 200)
    yield ("deepset_e2e", [py, "run_full_pipeline.py", "--slice", str(EE / "deepset_e2e_slice.jsonl"),
                           "--out", str(EE / "deepset_e2e.jsonl"), "--resume"], EE / "deepset_e2e.jsonl", 60)
    for gen in ("fused", "isolate"):
        yield (f"semantic40_{gen}", [py, "run_full_pipeline.py", "--slice", "semantic_slice_non_yes_no_40.jsonl",
                                     "--generator", gen, "--out", str(EE / f"semantic40_{gen}.jsonl"), "--resume"],
               EE / f"semantic40_{gen}.jsonl", 40)
    base = [py, "run_mitigation_ab.py", "--slice", "planted_attacks_extra224.jsonl", "--detector", "existing",
            "--run-dir", str(EXT)]
    yield ("ext_off", base + ["--off-only"], EXT / "off.jsonl", 224)
    yield ("ext_full", base + ["--skip-off", "--label", "full"], EXT / "on_full.jsonl", 224)
    yield ("ext_combo", base + ["--skip-off", "--label", COMBO, "--layers", "sanitize,grounding,refuse"],
           EXT / f"on_{COMBO}.jsonl", 224)


def nrows(p: Path) -> int:
    return sum(1 for l in p.open(encoding="utf-8") if l.strip()) if p.exists() else 0


def main() -> None:
    EE.mkdir(exist_ok=True)
    EXT.mkdir(parents=True, exist_ok=True)
    for name, cmd, out, n in jobs(sys.executable):
        if nrows(out) >= n:
            print(f"[skip] {name}")
            continue
        t0 = time.time()
        print(f"[run ] {name}", flush=True)
        with (EE / f"queue_{name}_log.txt").open("w", encoding="utf-8") as fh:
            rc = subprocess.call(cmd, cwd=str(HERE), stdout=fh, stderr=subprocess.STDOUT)
        msg = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {name} rc={rc} rows={nrows(out)}/{n} {time.time() - t0:.0f}s"
        print(msg, flush=True)
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(msg + "\n")
        if rc != 0:
            print(f"[fail] {name}; continuing with the next job")
    print("Queue finished.")


if __name__ == "__main__":
    main()
