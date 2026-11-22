#!/usr/bin/env python3
"""
run_seed_study.py -- re-run the headline Protocol B arms under extra generation seeds.

For each seed in SEEDS, runs six arms into mitigation_results/seed<SEED>/:
  attack: undefended, full stack, sanitize+grounding+refuse   (planted_attacks_200.jsonl)
  benign: undefended, full stack, sanitize+grounding+refuse   (200 QA rows, row-sampling seed fixed at 42)
The generation seed is passed through RAG_GEN_SEED (step_09_generator_llm.py). Seed 42 is the
existing ablation200/followup200 run, so it is not re-run here.

Resumable: an arm whose output already has N rows is skipped.

    .venv311\\Scripts\\python.exe run_seed_study.py            # run everything
    .venv311\\Scripts\\python.exe run_seed_study.py --dry-run
Then: python score_seed_study.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = [7, 1234]
ATTACK_SLICE = "planted_attacks_200.jsonl"
QA = "data/question-answer/test.jsonl"
N = 200
COMBO = ("combo_sanitize_grounding_refuse", ["--layers", "sanitize,grounding,refuse"])


def jobs(py: str):
    for seed in SEEDS:
        rd = HERE / "mitigation_results" / f"seed{seed}"
        r = str(rd)
        base_a = [py, "run_mitigation_ab.py", "--slice", ATTACK_SLICE, "--detector", "existing", "--run-dir", r]
        base_b = [py, "run_benign_cost.py", "--qa", QA, "--n", str(N), "--seed", "42", "--run-dir", r]
        yield seed, "attack_off", base_a + ["--off-only"], rd / "off.jsonl"
        yield seed, "benign_off", base_b + ["--off-only"], rd / "benign_off.jsonl"
        yield seed, "attack_full", base_a + ["--skip-off", "--label", "full"], rd / "on_full.jsonl"
        yield seed, "benign_full", base_b + ["--skip-off", "--label", "full"], rd / "benign_on.jsonl"
        lab, fl = COMBO
        yield seed, "attack_combo", base_a + ["--skip-off", "--label", lab] + fl, rd / f"on_{lab}.jsonl"
        yield seed, "benign_combo", base_b + ["--skip-off", "--label", lab] + fl, rd / f"benign_on_{lab}.jsonl"


def nrows(p: Path) -> int:
    return sum(1 for l in p.open(encoding="utf-8") if l.strip()) if p.exists() else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    log = HERE / "mitigation_results" / "seed_study_progress.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    for seed, name, cmd, out in jobs(sys.executable):
        out.parent.mkdir(parents=True, exist_ok=True)
        if nrows(out) >= N:
            print(f"[skip] seed={seed} {name} (complete)")
            continue
        print(f"[run ] seed={seed} {name}: {' '.join(cmd[1:])}")
        if args.dry_run:
            continue
        t0 = time.time()
        env = dict(os.environ, RAG_GEN_SEED=str(seed))
        with (out.parent / f"{name}_log.txt").open("w", encoding="utf-8") as fh:
            rc = subprocess.call(cmd, cwd=str(HERE), env=env, stdout=fh, stderr=subprocess.STDOUT)
        msg = f"{time.strftime('%H:%M:%S')} seed={seed} {name} rc={rc} rows={nrows(out)} {time.time() - t0:.0f}s"
        print(msg, flush=True)
        with log.open("a", encoding="utf-8") as fh:
            fh.write(msg + "\n")
        if rc != 0:
            print(f"[fail] {name} exited {rc}; see {out.parent / (name + '_log.txt')}")
            sys.exit(rc)
    print("Done. Now run: python score_seed_study.py")


if __name__ == "__main__":
    main()
