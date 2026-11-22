#!/usr/bin/env python3
"""
run_ablation_study.py -- per-layer ablation of the mitigation stack on planted-200 + 200 benign.

Run from `step4 dataset` with the project venv active (.venv311) and `ollama serve` running.
Resumable: a run whose output already has all rows is skipped, so Ctrl-C and restart are safe.
Each finished run is appended to mitigation_results/ablation200/ablation_progress.log.

Seven mitigation layers: sanitize, tighten, prompt, grounding, presidio, dlp, refuse.
  leave-one-out : FULL stack (all 7) minus ONE layer -> marginal value of that layer
  add-one-in    : exactly ONE layer on                -> standalone value of that layer
Everything is run under the CURRENT code (mitigation_pipeline.py and the fixed scorer), so
results are comparable with each other. Older runs in planted200/ predate the Sep-22 fixes
and the presidio layer; score_ablation_study.py lists them separately as cross-checks.

The undefended (OFF) arms are reused, not rerun:
  attack OFF : mitigation_results/planted200/off.jsonl
  benign OFF : mitigation_results/planted200/benign_off.jsonl   (seed 42, n=200)

Tiers (in this order, so partial results are useful early):
  1  attack, leave-one-out + full       8 runs x 200 rows  (~4.5 h)
  2  benign, leave-one-out + full       8 runs x 200 rows  (~4.5 h)
  3  attack, add-one-in                 7 runs x 200 rows  (~4 h)
  4  benign, add-one-in                 7 runs x 200 rows  (~4 h)
`noprompt` runs first: it tests whether the guarded prompt is what drives the ~45% benign
blocking of the full stack (suspected terse-polar-answer blind spot in the grounding judge).

USAGE
  python run_ablation_study.py --dry-run      # list commands + time estimate, runs nothing
  python run_ablation_study.py --tier 1
  python run_ablation_study.py --tier all
  python score_ablation_study.py              # table with CIs; partial results are fine
"""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "mitigation_results" / "ablation200"
ATTACK_SLICE = "planted_attacks_200.jsonl"
QA = "data/question-answer/test.jsonl"
N = 200
BASE_URL = "http://localhost:11434"
SRC_OFF_ATTACK = HERE / "mitigation_results" / "planted200" / "off.jsonl"
SRC_OFF_BENIGN = HERE / "mitigation_results" / "planted200" / "benign_off.jsonl"

LOO = [("noprompt", ["--no-prompt"]), ("full", []), ("norefuse", ["--no-refuse"]),
       ("nogrounding", ["--no-grounding"]), ("nosanitize", ["--no-sanitize"]),
       ("notighten", ["--no-tighten"]), ("nopresidio", ["--no-presidio"]), ("nodlp", ["--no-dlp"])]
ONLY = [(f"only_{x}", ["--only", x]) for x in
        ("refuse", "grounding", "sanitize", "tighten", "prompt", "presidio", "dlp")]

SEC_PER_ROW = 11.0   # earlier logged runs: 8-13 s/row


def attack_out(label): return RUN_DIR / f"on_{label}.jsonl"
def benign_out(label): return RUN_DIR / ("benign_on.jsonl" if label == "full" else f"benign_on_{label}.jsonl")


def rows_in(p: Path) -> int:
    if not p.is_file():
        return 0
    with open(p, encoding="utf-8", errors="replace") as f:
        return sum(1 for l in f if l.strip())


def jobs(tier):
    out = []
    if tier in ("1", "all"): out += [("attack", l, f) for l, f in LOO]
    if tier in ("2", "all"): out += [("benign", l, f) for l, f in LOO]
    if tier in ("3", "all"): out += [("attack", l, f) for l, f in ONLY]
    if tier in ("4", "all"): out += [("benign", l, f) for l, f in ONLY]
    return out


def command(kind, label, flags):
    py = sys.executable
    if kind == "attack":
        return [py, "run_mitigation_ab.py", "--slice", ATTACK_SLICE, "--detector", "existing",
                "--run-dir", str(RUN_DIR), "--skip-off", "--label", label] + flags
    return [py, "run_benign_cost.py", "--qa", QA, "--n", str(N), "--seed", "42",
            "--run-dir", str(RUN_DIR), "--skip-off", "--label", label] + flags


def preflight():
    """Warn (do not fail) if Ollama is unreachable or a needed model is not pulled."""
    try:
        with urllib.request.urlopen(BASE_URL + "/api/tags", timeout=5) as r:
            have = {m["name"] for m in json.load(r).get("models", [])}
    except Exception as e:
        print(f"[WARN] Ollama not reachable at {BASE_URL} ({e}). Start `ollama serve` first.")
        return False
    need = ["llama3.2:3b", "qwen2.5:3b", "llama-guard3:1b"]
    miss = [m for m in need if not any(h == m or h.startswith(m.split(':')[0] + ':') and m in h for h in have)]
    print(f"[ok] Ollama reachable; models: {sorted(have)}")
    if miss:
        print(f"[WARN] these usually-needed models were not found: {miss}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["1", "2", "3", "4", "all"], default="1")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for src in (SRC_OFF_ATTACK, SRC_OFF_BENIGN):
        if not src.is_file():
            raise SystemExit(f"missing undefended arm: {src}")
    todo = []
    for kind, label, flags in jobs(args.tier):
        out = attack_out(label) if kind == "attack" else benign_out(label)
        if rows_in(out) >= N:
            print(f"[skip] {kind:6} {label:12} already complete ({out.name})")
            continue
        todo.append((kind, label, flags, out))
    print(f"\n{len(todo)} runs to do; estimate ~{len(todo) * N * SEC_PER_ROW / 3600:.1f} h at {SEC_PER_ROW:.0f} s/row\n")
    for kind, label, flags, _ in todo:
        print("  " + " ".join(command(kind, label, flags)))
    if args.dry_run or not todo:
        return

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    for src, dst in [(SRC_OFF_ATTACK, RUN_DIR / "off.jsonl"), (SRC_OFF_BENIGN, RUN_DIR / "benign_off.jsonl")]:
        if not dst.is_file():
            shutil.copy2(src, dst)
    preflight()
    log = RUN_DIR / "ablation_progress.log"
    for i, (kind, label, flags, out) in enumerate(todo, 1):
        print(f"\n=== [{i}/{len(todo)}] {kind} {label} ===", flush=True)
        t0 = time.time()
        rc = subprocess.call(command(kind, label, flags), cwd=str(HERE))
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"kind": kind, "label": label, "rc": rc,
                                "minutes": round((time.time() - t0) / 60, 1), "rows": rows_in(out)}) + "\n")
        if rc != 0:
            print(f"[warn] {kind} {label} exited {rc}; continuing (rerun to resume)")
    print("\nDone. Now run:  python score_ablation_study.py")


if __name__ == "__main__":
    main()
