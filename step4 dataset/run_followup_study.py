#!/usr/bin/env python3
"""
run_followup_study.py -- follow-up to the 30-run ablation: (1) fresh undefended arms on the
CURRENT code, (2) layer combinations on the CURRENT code.

Why: in ablation200 the undefended arm was reused from July (older code). Two layers that cannot
change a yes/no answer (Presidio-only, DLP-only) still differed from it on 16/200 rows, so
comparisons against it are only reliable for large effects. Also, no combination of layers was run,
so "do seven layers beat the best two?" is unanswered.

Run from `step4 dataset` with the project venv (.venv311) and `ollama serve` running:
    .\.venv311\Scripts\python.exe run_followup_study.py --dry-run
    .\.venv311\Scripts\python.exe run_followup_study.py --tier all
    .\.venv311\Scripts\python.exe score_followup_study.py

Resumable at run granularity: a run whose output file already has 200 rows is skipped; a run that
was interrupted restarts from row 1 (the runners open their output with "w"). Existing
mitigation_results/ablation200 results are never touched. Output: mitigation_results/followup200/.

Tiers (in order):
  1  undefended attack + undefended benign on current code      2 runs  (~1.5 h)
  2  attack, layer combinations                                  4 runs  (~3 h)
  3  benign, layer combinations                                  4 runs  (~4 h)
  4  non-yes/no slice (semantic_slice_non_yes_no_40.jsonl, 40 rows): undefended + 9 configs
     on the attack arm only (~1.2 h). Tests whether the layer ranking in ablation200 survives
     outside yes/no flips. The slice is admitted only if the real scanners pass and do not
     redact it, so sanitization cannot act on it by construction (`only_sanitize` is a
     no-op control). With --tier all, tier 4 runs right after tier 1 so its result arrives early.
Combinations run most-informative first:
  sanitize+grounding+refuse   (= full stack minus prompt, tighten, presidio, dlp)
  grounding+refuse            (the old best two-layer pair)
  sanitize+grounding
  sanitize+refuse
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "mitigation_results" / "followup200"
ATTACK_SLICE = "planted_attacks_200.jsonl"
QA = "data/question-answer/test.jsonl"
N = 200
BASE_URL = "http://localhost:11434"

NY_SRC = "semantic_slice_non_yes_no_40.jsonl"          # built by build_semantic_slice.py
NY_SLICE = "semantic_slice_non_yes_no_40.mv.jsonl"       # same rows, kind -> multivector_attack
NY_DIR = HERE / "mitigation_results" / "followup_nonyesno40"
NY_N = 40
NY_CONFIGS = [("full", []), ("noprompt", ["--no-prompt"]), ("norefuse", ["--no-refuse"]),
              ("nogrounding", ["--no-grounding"]), ("only_grounding", ["--only", "grounding"]),
              ("only_refuse", ["--only", "refuse"]), ("only_prompt", ["--only", "prompt"]),
              ("only_sanitize", ["--only", "sanitize"]),
              ("combo_sanitize_grounding_refuse", ["--layers", "sanitize,grounding,refuse"])]

COMBOS = ["sanitize,grounding,refuse", "grounding,refuse", "sanitize,grounding", "sanitize,refuse"]
def combo_label(layers): return "combo_" + "_".join(layers.split(","))

SEC_ATTACK, SEC_BENIGN = 11.0, 16.0   # earlier logged runs: attack 6-13 s/row, benign 15-19 s/row


def rows_in(p: Path) -> int:
    if not p.is_file():
        return 0
    with open(p, encoding="utf-8", errors="replace") as f:
        return sum(1 for l in f if l.strip())


def prepare_nonyesno_slice():
    """run_mitigation_ab.py only processes kind in (None, 'multivector_attack'); the new slice's rows are
    'semantic_multivector'. Write a copy with the kind rewritten; every other field is unchanged."""
    src, dst = HERE / NY_SRC, HERE / NY_SLICE
    if not src.is_file():
        raise SystemExit(f"missing {src}; run build_semantic_slice.py first")
    rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    if len(rows) != NY_N:
        raise SystemExit(f"{src.name} has {len(rows)} rows, expected {NY_N}")
    for r in rows:
        r["kind"] = "multivector_attack"
    with open(dst, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def jobs(tier):
    """(kind, label, cmd, output_path, target_rows)"""
    py = sys.executable
    rd = str(RUN_DIR)
    out = []
    def t1():
        out.append(("attack", "off", [py, "run_mitigation_ab.py", "--slice", ATTACK_SLICE, "--detector", "existing",
                                      "--run-dir", rd, "--off-only"], RUN_DIR / "off.jsonl", N))
        out.append(("benign", "off", [py, "run_benign_cost.py", "--qa", QA, "--n", str(N), "--seed", "42",
                                      "--run-dir", rd, "--off-only"], RUN_DIR / "benign_off.jsonl", N))
    def t4():
        nd = str(NY_DIR)
        out.append(("attack", "ny40_off", [py, "run_mitigation_ab.py", "--slice", NY_SLICE, "--detector", "existing",
                                           "--run-dir", nd, "--off-only"], NY_DIR / "off.jsonl", NY_N))
        for lab, fl in NY_CONFIGS:
            out.append(("attack", f"ny40_{lab}", [py, "run_mitigation_ab.py", "--slice", NY_SLICE, "--detector", "existing",
                                                  "--run-dir", nd, "--skip-off", "--label", lab] + fl,
                        NY_DIR / f"on_{lab}.jsonl", NY_N))
    def t2():
        for c in COMBOS:
            lab = combo_label(c)
            out.append(("attack", lab, [py, "run_mitigation_ab.py", "--slice", ATTACK_SLICE, "--detector", "existing",
                                        "--run-dir", rd, "--skip-off", "--label", lab, "--layers", c],
                        RUN_DIR / f"on_{lab}.jsonl", N))
    def t3():
        for c in COMBOS:
            lab = combo_label(c)
            out.append(("benign", lab, [py, "run_benign_cost.py", "--qa", QA, "--n", str(N), "--seed", "42",
                                        "--run-dir", rd, "--skip-off", "--label", lab, "--layers", c],
                        RUN_DIR / f"benign_on_{lab}.jsonl", N))
    if tier in ("1", "all"): t1()
    if tier in ("4", "all"): t4()
    if tier in ("2", "all"): t2()
    if tier in ("3", "all"): t3()
    return out


def preflight():
    try:
        with urllib.request.urlopen(BASE_URL + "/api/tags", timeout=5) as r:
            have = sorted(m["name"] for m in json.load(r).get("models", []))
    except Exception as e:
        print(f"[WARN] Ollama not reachable at {BASE_URL} ({e}). Start `ollama serve` first.")
        return False
    print(f"[ok] Ollama reachable; models: {have}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["1", "2", "3", "4", "all"], default="all")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    todo = []
    for kind, label, cmd, out, target in jobs(args.tier):
        if rows_in(out) >= target:
            print(f"[skip] {kind:6} {label:34} already complete ({out.name})")
        else:
            todo.append((kind, label, cmd, out, target))
    est = sum(t * (SEC_ATTACK if k == "attack" else SEC_BENIGN) for k, _, _, _, t in todo) / 3600
    print(f"\n{len(todo)} runs to do; estimate ~{est:.1f} h\n")
    for _, _, cmd, _, _ in todo:
        print("  " + " ".join(cmd))
    if args.dry_run or not todo:
        return
    if any(l.startswith("ny40_") for _, l, *_ in todo):
        prepare_nonyesno_slice()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    preflight()
    log = RUN_DIR / "followup_progress.log"
    NY_DIR.mkdir(parents=True, exist_ok=True)
    for i, (kind, label, cmd, out, target) in enumerate(todo, 1):
        # runs that reuse an undefended arm need that arm complete first
        if label.startswith("ny40_"):
            if label != "ny40_off" and rows_in(NY_DIR / "off.jsonl") < NY_N:
                print(f"[stop] {NY_DIR.name}/off.jsonl is incomplete; run --tier 4 again."); break
        elif label != "off":
            need = RUN_DIR / ("off.jsonl" if kind == "attack" else "benign_off.jsonl")
            if rows_in(need) < N:
                print(f"[stop] {need.name} is incomplete ({rows_in(need)}/{N}); run --tier 1 first.")
                break
        print(f"\n=== [{i}/{len(todo)}] {kind} {label} ===", flush=True)
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=str(HERE))
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"kind": kind, "label": label, "rc": rc,
                                "minutes": round((time.time() - t0) / 60, 1), "rows": rows_in(out), "target": target}) + "\n")
        if rc != 0:
            print(f"[warn] {kind} {label} exited {rc}; continuing (rerun to resume)")
    print("\nDone. Now run:  python score_followup_study.py")


if __name__ == "__main__":
    main()
