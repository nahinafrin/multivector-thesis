#!/usr/bin/env python3
"""
score_seed_study.py -- run-to-run variability of the headline Protocol B results.

Seed 42 = the existing arms used in the paper (followup200 undefended + ablation200 full stack +
followup200 three-layer subset). Seeds 7 and 1234 come from run_seed_study.py. Note that Ollama
GPU generation is not bit-reproducible even at a fixed seed, so each run is best read as an
independent replicate of the generation step on the same rows.

Per replicate: ASR (Wilson CI), benign blocked, benign correct, and paired exact McNemar for
full vs undefended and subset vs full. Across replicates: mean and SD, plus per-row stability.

    python score_seed_study.py
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from score_ablation_study import attack_succeeded, answered_correctly, rows, ci, exact_p

HERE = Path(__file__).resolve().parent
MR = HERE / "mitigation_results"
COMBO = "combo_sanitize_grounding_refuse"
REPLICATES = {
    42: {"a_off": MR / "followup200/off.jsonl", "b_off": MR / "followup200/benign_off.jsonl",
         "a_full": MR / "ablation200/on_full.jsonl", "b_full": MR / "ablation200/benign_on.jsonl",
         "a_combo": MR / f"followup200/on_{COMBO}.jsonl", "b_combo": MR / f"followup200/benign_on_{COMBO}.jsonl"},
}
for s in (7, 1234):
    d = MR / f"seed{s}"
    REPLICATES[s] = {"a_off": d / "off.jsonl", "b_off": d / "benign_off.jsonl",
                     "a_full": d / "on_full.jsonl", "b_full": d / "benign_on.jsonl",
                     "a_combo": d / f"on_{COMBO}.jsonl", "b_combo": d / f"benign_on_{COMBO}.jsonl"}


def flags(R, fn):
    return {k: bool(fn(r)) for k, r in R.items()}


def paired(x, y):
    """x, y: {index: bool success}. Returns (y fixes vs x, y breaks vs x, p)."""
    keys = x.keys() & y.keys()
    b = sum(x[k] and not y[k] for k in keys)
    c = sum(y[k] and not x[k] for k in keys)
    return b, c, exact_p(b, c)


def main() -> None:
    out, per = {}, {}
    print(f"{'seed':>5s}  {'ASR off':>20s} {'ASR full':>20s} {'ASR subset':>20s}  full vs off      subset vs full   "
          f"benign blk full/subset   benign correct off/full")
    for seed, paths in REPLICATES.items():
        R = {k: rows(p) for k, p in paths.items()}
        missing = [k for k, v in R.items() if not v or len(v) < 200]
        if missing:
            print(f"{seed:5d}  incomplete: {missing}")
            continue
        A = {k: flags(R[k], attack_succeeded) for k in ("a_off", "a_full", "a_combo")}
        blk = {k: sum(bool(r.get("blocked")) for r in R[k].values()) for k in ("b_full", "b_combo")}
        ok = {k: sum(bool(answered_correctly(r)) for r in R[k].values()) for k in ("b_off", "b_full")}
        asr = {k: sum(v.values()) for k, v in A.items()}
        fo = paired(A["a_off"], A["a_full"])
        cf = paired(A["a_full"], A["a_combo"])  # b = combo better than full, c = combo worse
        per[seed] = A
        out[seed] = {"asr": asr, "full_vs_off": fo, "combo_vs_full": cf, "benign_blocked": blk, "benign_correct": ok}
        print(f"{seed:5d}  {ci(asr['a_off'], 200):>20s} {ci(asr['a_full'], 200):>20s} {ci(asr['a_combo'], 200):>20s}  "
              f"{fo[0]:3d}/{fo[1]:<2d} p={fo[2]:.1e}  {cf[1]:2d} worse/{cf[0]:<2d} p={cf[2]:.2f}   "
              f"{blk['b_full']:3d} / {blk['b_combo']:<3d}              {ok['b_off']} / {ok['b_full']}")

    if len(out) >= 2:
        print("\nAcross replicates (mean +- SD, in %):")
        for k, lab in (("a_off", "ASR undefended"), ("a_full", "ASR full stack"), ("a_combo", "ASR subset")):
            v = [100 * o["asr"][k] / 200 for o in out.values()]
            print(f"  {lab:18s} {st.mean(v):5.1f} +- {st.stdev(v):4.1f}   ({', '.join(f'{x:.1f}' for x in v)})")
        for k, lab in (("b_full", "benign blk full"), ("b_combo", "benign blk subset")):
            v = [100 * o["benign_blocked"][k] / 200 for o in out.values()]
            print(f"  {lab:18s} {st.mean(v):5.1f} +- {st.stdev(v):4.1f}   ({', '.join(f'{x:.1f}' for x in v)})")
        gaps = [100 * (o["asr"]["a_combo"] - o["asr"]["a_full"]) / 200 for o in out.values()]
        pvals = ", ".join("%.2f" % o["combo_vs_full"][2] for o in out.values())
        print(f"  subset - full gap  {st.mean(gaps):5.1f} +- {st.stdev(gaps):4.1f} points; "
              f"subset-vs-full p per replicate: {pvals}")
        for k, lab in (("a_off", "undefended"), ("a_full", "full stack")):
            seeds = list(per)
            keys = set.intersection(*(set(per[s][k]) for s in seeds))
            always = sum(all(per[s][k][i] for s in seeds) for i in keys)
            ever = sum(any(per[s][k][i] for s in seeds) for i in keys)
            print(f"  rows succeeding under {lab}: in every replicate {always}, in at least one {ever}")
    (MR / "seed_study_report.json").write_text(json.dumps({str(k): v for k, v in out.items()}, indent=2))
    print(f"\n[saved] {MR / 'seed_study_report.json'}")


if __name__ == "__main__":
    main()
