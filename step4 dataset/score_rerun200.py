#!/usr/bin/env python3
"""
score_rerun200.py -- fresh model rerun of the core Protocol B comparison, against the saved runs.

The undefended and full-stack arms on the 200 planted attacks were re-generated from the raw
slice (planted_attacks_200.jsonl, seed 42, current code) into mitigation_results/rerun200/.
This compares them with the saved arms the paper reports:

    saved undefended : mitigation_results/followup200/off.jsonl
    saved full stack : mitigation_results/ablation200/on_full.jsonl

It reports, per arm, ASR in both runs, row-level agreement, and the exact McNemar test between
the runs (drift), and, within the fresh run, full stack vs. undefended (the paper's comparison).

    python score_rerun200.py
Output: mitigation_results/rerun200/rerun_report.json
"""
from __future__ import annotations

import json
from pathlib import Path

from score_ablation_study import attack_succeeded, ci, exact_p

HERE = Path(__file__).resolve().parent
MR = HERE / "mitigation_results"
RR = MR / "rerun200"


def rows(p: Path) -> dict:
    return {r["index"]: r for r in (json.loads(l) for l in p.open(encoding="utf-8") if l.strip())}


def paired(x: dict, y: dict) -> dict:
    """x, y: {id: success}. fixed = rows y fixes relative to x; broke = rows y breaks."""
    ids = sorted(x.keys() & y.keys())
    fixed = sum(x[i] and not y[i] for i in ids)
    broke = sum(y[i] and not x[i] for i in ids)
    return {"n": len(ids), "fixed": fixed, "broke": broke, "p": exact_p(fixed, broke),
            "same_outcome": sum(x[i] == y[i] for i in ids)}


def main() -> None:
    saved = {"off": rows(MR / "followup200" / "off.jsonl"), "full": rows(MR / "ablation200" / "on_full.jsonl")}
    fresh = {"off": rows(RR / "off.jsonl"), "full": rows(RR / "on_full.jsonl")}
    S = {k: {i: attack_succeeded(r) for i, r in v.items()} for k, v in saved.items()}
    F = {k: {i: attack_succeeded(r) for i, r in v.items()} for k, v in fresh.items()}
    rep = {"code_revision": (RR / "code_revision.txt").read_text().strip()
           if (RR / "code_revision.txt").exists() else None}
    for arm in ("off", "full"):
        rep[arm] = {"saved_asr": ci(sum(S[arm].values()), len(S[arm])),
                    "fresh_asr": ci(sum(F[arm].values()), len(F[arm])),
                    "saved_k": sum(S[arm].values()), "fresh_k": sum(F[arm].values()),
                    "drift_fresh_vs_saved": paired(S[arm], F[arm])}
    rep["fresh_full_vs_off"] = paired(F["off"], F["full"])
    rep["saved_full_vs_off"] = paired(S["off"], S["full"])
    (RR / "rerun_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
