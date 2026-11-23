#!/usr/bin/env python3
"""
fit_gate_graded.py -- can the input gate recover missed injections from the graded DeBERTa margin?

Finding behind this script: on the deepset test split, 31 of the 32 injections the gate lets
through have a scanner score s_I of exactly 0.0, so no change of the fusion thresholds can catch
them (they take the fast ALLOW path). Stage 3 already computes a graded channel -- the DeBERTa
logit margin squashed with temperature T (graded_channels.py) -- for the multivector detector.
This script tests a within-architecture change: let the gate also fire on that graded score,

    flag  if  fusion decision != ALLOW  or  graded(prompt) >= t,

choosing t on the deepset TRAIN split only (largest recall while the graded flag adds at most
the budget of benign false positives beyond the gate's own), then reporting it on the deepset TEST split and on two benign sets the threshold never
saw: 200 rag-mini-wikipedia questions outside the evaluation slices (the pipeline's real benign
traffic) and the 80 Protocol A benign controls.

Inputs : external_eval/deepset_gate_rows.jsonl, external_eval/deepset_gate_train_rows.jsonl
         (run_deepset_gate_eval.py), external_eval/graded_heldout_benign.jsonl
         (collect_graded_heldout_benign.py), grounded_controller_graded.jsonl
Output : external_eval/gate_graded_report.json

    python fit_gate_graded.py [--fp-budget 0.02]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(1, str((HERE.parent / "dataset").resolve()))
from score_ablation_study import ci  # noqa: E402

EE = HERE / "external_eval"


def rows(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def add_graded(rs: list[dict], key: str = "text") -> None:
    from graded_channels import graded_score
    for r in rs:
        if "graded" not in r:
            r["graded"] = round(float(graded_score(r[key])), 4)


def rate(rs, pred) -> tuple[int, int]:
    return sum(bool(pred(r)) for r in rs), len(rs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fp-budget", type=float, default=0.02,
                    help="max ADDED benign flag rate (graded flags on rows the gate allows) on the deepset "
                         "train split when choosing t; the gate alone already flags 10/343 there")
    args = ap.parse_args()

    test = rows(EE / "deepset_gate_rows.jsonl")
    train = rows(EE / "deepset_gate_train_rows.jsonl")
    qa_benign = rows(EE / "graded_heldout_benign.jsonl")
    for r in qa_benign:
        r["graded"] = r["channels"]["query_vector"]
    pa = [r for r in rows(HERE / "grounded_controller_graded.jsonl") if r.get("kind") == "benign_control"]
    for r in pa:
        r["graded"] = float(r["gate"].get("injection_graded") or 0.0)
    print("[graded] scoring deepset train/test prompts on CPU ...", flush=True)
    add_graded(train)
    add_graded(test)

    gate_flag = lambda r: r["decision"] != "ALLOW"  # noqa: E731

    # choose t on TRAIN: maximise injection recall subject to the graded flag adding at most
    # `budget` benign false positives beyond those the gate already makes
    tr_inj = [r for r in train if r["label"] == 1]
    tr_ben = [r for r in train if r["label"] == 0]
    best = None
    for t in sorted({r["graded"] for r in train} | {1.01}):
        fp = sum(gate_flag(r) or r["graded"] >= t for r in tr_ben) / len(tr_ben)
        added = sum(not gate_flag(r) and r["graded"] >= t for r in tr_ben) / len(tr_ben)
        if added > args.fp_budget:
            continue
        rec = sum(gate_flag(r) or r["graded"] >= t for r in tr_inj) / len(tr_inj)
        if best is None or rec > best[1] or (rec == best[1] and t > best[0]):
            best = (t, rec, fp)
    t = best[0]
    combined = lambda r: gate_flag(r) or r["graded"] >= t  # noqa: E731

    def block(name, rs_inj, rs_ben):
        out = {}
        for lab, fn in (("gate_only", gate_flag), ("gate_plus_graded", combined)):
            a = rate(rs_inj, fn)
            b = rate(rs_ben, fn)
            out[lab] = {"injection_flagged": f"{a[0]}/{a[1]}", "injection_ci": ci(*a),
                        "benign_flagged": f"{b[0]}/{b[1]}", "benign_ci": ci(*b)}
        return out

    te_inj = [r for r in test if r["label"] == 1]
    te_ben = [r for r in test if r["label"] == 0]
    sm_inj = [r for r in te_inj if r["in_paper_sample"]]
    sm_ben = [r for r in te_ben if r["in_paper_sample"]]
    report = {
        "threshold_t": t, "chosen_on": "deepset train split", "fp_budget": args.fp_budget,
        "train": {"recall": round(best[1], 4), "benign_fp": round(best[2], 4),
                  "gate_only_recall": round(sum(map(gate_flag, tr_inj)) / len(tr_inj), 4),
                  "gate_only_benign_fp": round(sum(map(gate_flag, tr_ben)) / len(tr_ben), 4),
                  "n_injection": len(tr_inj), "n_benign": len(tr_ben)},
        "test_paper_sample_30_30": block("sample", sm_inj, sm_ben),
        "test_full_split": block("full", te_inj, te_ben),
        "rag_benign_questions": {k: rate(qa_benign, fn) for k, fn in
                                 (("graded_only_flags", lambda r: r["graded"] >= t),)},
        "protocolA_benign_controls": {k: rate(pa, fn) for k, fn in
                                      (("graded_only_flags", lambda r: r["graded"] >= t),)},
        "note": ("RAG and Protocol A benign rows report only the added graded flags; the existing gate "
                 "already flags 0/80 Protocol A benign rows."),
    }
    (EE / "gate_graded_report.json").write_text(json.dumps(report, indent=2))
    with (EE / "deepset_graded_scores.jsonl").open("w", encoding="utf-8") as fh:
        for split, rs in (("train", train), ("test", test)):
            for r in rs:
                fh.write(json.dumps({"split": split, "index": r["index"], "label": r["label"],
                                     "decision": r["decision"], "graded": r["graded"]}) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
