#!/usr/bin/env python3
"""
run_deepset_gate_eval.py -- input-gate test on deepset/prompt-injections (paper Table VIII).

Runs Steps 1, 2 and 3c (normalization + fusion gate, default thresholds tau_review=0.35,
tau_block=0.60) on prompts from the held-out *test* split of deepset/prompt-injections,
which was not used to calibrate the gate. Two samples are scored:

  * paper sample: 30 injection (label 1) + 30 benign (label 0) rows, drawn with
    random.Random(42) from the test split;
  * full test split: all 116 rows (60 injection, 56 benign), as a robustness check.

A prompt counts as "caught" at two strictness levels: BLOCK only (the gate refuses it), and
BLOCK or REVIEW (the gate flags it, which raises downstream strictness). Per-row outputs go
to external_eval/deepset_gate_rows.jsonl and the summary to external_eval/deepset_gate_report.json.

Input: ../row1_obfuscation_prompt_injection/deepset_prompt_injections_test.jsonl
       (create with `python download_datasets.py` from the repository root).

    python run_deepset_gate_eval.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(1, str((HERE.parent / "dataset").resolve()))

from pipeline_common import PipelineState  # noqa: E402
import step_01_user_input as s1  # noqa: E402
import step_02_normalization as s2  # noqa: E402
import step_03c_fusion_gate as s3c  # noqa: E402
from score_ablation_study import ci  # noqa: E402  (Wilson interval formatter)

DATA = HERE.parent / "row1_obfuscation_prompt_injection" / "deepset_prompt_injections_test.jsonl"
OUT = HERE / "external_eval"


def gate(text: str) -> dict:
    st = PipelineState(prompt=text, raw_prompt=text)
    st = s1.run(st)
    st = s2.run(st)
    st = s3c.run(st)
    return {"decision": st.meta.get("fusion_decision", "BLOCK" if st.blocked else "ALLOW"),
            "fusion_risk": round(float(st.scores.get("fusion_risk", 0.0)), 4),
            "injection_score": round(float(st.scores.get("injection_detection", 0.0)), 4),
            "llamaguard_soft": round(float(st.scores.get("llamaguard_soft", 0.0)), 4),
            "llamaguard_category": st.meta.get("llamaguard_category"),
            "override": st.meta.get("fusion_override")}


def summarize(rows: list[dict]) -> dict:
    out = {}
    for lab, name in ((1, "injection"), (0, "benign")):
        sub = [r for r in rows if r["label"] == lab]
        blk = sum(r["decision"] == "BLOCK" for r in sub)
        flg = sum(r["decision"] in ("BLOCK", "REVIEW") for r in sub)
        out[name] = {"n": len(sub), "blocked": blk, "blocked_or_review": flg,
                     "blocked_ci": ci(blk, len(sub)), "blocked_or_review_ci": ci(flg, len(sub))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--n-per-class", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.data).open(encoding="utf-8") if l.strip()]
    for i, r in enumerate(rows):
        r["index"] = i
    rng = random.Random(args.seed)
    sample = set()
    for lab in (1, 0):
        idx = [r["index"] for r in rows if r["label"] == lab]
        sample.update(rng.sample(idx, args.n_per_class))

    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    with (OUT / "deepset_gate_rows.jsonl").open("w", encoding="utf-8") as fh:
        for k, r in enumerate(rows, 1):
            g = gate(r["text"])
            rec = {"index": r["index"], "label": r["label"], "in_paper_sample": r["index"] in sample,
                   "text": r["text"], **g}
            r.update(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[{k:3d}/{len(rows)}] label={r['label']} {g['decision']:6s} R={g['fusion_risk']:.3f}", flush=True)

    report = {"source": str(Path(args.data).name), "seed": args.seed, "n_per_class": args.n_per_class,
              "paper_sample": summarize([r for r in rows if r["in_paper_sample"]]),
              "full_test_split": summarize(rows), "seconds": round(time.time() - t0, 1)}
    (OUT / "deepset_gate_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
