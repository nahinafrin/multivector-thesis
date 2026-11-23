#!/usr/bin/env python3
"""
collect_graded_heldout_benign.py -- graded multivector channels on held-out benign questions.

The graded multivector floors in the paper (query 0.228, context 0.30) were set from benign
rows of the Protocol A slice, so the 1/80 benign false-fire rate is in-sample. This script
collects the same two graded channels on benign questions that are NOT in the Protocol A slice,
so the floors can be refit out of sample (fit_graded_floors_heldout.py).

Per question it runs the pipeline's own Steps 1, 2, 4, 5, 6 (normalization, embedding, FAISS
retrieval, per-chunk sanitization; no LLM call) and reads the channels exactly as the
full pipeline does: input_channels(state) followed by the stored channel calibration.

    python collect_graded_heldout_benign.py --n 200
Output: external_eval/graded_heldout_benign.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import run_full_pipeline as rfp  # sets sys.path for ../dataset and loads the stage modules
from pipeline_common import PipelineState

HERE = Path(__file__).resolve().parent
OUT = HERE / "external_eval" / "graded_heldout_benign.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qa", default="data/question-answer/test.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    used = set()
    for f in ("adversarial_slice.jsonl", "planted_attacks_200.jsonl", "semantic_slice_non_yes_no_40.jsonl"):
        used |= {json.loads(l)["question"] for l in open(HERE / f, encoding="utf-8") if l.strip()}
    qa = [json.loads(l) for l in open(HERE / args.qa, encoding="utf-8") if l.strip()]
    pool = sorted({q["question"] for q in qa} - used)
    sample = random.Random(args.seed).sample(pool, min(args.n, len(pool)))
    print(f"[pool] {len(pool)} benign questions outside the evaluation slices; sampling {len(sample)}")

    rfp.ensure_index()
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for i, q in enumerate(sample, 1):
            st = PipelineState(prompt=q, raw_prompt=q)
            st = rfp.s1.run(st)
            st = rfp.s2.run(st)
            try:
                from graded_channels import graded_score
                st.scores["injection_graded"] = graded_score(st.prompt)
            except Exception as e:  # mirror step_03c: graded channel is optional
                print("graded_score failed:", e)
            st = rfp.s4.run(st)
            st = rfp.s5.run(st, k=args.k)
            st = rfp.s6.run(st)
            raw = rfp.input_channels(st)
            ch = rfp._CALIBRATION.apply_channels(raw)
            rec = {"index": i - 1, "question": q, "channels": {c: round(float(v), 4) for c, v in ch.items()},
                   "raw_channels": {c: round(float(v), 4) for c, v in raw.items()},
                   "context_scale": st.meta.get("context_scale")}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[{i:3d}/{len(sample)}] q={rec['channels'].get('query_vector')} c={rec['channels'].get('context_vector')}",
                  flush=True)
    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
