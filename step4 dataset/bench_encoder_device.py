#!/usr/bin/env python3
"""
bench_encoder_device.py -- per-stage latency of the CPU-side encoders, on CPU vs on the GPU.

The paper's latency table attributes most of the defense overhead to the grounding judge
(cross-encoder on the CPU) and leaves "move the encoders to the GPU" untested. This script
times the encoder-bearing stages on benign questions with no LLM call, so Ollama generation
does not affect the numbers:

    Step 4  query embedding      (sentence-transformers bi-encoder)
    Step 5  FAISS retrieval      (index search; no encoder, for reference)
    Step 6  per-chunk sanitize   (DeBERTa ONNX on CPU in both settings, for reference)
    Step 7  context reranking    (cross-encoder)
    Step 10 grounding judge      (cross-encoder; judged answer = the reference answer)

Device is chosen by RAG_ENCODER_DEVICE (default "cpu"; "cuda" needs a CUDA build of PyTorch,
e.g. the .venv311gpu environment). Run the same command in both environments:

    .venv311/Scripts/python.exe bench_encoder_device.py --n 50
    set RAG_ENCODER_DEVICE=cuda && .venv311gpu/Scripts/python.exe bench_encoder_device.py --n 50

Output: external_eval/encoder_latency_{device}.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

import run_full_pipeline as rfp  # sets sys.path for ../dataset and loads the stage modules
from pipeline_common import PipelineState

HERE = Path(__file__).resolve().parent
STAGES = ("s4_embed", "s5_retrieve", "s6_sanitize", "s7_rerank", "s10_judge")


def one(q: str, answer: str, k: int) -> dict:
    t = {}
    st = PipelineState(prompt=q, raw_prompt=q)
    st = rfp.s1.run(st)
    for name, fn in (("s4_embed", lambda s: rfp.s4.run(s)),
                     ("s5_retrieve", lambda s: rfp.s5.run(s, k=k)),
                     ("s6_sanitize", lambda s: rfp.s6.run(s)),
                     ("s7_rerank", lambda s: rfp.s7.run(s, top_n=3))):
        t0 = time.perf_counter()
        st = fn(st)
        t[name] = time.perf_counter() - t0
    st.meta["answer"] = answer
    t0 = time.perf_counter()
    rfp.s10.run(st)
    t["s10_judge"] = time.perf_counter() - t0
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    device = os.environ.get("RAG_ENCODER_DEVICE", "cpu")
    import torch
    qa = {json.loads(l)["question"]: json.loads(l) for l in open(HERE / "data/question-answer/test.jsonl",
                                                                 encoding="utf-8") if l.strip()}
    held = [json.loads(l)["question"] for l in open(HERE / "external_eval/graded_heldout_benign.jsonl",
                                                    encoding="utf-8") if l.strip()]
    qs = [(q, str(qa[q].get("answer", qa[q].get("ground_truth", "")))) for q in held if q in qa]
    rfp.ensure_index()
    for q, a in qs[:args.warmup]:
        one(q, a, args.k)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rows = []
    for i, (q, a) in enumerate(qs[args.warmup:args.warmup + args.n], 1):
        rows.append(one(q, a, args.k))
        print(f"[{i:3d}/{args.n}] " + " ".join(f"{s}={rows[-1][s]*1000:.0f}ms" for s in STAGES), flush=True)
    summ = {}
    for s in STAGES + ("encoders_total", "total"):
        if s == "encoders_total":
            v = [r["s4_embed"] + r["s7_rerank"] + r["s10_judge"] for r in rows]
        elif s == "total":
            v = [sum(r[x] for x in STAGES) for r in rows]
        else:
            v = [r[s] for r in rows]
        summ[s] = {"median_ms": round(1000 * float(np.median(v)), 1),
                   "p95_ms": round(1000 * float(np.percentile(v, 95)), 1)}
    rep = {"device": device, "torch": torch.__version__,
           "cuda_device": torch.cuda.get_device_name(0) if device == "cuda" else None,
           "peak_cuda_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if device == "cuda" else None,
           "n": len(rows), "warmup": args.warmup, "stages": summ}
    fp16 = device == "cuda" and os.environ.get("RAG_ENCODER_FP16") == "1"
    rep["fp16"] = fp16
    out = HERE / "external_eval" / f"encoder_latency_{device}{'_fp16' if fp16 else ''}.json"
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
