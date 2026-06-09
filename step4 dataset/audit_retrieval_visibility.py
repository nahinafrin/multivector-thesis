"""
audit_retrieval_visibility.py — make Steps 5-7 visible in the JSONL output.

The first frozen artifacts proved Steps 4-9 ran, but the JSONL schema only
carried `ranked_context`; it did not expose the raw retrieved chunks or the
post-sanitization chunks. This script reconstructs those retrieval-stage
artifacts from:

  - results.jsonl       : question list
  - gate_scores.jsonl   : empirical Step-3 risk scores
  - kb_wiki.faiss/texts : persisted Step-5 vector store

and writes retrieval_audit.jsonl with a structured `retrieval` object:

  {
    "index": 1,
    "retrieval": {
      "raw_chunks": [...],
      "sanitized_chunks": [...],
      "sanitization_strictness": 0.5,
      "sanitization_dropped": [...],
      "ranked_chunks": [...],
      "retrieval_scores": [...],
      "rerank_scores": [...],
      "rerank_min_score": 0.2
    }
  }

Run:
    python audit_retrieval_visibility.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pipeline_common import PipelineState
import kb_rag_mini_wikipedia as kb
import step_04_query_embedding as s4
import step_05_vector_search as s5
import step_06_context_sanitization as s6
import step_07_context_ranking as s7


def _trace_entry(state: PipelineState, step: str) -> dict:
    for entry in reversed(state.trace):
        if entry.get("step") == step:
            return entry
    return {}


def _load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit Steps 5-7 retrieval visibility.")
    ap.add_argument("--results", default="./results.jsonl")
    ap.add_argument("--gate", default="./gate_scores.jsonl")
    ap.add_argument("--out", default="./retrieval_audit.jsonl")
    ap.add_argument("--index", default="./kb_wiki")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = _load_jsonl(args.results)
    if args.limit:
        rows = rows[: args.limit]

    gate_by_index = {
        int(row["index"]): row.get("gate", {})
        for row in _load_jsonl(args.gate)
    }

    if not kb.load_persisted_index(args.index):
        raise RuntimeError(f"Could not load persisted index at {args.index}")

    out = Path(args.out)
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as fout:
        for i, row in enumerate(rows, start=1):
            idx = int(row.get("index", i))
            gate = gate_by_index.get(idx, {})
            risk = float(gate.get("risk_score", 0.0))

            st = PipelineState(prompt=row.get("question", ""))
            st.scores["injection_detection"] = risk
            st = s4.run(st)
            st = s5.run(st, k=args.k)
            raw_chunks = list(st.context)
            retrieval_scores = list(st.meta.get("retrieval_scores", []))

            st = s6.run(st)
            sanitized_chunks = list(st.context)
            s6_trace = _trace_entry(st, "step_06_context_sanitization")

            st = s7.run(st, top_n=args.top_n)
            ranked_chunks = list(st.ranked_context)
            rerank_scores = list(st.meta.get("rerank_scores", []))
            s7_trace = _trace_entry(st, "step_07_context_ranking")

            audit = {
                "index": idx,
                "question": row.get("question", ""),
                "gate_risk_score": risk,
                "retrieval": {
                    "raw_chunks": raw_chunks,
                    "raw_count": len(raw_chunks),
                    "retrieval_scores": retrieval_scores,
                    "sanitized_chunks": sanitized_chunks,
                    "sanitized_count": len(sanitized_chunks),
                    "sanitization_strictness": s6_trace.get("threshold"),
                    "sanitization_dropped": s6_trace.get("detail", []),
                    "ranked_chunks": ranked_chunks,
                    "ranked_count": len(ranked_chunks),
                    "rerank_scores": rerank_scores,
                    "rerank_min_score": s7_trace.get("min_score"),
                },
            }
            fout.write(json.dumps(audit, ensure_ascii=False) + "\n")
            fout.flush()

            if i % 50 == 0:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0.0
                eta = (len(rows) - i) / rate if rate > 0 else 0.0
                print(f"  audited {i}/{len(rows)} elapsed={elapsed:.0f}s "
                      f"rate={rate:.2f}/s eta={eta:.0f}s")

    print(f"[retrieval-audit] wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
