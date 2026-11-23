#!/usr/bin/env python3
"""
dump_undefended_contexts.py -- what the undefended Protocol B arm put in front of the generator.

Block-only defenses (ControlNET, an LLM-Guard firewall) either refuse a query or leave the
undefended pipeline's answer unchanged. Their outcome on a row is therefore fully determined by
(a) their decision on the inputs the undefended pipeline saw and (b) the undefended answer, which
is already stored (mitigation_results/followup200/{off,benign_off}.jsonl). This script recomputes
(a)'s inputs by replaying the undefended path of mitigation_pipeline.process_with_mitigation up to
the prompt, with no LLM call:

    normalize -> embed -> FAISS top-8 -> (+ planted chunk) -> rerank top-8   (no sanitization)

Retrieval and reranking are deterministic, so these are the passages the generator received.
Rows: the 200 planted attacks (planted_attacks_200.jsonl, same field handling as
run_mitigation_ab.py) and the 200 benign questions (run_benign_cost.py's seed-42 sample).

    python dump_undefended_contexts.py
Output: mitigation_results/baselines_protocolB/contexts_{attack,benign}.jsonl
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import mitigation_pipeline as mp
from pipeline_common import PipelineState
from run_mitigation_ab import _row_fields

HERE = Path(__file__).resolve().parent
OUT = HERE / "mitigation_results" / "baselines_protocolB"


def undefended_context(question: str, poison: str | None, query_suffix: str | None) -> tuple[str, list, list]:
    if query_suffix:
        question = f"{question} {query_suffix}"
    st = PipelineState(prompt=question, raw_prompt=question)
    st = mp.s2.run(mp.s1.run(st))
    st = mp.s5.run(mp.s4.run(st), k=8)
    retrieved = list(st.context)
    if poison:
        st.context = retrieved + [poison]
    st = mp.s7.run(st, top_n=8)          # permissive rerank, as in the undefended arm
    return question, retrieved, list(st.ranked_context or st.context)


def main() -> None:
    from run_full_pipeline import ensure_index
    ensure_index()
    OUT.mkdir(parents=True, exist_ok=True)
    attack = [json.loads(l) for l in open(HERE / "planted_attacks_200.jsonl", encoding="utf-8") if l.strip()]
    with (OUT / "contexts_attack.jsonl").open("w", encoding="utf-8") as fh:
        for i, row in enumerate(attack):
            f = _row_fields(row)
            q, retrieved, ranked = undefended_context(f["question"], f["poison_chunk"], f["query_suffix"])
            fh.write(json.dumps({"index": row.get("id", i), "question": q, "poison_chunk": f["poison_chunk"],
                                 "retrieved_top8": retrieved, "ranked_context": ranked}, ensure_ascii=False) + "\n")
            print(f"[attack {i + 1}/200]", flush=True)
    qa = [json.loads(l) for l in open(HERE / "data/question-answer/test.jsonl", encoding="utf-8") if l.strip()]
    qa = [r for r in qa if (r.get("question") or "").strip()]
    random.Random(42).shuffle(qa)                     # identical to run_benign_cost.run_arm
    with (OUT / "contexts_benign.jsonl").open("w", encoding="utf-8") as fh:
        for i, row in enumerate(qa[:200]):
            q, retrieved, ranked = undefended_context(row["question"], None, None)
            fh.write(json.dumps({"index": row.get("id", i), "question": q, "retrieved_top8": retrieved,
                                 "ranked_context": ranked}, ensure_ascii=False) + "\n")
            print(f"[benign {i + 1}/200]", flush=True)


if __name__ == "__main__":
    main()
