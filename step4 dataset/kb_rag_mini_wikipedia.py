"""
kb_rag_mini_wikipedia.py  —  Knowledge base loader for Steps 4-8
================================================================

Provides the KNOWLEDGE BASE that Steps 4-8 retrieve from and the QA test set
that evaluates retrieval. Uses rag-datasets/rag-mini-wikipedia:

  - subset "text-corpus"     : ~3,200 Wikipedia passages  -> the KB to index
  - subset "question-answer" : ~918 (question, answer) pairs -> test queries
                               with ground-truth answers

This is the data your retrieval/generation half of the methodology needs, and
it is SEPARATE from the attack/safety dataset (which is only for Step 3). The
attack dataset has prompts+labels; this has documents+QA — different jobs.

It does NOT reimplement Steps 4-8. It builds the FAISS index (via Step 5's
build_index, which embeds with Step 4) and yields test questions you can push
through the retrieval pipeline.

Setup:
    pip install datasets sentence-transformers faiss-cpu

Usage:
    # build the KB index from the Wikipedia corpus
    python kb_rag_mini_wikipedia.py --ingest --index ./kb_wiki

    # then run a retrieval test question through Steps 4->5->6->7->8
    python kb_rag_mini_wikipedia.py --index ./kb_wiki --test-questions 5
"""

from __future__ import annotations

import argparse
import json
import os
import time

from pipeline_common import PipelineState
import step_04_query_embedding as s4
import step_05_vector_search as s5
import step_06_context_sanitization as s6
import step_07_context_ranking as s7
import step_08_augmented_prompt as s8

HF_ID = "rag-datasets/rag-mini-wikipedia"


def load_corpus() -> list[str]:
    """Load the ~3.2k Wikipedia passages (the knowledge base)."""
    from datasets import load_dataset
    ds = load_dataset(HF_ID, "text-corpus", split="passages")
    # the passage text column is 'passage'
    col = "passage" if "passage" in ds.column_names else ds.column_names[0]
    passages = [r[col] for r in ds if isinstance(r[col], str) and r[col].strip()]
    print(f"[kb] loaded {len(passages)} Wikipedia passages from {HF_ID}")
    return passages


def load_qa(limit: int | None = None) -> list[dict]:
    """Load the (question, answer) test pairs with ground-truth answers."""
    from datasets import load_dataset
    ds = load_dataset(HF_ID, "question-answer", split="test")
    qcol = "question" if "question" in ds.column_names else ds.column_names[0]
    acol = "answer" if "answer" in ds.column_names else ds.column_names[1]
    pairs = []
    for i, r in enumerate(ds):
        if limit is not None and i >= limit:
            break
        pairs.append({"question": r[qcol], "ground_truth": r[acol]})
    print(f"[kb] loaded {len(pairs)} QA test pairs")
    return pairs


def ingest(index_path: str, dim: int = 512, use_hnsw: bool = False) -> None:
    """Embed the Wikipedia corpus (Step 4) and build the FAISS index (Step 5)."""
    passages = load_corpus()
    store = s5.build_index(passages, dim=dim, use_hnsw=use_hnsw)
    # persist so subsequent runs don't re-embed
    try:
        store.faiss.write_index(store.index, f"{index_path}.faiss")
        with open(f"{index_path}.texts", "w", encoding="utf-8") as f:
            for t in store.texts:
                f.write(t.replace("\n", " ") + "\n")
        print(f"[kb] index persisted to {index_path}.faiss (+ .texts)")
    except Exception as e:
        print(f"[kb] note: in-memory index built; persist skipped ({e})")


def load_persisted_index(index_path: str, dim: int = 512) -> bool:
    """Restore a previously persisted FAISS index into Step 5's module store.

    Returns True if both `<index_path>.faiss` and `<index_path>.texts` were
    found and loaded (so we can skip the ~10 min re-embed); False otherwise.
    """
    faiss_file = f"{index_path}.faiss"
    texts_file = f"{index_path}.texts"
    if not (os.path.exists(faiss_file) and os.path.exists(texts_file)):
        return False
    from pipeline_common import lazy_import
    faiss = lazy_import("faiss", "faiss-cpu")
    store = s5.FaissStore(dim=dim, use_hnsw=False)
    store.index = faiss.read_index(faiss_file)
    with open(texts_file, "r", encoding="utf-8") as f:
        store.texts = [line.rstrip("\n") for line in f if line.strip()]
    s5._STORE = store
    print(f"[kb] loaded persisted index: {len(store.texts)} passages from {faiss_file}")
    return True


def retrieve_pipeline(question: str, k: int = 5, top_n: int = 3,
                      input_risk: float = 0.0) -> PipelineState:
    """Run one question through Steps 4 -> 5 -> 6 -> 7 -> 8 (retrieval half)."""
    st = PipelineState(prompt=question, raw_prompt=question)
    st.scores["injection_detection"] = input_risk   # carried-forward risk (dual-checkpoint)
    st = s4.run(st)                  # Step 4: embed
    st = s5.run(st, k=k)             # Step 5: vector search
    st = s6.run(st)                  # Step 6: context sanitization (+canary)
    st = s7.run(st, top_n=top_n)     # Step 7: rerank
    st = s8.run(st)                  # Step 8: augmented prompt
    return st


def main() -> None:
    ap = argparse.ArgumentParser(description="rag-mini-wikipedia KB for Steps 4-8")
    ap.add_argument("--ingest", action="store_true", help="Build the FAISS index from the corpus.")
    ap.add_argument("--index", default="./kb_wiki")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--hnsw", action="store_true", help="Use HNSW (true ANN) instead of exact.")
    ap.add_argument("--test-questions", type=int, default=0,
                    help="Run N QA questions through the retrieval pipeline.")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--all", action="store_true",
                    help="Run the full QA set (ignore --test-questions limit).")
    ap.add_argument("--results", default="results.jsonl",
                    help="Where to write one JSON object per question (default: results.jsonl).")
    ap.add_argument("--resume", action="store_true",
                    help="Skip questions whose `index` is already present in --results "
                         "and append the rest (default: overwrite).")
    args = ap.parse_args()

    if args.ingest:
        ingest(args.index, dim=args.dim, use_hnsw=args.hnsw)

    run_qa = args.all or args.test_questions
    if run_qa:
        # Try the persisted index first; only re-embed if it's missing or unreadable.
        if s5._STORE is None and not load_persisted_index(args.index, dim=args.dim):
            ingest(args.index, dim=args.dim, use_hnsw=args.hnsw)

        limit = None if args.all else args.test_questions
        qa = load_qa(limit=limit)
        total = len(qa)
        results_path = os.path.abspath(args.results)

        done_indices: set[int] = set()
        if args.resume and os.path.exists(results_path):
            with open(results_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    try:
                        done_indices.add(int(json.loads(line)["index"]))
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
            print(f"[kb] resume: {len(done_indices)} questions already in {results_path}")

        open_mode = "a" if args.resume else "w"
        t0 = time.time()
        completed_now = 0
        with open(results_path, open_mode, encoding="utf-8") as fout:
            for idx, item in enumerate(qa, start=1):
                if idx in done_indices:
                    continue
                completed_now += 1
                st = retrieve_pipeline(item["question"], k=args.k, top_n=args.top_n)
                record = {
                    "index": idx,
                    "question": item["question"],
                    "ground_truth": item["ground_truth"],
                    "retrieved_count": len(st.context),
                    "reranked_count": len(st.ranked_context),
                    "ranked_context": st.ranked_context,
                    "retrieval_scores": st.meta.get("retrieval_scores", []),
                    "rerank_scores": st.meta.get("rerank_scores", []),
                    "augmented_prompt_chars": len(st.augmented_prompt),
                    "augmented_prompt": st.augmented_prompt,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()

                # Lightweight progress to stdout. Truncate context preview so
                # legacy Windows code pages can't crash the run on a stray char.
                elapsed = time.time() - t0
                rate = completed_now / elapsed if elapsed > 0 else 0.0
                remaining = total - idx
                eta = remaining / rate if rate > 0 else 0.0
                print(f"\n[{idx}/{total}] new={completed_now} elapsed={elapsed:.1f}s "
                      f"rate={rate:.2f} q/s eta={eta:.0f}s")
                print(f"Q: {item['question']}")
                print(f"  ground truth : {item['ground_truth']}")
                print(f"  retrieved {len(st.context)} -> reranked {len(st.ranked_context)} chunks")
                for i, c in enumerate(st.ranked_context):
                    safe = c[:90].encode("ascii", "replace").decode("ascii")
                    print(f"    [{i}] {safe}")
                print(f"  augmented prompt chars: {len(st.augmented_prompt)}")
        print(f"\n[kb] wrote {completed_now} new results "
              f"({len(done_indices) + completed_now}/{total} total) to {results_path}")


if __name__ == "__main__":
    main()
