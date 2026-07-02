#!/usr/bin/env python3
"""
kb_multisource.py  —  provenance-aware knowledge base over the enriched corpus
===============================================================================

A drop-in companion to kb_rag_mini_wikipedia.py that preserves each chunk's SOURCE
tag through indexing and retrieval, so trust-aware retrieval has provenance to work
with. Reuses the project's own embedder (Step 4) and FAISS store (Step 5) exactly;
the only addition is a parallel `sources` list aligned with `texts`.

Build the index once, then eval_trust_weighting.py / the adaptive pipeline can use it:

    python kb_multisource.py --ingest --corpus enriched_passages.jsonl --index ./kb_multisource
"""
from __future__ import annotations
import argparse, json, os

import step_04_query_embedding as s4
import step_05_vector_search as s5
from pipeline_common import PipelineState


class ProvStore:
    """Wraps the Step-5 vector store and keeps a sources[] list parallel to texts[]."""
    def __init__(self, store, sources: list[str]):
        self.store = store
        self.sources = sources


def load_corpus(path: str):
    texts, sources = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            t = r.get("text") or r.get("passage") or ""
            if t.strip():
                texts.append(t)
                sources.append(r.get("source", "unknown_web"))
    return texts, sources


def ingest(corpus_path: str, index_path: str, dim: int = 512) -> ProvStore:
    texts, sources = load_corpus(corpus_path)
    store = s5.build_index(texts, dim=dim)
    # Persist texts+sources alongside the FAISS index (mirrors kb_rag_mini's .texts).
    with open(f"{index_path}.texts", "w", encoding="utf-8") as f:
        for t in store.texts:
            f.write(t.replace("\n", " ") + "\n")
    with open(f"{index_path}.sources", "w", encoding="utf-8") as f:
        # align sources to store.texts order (build_index preserves input order)
        for src in sources:
            f.write(src + "\n")
    try:
        import faiss
        faiss.write_index(store.index, f"{index_path}.faiss")
    except Exception:
        pass
    print(f"[kb_multisource] indexed {len(texts)} chunks -> {index_path}.*")
    return ProvStore(store, sources)


def load_index(index_path: str, dim: int = 512) -> ProvStore:
    if not os.path.exists(f"{index_path}.texts"):
        raise FileNotFoundError(
            f"{index_path}.texts not found; run `python kb_multisource.py "
            f"--ingest --corpus enriched_passages.jsonl --index {index_path}` first")
    texts = [l.rstrip("\n") for l in open(f"{index_path}.texts", encoding="utf-8")]
    sources = [l.rstrip("\n") for l in open(f"{index_path}.sources", encoding="utf-8")] \
        if os.path.exists(f"{index_path}.sources") else ["unknown_web"] * len(texts)
    store = s5.build_index(texts, dim=dim)   # rebuild embeddings deterministically
    return ProvStore(store, sources)


def search(store: ProvStore, query: str, k: int = 8):
    """Return [(text, similarity, source), ...] for the query."""
    st = PipelineState(prompt=query, raw_prompt=query)
    st = s4.run(st)
    qvec = st.meta["query_embedding"] if "query_embedding" in st.meta else st.scores.get("query_embedding")
    # step_05's store exposes .search(qvec, k) -> [(text, score), ...]
    import numpy as np
    hits = store.store.search(np.asarray(st.meta.get("query_embedding", qvec)), k=k)
    # map text -> source via the aligned lists
    text_to_src = dict(zip(store.store.texts, store.sources))
    return [(text, score, text_to_src.get(text, "unknown_web")) for (text, score) in hits]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingest", action="store_true")
    ap.add_argument("--corpus", default="enriched_passages.jsonl")
    ap.add_argument("--index", default="./kb_multisource")
    args = ap.parse_args()
    if args.ingest:
        ingest(args.corpus, args.index)
    else:
        print("Use --ingest to build the index.")


if __name__ == "__main__":
    main()
