"""
step_05_vector_search.py  —  METHODOLOGY STEP 5: Vector Search
==============================================================

"Pinecone ... Approximate Nearest Neighbor (ANN) ... closest results (usually
through Cosine Similarity)."

FREE/LOCAL DEFAULT: FAISS. We use IndexFlatIP on L2-normalized vectors, which
makes inner product == cosine similarity. For large corpora swap IndexFlatIP
for IndexHNSWFlat to get true ANN (the methodology's "ANN" claim) instead of
exact search.

PAID SWAP (commented): Pinecone serverless index.

This step needs a knowledge base to search. For a security pipeline the KB is
your corpus of reference docs / policies; here we provide build_index() to
ingest texts and run() to retrieve top-k for the query embedding from Step 4.

Install (local): pip install faiss-cpu
Run standalone (builds a tiny demo index then queries it):
    python step_05_vector_search.py --demo
"""

from __future__ import annotations

import argparse
import numpy as np

from pipeline_common import PipelineState, lazy_import
from step_04_query_embedding import embed


class FaissStore:
    """Minimal local vector store: cosine via inner-product on normalized vecs."""

    def __init__(self, dim: int, use_hnsw: bool = False):
        faiss = lazy_import("faiss", "faiss-cpu")
        self.faiss = faiss
        self.dim = dim
        if use_hnsw:
            self.index = faiss.IndexHNSWFlat(dim, 32)   # true ANN
            self.index.metric_type = faiss.METRIC_INNER_PRODUCT
        else:
            self.index = faiss.IndexFlatIP(dim)         # exact cosine
        self.texts: list[str] = []

    def add(self, texts: list[str], vecs: np.ndarray) -> None:
        self.index.add(vecs.astype("float32"))
        self.texts.extend(texts)

    def search(self, qvec: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        q = qvec.reshape(1, -1).astype("float32")
        scores, idx = self.index.search(q, min(k, len(self.texts)))
        out = []
        for s, i in zip(scores[0], idx[0]):
            if i == -1:
                continue
            out.append((self.texts[i], float(s)))
        return out


# A module-level store you build once and reuse across queries.
_STORE: FaissStore | None = None


def build_index(texts: list[str], dim: int = 512, use_hnsw: bool = False) -> FaissStore:
    global _STORE
    vecs = np.vstack([embed(t, dim=dim) for t in texts])
    store = FaissStore(dim=dim, use_hnsw=use_hnsw)
    store.add(texts, vecs)
    _STORE = store
    return store


# ---- PAID SWAP: Pinecone --------------------------------------------------
# import os
# from pinecone import Pinecone, ServerlessSpec
# pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
# index = pc.Index("security-kb")
# def _pinecone_search(qvec, k=5):
#     res = index.query(vector=qvec.tolist(), top_k=k, include_metadata=True)
#     return [(m["metadata"]["text"], m["score"]) for m in res["matches"]]
# ---------------------------------------------------------------------------


def run(state: PipelineState, k: int = 5, dim: int = 512) -> PipelineState:
    if state.blocked:
        return state
    if _STORE is None:
        state.log("step_05_vector_search", warning="no index built; skipping")
        return state
    qvec = state.meta.get("embedding")
    if qvec is None:
        qvec = embed(state.prompt, dim=dim)
    hits = _STORE.search(np.asarray(qvec), k=k)
    state.context = [t for t, _ in hits]
    state.meta["retrieval_scores"] = [s for _, s in hits]
    state.log("step_05_vector_search", retrieved=len(hits),
              top_score=round(hits[0][1], 4) if hits else None)
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 5: Vector Search")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--prompt", default="how do I reset my account password")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()
    if args.demo:
        kb = [
            "To reset your password, go to Settings > Security > Reset Password.",
            "Our refund policy allows returns within 30 days of purchase.",
            "Two-factor authentication can be enabled in the Security settings.",
            "Business hours are 9am to 5pm, Monday through Friday.",
        ]
        build_index(kb, dim=512)
        st = PipelineState(prompt=args.prompt)
        st = run(st, k=args.k)
        for i, (t, s) in enumerate(zip(st.context, st.meta["retrieval_scores"])):
            print(f"  [{i}] score={s:.3f}  {t}")
    else:
        print("Use --demo, or import build_index()/run() in the orchestrator.")


if __name__ == "__main__":
    main()
