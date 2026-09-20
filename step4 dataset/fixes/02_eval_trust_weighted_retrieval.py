#!/usr/bin/env python3
"""
eval_trust_weighted_retrieval.py
=================================
TARGET LOCATION IN REPO:  step4 dataset/eval_trust_weighted_retrieval.py
(same directory as trust_aware_retrieval.py, build_multisource_corpus.py,
step_04_query_embedding.py)

WHY THIS SCRIPT EXISTS
-----------------------
trust_aware_retrieval.py and build_multisource_corpus.py already exist but are
explicitly flagged in the repo as "a no-op on single-source rag-mini" and
"un-evaluated". This is the missing piece: it actually retrieves against the
controlled multi-source corpus WITH and WITHOUT trust weighting and measures
whether trust weighting suppresses the planted poison chunks — which is the
concrete, testable step toward the open problem in the analysis report
(signature-free misinformation-style attacks succeed ~50% and nothing
model-derived separates them; the stated remaining direction is *external*
provenance / source-trust weighting).

WHAT IT MEASURES
-----------------
For every targeted question in poison_manifest.jsonl:
  1. Embed the question with the SAME embedder used elsewhere in the pipeline
     (BAAI/bge-m3 via step_04_query_embedding.embed) for consistency.
  2. Retrieve top-k by raw cosine similarity  -> "poison_in_topk_baseline"
  3. Retrieve top-k by trust_aware_retrieval.rescore()  -> "poison_in_topk_trust"
  4. Report the poison-suppression rate: how often trust weighting keeps the
     poison chunk OUT of top-k when raw similarity would have put it IN.

Run AFTER building the corpus:
    python build_multisource_corpus.py --passages data/text-corpus/passages.jsonl \
        --qa data/question-answer/test.jsonl --n-poison 60 \
        --out-corpus enriched_passages.jsonl --out-manifest poison_manifest.jsonl

Then:
    python eval_trust_weighted_retrieval.py \
        --corpus enriched_passages.jsonl --manifest poison_manifest.jsonl --k 3
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from step_04_query_embedding import embed
from trust_aware_retrieval import rescore, DEFAULT_TRUST


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def build_matrix(corpus: list[dict]) -> np.ndarray:
    print(f"[embed] encoding {len(corpus)} chunks with bge-m3 (this may take a while)...")
    vecs = [embed(r["text"], dim=512) for r in corpus]
    return np.stack(vecs)


def topk_raw(qvec: np.ndarray, mat: np.ndarray, corpus: list[dict], k: int) -> list[dict]:
    sims = mat @ qvec  # both already L2-normalized -> cosine
    order = np.argsort(-sims)[:k]
    return [{"id": corpus[i]["id"], "is_poison": corpus[i]["is_poison"],
             "similarity": float(sims[i])} for i in order]


def topk_trust(qvec: np.ndarray, mat: np.ndarray, corpus: list[dict], k: int) -> list[dict]:
    sims = mat @ qvec
    hits = [(corpus[i]["text"], float(sims[i]), corpus[i]["source"]) for i in range(len(corpus))]
    ranked = rescore(hits, table=DEFAULT_TRUST)[:k]
    # map back to ids via text match (small corpora only; fine for an eval script)
    text_to_row = {r["text"]: r for r in corpus}
    out = []
    for h in ranked:
        row = text_to_row.get(h["text"])
        out.append({"id": row["id"] if row else None, "is_poison": row["is_poison"] if row else None,
                    "trust_score": h["score"]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", default="trust_weighted_retrieval_report.json")
    args = ap.parse_args()

    corpus = read_jsonl(args.corpus)
    manifest = read_jsonl(args.manifest)
    mat = build_matrix(corpus)
    id_to_poison = {r["poison_id"]: r for r in manifest}
    poison_ids = set(id_to_poison)

    baseline_in, trust_in, n = 0, 0, 0
    suppressed_rows = []
    for m in manifest:
        qvec = embed(m["question"], dim=512)
        base_hits = topk_raw(qvec, mat, corpus, args.k)
        trust_hits = topk_trust(qvec, mat, corpus, args.k)

        base_has_poison = any(h["id"] == m["poison_id"] for h in base_hits)
        trust_has_poison = any(h["id"] == m["poison_id"] for h in trust_hits)
        n += 1
        baseline_in += int(base_has_poison)
        trust_in += int(trust_has_poison)
        if base_has_poison and not trust_has_poison:
            suppressed_rows.append(m["qid"])

    baseline_rate = baseline_in / n if n else 0.0
    trust_rate = trust_in / n if n else 0.0
    suppression_rate = (len(suppressed_rows) / baseline_in) if baseline_in else None

    print(f"\n=== Trust-weighted retrieval, n={n} targeted questions, k={args.k} ===")
    print(f"  poison in top-k, RAW similarity        : {baseline_in}/{n}  ({baseline_rate:.1%})")
    print(f"  poison in top-k, TRUST-WEIGHTED         : {trust_in}/{n}  ({trust_rate:.1%})")
    print(f"  of the raw-similarity hits, suppressed by trust weighting: "
          f"{len(suppressed_rows)}/{baseline_in}"
          + (f"  ({suppression_rate:.1%})" if suppression_rate is not None else ""))

    report = {
        "n": n, "k": args.k,
        "poison_in_topk_baseline": baseline_in, "poison_in_topk_baseline_rate": baseline_rate,
        "poison_in_topk_trust": trust_in, "poison_in_topk_trust_rate": trust_rate,
        "suppressed_by_trust": len(suppressed_rows),
        "suppression_rate_of_baseline_hits": suppression_rate,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[saved] {args.out}")
    print("\nIf suppression_rate is high, wire trust_aware_retrieval.rescore() into "
          "step_05_vector_search.py's real retrieval path (gated on a --trust-aware "
          "flag so the single-source rag-mini evaluation is unaffected), then re-run "
          "the semantic_slice (build_semantic_slice.py) end-to-end through the FULL "
          "pipeline to see whether the planted-false-fact success rate in "
          "score_attack_success.py actually drops below the current 50%.")


if __name__ == "__main__":
    main()
