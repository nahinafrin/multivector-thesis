"""
Reimplementation of a Sentinel-Strategist-style RAG defense baseline.

Faithful to the published method description as already characterized in this
project's own novelty-positioning-comparison.md:
  - Two-pass, SEQUENTIAL query-then-context scoring (NOT a joint/fused
    threshold like this project's own multivector.py)
  - Heuristic-only signals: lexical/cue-phrase density, embedding dispersion,
    a per-source trust score
  - No ensemble disagreement, no groundedness/faithfulness judge, no canary
  - Detect-once / activate-defenses-once: single-shot, no retry loop, no
    closed-loop controller

This is a REIMPLEMENTATION from the paper's described architecture, not the
authors' original (unpublished) code. Label it as such in the thesis.

Usage:
    python baseline_sentinel_strategist.py \
        --slice adversarial_slice.jsonl \
        --train train.jsonl \
        --out grounded_sentinel_strategist.jsonl \
        --retrieve-module step_05_vector_search --retrieve-fn retrieve_topk

The --retrieve-module/--retrieve-fn flags let you point this at YOUR repo's
real retrieval function (so both this baseline and the full pipeline search
the identical kb_wiki FAISS index) without editing this file's imports.
"""
import argparse
import importlib
import json
import re

import numpy as np
from sentence_transformers import SentenceTransformer

INJECTION_CUES = [
    r"\bignore (all|any|the) (previous|prior|above)\b",
    r"\bdisregard (any|all|the)\b.*\b(instructions|guidelines|sources)\b",
    r"\byou (must|should) (now|instead)\b",
    r"\bsystem prompt\b",
    r"\bnew instructions?\b",
    r"\bact as\b",
    r"\bpretend (you|to)\b",
    r"\broleplay\b",
    r"\bauthoritative (source|note|answer)\b",
    r"\bprioritize (any|this) note\b",
    r"\breviewer (guidance|note)\b",
    r"\bcorrected figure\b",
    r"\beditorial note\b",
    r"\bconclude that\b.*\bauthorized\b",
]
INJECTION_RE = re.compile("|".join(INJECTION_CUES), re.IGNORECASE)


def query_heuristic_score(query: str) -> float:
    """Query-side pass: cue-phrase density, normalized to [0, 1]."""
    hits = len(INJECTION_RE.findall(query or ""))
    return min(1.0, hits / 2.0)  # 2+ cue phrases -> score 1.0


def lexical_overlap(a: str, b: str) -> float:
    ta = set(re.findall(r"\w+", (a or "").lower()))
    tb = set(re.findall(r"\w+", (b or "").lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def context_heuristic_score(query, chunks, embedder, trust_by_chunk=None):
    """
    Context-side pass. Only reached if the query-side pass did NOT already
    block (this sequencing, not a joint fusion, is the point of the
    reimplementation).

    Combines, per chunk: cue-phrase density, embedding-dispersion from the
    retrieved-set centroid (an outlier chunk relative to its neighbors is
    treated as suspicious), inverse lexical overlap with the query, and a
    per-source trust discount. On this project's single-source rag-mini-
    wikipedia corpus every chunk's trust is 1.0 by construction, so the
    trust term never differentiates anything here -- report that explicitly,
    it's itself a fair point of comparison (this baseline's trust signal is
    inert on a single-source KB, same as this project's own
    trust_aware_retrieval.py before Addendum 7/Addendum 9's multi-source work).
    """
    if not chunks:
        return 0.0
    trust_by_chunk = trust_by_chunk or [1.0] * len(chunks)
    cue_scores = [min(1.0, len(INJECTION_RE.findall(c)) / 1.0) for c in chunks]
    embs = embedder.encode(list(chunks), normalize_embeddings=True)
    centroid = embs.mean(axis=0)
    dispersion = [float(1.0 - np.dot(e, centroid)) for e in embs]
    disp_norm = [min(1.0, d / 0.5) for d in dispersion]
    overlaps = [lexical_overlap(query, c) for c in chunks]
    low_overlap_risk = [1.0 - o for o in overlaps]
    per_chunk_risk = [
        (0.4 * cue + 0.35 * disp + 0.25 * lor) * trust
        for cue, disp, lor, trust in zip(cue_scores, disp_norm, low_overlap_risk, trust_by_chunk)
    ]
    return max(per_chunk_risk) if per_chunk_risk else 0.0


def calibrate_query_threshold(train_rows, target_fpr=0.05, floor=0.34):
    """
    Calibrate query_tau on the SAME train split C3RF is calibrated against,
    picking the lowest threshold that keeps the false-positive rate on
    benign training rows at or below target_fpr -- mirrors this project's
    own measure-once discipline (c3rf_protocol.py) so the final comparison
    is apples-to-apples rather than a threshold picked to flatter one side.
    """
    benign = [r for r in train_rows if r.get("safety") == "safe"]
    q_scores = sorted(query_heuristic_score(r.get("prompt", r.get("query", ""))) for r in benign)
    if not q_scores:
        return floor
    idx = min(int(len(q_scores) * (1 - target_fpr)), len(q_scores) - 1)
    return max(q_scores[idx], floor)


def run_row(row, embedder, retrieve_fn, query_tau, context_tau):
    query = row.get("prompt", row.get("query", ""))
    qs = query_heuristic_score(query)
    if qs >= query_tau:
        return {"blocked": True, "block_stage": "sentinel_query_pass", "baseline_score": qs}
    chunks = retrieve_fn(query)
    cs = context_heuristic_score(query, chunks, embedder)
    if cs >= context_tau:
        return {"blocked": True, "block_stage": "sentinel_context_pass", "baseline_score": cs}
    return {"blocked": False, "block_stage": None, "baseline_score": max(qs, cs)}


def load_retriever(module_name, fn_name):
    mod = importlib.import_module(module_name)
    return getattr(mod, fn_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--index", default="kb_wiki")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--embedder", default="BAAI/bge-m3", help="same embedder the project already loads")
    ap.add_argument("--context-tau", type=float, default=0.55, help="start here; re-grid-search on val.jsonl if time allows")
    ap.add_argument("--retrieve-module", required=True, help="e.g. step_05_vector_search")
    ap.add_argument("--retrieve-fn", required=True, help="e.g. retrieve_topk")
    args = ap.parse_args()

    retrieve_topk = load_retriever(args.retrieve_module, args.retrieve_fn)
    embedder = SentenceTransformer(args.embedder)

    train_rows = [json.loads(line) for line in open(args.train, encoding="utf-8")]
    query_tau = calibrate_query_threshold(train_rows)
    print(f"[calibration] query_tau={query_tau:.3f} context_tau={args.context_tau:.3f}")

    n_rows = 0
    with open(args.slice, encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as out:
        for line in f:
            row = json.loads(line)
            query = row.get("prompt", row.get("query", ""))
            raw_chunks = retrieve_topk(query, k=args.k, index_name=args.index)
            chunks = [c["text"] if isinstance(c, dict) else c for c in raw_chunks]
            result = run_row(row, embedder, lambda q, c=chunks: c, query_tau, args.context_tau)
            row["baseline"] = "sentinel_strategist_reimpl"
            row.update(result)
            out.write(json.dumps(row) + "\n")
            n_rows += 1

    print(f"[done] scored {n_rows} rows -> {args.out}")


if __name__ == "__main__":
    main()
