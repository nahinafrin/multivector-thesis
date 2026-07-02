"""
adapt_inband_for_separation_test.py — build scored JSONL for test_multivector_separation.py
using the same channel methodology as build_multivector_inband.py.

Attack rows: reuse pre-measured query_graded + context_estimate (pipeline max).
Benign rows: graded_score(question) + max(graded(chunk) for chunk in retrieved), no poison.

Usage:
    python adapt_inband_for_separation_test.py \\
        --inband multivector_inband.jsonl \\
        --benign-qa data/question-answer/test.jsonl \\
        --index ./kb_wiki --k 5 --n-benign 80 \\
        --out scored_inband.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from build_multivector_inband import (
    _ensure_index,
    _graded_cached,
    _read_jsonl,
    _retrieve,
)


def load_jsonl(path: Path) -> list[dict]:
    return list(_read_jsonl(str(path)))


def inband_attack_rows(path: Path) -> list[dict]:
    rows = []
    for r in load_jsonl(path):
        m = r.get("measured") or {}
        if "query_graded" not in m or "context_estimate" not in m:
            continue
        rows.append({
            "id": r.get("id"),
            "kind": "multivector_attack_inband",
            "channels": {
                "query_vector": float(m["query_graded"]),
                "context_vector": float(m["context_estimate"]),
            },
        })
    return rows


def benign_from_qa(
    qa_path: Path,
    *,
    index_path: str,
    k: int,
    n_benign: int,
    seed: int,
) -> list[dict]:
    """Pipeline-faithful benign controls: same retrieval + max-graded context as inband."""
    _ensure_index(index_path)
    pool = [r for r in load_jsonl(qa_path) if (r.get("question") or "").strip()]
    rng = random.Random(seed)
    rng.shuffle(pool)

    rows = []
    for i, r in enumerate(pool[:n_benign]):
        prompt = r["question"]
        chunks = _retrieve(prompt, k)
        rows.append({
            "id": f"benign_{i}",
            "kind": "benign_control",
            "channels": {
                "query_vector": round(_graded_cached(prompt), 4),
                "context_vector": round(
                    max((_graded_cached(c) for c in chunks), default=0.0), 4
                ),
            },
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inband", default="multivector_inband.jsonl")
    ap.add_argument("--benign-qa", default="data/question-answer/test.jsonl")
    ap.add_argument("--index", default="./kb_wiki")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-benign", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="scored_inband.jsonl")
    args = ap.parse_args()

    inband_path = Path(args.inband)
    if not inband_path.exists():
        raise SystemExit(f"Missing {inband_path}. Run build_multivector_inband.py first.")

    attack = inband_attack_rows(inband_path)
    print(f"Loaded {len(attack)} in-band attack rows from {inband_path}")

    benign_path = Path(args.benign_qa)
    if not benign_path.exists():
        raise SystemExit(f"Missing benign QA file: {benign_path}")

    print(f"Scoring {args.n_benign} benign rows (retrieve k={args.k}, max-graded context)...")
    benign = benign_from_qa(
        benign_path,
        index_path=args.index,
        k=args.k,
        n_benign=args.n_benign,
        seed=args.seed,
    )

    scored = attack + benign
    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        for row in scored:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(scored)} rows ({len(attack)} attack, {len(benign)} benign) -> {out_path}")


if __name__ == "__main__":
    main()
