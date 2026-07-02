"""
score_conjunctive_synth.py — grade conjunctive_synth.jsonl + benign controls
into the JSONL format expected by test_multivector_separation.py.

Usage:
    python score_conjunctive_synth.py
    python score_conjunctive_synth.py --limit 50   # quick smoke test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from graded_channels import graded_score


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _context_score_from_retrieval(r: dict) -> float:
    """Match pipeline: context_vector = max graded score over ranked chunks."""
    chunks = (r.get("retrieval") or {}).get("ranked_chunks") or []
    if not chunks:
        return 0.0
    return max(graded_score(c) for c in chunks)


def benign_from_grounded(path: Path) -> list[dict]:
    """Score benign rows fresh on the graded scale — do not reuse stored channels."""
    rows = []
    for r in load_jsonl(path):
        if r.get("kind") != "benign_control":
            continue
        prompt = r.get("question", r.get("prompt", ""))
        rows.append({
            "id": r.get("index"),
            "kind": "benign_control",
            "channels": {
                "query_vector": round(graded_score(prompt), 4),
                "context_vector": round(_context_score_from_retrieval(r), 4),
            },
        })
    return rows


def score_conjunctive(path: Path, *, limit: int | None) -> list[dict]:
    out = []
    for i, r in enumerate(load_jsonl(path)):
        if limit is not None and i >= limit:
            break
        prompt = r.get("prompt", "")
        ctx = r.get("injected_context", "")
        meta = r.get("meta") or {}
        out.append({
            "id": r.get("id"),
            "kind": r.get("kind", "multivector_attack_synth"),
            "split_weight": meta.get("split_weight"),
            "strategy": meta.get("strategy"),
            "channels": {
                "query_vector": round(graded_score(prompt), 4),
                "context_vector": round(graded_score(ctx), 4),
            },
        })
        if (i + 1) % 25 == 0:
            print(f"  scored {i + 1} conjunctive rows...", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conjunctive", default="conjunctive_synth.jsonl")
    ap.add_argument("--benign-from", default="grounded_controller.jsonl",
                    help="Reuse pipeline-scored benign channel values from a run JSONL")
    ap.add_argument("--out", default="scored_conjunctive.jsonl")
    ap.add_argument("--limit", type=int, default=None,
                    help="Score only the first N conjunctive rows (smoke test)")
    args = ap.parse_args()

    conj_path = Path(args.conjunctive)
    if not conj_path.exists():
        raise SystemExit(f"Missing {conj_path}. Run:\n"
                         "  python adversarial_conjunctive_generator.py "
                         "--attacks-file attacks_source.txt --out conjunctive_synth.jsonl")

    print(f"Scoring conjunctive rows from {conj_path}...")
    scored = score_conjunctive(conj_path, limit=args.limit)

    benign_path = Path(args.benign_from)
    if benign_path.exists():
        benign = benign_from_grounded(benign_path)
        print(f"Added {len(benign)} benign_control rows from {benign_path}")
        scored.extend(benign)
    else:
        print(f"Warning: {benign_path} not found — no benign controls in output")

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        for row in scored:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_conj = sum(1 for r in scored if "multivector_attack" in r.get("kind", ""))
    n_benign = sum(1 for r in scored if r.get("kind") == "benign_control")
    print(f"Wrote {len(scored)} rows ({n_conj} conjunctive, {n_benign} benign) to {out_path}")


if __name__ == "__main__":
    main()
