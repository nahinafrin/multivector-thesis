"""Compare stored multivector.channels vs fresh graded_score on same benign rows."""
from __future__ import annotations

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


def fresh_context_score(r: dict) -> float:
    chunks = (r.get("retrieval") or {}).get("ranked_chunks") or []
    if not chunks:
        return 0.0
    return max(graded_score(c) for c in chunks)


def main() -> None:
    path = Path("grounded_controller.jsonl")
    benign = [r for r in load_jsonl(path) if r.get("kind") == "benign_control"][:10]

    print(f"{'idx':>4}  {'stored_q':>8} {'fresh_q':>8}  {'stored_c':>8} {'fresh_c':>8}  delta_c")
    deltas = []
    for r in benign:
        stored = (r.get("multivector") or {}).get("channels") or {}
        sq = float(stored.get("query_vector", 0))
        sc = float(stored.get("context_vector", 0))
        fq = round(graded_score(r.get("question", "")), 4)
        fc = round(fresh_context_score(r), 4)
        deltas.append(sc - fc)
        print(f"{r.get('index', '?'):>4}  {sq:8.4f} {fq:8.4f}  {sc:8.4f} {fc:8.4f}  {sc - fc:+.4f}")

    avg_delta = sum(deltas) / len(deltas) if deltas else 0.0
    print(f"\nmean(stored_context - fresh_context) = {avg_delta:+.4f}")
    if avg_delta > 0.05:
        print("=> CONFIRMED: stored context_vector is systematically HIGHER than fresh graded_score")
    elif avg_delta < -0.05:
        print("=> stored context is LOWER than fresh (unexpected)")
    else:
        print("=> scales roughly agree on this sample")


if __name__ == "__main__":
    main()
