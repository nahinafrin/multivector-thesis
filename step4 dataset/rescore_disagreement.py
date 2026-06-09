"""
rescore_disagreement.py — re-score Step 9 disagreement offline.

Reads an existing generated.jsonl (with per-row `candidates`), runs the new
SEMANTIC disagreement scorer on those candidates, and rewrites the file in
place after taking a `.bak` backup. The previous score is preserved on each
row as `disagreement_legacy` so before/after distributions can be compared
without re-invoking any LLM.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from step_09_generator_llm import compute_disagreement


def _bucket(x: float) -> str:
    # Coarse bands for distribution printing.
    if x < 0.05:
        return "0.00-0.05"
    if x < 0.15:
        return "0.05-0.15"
    if x < 0.30:
        return "0.15-0.30"
    if x < 0.50:
        return "0.30-0.50"
    if x < 0.80:
        return "0.50-0.80"
    return "0.80-1.00"


def main() -> None:
    ap = argparse.ArgumentParser(description="Rescore Step 9 disagreement semantically.")
    ap.add_argument("--in", dest="in_path", default="./generated.jsonl")
    ap.add_argument("--out", default=None,
                    help="Where to write rescored rows. Defaults to overwriting "
                         "--in after taking a .bak backup.")
    args = ap.parse_args()

    src = Path(args.in_path)
    rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]

    backup: Path | None = None
    if args.out:
        dst = Path(args.out)
    else:
        backup = src.with_suffix(src.suffix + ".bak")
        shutil.copyfile(src, backup)
        dst = src

    old_dist: Counter[str] = Counter()
    new_dist: Counter[str] = Counter()
    n_changed = 0

    with open(dst, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows, start=1):
            cands = row.get("candidates") or {}
            old = float(row.get("disagreement", 0.0))
            new = compute_disagreement(cands) if len(cands) >= 2 else old

            row["disagreement_legacy"] = old
            row["disagreement"] = new

            if abs(new - old) > 1e-6:
                n_changed += 1
            old_dist[_bucket(old)] += 1
            new_dist[_bucket(new)] += 1

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 200 == 0:
                print(f"  rescored {i}/{len(rows)}")

    print(f"\n[rescore] rewrote {len(rows)} rows; {n_changed} disagreement scores changed")
    if backup:
        print(f"[rescore] backup at {backup}")

    print("\nLegacy (whitespace) distribution:")
    for band in ["0.00-0.05", "0.05-0.15", "0.15-0.30",
                 "0.30-0.50", "0.50-0.80", "0.80-1.00"]:
        print(f"  {band:11s} : {old_dist[band]}")
    print("\nSemantic distribution:")
    for band in ["0.00-0.05", "0.05-0.15", "0.15-0.30",
                 "0.30-0.50", "0.50-0.80", "0.80-1.00"]:
        print(f"  {band:11s} : {new_dist[band]}")


if __name__ == "__main__":
    main()
