"""
apply_retrieval_to_grounded.py — merge retrieval_audit.jsonl into grounded.jsonl.

This makes Steps 5-7 examiner-visible in the final row schema by adding a
top-level `retrieval` object to every grounded row. It preserves the existing
flat `ranked_context` field for backward compatibility.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge retrieval audit into grounded output.")
    ap.add_argument("--grounded", default="./grounded.jsonl")
    ap.add_argument("--audit", default="./retrieval_audit.jsonl")
    ap.add_argument("--out", default=None,
                    help="Default: overwrite --grounded after .bak2 backup.")
    args = ap.parse_args()

    grounded_path = Path(args.grounded)
    audit_by_index = {
        int(row["index"]): row["retrieval"]
        for row in _load_jsonl(args.audit)
    }
    rows = _load_jsonl(args.grounded)

    if args.out:
        out_path = Path(args.out)
        backup = None
    else:
        out_path = grounded_path
        backup = grounded_path.with_suffix(grounded_path.suffix + ".retrieval.bak")
        shutil.copyfile(grounded_path, backup)

    missing = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for row in rows:
            idx = int(row.get("index", -1))
            retrieval = audit_by_index.get(idx)
            if retrieval is None:
                missing += 1
            else:
                row["retrieval"] = retrieval
                row["ranked_context"] = retrieval.get("ranked_chunks", row.get("ranked_context", []))
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[apply-retrieval] wrote {len(rows)} rows -> {out_path}")
    if backup:
        print(f"[apply-retrieval] backup at {backup}")
    print(f"[apply-retrieval] rows missing retrieval audit: {missing}")


if __name__ == "__main__":
    main()
