"""Driver that exercises `gap_fill_sources.load_gap_fill_sources` end-to-end.

Two modes:

    python run_gap_fill.py --probe          # small caps, validates access/columns
    python run_gap_fill.py --compile        # full caps, writes JSONL per attack

The compile mode mirrors how `build_new_dataset.py` would consume these rows,
but persists them standalone so the gap-fill data can be inspected before
being merged into the main pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from build_new_dataset import Normalizer
from gap_fill_sources import GAP_SOURCES, load_gap_fill_sources


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--probe", action="store_true",
                   help="small caps to validate column mapping and gating")
    p.add_argument("--compile", action="store_true",
                   help="full caps, write JSONL output")
    p.add_argument("--out-dir", default="gap_fill_output")
    p.add_argument("--max-rows-per-source", type=int, default=0,
                   help="HF-level cap; 0 means take everything")
    p.add_argument("--per-source-cap", type=int, default=2000,
                   help="when --keep-positive-only or --benign-cap-per-source "
                        "is set, caps POSITIVES only; otherwise caps total rows")
    p.add_argument("--only", nargs="*", default=None,
                   help="optional subset of source keys")
    p.add_argument("--keep-positive-only", action="store_true",
                   help="drop benign rows entirely; per-source-cap caps positives")
    p.add_argument("--benign-cap-per-source", type=int, default=None,
                   help="independent benign cap (positives uncoupled). "
                        "Implies independent-cap mode.")
    p.add_argument("--min-len", type=int, default=20)
    p.add_argument("--max-len", type=int, default=2000)
    args = p.parse_args()

    if not (args.probe or args.compile):
        p.error("pass --probe or --compile")

    if args.probe:
        max_rows = 200
        per_source_cap = 50
        out_dir = Path(args.out_dir) / "probe"
    else:
        max_rows = args.max_rows_per_source
        per_source_cap = args.per_source_cap
        out_dir = Path(args.out_dir)

    print(f"[run_gap_fill] mode={'probe' if args.probe else 'compile'}")
    print(f"[run_gap_fill] max_rows_per_source={max_rows} "
          f"per_source_cap={per_source_cap} "
          f"keep_positive_only={args.keep_positive_only} "
          f"benign_cap_per_source={args.benign_cap_per_source}")
    if args.only:
        unknown = [k for k in args.only if k not in GAP_SOURCES]
        if unknown:
            print(f"[run_gap_fill] WARN unknown --only keys: {unknown}")

    normalizer = Normalizer(min_len=args.min_len, max_len=args.max_len)
    pools, safe_rows, counts, loaded = load_gap_fill_sources(
        max_rows_per_source=max_rows,
        normalizer=normalizer,
        per_source_cap=per_source_cap,
        only=args.only,
        keep_positive_only=args.keep_positive_only,
        benign_cap_per_source=args.benign_cap_per_source,
    )

    print("\n[run_gap_fill] === summary ===")
    print(f"loaded sources: {sorted(loaded)}")
    missing = sorted(set(GAP_SOURCES) - loaded - set(args.only or []))
    if args.only is None and missing:
        print(f"FAILED sources: {missing}")
    print(f"rows per source: {dict(counts)}")
    print(f"pool sizes: { {k: len(v) for k, v in pools.items()} }")
    print(f"safe rows (folded into benign_clear): {len(safe_rows)}")

    out_dir.mkdir(parents=True, exist_ok=True)

    by_source: dict[str, list] = {}
    for atk, rows in pools.items():
        for r in rows:
            by_source.setdefault(r.source_dataset, []).append(r)

    for source_name, rows in by_source.items():
        _write_jsonl(out_dir / f"{source_name}.jsonl", rows)
        print(f"  wrote {len(rows):>6} rows -> {out_dir / (source_name + '.jsonl')}")

    for atk, rows in pools.items():
        _write_jsonl(out_dir / "by_attack" / f"{atk}.jsonl", rows)

    summary = {
        "mode": "probe" if args.probe else "compile",
        "max_rows_per_source": max_rows,
        "per_source_cap": per_source_cap,
        "keep_positive_only": args.keep_positive_only,
        "benign_cap_per_source": args.benign_cap_per_source,
        "only": args.only,
        "loaded": sorted(loaded),
        "failed": sorted(set(GAP_SOURCES) - loaded) if args.only is None else [],
        "rows_per_source": dict(counts),
        "pool_sizes": {k: len(v) for k, v in pools.items()},
        "safe_rows": len(safe_rows),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[run_gap_fill] wrote summary -> {out_dir / 'summary.json'}")

    return 0 if loaded else 1


if __name__ == "__main__":
    sys.exit(main())
