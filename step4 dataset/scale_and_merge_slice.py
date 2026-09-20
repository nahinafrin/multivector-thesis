#!/usr/bin/env python3
"""scale_and_merge_slice.py - grow a small evaluation slice and merge it with
the existing one, deduplicated by prompt text.

Wraps whichever slice builder you already have (build_semantic_slice.py for
the n=8 misinformation slice, build_planted_attacks.py for the n=30 mitigation
slice) rather than reimplementing construction logic - this script's only job
is to call the builder at a larger --n, merge with what you already scored,
and dedup so you don't have to throw away the existing n=8/n=30 results.

USAGE
------
    # grow the semantic (misinformation) slice from 8 to 40 rows:
    python scale_and_merge_slice.py \
        --builder build_semantic_slice.py --builder-args "--out semantic_slice_v2.jsonl --n 40" \
        --existing semantic_slice.jsonl \
        --out semantic_slice_merged.jsonl

    # grow the planted-attack mitigation slice from 30 to 150:
    python scale_and_merge_slice.py \
        --builder build_planted_attacks.py \
        --builder-args "--qa-jsonl data/question-answer/test.jsonl --n 150 --yesno-only --out planted_attacks_v3.jsonl" \
        --existing planted_attacks_v2.jsonl \
        --out planted_attacks_merged.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path


def read_jsonl(path: str) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def dedup_key(row: dict) -> str:
    text = row.get("prompt") or row.get("question") or json.dumps(row, sort_keys=True)
    return re.sub(r"\s+", " ", str(text).strip().lower())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--builder", required=True, help="e.g. build_semantic_slice.py")
    ap.add_argument("--builder-args", required=True, help="quoted CLI args for the builder")
    ap.add_argument("--existing", required=True, help="the current, already-scored slice")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[run] python {args.builder} {args.builder_args}")
    result = subprocess.run([sys.executable, args.builder, *shlex.split(args.builder_args)])
    if result.returncode != 0:
        print("[error] builder failed - fix it before merging.", file=sys.stderr)
        sys.exit(result.returncode)

    # The builder's --out path is embedded in builder-args; extract it so we
    # know what to read back without re-parsing every builder's own argparse.
    new_out = None
    parts = shlex.split(args.builder_args)
    if "--out" in parts:
        new_out = parts[parts.index("--out") + 1]
    if not new_out:
        print("[error] could not find --out in --builder-args; pass it explicitly.", file=sys.stderr)
        sys.exit(1)

    existing_rows = read_jsonl(args.existing)
    new_rows = read_jsonl(new_out)
    seen = {dedup_key(r) for r in existing_rows}

    merged = list(existing_rows)
    added = 0
    for r in new_rows:
        k = dedup_key(r)
        if k in seen:
            continue
        seen.add(k)
        merged.append(r)
        added += 1

    with open(args.out, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[merge] existing={len(existing_rows)} new_from_builder={len(new_rows)} "
          f"added={added} total={len(merged)} -> {args.out}")
    print("\nNEXT STEP: re-run run_full_pipeline.py + score_attack_success.py + "
          "grounding_separation_probe.py on --out, and report the CIs via "
          "stats_utils.wilson_ci / bootstrap_ci instead of a bare point estimate.")


if __name__ == "__main__":
    main()
