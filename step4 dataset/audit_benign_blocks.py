#!/usr/bin/env python3
"""Summarize benign block reasons from a benign ON-arm JSONL."""
from __future__ import annotations
import argparse, json
from collections import Counter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="path", required=True)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.path, encoding="utf-8") if l.strip()]
    blocked = [r for r in rows if r.get("blocked")]
    reasons = Counter()
    mit = Counter()
    for r in blocked:
        br = r.get("block_reason") or "unknown"
        reasons[br.split(":")[0] if ":" in br else br] += 1
        for layer in r.get("mitigation_applied") or []:
            mit[layer] += 1
        if r.get("block_reason") and "refuse_on_detection" in str(r.get("block_reason")):
            mit["refuse_on_detection (block)"] += 1
    print(f"rows={len(rows)} blocked={len(blocked)} ({100*len(blocked)/len(rows):.1f}%)")
    print("\nblock_reason (prefix):")
    for k, v in reasons.most_common():
        print(f"  {k}: {v}")
    if mit:
        print("\nmitigation_applied on blocked rows:")
        for k, v in mit.most_common():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
