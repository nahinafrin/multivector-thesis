"""
extract_repro_test_slice.py

Pulls a small (~20-24 row), stratified sample of REAL questions + their
ORIGINAL recorded verdicts from grounded_controller.jsonl -- the file
behind the locked internal 130/130 result -- so those exact verdicts can
be compared against what the CURRENT, freshly-rebuilt .venv311 environment
produces for the same questions through the same (unmodified this session)
controller code path.

This only extracts the comparison baseline -- it does not run anything.

Usage:
    python extract_repro_test_slice.py
"""
from __future__ import annotations

import json
import random
from collections import defaultdict

random.seed(42)

rows = [json.loads(l) for l in open("grounded_controller.jsonl", encoding="utf-8")]
by_kind: dict[str, list[dict]] = defaultdict(list)
for r in rows:
    by_kind[r["kind"]].append(r)

sample: list[dict] = []
for kind, items in by_kind.items():
    n = min(4, len(items))
    sample.extend(random.sample(items, n))

slice_rows = []
baseline = []
for r in sample:
    rid = r.get("index", r.get("id"))
    slice_rows.append({
        "kind": r["kind"],
        "question": r["question"],
        "ground_truth": r.get("ground_truth"),
        "true_answer": r.get("true_answer"),
        "success_marker": r.get("success_marker"),
        "poison_chunk": r.get("retrieval", {}).get("poison_injected"),
        "attack_type": r.get("attack_type", ""),
        "expectation": r.get("expectation", {}),
        "id": rid,
    })
    baseline.append({
        "id": rid,
        "kind": r["kind"],
        "question": r["question"],
        "original_blocked": r.get("blocked"),
        "original_block_stage": r.get("block_stage"),
    })

with open("repro_test_slice.jsonl", "w", encoding="utf-8") as f:
    for r in slice_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with open("repro_test_baseline.json", "w", encoding="utf-8") as f:
    json.dump(baseline, f, indent=2, ensure_ascii=False)

print(f"wrote {len(slice_rows)} rows across {len(by_kind)} kinds")
print("  -> repro_test_slice.jsonl   (feed this through the controller path)")
print("  -> repro_test_baseline.json (original verdicts to compare against)")
