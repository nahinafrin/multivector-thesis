"""
extract_repro_test_slice.py (v2 -- multivector-only, full n)

run_mitigation_ab.py is scoped specifically to multivector_attack rows by
design ("Mitigation OFF vs ON on multi-vector attacks" per its own --help
text) -- that's why the earlier 4-per-kind stratified slice only produced
4 output rows; the other 20 were filtered by the tool itself, not dropped
by a bug.

This replaces that slice with ALL 30 multivector_attack rows from
grounded_controller.jsonl -- full n for the one kind this tool supports,
and the class the thesis's core claims center on regardless.

Overwrites repro_test_slice.jsonl / repro_test_baseline.json so the
existing compare_repro_test_v2.py works completely unmodified.

Usage:
    python extract_repro_test_slice.py
"""
from __future__ import annotations

import json

rows = [json.loads(l) for l in open("grounded_controller.jsonl", encoding="utf-8")]
mv_rows = [r for r in rows if r["kind"] == "multivector_attack"]
print(f"found {len(mv_rows)} multivector_attack rows")

slice_rows = []
baseline = []
for r in mv_rows:
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

print(f"wrote {len(slice_rows)} rows -> repro_test_slice.jsonl / repro_test_baseline.json")
