"""
inspect_repro_mismatches.py

Full side-by-side detail for each of the 5 reproducibility mismatches --
prints the full grounding/generation/multivector sub-dicts rather than
guessing at individual field names within them, so nothing gets missed.

Usage:
    python inspect_repro_mismatches.py
"""
from __future__ import annotations

import json

MISMATCH_IDS = [106, 129, 172, 201, 206]

orig_rows = {
    r.get("index", r.get("id")): r
    for r in (json.loads(l) for l in open("grounded_controller.jsonl", encoding="utf-8"))
    if r.get("kind") == "multivector_attack"
}
new_rows = {
    r.get("index"): r
    for r in (json.loads(l) for l in open("repro_test_on4.jsonl", encoding="utf-8") if l.strip())
}


def answer_text(r: dict) -> str:
    return r.get("final_response") or (r.get("generation") or {}).get("answer") or ""


for rid in MISMATCH_IDS:
    o = orig_rows.get(rid, {})
    n = new_rows.get(rid, {})
    print("=" * 90)
    print(f"id={rid}")
    print(f"question: {o.get('question', n.get('question', ''))!r}")
    print()
    print(f"ORIGINAL: blocked={o.get('blocked')}  block_stage={o.get('block_stage')!r}")
    print(f"  block_reason: {o.get('block_reason')!r}")
    print(f"  grounding: {o.get('grounding')}")
    print(f"  multivector: {o.get('multivector')}")
    print(f"  generation.disagreement: {(o.get('generation') or {}).get('disagreement')}")
    print(f"  answer: {answer_text(o)[:200]!r}")
    print()
    print(f"NEW:      blocked={n.get('blocked')}  block_stage={n.get('block_stage')!r}")
    print(f"  block_reason: {n.get('block_reason')!r}")
    print(f"  grounding: {n.get('grounding')}")
    print(f"  multivector: {n.get('multivector')}")
    print(f"  generation.disagreement: {(n.get('generation') or {}).get('disagreement')}")
    print(f"  answer: {answer_text(n)[:200]!r}")
    print()
