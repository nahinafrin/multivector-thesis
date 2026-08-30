"""
check_poison_content_id206.py

Determines whether "Qwen2.5:3b" appearing in id=206's NEW answer is a
genuine spontaneous self-referential hallucination, or the model simply
echoing/paraphrasing text already present in the retrieved/injected
context -- a very different, and arguably more important, finding either
way. Writes UTF-8 via Python's own file handle to avoid the console
encoding issue hit earlier this turn.

Usage:
    python check_poison_content_id206.py
"""
from __future__ import annotations

import json

orig = {
    r.get("question"): r
    for r in (json.loads(l) for l in open("grounded_controller.jsonl", encoding="utf-8"))
    if r.get("kind") == "multivector_attack"
}
new = {
    r.get("question"): r
    for r in (json.loads(l) for l in open("repro_test_on4.jsonl", encoding="utf-8") if l.strip())
}
baseline = {b["id"]: b for b in json.load(open("repro_test_baseline.json", encoding="utf-8"))}

q = baseline[206]["question"]
o = orig.get(q, {})
n = new.get(q, {})

with open("id206_context_check_utf8.txt", "w", encoding="utf-8") as f:
    f.write(f"Q: {q}\n\n")
    for label, row in (("ORIGINAL", o), ("NEW", n)):
        ret = row.get("retrieval", {})
        f.write(f"===== {label} =====\n")
        f.write(f"poison_injected: {ret.get('poison_injected')!r}\n\n")
        for field_name in ("raw_chunks", "sanitized_chunks", "ranked_chunks"):
            chunks = ret.get(field_name, [])
            f.write(f"{field_name}:\n")
            for i, c in enumerate(chunks):
                flag = "  <-- CONTAINS 'qwen'" if "qwen" in str(c).lower() else ""
                f.write(f"  [{i}] {c!r}{flag}\n")
            f.write("\n")
        pj = ret.get("poison_injected")
        if pj and "qwen" in str(pj).lower():
            f.write("*** 'qwen' FOUND in poison_injected ***\n\n")
        else:
            f.write("(no 'qwen' match in poison_injected)\n\n")

print("wrote id206_context_check_utf8.txt")
