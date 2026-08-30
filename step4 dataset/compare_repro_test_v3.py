"""
compare_repro_test_v3.py

Joins on the original row id/index rather than question text, to avoid
the dict-key-collision risk found when checking repro_test_on4.jsonl:
two baseline rows (ids 201 and 206) share what LOOKS like the same
question text when truncated for display, and a question-text-keyed dict
silently drops one of two colliding entries.

Checks both "id" and "index" fields in the new output, since different
scripts this session have used different field names for the same concept.

Usage:
    python compare_repro_test_v3.py [results_file.jsonl]
"""
from __future__ import annotations

import json
import sys


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "repro_test_on4.jsonl"

    baseline_list = json.load(open("repro_test_baseline.json", encoding="utf-8"))
    baseline = {b["id"]: b for b in baseline_list}
    new_rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

    matched = 0
    mismatched = []
    unmatched = []
    for r in new_rows:
        rid = r.get("id")
        if rid is None:
            rid = r.get("index")
        b = baseline.get(rid)
        if b is None:
            unmatched.append(rid)
            continue
        new_blocked = r.get("blocked")
        if new_blocked == b["original_blocked"]:
            matched += 1
        else:
            mismatched.append((rid, b["kind"], b["question"][:70], b["original_blocked"], new_blocked))

    total = matched + len(mismatched)
    print(f"blocked-decision match (id-joined): {matched}/{total}")
    print(f"  (baseline={len(baseline_list)} rows, new_rows={len(new_rows)}, "
          f"unmatched id/index values: {unmatched})")

    if mismatched:
        print("\nMISMATCHES:")
        for rid, kind, q, orig, new in mismatched:
            print(f"  id={rid} kind={kind:18} orig_blocked={orig!s:5} new_blocked={new!s:5}  Q: {q}")
    elif total > 0:
        print("\nAll matched decisions agree.")


if __name__ == "__main__":
    main()
