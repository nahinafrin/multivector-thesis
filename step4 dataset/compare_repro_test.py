"""
compare_repro_test.py

Compares the CURRENT .venv311 environment's blocked/not-blocked verdicts
on repro_test_slice.jsonl against the ORIGINAL verdicts already recorded
in grounded_controller.jsonl for those same exact questions -- a direct
functional reproducibility check of the rebuilt environment against the
one that produced the locked internal numbers.

Usage:
    python compare_repro_test.py <new_results_file.jsonl>
"""
from __future__ import annotations

import json
import sys


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python compare_repro_test.py <new_results_file.jsonl>")
        return

    baseline = {b["id"]: b for b in json.load(open("repro_test_baseline.json", encoding="utf-8"))}
    new_rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]

    matched = 0
    mismatched = []
    for r in new_rows:
        rid = r.get("id", r.get("index"))
        b = baseline.get(rid)
        if b is None:
            continue
        new_blocked = r.get("blocked")
        if new_blocked == b["original_blocked"]:
            matched += 1
        else:
            mismatched.append((rid, b["kind"], b["question"][:60], b["original_blocked"], new_blocked))

    total = matched + len(mismatched)
    print(f"blocked-decision match: {matched}/{total}")
    if mismatched:
        print("\nMISMATCHES (original environment vs current environment):")
        for rid, kind, q, orig, new in mismatched:
            print(f"  id={rid} kind={kind:18} orig_blocked={orig!s:5} new_blocked={new!s:5}  Q: {q}")
        print("\nA mismatch here means the rebuilt environment behaves "
              "differently from whatever produced the locked internal numbers "
              "-- worth reporting honestly rather than treating those numbers "
              "as unconditionally reproducible.")
    else:
        print("All blocked/not-blocked decisions matched -- strong evidence "
              "the current pinned environment reproduces the original one "
              "functionally, even though it isn't literally the same venv.")


if __name__ == "__main__":
    main()
