"""
compare_repro_test_v2.py

Same purpose as compare_repro_test.py, but joins on QUESTION TEXT instead
of a numeric id/index field. run_mitigation_ab.py appears to use its own
id/index numbering rather than preserving the original grounded_controller
.jsonl row indices repro_test_baseline.json was built with -- question
text is a far more robust join key here, unlikely to collide across ~24
distinct trivia questions.

Usage:
    python compare_repro_test_v2.py [results_file.jsonl]
    (defaults to repro_test_on.jsonl if no argument given)
"""
from __future__ import annotations

import json
import sys


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "repro_test_on.jsonl"

    baseline_list = json.load(open("repro_test_baseline.json", encoding="utf-8"))
    baseline = {b["question"]: b for b in baseline_list}
    new_rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

    matched = 0
    mismatched = []
    unmatched_new = []
    for r in new_rows:
        q = r.get("question")
        b = baseline.get(q)
        if b is None:
            unmatched_new.append(q)
            continue
        new_blocked = r.get("blocked")
        if new_blocked == b["original_blocked"]:
            matched += 1
        else:
            mismatched.append((b["id"], b["kind"], q[:60] if q else "", b["original_blocked"], new_blocked))

    total = matched + len(mismatched)
    print(f"blocked-decision match (question-text joined): {matched}/{total}")
    print(f"  (baseline had {len(baseline_list)} rows, new output had {len(new_rows)} rows, "
          f"{len(unmatched_new)} new rows had no question-text match at all)")

    if unmatched_new:
        print("\nnew-output questions with NO baseline match "
              "(check for whitespace/truncation differences, or genuinely different rows):")
        for q in unmatched_new[:5]:
            print(f"  {q!r}")

    if mismatched:
        print("\nMISMATCHES (original environment vs current environment):")
        for rid, kind, q, orig, new in mismatched:
            print(f"  id={rid} kind={kind:18} orig_blocked={orig!s:5} new_blocked={new!s:5}  Q: {q}")
    elif total > 0:
        print("\nAll matched decisions agree.")


if __name__ == "__main__":
    main()
