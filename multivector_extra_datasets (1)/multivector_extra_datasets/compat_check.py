#!/usr/bin/env python3
"""
compat_check.py — non-destructive compatibility test for external datasets

Purpose: check whether your pipeline (run_full_pipeline.py) can ingest and
process rows from an external dataset (BIPIA, deepset, Mindgard, TAB,
ai4privacy, SafeRAG), WITHOUT modifying anything in your existing project:

  - Does NOT edit build_adversarial_slice.py, run_full_pipeline.py, or any
    other script in your repo.
  - Does NOT overwrite adversarial_slice.jsonl or any existing results file.
  - Does NOT write into your FAISS index — retrieval against kb_wiki is
    read-only in the normal pipeline; this script never touches index files.
  - Only ever CREATES new files, all prefixed "compat_" so they're impossible
    to confuse with your real data and trivial to delete afterward:
        compat_<rowname>_slice.jsonl   <- the adapted input slice (new)
        compat_<rowname>_out.jsonl     <- the pipeline's output (new)

If anything looks wrong, the fix is: delete the compat_*.jsonl files and
re-run. Nothing else in the repo is ever touched.

--- Step 0: confirm your REAL slice schema first (don't skip this) ---

Before running this script, look at one real row of your own, already-working
adversarial_slice.jsonl so you know exactly which field names
run_full_pipeline.py expects:

    PowerShell:
        Get-Content adversarial_slice.jsonl -TotalCount 1

This script ships with a best-guess schema (a "prompt" field, optionally
"context" for RAG-poisoning-style rows, "kind" and "expectation.should_block"
metadata, matching the fields referenced throughout your project's own
addenda) — but "best-guess" is exactly the kind of assumption that has caused
real bugs in this project before (field-name mismatches were the root cause
of two separate scorer bugs). Open this file and check/edit the
SLICE_SCHEMA section below against your real row before trusting the output.

--- Usage ---

Dry run first (just shows you what would be written, touches no files):

    python compat_check.py --external row4_indirect_injection_rag_poisoning/bipia_indirect_injection.jsonl --rowname bipia_row4 --dry-run --limit 5

Once the dry-run output looks right, write the new slice file for real:

    python compat_check.py --external row4_indirect_injection_rag_poisoning/bipia_indirect_injection.jsonl --rowname bipia_row4 --limit 20

Then, separately and manually (this script does NOT invoke your pipeline for
you, on purpose — you stay in control of when your own code actually runs):

    python run_full_pipeline.py --slice compat_bipia_row4_slice.jsonl --out compat_bipia_row4_out.jsonl --index kb_wiki

Inspect compat_bipia_row4_out.jsonl the same way you'd inspect any other
pipeline output. When you're done, delete the compat_*.jsonl files — nothing
elsewhere in the repo was ever modified.
"""

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# SLICE_SCHEMA — EDIT THIS to match your real adversarial_slice.jsonl fields
# before trusting the output. This is a best guess based on field names
# referenced in the project's own addenda (prompt, kind, expectation.should_block).
# ---------------------------------------------------------------------------
SLICE_SCHEMA = {
    "prompt_field": "prompt",              # the field your pipeline reads as the user query/text
    "context_field": None,                 # set to e.g. "context" if your schema has a separate
                                            # poisoned-context field for RAG-poisoning-style rows;
                                            # leave None if poison text should be folded into "prompt"
    "kind_value": "external_compat_test",  # a clearly-labeled kind so these rows are never mistaken
                                            # for your own hand-built categories in any downstream report
    "should_block": True,                  # every row here is an attack construction, so True is the
                                            # honest default expectation
}

# Candidate field names to try, per known dataset, when auto-extracting the
# attack text out of each external dataset's own (different) schema. Extend
# this if a dataset you add has different field names than expected —
# inspect one real row first (the script prints one before processing).
CANDIDATE_TEXT_FIELDS = [
    "prompt", "text", "input", "query", "question", "original_prompt",
    "modified_prompt", "attack_prompt", "instruction",
]
CANDIDATE_CONTEXT_FIELDS = [
    "context", "document", "chunk", "passage", "retrieved_text", "poison",
]


def sniff_first_row(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        first = f.readline().strip()
    if not first:
        raise SystemExit(f"{path} appears to be empty.")
    return json.loads(first)


def extract_text(row: dict, candidates: list) -> str | None:
    for key in candidates:
        if key in row and isinstance(row[key], str) and row[key].strip():
            return row[key]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--external", required=True, help="Path to a downloaded external dataset JSONL file.")
    ap.add_argument("--rowname", required=True, help="Short label, e.g. 'bipia_row4' — used in output filenames.")
    ap.add_argument("--limit", type=int, default=20, help="How many rows to adapt (default 20 — start small).")
    ap.add_argument("--dry-run", action="store_true", help="Print what would be written; create no files.")
    args = ap.parse_args()

    src = Path(args.external)
    if not src.exists():
        raise SystemExit(f"File not found: {src}")

    first_row = sniff_first_row(src)
    print("First real row of your external dataset (inspect this before trusting the mapping):")
    print(json.dumps(first_row, ensure_ascii=False, indent=2)[:1000])
    print()

    text_key = None
    for key in CANDIDATE_TEXT_FIELDS:
        if key in first_row:
            text_key = key
            break
    if text_key is None:
        print("WARNING: none of the candidate text-field names matched this dataset's real schema.")
        print(f"  Candidates tried: {CANDIDATE_TEXT_FIELDS}")
        print(f"  Real fields present: {list(first_row.keys())}")
        print("  Edit CANDIDATE_TEXT_FIELDS in this script to add the real field name, then re-run.")
        sys.exit(1)
    print(f"Using '{text_key}' as the attack-text field for this dataset.\n")

    context_key = None
    for key in CANDIDATE_CONTEXT_FIELDS:
        if key in first_row:
            context_key = key
            break

    out_rows = []
    n_seen = 0
    n_skipped = 0
    with open(src, encoding="utf-8") as f:
        for line in f:
            if n_seen >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_seen += 1
            text = extract_text(row, [text_key])
            if not text:
                n_skipped += 1
                continue

            new_row = {
                SLICE_SCHEMA["prompt_field"]: text,
                "kind": SLICE_SCHEMA["kind_value"],
                "expectation": {"should_block": SLICE_SCHEMA["should_block"]},
                "_source_dataset": src.name,
                "_source_row_index": n_seen - 1,
            }
            if SLICE_SCHEMA["context_field"] and context_key:
                new_row[SLICE_SCHEMA["context_field"]] = row.get(context_key, "")

            out_rows.append(new_row)

    print(f"Adapted {len(out_rows)} rows ({n_skipped} skipped for missing/empty text field).\n")

    if args.dry_run:
        print("--dry-run set: no files written. Sample of what would be produced:\n")
        for r in out_rows[:3]:
            print(json.dumps(r, ensure_ascii=False, indent=2))
        print(f"\n... and {max(0, len(out_rows) - 3)} more rows.")
        print("\nIf this looks right, re-run without --dry-run to write compat_<rowname>_slice.jsonl.")
        return

    out_path = Path(f"compat_{args.rowname}_slice.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(out_rows)} rows -> {out_path}")
    print(f"\nNo existing files were modified. To actually test the pipeline against these rows, run:")
    print(f"  python run_full_pipeline.py --slice {out_path} --out compat_{args.rowname}_out.jsonl --index kb_wiki")
    print(f"\nWhen finished, delete compat_{args.rowname}_slice.jsonl and compat_{args.rowname}_out.jsonl "
          f"— nothing else was touched.")


if __name__ == "__main__":
    main()
