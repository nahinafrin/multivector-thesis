#!/usr/bin/env python3
"""
compat_check.py -- non-destructive compatibility test for external datasets.
v2: uses REAL per-row ground truth (deepset's "label" field) instead of a
hardcoded True, and stamps the real "safety" field step_03c_fusion_gate.py's
--eval reads (confirmed from source: gt = row.get("safety","").lower()=="unsafe").
Only ever CREATES new compat_*.jsonl files; nothing else in the repo is touched.
"""

import argparse
import json
import sys
from pathlib import Path

SLICE_SCHEMA = {
    "prompt_field": "prompt",
    "context_field": None,
    "kind_value": "external_compat_test",
    "should_block_fallback": True,
    "extra_ground_truth_fields": {
        "safety": {"positive": "unsafe", "negative": "safe"},
    },
}

CANDIDATE_TEXT_FIELDS = [
    "prompt", "text", "input", "query", "question", "original_prompt",
    "modified_prompt", "attack_prompt", "instruction",
]
CANDIDATE_CONTEXT_FIELDS = [
    "context", "document", "chunk", "passage", "retrieved_text", "poison",
]
CANDIDATE_LABEL_FIELDS = [
    "label", "is_injection", "is_attack", "attack", "malicious", "class", "target",
]
POSITIVE_LABEL_VALUES = {1, "1", True, "true", "True", "injection", "malicious", "attack", "unsafe"}
NEGATIVE_LABEL_VALUES = {0, "0", False, "false", "False", "legitimate", "benign", "safe"}


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


def resolve_label(row: dict, label_key, fallback: bool):
    if label_key is not None and label_key in row:
        raw = row[label_key]
        if raw in POSITIVE_LABEL_VALUES:
            return True, raw, False
        if raw in NEGATIVE_LABEL_VALUES:
            return False, raw, False
        print(f"  WARNING: unrecognized label value {raw!r} in field '{label_key}'; falling back to {fallback}.")
        return fallback, raw, True
    return fallback, None, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", required=True)
    ap.add_argument("--rowname", required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.external)
    if not src.exists():
        raise SystemExit(f"File not found: {src}")

    first_row = sniff_first_row(src)
    print("First real row:")
    print(json.dumps(first_row, ensure_ascii=False, indent=2)[:1000])
    print()

    text_key = next((k for k in CANDIDATE_TEXT_FIELDS if k in first_row), None)
    if text_key is None:
        print(f"WARNING: no text field matched. Real fields present: {list(first_row.keys())}")
        sys.exit(1)
    print(f"Using '{text_key}' as the attack-text field.")

    context_key = next((k for k in CANDIDATE_CONTEXT_FIELDS if k in first_row), None)
    label_key = next((k for k in CANDIDATE_LABEL_FIELDS if k in first_row), None)
    if label_key:
        print(f"Using '{label_key}' as a REAL ground-truth label field.")
    else:
        print(f"No label field found; every row falls back to should_block={SLICE_SCHEMA['should_block_fallback']}.")
    print()

    out_rows, n_seen, n_skipped, n_pos, n_neg, n_fallback = [], 0, 0, 0, 0, 0
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
            is_positive, raw_label, used_fallback = resolve_label(row, label_key, SLICE_SCHEMA["should_block_fallback"])
            if used_fallback:
                n_fallback += 1
            n_pos += 1 if is_positive else 0
            n_neg += 0 if is_positive else 1
            new_row = {
                SLICE_SCHEMA["prompt_field"]: text,
                "kind": SLICE_SCHEMA["kind_value"],
                "expectation": {"should_block": is_positive},
                "_source_dataset": src.name,
                "_source_row_index": n_seen - 1,
                "_source_label": raw_label,
            }
            if SLICE_SCHEMA["context_field"] and context_key:
                new_row[SLICE_SCHEMA["context_field"]] = row.get(context_key, "")
            for field_name, mapping in SLICE_SCHEMA["extra_ground_truth_fields"].items():
                new_row[field_name] = mapping["positive"] if is_positive else mapping["negative"]
            out_rows.append(new_row)

    print(f"Adapted {len(out_rows)} rows ({n_skipped} skipped).")
    print(f"Label distribution: {n_pos} positive, {n_neg} negative, {n_fallback} fallback.\n")

    if args.dry_run:
        for r in out_rows[:3]:
            print(json.dumps(r, ensure_ascii=False, indent=2))
        return

    out_path = Path(f"compat_{args.rowname}_slice.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(out_rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
