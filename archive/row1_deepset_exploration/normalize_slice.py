#!/usr/bin/env python3
"""
normalize_slice.py -- the mitigation: reverses obfuscate_slice.py's three
techniques (base64 decode, strip invisible Unicode, NFKC normalize) before
text reaches the classifier. Only CREATES a new file.
"""

import argparse
import base64
import json
import re
import unicodedata
from pathlib import Path

BASE64_SPAN_RE = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")


def decode_base64_spans(text: str) -> str:
    def try_decode(match: re.Match) -> str:
        candidate = match.group(0)
        try:
            decoded = base64.b64decode(candidate, validate=True).decode("utf-8")
        except Exception:
            return candidate
        printable = sum(1 for c in decoded if c.isprintable())
        if len(decoded) == 0 or printable / len(decoded) < 0.85:
            return candidate
        return f"{candidate} [DECODED: {decoded}]"
    return BASE64_SPAN_RE.sub(try_decode, text)


def strip_invisible_chars(text: str) -> str:
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


def nfkc_normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def normalize(text: str) -> str:
    return nfkc_normalize(strip_invisible_chars(decode_base64_spans(text)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", required=True)
    ap.add_argument("--rowname", required=True)
    args = ap.parse_args()

    src = Path(args.external)
    if not src.exists():
        raise SystemExit(f"File not found: {src}")

    out_rows = []
    n_changed = 0
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            original_prompt = row.get("prompt", "")
            normalized_prompt = normalize(original_prompt)
            if normalized_prompt != original_prompt:
                n_changed += 1
            new_row = dict(row)
            new_row["prompt"] = normalized_prompt
            new_row["_pre_normalization_prompt"] = original_prompt
            out_rows.append(new_row)

    out_path = Path(f"compat_{args.rowname}_obfuscated_normalized_slice.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Normalized {len(out_rows)} rows ({n_changed} changed) -> {out_path}")


if __name__ == "__main__":
    main()
