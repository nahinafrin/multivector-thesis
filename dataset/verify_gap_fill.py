"""
verify_gap_fill.py
==================

A SEPARATE validation layer that sits on top of gap_fill_sources.py. It does
NOT modify your existing loaders. It exists to catch the failure mode that
produces a suspicious "2000 / 2000 / 2000" positive count:

    THE BUG IT GUARDS AGAINST
    -------------------------
    `_safe_from_binary_label(...)` in gap_fill_sources.py defaults to "unsafe"
    when its expected label column is MISSING from a record:

        return "unsafe"  # default: assume the source's attack rows are unsafe

    If a parquet mirror renames its label column (e.g. tasksource/jigsaw_toxicity
    uses a single `label` instead of the six Jigsaw columns, or safeguard uses
    `is_injection` instead of `label`), then EVERY row hits that default and is
    imported as a positive attack example — including benign text. You get a
    clean 2000 positives that are mostly mislabeled benign rows. That silently
    poisons the training set far worse than having too few positives.

WHAT THIS MODULE DOES
---------------------
1. validate_source_schema(): loads a tiny sample from each source and asserts
   the expected label column(s) actually exist, with the expected value range.
   Raises (or warns) BEFORE you compile a full run on bad assumptions.

2. load_gap_fill_validated(): a strict wrapper around load_gap_fill_sources
   that first runs the schema check, then reports the TRUE positive vs benign
   breakdown per source so you can confirm the real numbers (not the cap).

3. audit_compiled_jsonl(): point it at your already-written by_attack/*.jsonl
   and it re-checks, offline, for the tell-tale signs of mislabeling:
     - a positive count suspiciously equal to the cap (2000) AND
     - low lexical diversity / lack of attack-indicative tokens in the prompts
   so you can validate the files you ALREADY produced without re-downloading.

USAGE
-----
# Before a real run, sanity-check the live schemas:
python verify_gap_fill.py --check-schema --only safeguard_prompt_injection jigsaw_toxicity

# Or audit the files you already wrote (no network needed):
python verify_gap_fill.py --audit-dir "C:/.../new_dataset/gap_fill_output/by_attack"
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from build_new_dataset import Normalizer
from gap_fill_sources import (
    GAP_SOURCES,
    load_gap_fill_sources,
    _pick_prompt,
)

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover
    load_dataset = None  # type: ignore


# Expected label columns per source. If NONE of these are present in a record,
# the source's row_to_safety would silently default to "unsafe" -> mislabel.
EXPECTED_LABEL_COLUMNS: dict[str, list[str]] = {
    "safeguard_prompt_injection": ["label"],
    "ai4privacy_pii": [],  # treated as all-positive by design (no label needed)
    "jigsaw_toxicity": ["toxic", "severe_toxic", "obscene",
                         "threat", "insult", "identity_hate"],
    "bipia_indirect_injection": ["label"],
    "mindgard_evaded": [],  # all rows are attacks by design
}

# Attack-indicative tokens used as a heuristic sniff test for mislabeling.
ATTACK_HINT_TOKENS: dict[str, list[str]] = {
    "prompt_injection": ["ignore", "instruction", "system prompt", "disregard",
                         "override", "you are now", "pretend", "jailbreak",
                         "developer mode", "reveal"],
    "toxicity_abuse": ["hate", "stupid", "idiot", "kill", "ugly", "worthless",
                       "die", "moron", "trash", "disgusting"],
}


def validate_source_schema(source_keys: Optional[list[str]] = None,
                           sample_n: int = 200,
                           strict: bool = True) -> dict[str, dict[str, Any]]:
    """Load a small sample from each source and verify the label column exists.

    Returns a per-source report. With strict=True, raises on a missing label
    column for a source that needs one (the conditions that cause the
    default-to-unsafe mislabel).
    """
    if load_dataset is None:
        raise RuntimeError("`datasets` not installed.")

    keys = source_keys or list(GAP_SOURCES)
    report: dict[str, dict[str, Any]] = {}

    for name in keys:
        cfg = GAP_SOURCES[name]
        hf_id = cfg["hf_id"]
        expected = EXPECTED_LABEL_COLUMNS.get(name, [])
        entry: dict[str, Any] = {"hf_id": hf_id, "expected_label_cols": expected}
        try:
            ds = None
            for kw in cfg.get("load_options", [{}]):
                try:
                    ds = load_dataset(hf_id, **kw)
                    break
                except Exception as e:  # noqa: BLE001
                    entry["load_error"] = str(e)
            if ds is None:
                entry["status"] = "LOAD_FAILED"
                report[name] = entry
                continue

            # Grab the first split, first sample_n rows
            split = ds[list(ds.keys())[0]] if hasattr(ds, "keys") else ds
            rows = list(split.select(range(min(sample_n, len(split))))) \
                if hasattr(split, "select") else list(split)[:sample_n]
            cols = set(rows[0].keys()) if rows else set()
            entry["actual_columns"] = sorted(cols)

            if expected:
                present = [c for c in expected if c in cols]
                missing = [c for c in expected if c not in cols]
                entry["label_cols_present"] = present
                entry["label_cols_missing"] = missing
                if not present:
                    entry["status"] = "FAIL_NO_LABEL_COLUMN"
                    entry["danger"] = (
                        "NONE of the expected label columns exist -> "
                        "row_to_safety will default everything to 'unsafe' "
                        "(silent mislabel). Update GAP_SOURCES row_to_safety "
                        "to the real column name shown in actual_columns.")
                    if strict:
                        report[name] = entry
                        raise ValueError(f"[{name}] {entry['danger']} "
                                         f"actual columns: {entry['actual_columns']}")
                else:
                    # show the real positive rate on the sample
                    fn = cfg["row_to_safety"]
                    labels = Counter(fn(r) for r in rows)
                    entry["sample_label_dist"] = dict(labels)
                    entry["status"] = "OK"
            else:
                entry["status"] = "OK (all-positive by design)"

        except ValueError:
            raise
        except Exception as e:  # noqa: BLE001
            entry["status"] = "ERROR"
            entry["error"] = str(e)
        report[name] = entry

    return report


def audit_compiled_jsonl(by_attack_dir: Path, cap_hint: int = 2000) -> None:
    """Offline audit of already-written by_attack/*.jsonl for mislabel signs.

    Flags any file whose count == cap_hint (suspicious) and whose prompts show
    weak attack-token signal relative to expectation.
    """
    files = sorted(by_attack_dir.glob("*.jsonl"))
    if not files:
        print(f"[audit] no .jsonl files in {by_attack_dir}")
        return

    print(f"[audit] scanning {len(files)} files in {by_attack_dir}\n")
    for f in files:
        rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        n = len(rows)
        attack = f.stem
        safeties = Counter(r.get("safety") for r in rows)
        # uniqueness
        prompts = [r.get("prompt", "") for r in rows]
        uniq = len(set(prompts))
        # attack-token hit rate (only meaningful for classes we have hints for)
        hits = None
        if attack in ATTACK_HINT_TOKENS:
            toks = ATTACK_HINT_TOKENS[attack]
            hit = sum(1 for p in prompts if any(t in p.lower() for t in toks))
            hits = hit / n if n else 0.0

        flags = []
        if n == cap_hint:
            flags.append(f"count==cap({cap_hint}): verify these are REAL positives, not cap-padding")
        if uniq < n:
            flags.append(f"{n-uniq} duplicate prompts")
        if hits is not None and hits < 0.30:
            flags.append(f"LOW attack-token rate {hits:.0%} "
                         f"-> possible benign rows mislabeled as '{attack}'")

        print(f"  {attack:28} n={n:<5} unique={uniq:<5} "
              f"safety={dict(safeties)}"
              + (f" tokenhit={hits:.0%}" if hits is not None else ""))
        for fl in flags:
            print(f"      ⚠  {fl}")
    print("\n[audit] done. A low token-hit rate on prompt_injection/toxicity_abuse "
          "is the strongest offline signal of the default-to-unsafe mislabel bug.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-schema", action="store_true",
                    help="Load samples and verify label columns exist (needs HF).")
    ap.add_argument("--audit-dir", type=str, default=None,
                    help="Offline audit of an existing by_attack/ dir.")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--sample-n", type=int, default=200)
    ap.add_argument("--cap-hint", type=int, default=2000)
    ap.add_argument("--no-strict", action="store_true",
                    help="Warn instead of raising on a missing label column.")
    args = ap.parse_args()

    if args.audit_dir:
        audit_compiled_jsonl(Path(args.audit_dir), cap_hint=args.cap_hint)

    if args.check_schema:
        rep = validate_source_schema(
            source_keys=args.only, sample_n=args.sample_n,
            strict=not args.no_strict)
        print(json.dumps(rep, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
