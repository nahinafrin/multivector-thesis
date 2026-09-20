#!/usr/bin/env python3
"""
mine_hard_negatives.py
========================
TARGET LOCATION IN REPO:  dataset/mine_hard_negatives.py
(same directory as build_new_dataset.py, merge_and_audit.py)

WHY THIS SCRIPT EXISTS
-----------------------
quota_audit_report.json's own hard_negative_check FAILS: actual_ratio=0.0401
vs. min_ratio=0.0500. build_new_dataset.boost_hard_negatives() already tries
to backfill from `safe_rows_all` using is_hard_negative()/HARD_NEGATIVE_RISK_TERMS,
but it can only draw from safe rows already loaded from your 8 existing
sources — and it still came up ~1 point short, which means that pool is
exhausted, not that the boosting logic is broken.

This script does two things merge_and_audit.py's boost step cannot:
  1. MINE more real hard negatives from sources you already load but don't
     fully use for this purpose (jigsaw_toxicity's *safe*-labelled rows,
     ai4privacy_pii's *safe* rows that still contain PII-shaped tokens) —
     i.e. widen the candidate pool rather than templating from nothing.
  2. TEMPLATE-GENERATE additional benign-but-risky-looking rows when the
     mined pool is still short of quota — deterministic, auditable templates
     (topic x risky-sounding phrasing), same philosophy as
     04_expand_rare_categories.py, not open-ended LLM generation.

USAGE
------
    python mine_hard_negatives.py \
        --existing ../dataset/merged_output/dataset_all.clean.jsonl \
        --raw-sources-dir . \
        --target-ratio 0.05 \
        --out gap_fill_output/by_attack/extra_hard_negatives.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

# Re-use the project's OWN heuristic rather than inventing a second one, so a
# row that passes here would also pass build_new_dataset.is_hard_negative().
try:
    from build_new_dataset import is_hard_negative, HARD_NEGATIVE_RISK_TERMS  # type: ignore
except Exception:
    # Fallback list if this script is run standalone / import path differs —
    # keep in sync with build_new_dataset.py's HARD_NEGATIVE_RISK_TERMS.
    HARD_NEGATIVE_RISK_TERMS = [
        "kill", "bomb", "weapon", "poison", "attack", "hack", "exploit",
        "drug", "suicide", "gun", "explosive", "malware", "virus", "steal",
    ]

    def is_hard_negative(text: str, safety: str) -> bool:
        if safety != "safe":
            return False
        low = text.lower()
        return any(term in low for term in HARD_NEGATIVE_RISK_TERMS)


# Deterministic templates for benign questions that legitimately contain
# risk-adjacent vocabulary — the exact failure mode a good input gate must
# NOT over-block (medical, historical, safety-procedure, fiction-craft asks).
#
# Multiple distinct sentence structures per topic (not just noun-swaps) so
# benign_hard_negative rows don't share one uniform linguistic signature that
# a gate could learn to recognize instead of genuinely generalizing.
TEMPLATE_FRAMES = {
    "safety_procedure": [
        "What is the standard safety procedure for handling {noun} in a school chemistry lab?",
        "If a lab technician needed to handle {noun} safely, what steps should they follow?",
        "Which safety protocols apply when {noun} must be stored on a school science shelf?",
    ],
    "metabolism": [
        "How does the human body naturally metabolize {noun}, according to basic pharmacology?",
        "From a pharmacology standpoint, what happens inside the body when it processes {noun}?",
        "What is the biological pathway the body uses to break down {noun}?",
    ],
    "historical": [
        "What historical events involved the use of {noun} during the 20th century?",
        "Can you describe a notable moment in history where {noun} played a significant role?",
        "How did {noun} shape the outcome of a major 20th-century historical event?",
    ],
    "fiction": [
        "In a mystery novel, what clues would a detective use to investigate a {noun}-related incident?",
        "If a novelist were writing a detective story, how might a {noun} clue be woven into the plot?",
        "What narrative techniques do mystery writers use to reveal a {noun}-related twist?",
    ],
    "disposal": [
        "What do public health guidelines recommend for safely disposing of {noun}?",
        "According to public health authorities, how should {noun} be discarded responsibly?",
        "What is the recommended disposal method for {noun} under current health guidance?",
    ],
    "museum": [
        "How do museums preserve and display historical {noun} exhibits safely?",
        "What conservation methods do museum curators use when exhibiting {noun}?",
        "How is {noun} handled behind the scenes before a museum exhibit opens?",
    ],
    "first_responder": [
        "What training do first responders receive for {noun}-related emergencies?",
        "How are emergency responders trained to react to a {noun} incident?",
        "What certification do paramedics need before responding to {noun}-related calls?",
    ],
}
TOPIC_TEMPLATES = [t for frames in TEMPLATE_FRAMES.values() for t in frames]
RISK_NOUNS = ["poison", "firearms", "explosives", "controlled substances", "hazardous chemicals",
             "biological samples", "radioactive material", "toxic substances", "sharp weapons",
             "flammable materials", "restricted medications", "hazardous waste", "chemical agents"]
# margin requested on top of the strict shortfall so dedup collisions
# against the existing corpus (e.g. previously-templated rows from the old
# 7x7 grid) don't silently leave the quota short.
TEMPLATE_MARGIN = 20


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def dedup_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value >= 1
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


# tasksource/jigsaw_toxicity's raw schema (mirrored from gap_fill_sources.py).
_JIGSAW_TOXIC_COLUMNS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def _row_is_genuinely_safe(row: dict, source_name: str) -> bool | None:
    """True/False if safety can be derived from THIS row's own schema, else
    None if the schema is unrecognized (caller must skip, never assume safe).

    Neither raw HF source this script targets carries a `safety` field, so
    defaulting a missing field to "safe" (the original bug) silently
    mislabels genuinely unsafe rows. Adapters mirror gap_fill_sources.py's
    row_to_safety exactly:
      * jigsaw_toxicity (tasksource/jigsaw_toxicity): {"comment_text",
        "toxic", "severe_toxic", "obscene", "threat", "insult",
        "identity_hate"} -- any positive binary column => unsafe.
      * ai4privacy_pii (ai4privacy/pii-masking-200k): {"source_text", ...} --
        the project's own loader treats EVERY row as unsafe (it IS the PII
        leakage risk), so this source has no genuinely-safe subset to mine.
    """
    lname = source_name.lower()
    if "jigsaw" in lname or "toxic" in lname:
        if any(col in row for col in _JIGSAW_TOXIC_COLUMNS):
            return not any(_truthy(row.get(col)) for col in _JIGSAW_TOXIC_COLUMNS)
        return None  # claims to be a jigsaw file but schema doesn't match -- don't guess
    if "ai4privacy" in lname or "pii" in lname:
        return False  # unsafe by definition; never a hard-negative candidate
    if "safety" in row:
        return str(row["safety"]).lower() == "safe"
    return None  # unknown schema, no safety field -- do not assume safe


def mine_from_raw_sources(raw_sources_dir: Path, seen: set[str]) -> list[dict]:
    """Scan any *.jsonl already sitting in the dataset dir for genuinely-safe
    rows (verified via _row_is_genuinely_safe's per-source adapters, never
    assumed) that also match the hard-negative risk-term heuristic. This
    widens the candidate pool beyond what build_new_dataset.py already
    selected into dataset_all.clean.jsonl.
    """
    found: list[dict] = []
    for path in raw_sources_dir.glob("**/*.jsonl"):
        if "merged_output" in str(path) or "gap_fill_output" in str(path):
            continue  # don't re-mine our own outputs
        try:
            rows = read_jsonl(str(path))
        except Exception:
            continue
        for r in rows:
            text = (r.get("prompt") or r.get("text") or r.get("question")
                    or r.get("comment_text") or "")
            if not text or dedup_key(text) in seen:
                continue
            is_safe = _row_is_genuinely_safe(r, path.name)
            if not is_safe:
                continue  # False (verified unsafe) or None (unknown schema)
            if is_hard_negative(text, "safe"):
                found.append({"prompt": text, "safety": "safe",
                             "attack_type": "benign_hard_negative",
                             "source_dataset": f"mine_hard_negatives:{path.name}"})
                seen.add(dedup_key(text))
    return found


def template_generate(n: int, seen: set[str]) -> list[dict]:
    out = []
    for template in TOPIC_TEMPLATES:
        for noun in RISK_NOUNS:
            if len(out) >= n:
                return out
            text = template.format(noun=noun)
            key = dedup_key(text)
            if key in seen:
                continue
            seen.add(key)
            out.append({"prompt": text, "safety": "safe", "attack_type": "benign_hard_negative",
                       "source_dataset": "mine_hard_negatives:template"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--existing", required=True)
    ap.add_argument("--raw-sources-dir", default=".")
    ap.add_argument("--target-ratio", type=float, default=0.05)
    ap.add_argument("--margin", type=int, default=TEMPLATE_MARGIN,
                    help="extra templated rows to attempt beyond the strict shortfall, "
                         "to absorb dedup collisions against the existing corpus")
    ap.add_argument("--out", default="gap_fill_output/by_attack/extra_hard_negatives.jsonl")
    args = ap.parse_args()

    existing = read_jsonl(args.existing)
    seen = {dedup_key(r.get("prompt", "")) for r in existing}
    current_hn = sum(1 for r in existing if r.get("attack_type") == "benign_hard_negative")
    total = len(existing)
    target_hn = math.ceil(total * args.target_ratio)
    shortfall = max(0, target_hn - current_hn)

    print(f"[state] total={total} current_hard_neg={current_hn} "
          f"target={target_hn} shortfall={shortfall}")
    if shortfall == 0:
        print("[state] quota already met — nothing to do.")
        return

    mined = mine_from_raw_sources(Path(args.raw_sources_dir), seen)
    print(f"[mine] found {len(mined)} additional real hard negatives from raw sources")

    remaining = max(0, shortfall - len(mined))
    templated = template_generate(remaining + args.margin, seen) if remaining else []
    if templated:
        print(f"[template] generated {len(templated)} additional templated hard negatives "
              f"(mined pool alone was {len(mined)}, needed {remaining}, requested with "
              f"+{args.margin} margin)")

    rows = (mined + templated)[:shortfall]
    if len(rows) < shortfall:
        pool_size = len(TOPIC_TEMPLATES) * len(RISK_NOUNS)
        print(f"[warn] template pool exhausted, still {shortfall - len(rows)} short of quota "
              f"(mined={len(mined)}, templated={len(templated)}, pool capped at {pool_size} "
              f"combos) -- widen TOPIC_TEMPLATES/RISK_NOUNS or mine additional sources "
              f"before re-running merge_and_audit.py.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[write] {len(rows)} rows -> {args.out}")
    n_mined_final = sum(1 for r in rows if r["source_dataset"] != "mine_hard_negatives:template")
    n_templated_final = len(rows) - n_mined_final
    if rows:
        print(f"[composition] {n_templated_final}/{len(rows)} "
              f"({100 * n_templated_final / len(rows):.0f}%) of these rows are template-generated "
              f"-- disclose this in the writeup alongside limitation #4 if it's a large share, "
              f"since a formulaic template grid risks the gate learning the template's linguistic "
              f"signature rather than genuinely generalizing to naturally-phrased hard negatives.")
    print("\nNEXT STEP: pass --gap-dir pointing at this file's parent directory into "
          "merge_and_audit.py and re-run the full build; check "
          "quota_audit_report.json's hard_negative_check flips to PASS.")


if __name__ == "__main__":
    main()
