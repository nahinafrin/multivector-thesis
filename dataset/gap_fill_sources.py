"""
gap_fill_sources.py
===================

Drop-in companion to `build_new_dataset.py`. Adds *real* data for the attack
classes that the main pipeline starves because their rows never match the
keyword/label heuristics (prompt_injection, indirect_injection, obfuscation,
pii_leakage, toxicity_abuse).

WHY A SEPARATE LOADER?
----------------------
`load_all_sources()` in build_new_dataset.py *infers* attack_type from keyword
patterns. BeaverTails rows simply don't contain "ignore previous instructions"
markers, so prompt_injection / indirect_injection pools end up at 8-16 rows.
The loaders here instead FORCE the attack_type per source (a BIPIA row is
always indirect_injection), which is the only way to actually fill those pools.

USAGE
-----
In build_new_dataset.py, after the existing `load_all_sources(...)` call:

    from gap_fill_sources import load_gap_fill_sources, merge_into_pools

    gap_pools, gap_safe, gap_counts, gap_loaded = load_gap_fill_sources(
        max_rows_per_source=args.max_rows_per_source,
        normalizer=normalizer,
        # cap each new source so a 70k-row set can't swamp the build;
        # tune to taste or pass None to take everything (then let the
        # global --dataset-cap + per-attack overfill trim it).
        per_source_cap=2000,
    )
    merge_into_pools(pools, safe_rows, source_counts, loaded,
                     gap_pools, gap_safe, gap_counts, gap_loaded)

Then run the pipeline normally. The new sources flow through the SAME
overfill ceiling, dataset cap, dedup, audit and split logic as everything else.

NOTE ON SCHEMAS: these mappings are written against each dataset's published
card (column names verified from the HF dataset pages, Feb 2026). HF schemas
occasionally change; if a load yields 0 rows, print one record's .keys() and
adjust the `prompt_col` / label logic for that source below.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any, Callable, Optional

# Reuse the exact same primitives as the main pipeline so the rows are
# byte-for-byte compatible with everything downstream.
from build_new_dataset import (
    UnifiedSample,
    Normalizer,
    ATTACK_TO_RISK_DOMAIN,
    tier_for_attack,
    dataset_rows,
    find_prompt,
)

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover
    load_dataset = None  # type: ignore


# --------------------------------------------------------------------------- #
# SOURCE DEFINITIONS
# --------------------------------------------------------------------------- #
# Each entry describes ONE huggingface source and how to turn its rows into
# UnifiedSamples with a FORCED attack_type. `selector` decides, per row,
# whether it is the attack-positive class or a benign/clear row (so a single
# source can contribute both to its target class and to benign_clear).
#
# Fields:
#   hf_id           : huggingface dataset id
#   load_options    : list of kwargs dicts tried in order (mirrors the main
#                     pipeline's DATASET_LOAD_OPTIONS pattern)
#   prompt_cols     : ordered candidate columns to read the prompt text from
#   attack_type     : the class this source is here to FILL
#   row_to_safety   : fn(record) -> "safe" | "unsafe" | None(skip)
#                     when "safe", the row is reclassified as benign_clear
# --------------------------------------------------------------------------- #


def _truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v >= 1
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "injection", "unsafe",
                                     "harmful", "toxic", "malicious", "jailbreak"}
    return False


def _safe_from_binary_label(label_keys: list[str], positive_is_unsafe: bool = True):
    """Build a row_to_safety fn from a binary label column."""
    def fn(rec: dict[str, Any]) -> Optional[str]:
        for k in label_keys:
            if k in rec and rec[k] is not None:
                pos = _truthy(rec[k])
                if positive_is_unsafe:
                    return "unsafe" if pos else "safe"
                return "safe" if pos else "unsafe"
        # Missing expected labels should never be interpreted as unsafe. That
        # silently turns benign rows into positives if a dataset mirror changes
        # schema. Skip the row; schema validation should catch this upstream.
        return None
    return fn


GAP_SOURCES: dict[str, dict[str, Any]] = {
    # ---- 1. PROMPT INJECTION (you have ~16 rows) ----------------------------
    # xTRam1/safe-guard-prompt-injection: ~7k safe + 3k injection.
    # Card schema: {"text": str, "label": int}  (1 = injection, 0 = safe)
    "safeguard_prompt_injection": {
        "hf_id": "xTRam1/safe-guard-prompt-injection",
        "load_options": [{"split": "train"}, {}],
        "prompt_cols": ["text", "prompt"],
        "attack_type": "prompt_injection",
        "row_to_safety": _safe_from_binary_label(["label"], positive_is_unsafe=True),
    },

    # ---- 2. INDIRECT INJECTION (you have 8 synthetic rows) ------------------
    # MAlmasabi/Indirect-Prompt-Injection-BIPIA-GPT: 70k rows.
    # Card schema: {"context": str, "user_intent": str, "label": int, "source": str}
    # label 1 = malicious instruction embedded in context.
    # We use `context` as the prompt because that's where the indirect
    # injection lives (the "invisible text in a CV" your methodology cites).
    "bipia_indirect_injection": {
        "hf_id": "MAlmasabi/Indirect-Prompt-Injection-BIPIA-GPT",
        "load_options": [{"split": "train"}, {}],
        "prompt_cols": ["context", "text", "prompt"],
        "attack_type": "indirect_injection",
        "row_to_safety": _safe_from_binary_label(["label"], positive_is_unsafe=True),
    },

    # ---- 3. OBFUSCATION / EVASION (you have 8 synthetic rows) ---------------
    # Mindgard/evaded-prompt-injection-and-jailbreak-samples.
    # Card schema includes original + modified prompt + attack_name.
    # The MODIFIED prompt is the obfuscated one. Emoji-smuggling rows are
    # base64-encoded and some contain injected unicode -> normalize handles it.
    "mindgard_evaded": {
        "hf_id": "Mindgard/evaded-prompt-injection-and-jailbreak-samples",
        "load_options": [{"split": "train"}, {}],
        "prompt_cols": ["modified_prompt", "modified", "prompt", "text"],
        "attack_type": "obfuscation",
        "row_to_safety": lambda rec: "unsafe",  # all rows are attacks
    },

    # ---- 4. PII LEAKAGE (you have 47 rows) ----------------------------------
    # ai4privacy/pii-masking-200k. Card schema includes "source_text" (the
    # text containing PII) and "masked_text". We treat PII-bearing source_text
    # as the risky prompt. These are "unsafe" in the leakage sense.
    "ai4privacy_pii": {
        "hf_id": "ai4privacy/pii-masking-200k",
        "load_options": [{"split": "train"}, {}],
        "prompt_cols": ["source_text", "unmasked_text", "text", "prompt"],
        "attack_type": "pii_leakage",
        "row_to_safety": lambda rec: "unsafe",
    },

    # ---- 5. TOXICITY/ABUSE  (you have 36 rows) + KAGGLE PROVENANCE ----------
    # google/jigsaw_toxicity_pred is the canonical HF entry for the Jigsaw
    # Kaggle competition, but it ships as a legacy *.py loading script which
    # `datasets >= 4.x` no longer supports. `tasksource/jigsaw_toxicity` is a
    # straight parquet mirror with the IDENTICAL schema, so we try it first
    # and fall back to the google entry for older `datasets` installs.
    # Schema: {"comment_text": str, "toxic": int, "severe_toxic": int,
    #          "obscene": int, "threat": int, "insult": int,
    #          "identity_hate": int}. Any positive label => toxic.
    "jigsaw_toxicity": {
        "hf_id": "tasksource/jigsaw_toxicity",
        "load_options": [{"split": "train"}, {}],
        "prompt_cols": ["comment_text", "text", "prompt"],
        "attack_type": "toxicity_abuse",
        "row_to_safety": lambda rec: (
            "unsafe" if any(_truthy(rec.get(c)) for c in
                            ["toxic", "severe_toxic", "obscene",
                             "threat", "insult", "identity_hate"])
            else "safe"
        ),
    },
}


# --------------------------------------------------------------------------- #
# LOADER
# --------------------------------------------------------------------------- #


def _pick_prompt(rec: dict[str, Any], prompt_cols: list[str]) -> str:
    for c in prompt_cols:
        v = rec.get(c)
        if isinstance(v, str) and v.strip():
            return v
    # fall back to the pipeline's generic finder
    return find_prompt(rec)


def _build_row(prompt: str, safety: str, forced_attack: str,
               source_name: str) -> UnifiedSample:
    """Construct a UnifiedSample with a FORCED attack label.

    Safe rows from any source become benign_clear (they still pad the benign
    pool, which the audit needs). Unsafe rows keep the source's target class.
    """
    if safety == "safe":
        primary = "benign_clear"
    else:
        primary = forced_attack

    risk_domain = ATTACK_TO_RISK_DOMAIN.get(primary, "policy_safety")
    tier = tier_for_attack(primary)
    return UnifiedSample(
        prompt=prompt,
        safety=safety,
        attack_type=primary,
        attack_types=[primary],
        risk_domain=risk_domain,
        risk_domains=[risk_domain],
        is_adversarial=primary not in ("benign_clear", "benign_hard_negative"),
        source_dataset=source_name,
        tier=tier,
    )


def load_gap_fill_sources(
    max_rows_per_source: int,
    normalizer: Normalizer,
    per_source_cap: Optional[int] = 2000,
    only: Optional[list[str]] = None,
    keep_positive_only: bool = False,
    benign_cap_per_source: Optional[int] = None,
) -> tuple[dict[str, list[UnifiedSample]], list[UnifiedSample], Counter, set[str]]:
    """Load the gap-filling sources.

    Args:
        max_rows_per_source:    HF-level download cap (mirrors main pipeline flag).
        normalizer:             the SAME Normalizer instance the main build uses.
        per_source_cap:         after normalization, keep at most this many rows
                                per source (post-shuffle). When the new
                                independent-cap modes are active (see below),
                                this becomes the MAX POSITIVES per source.
                                None disables the cap.
        only:                   optional list of source keys to load (debugging).
        keep_positive_only:     drop benign rows entirely. The early build had a
                                cap-vs-label-split bug where per_source_cap
                                counted every row (positive + benign) toward the
                                same limit, so low-base-rate sources (e.g. ~15%
                                injection in xTRam1) starved the positive pool.
                                Set this True to keep ALL positives and ZERO
                                benigns. per_source_cap then caps positives.
        benign_cap_per_source:  independent cap on benigns. Use this when you
                                still want some in-domain benigns alongside the
                                positives (e.g. 200 negatives from
                                xTRam1 to give the model contrastive context).
                                If set, positives and benigns are counted
                                separately; per_source_cap caps positives only.

    Cap semantics:
        - Default (both new flags off): legacy behaviour — per_source_cap caps
          the COMBINED row count, first-come-first-served. Backwards compatible.
        - keep_positive_only=True OR benign_cap_per_source is not None:
          INDEPENDENT caps. Positives capped at per_source_cap, benigns capped
          at benign_cap_per_source (0 when keep_positive_only=True).

    Returns:
        (pools_by_attack, safe_rows, source_counts, loaded)
        — same shapes the main pipeline expects, ready for merge_into_pools().
    """
    if load_dataset is None:
        raise RuntimeError("`datasets` not installed. pip install datasets")

    pools: dict[str, list[UnifiedSample]] = defaultdict(list)
    safe_rows: list[UnifiedSample] = []
    source_counts: Counter = Counter()
    loaded: set[str] = set()

    items = GAP_SOURCES.items()
    if only:
        items = [(k, v) for k, v in items if k in only]

    # Decide which cap regime is in effect. Independent caps activate as soon
    # as either of the new flags is used; otherwise we preserve the original
    # combined-cap semantics so existing callers behave identically.
    independent_caps = keep_positive_only or benign_cap_per_source is not None
    if keep_positive_only:
        effective_benign_cap: Optional[int] = 0
    else:
        effective_benign_cap = benign_cap_per_source  # may be None ⇒ unbounded

    for source_name, cfg in items:
        hf_id = cfg["hf_id"]
        print(f"[gap-load] {source_name} <- {hf_id}")
        ds = None
        last_error: Optional[Exception] = None
        for kwargs in cfg.get("load_options", [{}]):
            try:
                ds = load_dataset(hf_id, **kwargs)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        if ds is None:
            print(f"  WARN: failed to load {source_name}: {last_error}")
            continue
        loaded.add(source_name)

        records = dataset_rows(ds)
        random.shuffle(records)
        if max_rows_per_source and max_rows_per_source > 0:
            records = records[:max_rows_per_source]

        prompt_cols: list[str] = cfg["prompt_cols"]
        forced_attack: str = cfg["attack_type"]
        row_to_safety: Callable[[dict[str, Any]], Optional[str]] = cfg["row_to_safety"]

        positive_kept = 0
        benign_kept = 0

        def pos_full() -> bool:
            return per_source_cap is not None and positive_kept >= per_source_cap

        def ben_full() -> bool:
            if effective_benign_cap is None:
                return False
            return benign_kept >= effective_benign_cap

        for rec in records:
            prompt = normalizer.normalize(_pick_prompt(rec, prompt_cols))
            if not prompt or not normalizer.keep(prompt):
                continue
            safety = row_to_safety(rec)
            if safety is None:
                continue

            is_positive = safety != "safe"

            if independent_caps:
                if is_positive:
                    if pos_full():
                        # No more positives wanted. If benigns are also done,
                        # stop reading this source entirely.
                        if ben_full() or effective_benign_cap == 0:
                            break
                        continue
                else:
                    if effective_benign_cap == 0 or ben_full():
                        # No more benigns wanted. If positives are also done,
                        # stop reading this source entirely.
                        if pos_full():
                            break
                        continue
            else:
                # Legacy combined cap (positives + benigns share per_source_cap).
                if per_source_cap is not None and (positive_kept + benign_kept) >= per_source_cap:
                    break

            row = _build_row(prompt, safety, forced_attack, source_name)
            pools[row.attack_type].append(row)
            if row.safety == "safe":
                safe_rows.append(row)
                benign_kept += 1
            else:
                positive_kept += 1
            source_counts[source_name] += 1

        print(f"  ingested {source_counts[source_name]} rows "
              f"({positive_kept} positives + {benign_kept} benigns, "
              f"forced attack_type='{forced_attack}') from {source_name}")

    return pools, safe_rows, source_counts, loaded


def merge_into_pools(
    pools: dict[str, list[UnifiedSample]],
    safe_rows: list[UnifiedSample],
    source_counts: Counter,
    loaded: set[str],
    gap_pools: dict[str, list[UnifiedSample]],
    gap_safe: list[UnifiedSample],
    gap_counts: Counter,
    gap_loaded: set[str],
) -> None:
    """Merge gap-fill results into the main pipeline's structures in place."""
    for atk, rows in gap_pools.items():
        pools.setdefault(atk, []).extend(rows)
    safe_rows.extend(gap_safe)
    source_counts.update(gap_counts)
    loaded.update(gap_loaded)


if __name__ == "__main__":
    # Smoke test for column-mapping + new positive-only switch.
    norm = Normalizer(min_len=20, max_len=2000)
    p, s, c, l = load_gap_fill_sources(
        max_rows_per_source=500, normalizer=norm,
        per_source_cap=200, only=["safeguard_prompt_injection"],
        keep_positive_only=True,
    )
    print("loaded:", l)
    print("counts:", dict(c))
    print("pool sizes:", {k: len(v) for k, v in p.items()})
