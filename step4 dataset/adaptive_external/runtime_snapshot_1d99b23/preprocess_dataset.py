"""
preprocess_dataset.py
=====================

Implements the PRE-PROCESSING stage of the methodology (Chapter 3), i.e.
everything that runs BEFORE Injection Detection (Step 3):

    Normalization
      - ftfy            : repair mojibake / broken encoding   (Speer, 2019)
      - hidden-char     : strip zero-width / control / bidi formatting tricks
      - whitespace      : collapse runs, normalize unicode spaces, trim
      - unicode (NFKC)  : canonical form so look-alike glyphs unify
      - spaCy           : linguistic normalization + contraction expansion
                          (Honnibal & Montani, 2017)

    Heuristic Length Filtering                              (Butcher et al., 2025)
      - keep 40 <= len(chars) <= 2000   (methodology's stated window)
      - rows outside the window are REPORTED, then dropped (or flagged), so you
        can see how many real rows violate the 40-char floor before committing.

Design notes
------------
* The methodology states a 40-character minimum, but ~14% of the real corpus is
  shorter than that. This script does NOT silently delete: it counts and reports
  every drop, and `--min-len` lets you choose the threshold. Use --flag-only to
  keep all rows and merely add a `length_ok` boolean instead of dropping.
* spaCy is used via the small English model when available; if the model isn't
  installed it falls back to a blank tokenizer so the script still runs (you
  lose lemma-based normalization but keep tokenization + contraction expansion).
* Contraction expansion is dictionary-based (deterministic, no model needed),
  applied before spaCy so "don't" -> "do not" consistently.

Usage
-----
    pip install spacy ftfy
    python -m spacy download en_core_web_sm        # optional but recommended

    # JSONL in -> cleaned JSONL out (operates on the "prompt" field)
    python preprocess_dataset.py \
        --in  ./merged_output/dataset_all.jsonl \
        --out ./merged_output/dataset_all.clean.jsonl \
        --min-len 40 --max-len 2000

    # Keep everything, just annotate length_ok / cleaning changes:
    python preprocess_dataset.py --in data.jsonl --out data.clean.jsonl --flag-only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import ftfy
except ImportError:  # pragma: no cover
    ftfy = None

try:
    import spacy
except ImportError:  # pragma: no cover
    spacy = None


# --------------------------------------------------------------------------- #
# Hidden / formatting-trick characters to strip
# --------------------------------------------------------------------------- #
# Zero-width, bidi controls, BOM, and other invisible characters that attackers
# use to smuggle instructions or evade filters. Removing these is part of the
# methodology's "removing whitespace, hidden characters, and formatting tricks".
_HIDDEN_CHARS = [
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\u2060",  # word joiner
    "\ufeff",  # BOM / zero-width no-break space
    "\u00ad",  # soft hyphen
    "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",  # bidi overrides
    "\u2066", "\u2067", "\u2068", "\u2069",            # bidi isolates
    "\u180e",  # mongolian vowel separator
]
_HIDDEN_RE = re.compile("|".join(map(re.escape, _HIDDEN_CHARS)))

# Other unicode space variants -> normal space
_UNICODE_SPACE_RE = re.compile(
    "[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]"
)
# Control chars except tab/newline/carriage-return
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Collapse any run of whitespace to a single space
_WS_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# Contraction expansion (deterministic, model-free)
# --------------------------------------------------------------------------- #
_CONTRACTIONS = {
    "won't": "will not", "can't": "cannot", "n't": " not",
    "'re": " are", "'ve": " have", "'ll": " will", "'d": " would",
    "'m": " am", "let's": "let us", "y'all": "you all",
    "ain't": "is not", "gonna": "going to", "wanna": "want to",
    "gotta": "got to", "i'm": "i am", "it's": "it is",
    "he's": "he is", "she's": "she is", "that's": "that is",
    "there's": "there is", "what's": "what is", "who's": "who is",
}
_CONTRACTION_RE = re.compile(
    "|".join(re.escape(k) for k in sorted(_CONTRACTIONS, key=len, reverse=True)),
    flags=re.IGNORECASE,
)


def expand_contractions(text: str) -> str:
    def _sub(m: re.Match) -> str:
        token = m.group(0)
        repl = _CONTRACTIONS.get(token.lower(), token)
        # preserve leading capitalization
        if token[:1].isupper():
            repl = repl[:1].upper() + repl[1:]
        return repl
    return _CONTRACTION_RE.sub(_sub, text)


# --------------------------------------------------------------------------- #
# spaCy loader with graceful fallback
# --------------------------------------------------------------------------- #
def _load_spacy(model: str = "en_core_web_sm"):
    if spacy is None:
        print("[warn] spaCy not installed; skipping linguistic normalization.",
              file=sys.stderr)
        return None
    try:
        nlp = spacy.load(model, disable=["ner", "parser"])
        print(f"[spacy] loaded '{model}' (lemmatization enabled)")
        return nlp
    except Exception:
        # Fall back to a blank English tokenizer: still tokenizes & re-spaces,
        # but no lemmas. Keeps the pipeline runnable without the downloaded model.
        print(f"[warn] spaCy model '{model}' not found; using blank tokenizer "
              f"(run `python -m spacy download {model}` for full normalization).",
              file=sys.stderr)
        return spacy.blank("en")


# --------------------------------------------------------------------------- #
# Core normalizer
# --------------------------------------------------------------------------- #
@dataclass
class CleanStats:
    total: int = 0
    ftfy_fixed: int = 0
    hidden_stripped: int = 0
    empty_after_clean: int = 0
    too_short: int = 0
    too_long: int = 0
    kept: int = 0
    dropped: int = 0
    short_examples: list[str] = field(default_factory=list)


class Preprocessor:
    """Methodology-faithful normalization + heuristic length filter."""

    def __init__(self, min_len: int = 40, max_len: int = 2000,
                 use_spacy: bool = True, lemmatize: bool = False,
                 spacy_model: str = "en_core_web_sm"):
        self.min_len = min_len
        self.max_len = max_len
        self.lemmatize = lemmatize
        self.nlp = _load_spacy(spacy_model) if use_spacy else None

    # -- text-level cleaning ------------------------------------------------ #
    def normalize_text(self, text: str, stats: Optional[CleanStats] = None) -> str:
        if not isinstance(text, str):
            text = str(text)
        original = text

        # 1) ftfy: repair mojibake / broken encoding
        if ftfy is not None:
            text = ftfy.fix_text(text)
            if stats is not None and text != original:
                stats.ftfy_fixed += 1

        # 2) Unicode canonicalization (NFKC) so look-alike glyphs unify
        text = unicodedata.normalize("NFKC", text)

        # 3) Strip hidden / formatting-trick characters
        before_hidden = text
        text = _HIDDEN_RE.sub("", text)
        text = _CONTROL_RE.sub("", text)
        if stats is not None and text != before_hidden:
            stats.hidden_stripped += 1

        # 4) Normalize exotic spaces, collapse whitespace, trim
        text = _UNICODE_SPACE_RE.sub(" ", text)
        text = _WS_RE.sub(" ", text).strip()

        # 5) Expand contractions (deterministic)
        text = expand_contractions(text)

        # 6) spaCy linguistic normalization
        if self.nlp is not None:
            doc = self.nlp(text)
            if self.lemmatize and doc.has_annotation("LEMMA"):
                text = " ".join(tok.lemma_ if tok.lemma_ != "-PRON-" else tok.text
                                for tok in doc)
            else:
                # token re-join normalizes spacing around punctuation
                text = " ".join(tok.text for tok in doc)
            text = _WS_RE.sub(" ", text).strip()

        return text

    # -- length filter ------------------------------------------------------ #
    def length_ok(self, text: str) -> Optional[str]:
        n = len(text)
        if n < self.min_len:
            return "too_short"
        if n > self.max_len:
            return "too_long"
        return None

    # -- row-level ---------------------------------------------------------- #
    def process_rows(self, rows: list[dict], prompt_field: str = "prompt",
                     flag_only: bool = False) -> tuple[list[dict], CleanStats]:
        stats = CleanStats(total=len(rows))
        out: list[dict] = []
        for r in rows:
            raw = r.get(prompt_field, "")
            clean = self.normalize_text(raw, stats)

            if not clean:
                stats.empty_after_clean += 1
                if not flag_only:
                    stats.dropped += 1
                    continue

            verdict = self.length_ok(clean)
            if verdict == "too_short":
                stats.too_short += 1
                if len(stats.short_examples) < 10:
                    stats.short_examples.append(clean)
            elif verdict == "too_long":
                stats.too_long += 1

            row = dict(r)
            row[prompt_field] = clean
            if flag_only:
                row["length_ok"] = verdict is None
                row["char_len"] = len(clean)
                out.append(row)
                stats.kept += 1
            else:
                if verdict is None:
                    out.append(row)
                    stats.kept += 1
                else:
                    stats.dropped += 1
        return out, stats


# --------------------------------------------------------------------------- #
# Single-function entry point for training + inference reuse
# --------------------------------------------------------------------------- #
# A process-wide singleton so spaCy is loaded once even when called row-by-row
# from a serving path. Override defaults by calling `get_inference_normalizer`
# with explicit kwargs before the first `normalize_prompt` call.
_INFERENCE_NORMALIZER: Optional["Preprocessor"] = None


def get_inference_normalizer(
    min_len: int = 40,
    max_len: int = 2000,
    use_spacy: bool = True,
    lemmatize: bool = False,
    spacy_model: str = "en_core_web_sm",
) -> "Preprocessor":
    """Return (and lazily build) the shared inference-time Preprocessor.

    Pass explicit kwargs only on the first call. The same instance is reused
    for every subsequent `normalize_prompt` invocation so spaCy is loaded
    exactly once per process.
    """
    global _INFERENCE_NORMALIZER
    if _INFERENCE_NORMALIZER is None:
        _INFERENCE_NORMALIZER = Preprocessor(
            min_len=min_len, max_len=max_len,
            use_spacy=use_spacy, lemmatize=lemmatize,
            spacy_model=spacy_model,
        )
    return _INFERENCE_NORMALIZER


def normalize_prompt(text: str) -> str:
    """Apply the canonical training-time normalization to a single prompt.

    Identical to the pre-split normalization run inside merge_and_audit.py, so
    the live inference pipeline can call this and be byte-for-byte aligned
    with the training distribution.
    """
    return get_inference_normalizer().normalize_text(text)


def print_report(stats: CleanStats, min_len: int, max_len: int,
                 flag_only: bool) -> None:
    print("\n=== PREPROCESSING REPORT ===")
    print(f"  input rows           : {stats.total}")
    print(f"  ftfy repaired        : {stats.ftfy_fixed}")
    print(f"  hidden chars stripped: {stats.hidden_stripped}")
    print(f"  empty after clean    : {stats.empty_after_clean}")
    print(f"  length window        : [{min_len}, {max_len}] chars")
    print(f"  below min ({min_len}c)   : {stats.too_short}")
    print(f"  above max ({max_len}c) : {stats.too_long}")
    if flag_only:
        print(f"  kept (all, flagged)  : {stats.kept}")
    else:
        print(f"  kept                 : {stats.kept}")
        print(f"  dropped              : {stats.dropped}")
    if stats.too_short and stats.short_examples:
        print(f"\n  sample sub-{min_len}-char prompts that "
              f"{'were flagged' if flag_only else 'were dropped'}:")
        for ex in stats.short_examples[:5]:
            print(f"    ({len(ex):>3}c) {ex!r}")
    if stats.too_short:
        pct = stats.too_short / stats.total * 100 if stats.total else 0
        print(f"\n  NOTE: {pct:.1f}% of rows fall below the {min_len}-char floor. "
              f"If this is high, either lower --min-len to match the data, or "
              f"update the methodology's stated minimum to match what you keep.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Preprocess (normalize + length-filter) the dataset per methodology.")
    ap.add_argument("--in", dest="inp", required=True, help="Input JSONL.")
    ap.add_argument("--out", dest="outp", required=True, help="Output JSONL.")
    ap.add_argument("--prompt-field", default="prompt")
    ap.add_argument("--min-len", type=int, default=40)
    ap.add_argument("--max-len", type=int, default=2000)
    ap.add_argument("--no-spacy", action="store_true",
                    help="Skip spaCy entirely (regex/ftfy cleaning only).")
    ap.add_argument("--lemmatize", action="store_true",
                    help="Apply lemmatization (needs the full spaCy model).")
    ap.add_argument("--flag-only", action="store_true",
                    help="Don't drop rows; add length_ok/char_len fields instead.")
    ap.add_argument("--spacy-model", default="en_core_web_sm")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]
    pre = Preprocessor(min_len=args.min_len, max_len=args.max_len,
                       use_spacy=not args.no_spacy, lemmatize=args.lemmatize,
                       spacy_model=args.spacy_model)
    cleaned, stats = pre.process_rows(rows, prompt_field=args.prompt_field,
                                      flag_only=args.flag_only)

    out_path = Path(args.outp)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in cleaned:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print_report(stats, args.min_len, args.max_len, args.flag_only)
    print(f"\n[write] {len(cleaned)} rows -> {out_path}")


if __name__ == "__main__":
    main()
