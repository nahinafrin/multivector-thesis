"""
step_02_normalization.py  —  METHODOLOGY STEP 2: Normalization
==============================================================

"...removing whitespaces, hidden characters, and formatting tricks... spaCy
for Linguistic Normalization... ftfy for mojibake... Heuristic Length Filtering
(min 40, max 2000 chars)."

This step reuses the SAME normalization logic as your offline preprocessing
(preprocess_dataset.py) so that training-time and inference-time normalization
are IDENTICAL — avoiding train/serve skew. It imports Preprocessor rather than
reimplementing it.

Behaviour at inference time:
  - normalize the prompt (ftfy + NFKC + hidden-char strip + spaCy token-rejoin)
  - apply the length heuristic: if outside [min_len, max_len], by default we
    FLAG (state.meta['length_ok']=False) but DO NOT block — because, per our
    dataset analysis, short prompts are often genuine attacks that must still
    reach the Injection Detection scanner (Step 3). Set --drop-short to instead
    block over/under-length prompts outright.

Requires preprocess_dataset.py on the path.

Run standalone:
    python step_02_normalization.py --prompt "don't    ignore   this\u200b request"
"""

from __future__ import annotations

import argparse

from pipeline_common import PipelineState

# Reuse the exact normalizer from the offline preprocessing script.
try:
    from preprocess_dataset import Preprocessor
except ImportError as e:
    raise ImportError(
        "step_02 needs preprocess_dataset.py on the path "
        "(the same file you used to clean the dataset)."
    ) from e


# Build one shared Preprocessor (loads spaCy once). Configure via set_config().
_PRE: Preprocessor | None = None
_CFG = dict(min_len=40, max_len=2000, use_spacy=True, lemmatize=False)


def set_config(**kwargs) -> None:
    global _PRE, _CFG
    _CFG.update(kwargs)
    _PRE = None  # force rebuild on next use


def _pre() -> Preprocessor:
    global _PRE
    if _PRE is None:
        _PRE = Preprocessor(**_CFG)
    return _PRE


def run(state: PipelineState, drop_short: bool = False) -> PipelineState:
    pre = _pre()
    before = state.prompt
    state.raw_prompt = state.raw_prompt or before
    state.prompt = pre.normalize_text(before)

    verdict = pre.length_ok(state.prompt)        # None | "too_short" | "too_long"
    state.meta["length_ok"] = verdict is None
    state.meta["char_len"] = len(state.prompt)
    state.log("step_02_normalization",
              chars_before=len(before), chars_after=len(state.prompt),
              length_verdict=verdict or "ok")

    if verdict is not None and drop_short:
        state.block("step_02_normalization",
                    f"length {verdict} (len={len(state.prompt)}, "
                    f"window=[{pre.min_len},{pre.max_len}])")
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 2: Normalization")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--min-len", type=int, default=40)
    ap.add_argument("--max-len", type=int, default=2000)
    ap.add_argument("--no-spacy", action="store_true")
    ap.add_argument("--drop-short", action="store_true")
    args = ap.parse_args()

    set_config(min_len=args.min_len, max_len=args.max_len,
               use_spacy=not args.no_spacy, lemmatize=False)
    st = PipelineState(prompt=args.prompt, raw_prompt=args.prompt)
    st = run(st, drop_short=args.drop_short)
    print(f"raw   : {st.raw_prompt!r}")
    print(f"clean : {st.prompt!r}")
    print(f"length_ok={st.meta['length_ok']} blocked={st.blocked} "
          f"reason={st.block_reason!r}")


if __name__ == "__main__":
    main()
