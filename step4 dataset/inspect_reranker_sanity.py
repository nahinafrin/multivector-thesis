#!/usr/bin/env python3
"""
inspect_reranker_sanity.py -- is step_07's cross-encoder actually discriminating
relevance at all, or is it degenerate?

WHY: inspect_row8_incidental_rerank.py returned ~0.5000 for all 30 rows,
varying only in the 4th decimal place, even though the 30 poison_chunks are
genuinely different personal-data records (different names, emails, dates,
addresses...) scored against the SAME query. A real cross-encoder should not
produce near-identical output across genuinely different (query, doc) pairs
this consistently. sigmoid(x) = 0.5 exactly when the raw pre-sigmoid logit is
0 -- so this pattern looks like the model is returning a raw score near 0
for every input, which is what an UNTRAINED / randomly-initialized model (or
one whose pretrained head failed to load) tends to do, not a real relevance
judgment.

This is a big deal if true: step_07_context_ranking.py's BASE_MIN_SCORE
(0.20) / STRICT_MIN_SCORE (0.40) thresholds -- and therefore every context-
ranking decision in the ENTIRE project, not just Row 8 -- assume this
reranker actually discriminates relevance. If it doesn't, "risk-adaptive
retrieval tightening" may not have been doing anything real in any prior
experiment either.

This script checks with an obvious positive control (a chunk that is
clearly, maximally relevant to the query) and an obvious negative control
(a chunk that is clearly irrelevant gibberish), plus the raw (pre-sigmoid)
score so we can see the actual model output, not just the squashed version.

USAGE (from `step4 dataset`, .venv311 active -- no Ollama needed):
    python inspect_reranker_sanity.py
"""
from __future__ import annotations
import sys

sys.path.insert(0, "../dataset")
from step_07_context_ranking import _get_reranker, _sigmoid  # noqa: E402

QUERY = "Please summarize the key points of the reference material above."

PAIRS = [
    ("obvious MATCH (should score high)",
     "The reference material above summarizes three key points: first, the "
     "historical background; second, the main findings; and third, the "
     "conclusions drawn by the authors."),
    ("obvious MISMATCH (should score low)",
     "The quick brown fox jumps over the lazy dog while purple elephants "
     "dance on the moon eating spaghetti tacos."),
    ("a real PII blob, for comparison",
     "Name: John Smith. Email: john.smith1984@example.com. DOB: 05/07/2010. "
     "Address: 123 Hill Road, 28170-6392. Card: 6225427220412963."),
]


def main() -> None:
    model = _get_reranker()
    print(f"Reranker loaded: {model}\n")
    raw_scores = model.predict([(QUERY, text) for _, text in PAIRS])
    for (label, text), raw in zip(PAIRS, raw_scores):
        sig = _sigmoid(float(raw))
        print(f"{label}")
        print(f"  raw (pre-sigmoid) = {float(raw):.4f}   sigmoid = {sig:.4f}")
        print(f"  text: {text[:70]!r}")
        print()

    print(
        "If the MATCH and MISMATCH rows come back close to each other (both "
        "near 0.5, or otherwise not clearly separated), the reranker itself "
        "is not discriminating relevance -- likely a model-loading problem "
        "(pretrained weights not actually loaded, network/cache issue during "
        "first download, or a version mismatch), not anything specific to "
        "Row 8's PII data. If MATCH scores clearly higher than MISMATCH but "
        "the PII blob still lands near 0.5, the reranker works in general but "
        "treats structured PII-shaped text as ambiguous input -- a narrower, "
        "still-worth-noting finding."
    )


if __name__ == "__main__":
    main()
