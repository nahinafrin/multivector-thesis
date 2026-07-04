"""
trust_aware_retrieval.py  —  final_score = semantic_similarity x source_trust
=============================================================================

Discourages retrieving low-quality / potentially poisoned chunks by weighting
each chunk's similarity by the trustworthiness of its source.

HONEST EVALUATION CAVEAT (must go in the thesis)
------------------------------------------------
rag-mini-wikipedia is a SINGLE source, so every chunk would get the same trust
score and this feature changes NOTHING measurable on the current corpus. To
evaluate it you need a multi-source corpus (e.g. Wikipedia + a forum dump +
injected "unknown-web" poison chunks). Two honest options:
  (a) Build that multi-source corpus and isolate the effect, OR
  (b) Present trust-aware retrieval as a designed component you do NOT empirically
      isolate on this corpus, and say so explicitly.
Do not claim a benefit you cannot show. This module supports (a) when you have
provenance, and degrades gracefully to a no-op when every source is identical.
"""
from __future__ import annotations

# Default trust table. Keys are source tags attached to each chunk's metadata.
# Tune to your corpus; values are multiplicative weights in (0, 1].
DEFAULT_TRUST = {
    "internal_kb": 1.00,
    "wikipedia": 0.95,
    "research_paper": 0.90,
    "news": 0.70,
    "community_forum": 0.50,
    "unknown_web": 0.30,
}
DEFAULT_TRUST_FALLBACK = 0.50   # used when a chunk has no known source tag


def trust_for(source: str | None, table: dict[str, float] = DEFAULT_TRUST,
              fallback: float = DEFAULT_TRUST_FALLBACK) -> float:
    if not source:
        return fallback
    return table.get(str(source).strip().lower(), fallback)


def rescore(hits, *, table: dict[str, float] = DEFAULT_TRUST,
            fallback: float = DEFAULT_TRUST_FALLBACK):
    """Re-weight (text, similarity, source) hits by source trust.

    Accepts a list of (text, similarity, source) OR (text, similarity) tuples.
    Returns a list of dicts sorted by trust-weighted score, descending:
        {"text", "similarity", "source", "trust", "score"}
    When sources are all identical/absent, ordering is unchanged (graceful no-op),
    so wiring this in cannot hurt the single-source baseline.
    """
    out = []
    for h in hits:
        if len(h) == 3:
            text, sim, source = h
        else:
            text, sim = h
            source = None
        t = trust_for(source, table, fallback)
        out.append({"text": text, "similarity": float(sim), "source": source,
                    "trust": t, "score": float(sim) * t})
    out.sort(key=lambda d: d["score"], reverse=True)
    return out


if __name__ == "__main__":
    demo = [
        ("clean fact A", 0.82, "wikipedia"),
        ("poison chunk", 0.88, "unknown_web"),
        ("paper claim", 0.79, "research_paper"),
    ]
    for d in rescore(demo):
        print(f"{d['score']:.3f}  sim={d['similarity']:.2f} x trust={d['trust']:.2f}"
              f"  [{d['source']}]  {d['text']}")
