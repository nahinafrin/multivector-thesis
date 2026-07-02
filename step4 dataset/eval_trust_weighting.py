#!/usr/bin/env python3
"""
eval_trust_weighting.py  —  the NEW result trust-aware retrieval makes possible
===============================================================================

Measures, on the controlled multi-source corpus, whether weighting retrieval by
source trust keeps poison chunks OUT of the context. Compares two conditions on
the SAME retrievals:

    baseline : rank by semantic similarity only
    trust    : rank by similarity x source_trust   (trust_aware_retrieval.rescore)

For each targeted question it asks: did the poison chunk for that question survive
into the top-N context window? The headline metric is POISON ADMISSION RATE (lower
is better) under each condition, with a Wilson 95% CI and a McNemar paired test —
because the two conditions see identical retrievals, so the comparison is paired.

It also reports CLEAN-CHUNK DISPLACEMENT: how many relevant high-trust clean chunks
trust weighting pushed OUT of top-N (the benign cost), so the trade-off is measured
rather than assumed.

This needs the FAISS index over the enriched corpus. Build it once with the loader
patch in INTEGRATION_corpus.md (kb stores (text, source) instead of bare text), then:

    python eval_trust_weighting.py \
        --manifest poison_manifest.jsonl \
        --index ./kb_multisource --top-n 3 --k 8
"""
from __future__ import annotations
import argparse, json, math


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = (z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / d
    return (round(100*p, 1), round(100*max(0, c-h), 1), round(100*min(1, c+h), 1))


def mcnemar(b, c):
    if b + c == 0:
        return (0.0, "tie")
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return (round(chi2, 3), "trust" if c > b else ("baseline" if b > c else "tie"))


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser(description="Evaluate trust-weighting on poison admission")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--index", default="./kb_multisource",
                    help="FAISS index over the ENRICHED corpus (text+source).")
    ap.add_argument("--qa", default="data/question-answer/test.jsonl")
    ap.add_argument("--k", type=int, default=8, help="candidates retrieved")
    ap.add_argument("--top-n", type=int, default=3, help="context window size")
    args = ap.parse_args()

    # The provenance-aware store and embedder come from the corpus loader patch.
    import kb_multisource as kb            # see INTEGRATION_corpus.md
    from trust_aware_retrieval import rescore

    store = kb.load_index(args.index)
    questions = {q["id"]: q["question"] for q in read_jsonl(args.qa)}
    manifest = list(read_jsonl(args.manifest))

    base_admit = trust_admit = 0
    only_base = only_trust = 0          # discordant pairs for McNemar
    displaced = 0                        # clean high-trust chunks trust pushed out
    n = 0

    for row in manifest:
        q = questions.get(row["qid"]) or row.get("question")
        if not q:
            continue
        n += 1
        # Retrieve k candidates as (text, similarity, source).
        hits = kb.search(store, q, k=args.k)         # -> [(text, sim, source), ...]

        base_topn = [h for h in sorted(hits, key=lambda x: x[1], reverse=True)][:args.top_n]
        trust_topn = rescore(hits)[:args.top_n]

        def is_poison(text):
            return row["marker"] in text or _looks_like(text, row)

        b_in = any(is_poison(t) for (t, *_ ) in base_topn)
        t_in = any(is_poison(d["text"]) for d in trust_topn)
        base_admit += int(b_in)
        trust_admit += int(t_in)
        if b_in and not t_in:
            only_base += 1            # trust FIXED this one
        elif t_in and not b_in:
            only_trust += 1           # trust BROKE this one

        # benign cost: high-trust clean chunks in base top-N that trust dropped
        base_clean_ht = {t for (t, sim, src) in base_topn
                         if not is_poison(t) and src in ("internal_kb", "wikipedia", "research_paper")}
        trust_texts = {d["text"] for d in trust_topn}
        displaced += len(base_clean_ht - trust_texts)

    print(f"\n=== Poison admission into top-{args.top_n} (n={n}) ===")
    ba, tr = wilson(base_admit, n), wilson(trust_admit, n)
    print(f"  similarity only      : {base_admit}/{n} = {ba[0]}%  CI[{ba[1]},{ba[2]}]")
    print(f"  similarity x trust   : {trust_admit}/{n} = {tr[0]}%  CI[{tr[1]},{tr[2]}]")
    chi2, better = mcnemar(only_trust, only_base)
    print(f"  paired: trust-fixed={only_base}  trust-broke={only_trust}  "
          f"McNemar chi2={chi2} favours {better}")
    print(f"\n=== Benign cost ===")
    print(f"  high-trust clean chunks displaced from top-N by trust weighting: {displaced}")
    print(f"\nMarkdown:\n")
    print(f"| Condition | Poison admission | 95% CI |")
    print(f"|---|---|---|")
    print(f"| Similarity only | {ba[0]}% | [{ba[1]}, {ba[2]}] |")
    print(f"| Similarity x trust | {tr[0]}% | [{tr[1]}, {tr[2]}] |")


def _looks_like(text, row):
    """Fallback poison match if the marker was paraphrased away: topical overlap
    with the targeted question plus an imperative verb. Conservative."""
    q = (row.get("question") or "").lower()
    t = text.lower()
    imperative = any(w in t for w in ("ignore", "override", "must include",
                                      "append", "disregard", "system"))
    overlap = len(set(q.split()) & set(t.split())) >= 3
    return imperative and overlap


if __name__ == "__main__":
    main()
