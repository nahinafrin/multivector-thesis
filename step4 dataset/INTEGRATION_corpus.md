# INTEGRATION_corpus.md — making trust-aware retrieval an evaluated result

This turns trust-aware retrieval from a described component into a measured one.
All files drop into `step4 dataset/`. Run in order.

---

## What this gives the thesis

A new, self-contained experiment with a clean result table:

> On a controlled multi-source corpus, weighting retrieval by source trust reduces
> poison admission into the context from X% to Y% (paired McNemar), at a benign
> cost of Z displaced high-trust clean chunks.

That replaces the honest-but-weak caveat ("trust weighting is inert on single-source
data") with an actual number — and the controlled-provenance limitation is stated,
so it stays rigorous.

---

## Step 1 — apply the three pipeline patches (also needed by the adaptive harness)

From the repo's `step4 dataset/` directory:

```bash
patch step_06_context_sanitization.py < step_06_context_sanitization.patch.diff
patch step_07_context_ranking.py      < step_07_context_ranking.patch.diff
patch step_08_augmented_prompt.py     < step_08_augmented_prompt.patch.diff
```

Each is a few lines, backward-compatible (no tier metadata ⇒ original behaviour),
and verified to apply cleanly against the current `main`. They let the adaptive
orchestrator pin per-tier sanitization strictness, rerank floor, and prompt profile.

---

## Step 2 — build the controlled multi-source corpus

```bash
python build_multisource_corpus.py \
  --passages data/text-corpus/passages.jsonl \
  --qa       data/question-answer/test.jsonl \
  --n-poison 60 --low-trust-poison-frac 0.75 \
  --out-corpus enriched_passages.jsonl \
  --out-manifest poison_manifest.jsonl
```

This assigns provenance to the 3,200 real Wikipedia passages (spread across all six
trust tiers so clean chunks exist at every trust level) and injects 60 labelled
poison chunks, each topically wrapped around a target question and carrying a unique
marker. 75% of poison sits on low-trust sources, 25% on mid-trust — so trust is
*correlated* with poison but not identical to it, which is what makes the result
non-trivial.

Verified output on the real corpus:
```
3260 chunks (3200 clean + 60 poison)
chunks by source: wikipedia 1321, internal_kb 485, news 488, research_paper 470,
                  community_forum 259, unknown_web 237
poison by source: community_forum 24, news 17, unknown_web 19
```

---

## Step 3 — index the enriched corpus (provenance-aware)

```bash
python kb_multisource.py --ingest \
  --corpus enriched_passages.jsonl --index ./kb_multisource
```

`kb_multisource.py` reuses Step 4 (embedding) and Step 5 (FAISS) unchanged; it only
keeps a `sources[]` list aligned with the texts, persisted as `kb_multisource.sources`
beside the index, so retrieval can return `(text, similarity, source)` triples.

---

## Step 4 — run the trust-weighting experiment

```bash
python eval_trust_weighting.py \
  --manifest poison_manifest.jsonl \
  --index ./kb_multisource --k 8 --top-n 3
```

It compares two rankings on the SAME retrievals — similarity only vs
similarity × trust (from `trust_aware_retrieval.rescore`) — and reports:
- poison admission rate under each, with Wilson 95% CIs;
- a paired McNemar test (valid because both conditions see identical candidates);
- benign cost: high-trust clean chunks trust weighting displaced from the window.

---

## Step 5 (optional) — fold trust weighting into the adaptive pipeline

To use trust-weighted retrieval inside `run_full_pipeline_adaptive.py`, swap the
plain Step-5 search for `kb_multisource.search` + `trust_aware_retrieval.rescore`
in `_apply_similarity_threshold` (one call site). Then the three-way experiment
also reflects trust weighting on the poisoned-context cohort. Keep it OFF for the
single-source rag-mini runs (where it is inert) and ON only for multi-source runs,
and report which corpus each result used.

---

## Honest framing for the write-up (do not omit)

- Provenance is **assigned by construction**, not naturally observed. This is a
  controlled isolation of the trust mechanism under known ground truth — a valid
  experiment, but name the limitation and present a naturally-sourced corpus as
  future work.
- Because poison is correlated with (not equal to) low trust, the result measures a
  genuine similarity-vs-trust trade-off rather than a trivial "drop all low-trust".
- Report BOTH sides: poison kept out (security gain) and clean high-trust chunks
  displaced (benign cost). A one-sided "poison dropped" number overstates the case.
```
