"""
step_09_generator_llm.py  —  METHODOLOGY STEP 9: Generator LLM (Parallel Intelligence)
======================================================================================

"...Parallel Intelligence ... passed to a varied array of text-only models such
as Llama 4, Mistral 3, and GPT-4o Mini, all at once through LangChain's
RunnableParallel ... avoid Single Model Bias ... choose the most common answer,
or a fusing model."

FREE/LOCAL DEFAULT: three local Ollama models run concurrently via
RunnableParallel. No API keys.

POINT OF DIFFERENCE (methodology): the DISAGREEMENT between models is computed
and stored on state.scores["ensemble_disagreement"]. Step 10 reads this to
raise the grounding threshold when models diverge — treating disagreement as a
risk signal. This is absent from the LangChain/LlamaIndex baselines.

INPUTS FROM UPSTREAM:
  state.augmented_prompt  : from Step 8 (preferred)
  state.ranked_context    : from Step 7 (used if augmented_prompt absent)
  state.prompt            : fallback

--from-results mode reads your results.jsonl from Steps 4-8 directly, so you
can run the generation half of the pipeline without re-running retrieval.

Setup:
    pip install langchain-ollama langchain-core
    ollama pull llama3.2:3b
    ollama pull mistral:7b
    ollama pull qwen2.5:3b

Run:
    python step_09_generator_llm.py --demo
    python step_09_generator_llm.py --from-results ./results.jsonl --out ./generated.jsonl --limit 10
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from pipeline_common import PipelineState

DEFAULT_ENSEMBLE = {
    "llama3.2:3b": "logic",
    "mistral:7b":  "styling",
    "qwen2.5:3b":  "generalism",
}


# --------------------------------------------------------------------------- #
# Semantic disagreement scorer
# --------------------------------------------------------------------------- #
# The original implementation normalized on whitespace/case and counted unique
# strings, so any stylistic rewording (extra word, different punctuation) was
# treated as disagreement. On rag-mini this saturated at 1.0 in ~80% of rows
# and made the signal useless to Step 10. We now use sentence-transformer
# embeddings + average pairwise cosine similarity, which captures *meaning*
# rather than surface form. See `compute_disagreement_legacy` below for the
# previous behaviour (kept for reproducibility / ablation comparisons).

_DISAGREE_MODEL = None
_DISAGREE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _get_disagree_model(name: str = _DISAGREE_MODEL_NAME):
    global _DISAGREE_MODEL
    if _DISAGREE_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _DISAGREE_MODEL = SentenceTransformer(name)
    return _DISAGREE_MODEL


# --------------------------------------------------------------------------- #
# Core generation helpers
# --------------------------------------------------------------------------- #

def _build_runnable_parallel(models: dict[str, str], base_url: str):
    from langchain_ollama import ChatOllama
    from langchain_core.runnables import RunnableParallel, RunnableLambda
    branches = {}
    for model_name in models:
        llm = ChatOllama(model=model_name, base_url=base_url, temperature=0.2)
        branches[model_name] = llm | RunnableLambda(lambda m: m.content)
    return RunnableParallel(branches)


def generate_candidates(augmented_prompt: str,
                        models: dict[str, str] = DEFAULT_ENSEMBLE,
                        base_url: str = "http://localhost:11434") -> dict[str, str]:
    """Run all models concurrently; return {model_name: answer}."""
    parallel = _build_runnable_parallel(models, base_url)
    return parallel.invoke(augmented_prompt)


def generate_single(augmented_prompt: str,
                    model: str = "llama3.2:3b",
                    base_url: str = "http://localhost:11434") -> str:
    """One-model generation for controlled A/B arms (fixed decoding)."""
    out = generate_candidates(augmented_prompt, models={model: model}, base_url=base_url)
    return next(iter(out.values())) if out else ""


def compute_disagreement(candidates: dict[str, str]) -> float:
    """Return a [0,1] *semantic* disagreement score.

    0.0 = all models said the same thing (mean pairwise cosine ~= 1).
    1.0 = mutually contradictory answers (mean pairwise cosine ~= 0).

    Stored on state.scores["ensemble_disagreement"] so Step 10 can raise the
    grounding threshold when models actually diverge in meaning — not when
    they merely paraphrase each other.
    """
    if len(candidates) <= 1:
        return 0.0
    answers = [v.strip() for v in candidates.values() if v and v.strip()]
    if len(answers) <= 1:
        return 0.0
    from sentence_transformers import util
    model = _get_disagree_model()
    embeddings = model.encode(answers, normalize_embeddings=True,
                              convert_to_tensor=True, show_progress_bar=False)
    n = len(embeddings)
    sims: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(float(util.cos_sim(embeddings[i], embeddings[j])))
    avg_sim = sum(sims) / len(sims)
    return round(max(0.0, min(1.0, 1.0 - avg_sim)), 4)


def compute_disagreement_legacy(candidates: dict[str, str]) -> float:
    """Original whitespace-uniqueness scorer; kept for ablation reproduction."""
    if len(candidates) <= 1:
        return 0.0
    norm = [" ".join(v.lower().split()) for v in candidates.values()]
    unique = len(set(norm))
    return round((unique - 1) / (len(norm) - 1), 4)


def select_majority(candidates: dict[str, str]) -> str:
    norm = {k: " ".join(v.lower().split()) for k, v in candidates.items()}
    winner_norm, _ = Counter(norm.values()).most_common(1)[0]
    for k, v in norm.items():
        if v == winner_norm:
            return candidates[k]
    return next(iter(candidates.values()))


def select_fusion(candidates: dict[str, str],
                  fuser_model: str = "qwen2.5:3b",
                  base_url: str = "http://localhost:11434") -> str:
    from langchain_ollama import ChatOllama
    joined = "\n\n".join(f"[{k}]: {v}" for k, v in candidates.items())
    fuse_prompt = (
        "You are a fusion judge. Produce one accurate, concise answer that "
        "combines the strongest factual points from the models below. "
        "Do not invent facts.\n\n"
        f"{joined}\n\nFused answer:"
    )
    llm = ChatOllama(model=fuser_model, base_url=base_url, temperature=0.0)
    return llm.invoke(fuse_prompt).content


# ---- PAID SWAP (commented): hosted models -------------------------------- #
# from langchain_openai import ChatOpenAI
# branches = {
#   "gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini") | RunnableLambda(lambda m:m.content),
# }
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Pipeline-contract entry
# --------------------------------------------------------------------------- #

def run(state: PipelineState, strategy: str = "fusion",
        models: dict[str, str] = DEFAULT_ENSEMBLE,
        base_url: str = "http://localhost:11434") -> PipelineState:
    if state.blocked:
        return state

    # Use augmented prompt from Step 8 if present; fall back gracefully.
    prompt = state.augmented_prompt or state.prompt
    candidates = generate_candidates(prompt, models, base_url)
    state.candidates = candidates

    # --- DISAGREEMENT SIGNAL (methodology point of difference) ---
    disagreement = compute_disagreement(candidates)
    state.scores["ensemble_disagreement"] = disagreement
    # Unified full-pipeline contract: Step 10 reads `scores["disagreement"]`.
    state.scores["disagreement"] = disagreement

    if strategy == "majority":
        state.output = select_majority(candidates)
    else:
        state.output = select_fusion(candidates, base_url=base_url)
    # Unified full-pipeline contract: Steps 10-13 read the answer from meta.
    state.meta["answer"] = state.output
    state.meta["candidates"] = candidates

    state.log("step_09_generator_llm",
              strategy=strategy, models=list(models),
              n_candidates=len(candidates),
              disagreement=disagreement)
    return state


# --------------------------------------------------------------------------- #
# --from-results: read Steps 4-8 output and generate answers
# --------------------------------------------------------------------------- #

def run_from_results(results_path: str, out_path: str,
                     strategy: str = "fusion",
                     base_url: str = "http://localhost:11434",
                     limit: int | None = None,
                     resume: bool = False) -> None:
    """Read results.jsonl from Steps 4-8 and run generation on each row.

    Each input row must have: question, ranked_context (list of strings).
    Output adds: answer, candidates, disagreement.
    """
    rows = [json.loads(l) for l in open(results_path, encoding="utf-8") if l.strip()]
    if limit:
        rows = rows[:limit]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    import step_08_augmented_prompt as s8

    done_indices: set[int] = set()
    if resume and out.exists():
        with open(out, encoding="utf-8") as existing:
            for line in existing:
                if not line.strip():
                    continue
                try:
                    done_indices.add(int(json.loads(line)["index"]))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
        print(f"[step09] resume: {len(done_indices)} existing rows in {out_path}")

    pending_rows = [
        (i, row) for i, row in enumerate(rows)
        if int(row.get("index", i)) not in done_indices
    ]

    mode = "a" if resume else "w"
    with open(out, mode, encoding="utf-8") as f:
        for completed, (i, row) in enumerate(pending_rows, start=1):
            st = PipelineState(prompt=row.get("question", ""))
            st.ranked_context = row.get("ranked_context", row.get("context", []))
            # Build augmented prompt exactly as Step 8 does
            st = s8.run(st)
            # Generate
            try:
                st = run(st, strategy=strategy, base_url=base_url)
                out_row = {
                    "index": row.get("index", i),
                    "question": row.get("question", ""),
                    "ground_truth": row.get("ground_truth", ""),
                    "answer": st.output,
                    "candidates": st.candidates,
                    "disagreement": st.scores.get("ensemble_disagreement", 0.0),
                    "ranked_context": st.ranked_context,
                }
            except Exception as e:
                out_row = {
                    "index": row.get("index", i),
                    "question": row.get("question", ""),
                    "ground_truth": row.get("ground_truth", ""),
                    "answer": "",
                    "error": str(e),
                }
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            f.flush()
            if completed % 10 == 0:
                print(f"  ...generated {completed}/{len(pending_rows)} new rows")
    print(f"[step09] wrote {len(pending_rows)} new rows -> {out_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Step 9: Ensemble Generator LLM")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--demo", action="store_true",
                   help="Single-prompt demo (needs Ollama running).")
    g.add_argument("--from-results", dest="results",
                   help="Path to results.jsonl from Steps 4-8.")
    ap.add_argument("--out", default="./generated.jsonl")
    ap.add_argument("--strategy", choices=["majority", "fusion"], default="fusion")
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="Append to --out and skip indices already present.")
    args = ap.parse_args()

    if args.demo:
        st = PipelineState(prompt="In one sentence, what is two-factor authentication?")
        st.augmented_prompt = st.prompt
        st = run(st, strategy=args.strategy, base_url=args.base_url)
        print("=== candidates ===")
        for k, v in st.candidates.items():
            print(f"  [{k}] {v[:120]}")
        print(f"\n=== selected ({args.strategy}) ===\n{st.output[:300]}")
        print(f"\ndisagreement score: {st.scores['ensemble_disagreement']:.4f}")
        print("(0.0 = all models agree, 1.0 = all models gave unique answers)")
    else:
        run_from_results(args.results, args.out,
                         strategy=args.strategy,
                         base_url=args.base_url,
                         limit=args.limit,
                         resume=args.resume)


if __name__ == "__main__":
    main()
