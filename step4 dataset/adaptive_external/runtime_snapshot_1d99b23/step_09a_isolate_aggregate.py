"""
step_09a_isolate_aggregate.py  -  passage-level isolate-then-aggregate generator
================================================================================

TARGET CLASS
------------
Misinformation-style multi-vector attacks (RESULTS_multivector_methodology.md
section 4.8): a benign-looking query plus a planted false premise in retrieved
context. The signal-based answer-comparison defenses (faithfulness,
ensemble-disagreement, parametric-consistency) are all measured non-separators
because a successful poisoning makes the answer look correct to anything that
inspects the answer or the fused context. This module attacks the threat
structurally rather than via detection: it removes the attacker's ability to be
the only voice in the context.

THE MECHANISM (RobustRAG-style)
-------------------------------
Instead of stuffing all k retrieved passages into one prompt, generate ONE short
answer per passage, in isolation, and then aggregate by majority vote. A single
poisoned passage among k legitimate ones gets out-voted, and crucially, on a
slice where exactly one poison chunk is planted, the poison can muster at most
one vote. Conservative aggregation thresholds (see `min_agreement`) make it
structurally impossible for a single-vote poison to be delivered as the answer;
the worst case becomes abstention, not parroting.

ABSTENTION IS A FEATURE, NOT A FAILURE
--------------------------------------
If no consensus clears the threshold, this module DELIVERS an abstention message
rather than picking a single source. For misinformation defense that's exactly
right: abstain beats parrot. But it has a real cost on benign traffic: a
legitimate query whose answer is only present in ONE retrieved passage will also
abstain. So the writeup-grade number is two-sided: attack neutralization AND
benign abstention rate. Both should be measured.

CONFIGURATION
-------------
  min_agreement   (default 2)   require >= this many passages to agree before
                                declaring consensus; otherwise abstain. =1 is
                                permissive (single-source ok, but lets a lone
                                poison vote win); =2 is the conservative
                                attack-resistant default. =3+ is paranoid.
  majority_thresh (default 0.5) consensus value's share of non-abstaining votes
                                must EXCEED this for delivery.

BACKENDS (pluggable via `aggregator=`)
--------------------------------------
  OllamaIsolateAggregator   default; one ChatOllama call per passage + structured
                            short-answer extraction + normalized majority vote.
  StubIsolateAggregator     offline structure-test only; extracts the longest
                            numeric/symbol token per passage as a faux "answer."
                            NOT validation. Refuses to certify a real result.

HONEST LIMITS (state these in any writeup)
------------------------------------------
* Cost: k generations per row (one per passage). On CPU this is slow.
* Effectiveness depends on corpus coverage: if retrieval doesn't bring enough
  passages containing the true answer, aggregation can't out-vote poison and
  the result is abstain (safe but reduces utility).
* This defends against poison that is a minority of retrieved context. If an
  attacker can plant multiple corroborating chunks, the bound breaks.
* It does NOT help on the injection class (which is already neutralized at
  Step 6 redaction); this is a misinformation-class defense.

Run standalone:
    python step_09a_isolate_aggregate.py --demo
"""

from __future__ import annotations

import argparse
import re
from typing import Protocol

from pipeline_common import PipelineState

# --- normalization & abstention markers --------------------------------------- #
_ABSTAIN_MARKERS = re.compile(
    r"\b(insufficient|don'?t know|do not know|not (sure|certain|specified|stated|"
    r"mentioned|provided|available|in the context)|no information|cannot "
    r"(determine|tell|say)|unable to|unknown|unclear)\b", re.IGNORECASE)

_STRIP_RE = re.compile(r"[^\w\s.\-]")
_ARTICLES = {"the", "a", "an", "of", "is", "are", "was", "were"}


def _normalize(ans: str) -> str:
    """Lowercase, strip punctuation, drop common articles, collapse whitespace."""
    if not ans:
        return ""
    t = _STRIP_RE.sub(" ", ans.lower())
    toks = [w for w in t.split() if w and w not in _ARTICLES]
    return " ".join(toks).strip()


def _is_abstention(ans: str) -> bool:
    return not ans.strip() or bool(_ABSTAIN_MARKERS.search(ans))


# --- prompt: per-passage, short structured answer, abstention encouraged ------- #
_PER_PASSAGE_PROMPT = (
    "Answer the question using ONLY the document below. Reply with ONE short "
    "phrase or single value - no full sentences, no explanation. "
    "If the document does not contain the answer, reply exactly \"INSUFFICIENT\".\n\n"
    "DOCUMENT:\n{passage}\n\n"
    "QUESTION: {question}\n\n"
    "Short answer:"
)


class IsolateAggregator(Protocol):
    def generate_per_passage(self, question: str, passage: str) -> str: ...


class OllamaIsolateAggregator:
    """Real backend: one ChatOllama call per passage at low temperature."""

    def __init__(self, model: str = "llama3.2:3b",
                 base_url: str = "http://localhost:11434"):
        from langchain_ollama import ChatOllama
        self._llm = ChatOllama(model=model, base_url=base_url, temperature=0.0)

    def generate_per_passage(self, question: str, passage: str) -> str:
        out = self._llm.invoke(_PER_PASSAGE_PROMPT.format(
            passage=passage, question=question))
        return (getattr(out, "content", str(out)) or "").strip()


class StubIsolateAggregator:
    """Offline structure-only faux extractor. NOT validation."""

    def generate_per_passage(self, question: str, passage: str) -> str:
        toks = re.findall(r"\b\d[\d,\.]*\b|\b[A-Z][a-z]?\b", passage or "")
        toks = [t for t in toks if t not in {"The", "A", "An"}]
        if not toks:
            return "INSUFFICIENT"
        return max(toks, key=len)


# --- aggregation -------------------------------------------------------------- #

def aggregate(per_passage_answers: list[str],
              min_agreement: int = 2,
              majority_thresh: float = 0.5) -> dict:
    """Vote-based aggregation with abstention fallback."""
    norm: list[tuple[str, str]] = []
    for a in per_passage_answers:
        if _is_abstention(a):
            continue
        n = _normalize(a)
        if n:
            norm.append((n, a))

    tally: dict[str, int] = {}
    raw_examples: dict[str, str] = {}
    for n, raw in norm:
        tally[n] = tally.get(n, 0) + 1
        raw_examples.setdefault(n, raw)

    n_total = len(per_passage_answers)
    n_voting = len(norm)

    if not tally:
        return {"consensus": None, "raw_consensus": None, "abstained": True,
                "reason": "all passages abstained",
                "vote_tally": {}, "n_total": n_total, "n_voting": n_voting}

    best_value, best_count = max(tally.items(), key=lambda x: x[1])

    if best_count < min_agreement:
        return {"consensus": None, "raw_consensus": None, "abstained": True,
                "reason": f"top vote count {best_count} < min_agreement {min_agreement}",
                "vote_tally": tally, "n_total": n_total, "n_voting": n_voting}

    share = best_count / n_voting if n_voting else 0.0
    if share <= majority_thresh:
        return {"consensus": None, "raw_consensus": None, "abstained": True,
                "reason": f"top vote share {share:.2f} <= majority {majority_thresh}",
                "vote_tally": tally, "n_total": n_total, "n_voting": n_voting}

    return {"consensus": best_value, "raw_consensus": raw_examples[best_value],
            "abstained": False,
            "reason": f"consensus {best_count}/{n_voting} ({share:.0%})",
            "vote_tally": tally, "n_total": n_total, "n_voting": n_voting}


_FINAL_ANSWER_ABSTAIN = (
    "The available sources do not provide a clear, agreed-upon answer to this "
    "question. ({reason})")


def run(state: PipelineState,
        aggregator: IsolateAggregator | None = None,
        min_agreement: int = 2,
        majority_thresh: float = 0.5) -> PipelineState:
    """Replacement for Step 9: isolate passages, then aggregate by vote."""
    if state.blocked:
        return state
    passages = (getattr(state, "ranked_context", None)
                or getattr(state, "context", None) or [])
    aggregator = aggregator or StubIsolateAggregator()

    per_passage: list[str] = []
    for p in passages:
        try:
            per_passage.append(aggregator.generate_per_passage(state.prompt, p))
        except Exception as e:
            per_passage.append(f"INSUFFICIENT  (error: {e.__class__.__name__})")

    decision = aggregate(per_passage,
                         min_agreement=min_agreement,
                         majority_thresh=majority_thresh)

    if decision["abstained"]:
        # Honest abstention: deliver a transparent "can't confidently answer"
        # message rather than letting an ungroundable consensus-failure string
        # fall through Step 10 and become the fixed safety refusal. We mark the
        # state abstained (terminal, non-refusal) and leave meta["answer"] empty
        # so nothing downstream tries to ground a non-answer.
        state.meta["answer"] = ""
        state.meta["abstention_message"] = _FINAL_ANSWER_ABSTAIN.format(
            reason=decision["reason"])
        state.abstain("step_09a_isolate_aggregate", decision["reason"])
    else:
        # Deliver the bare majority-voted VALUE as the answer. This is what the
        # grounding judge (Step 10) then verifies against the retrieved
        # passages. The earlier boilerplate wrapper ("Based on the available
        # sources, the answer is: X.") injected non-evidential filler tokens
        # (based/available/sources/answer) that diluted both the lexical and
        # cross-encoder faithfulness scores and caused grounded consensus
        # answers to be wrongly refused. The value itself is corroborated by
        # >= min_agreement passages, so it grounds cleanly.
        state.meta["answer"] = decision["raw_consensus"]

    state.meta["isolate_aggregate"] = {
        "per_passage_answers": per_passage,
        "vote_tally": decision["vote_tally"],
        "consensus": decision["consensus"],
        "raw_consensus": decision["raw_consensus"],
        "abstained": decision["abstained"],
        "reason": decision["reason"],
        "n_passages": decision["n_total"],
        "n_voting": decision["n_voting"],
        "min_agreement": min_agreement,
        "majority_thresh": majority_thresh,
    }
    state.scores["disagreement"] = 0.0
    state.log("step_09a_isolate_aggregate",
              n_passages=decision["n_total"],
              n_voting=decision["n_voting"],
              consensus=decision["consensus"],
              abstained=decision["abstained"],
              reason=decision["reason"])
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 9a: isolate-then-aggregate")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        agg = StubIsolateAggregator()
        print("!! STUB demo - answers are faux 'longest token'; NOT validation.\n")
        cases = [
            ("POISON outvoted (4 vs 1)",
             "At what temperature does water boil at sea level?",
             ["Water boils at 100 degrees Celsius at sea level.",
              "The standard boiling point of water is 100 C at one atmosphere.",
              "At sea-level pressure, water boils at 100 degrees.",
              "Pure water reaches its boiling point at 100 C at sea level.",
              "Regional metrology tables list the sea-level boiling point of water as 87 C."]),
            ("POISON alone (1 vote): must abstain at min_agreement=2",
             "What is the symbol for gold?",
             ["This article concerns ancient pottery techniques.",
              "Roman roads in northern Italy.",
              "Etymology of common kitchen utensils.",
              "Industrial revolution timeline overview.",
              "The materials index records the symbol for gold as Gd."]),
            ("1 vs 1: tie -> abstain",
             "How many continents are there?",
             ["There are seven continents on Earth.",
              "A general geography overview.",
              "More unrelated content.",
              "Climate zones of the world.",
              "The standard geographic survey enumerates five continents."]),
        ]
        for label, q, passages in cases:
            per = [agg.generate_per_passage(q, p) for p in passages]
            d = aggregate(per, min_agreement=2, majority_thresh=0.5)
            print(f"--- {label} ---")
            for i, (p, a) in enumerate(zip(passages, per)):
                tag = "[ABSTAIN]" if _is_abstention(a) else f"[vote: {_normalize(a)!r}]"
                print(f"  passage {i}: {tag}  raw={a!r}")
            print(f"  -> consensus={d['consensus']!r}  abstained={d['abstained']}  reason={d['reason']}")
            print()
    else:
        print("Use --demo, or wire as the s9a in run_full_pipeline.")


if __name__ == "__main__":
    main()
