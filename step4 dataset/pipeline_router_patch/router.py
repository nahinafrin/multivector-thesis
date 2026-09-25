"""
router.py -- per-query switch between v2 (deployed fused generator) and v3 (verified isolate-and-vote).

Signal (label-free, attacker-agnostic): do the retrieved passages DISAGREE about the answer?
Every query first gets the isolate step's per-passage short answers (step_09a, 5 small LLM calls).
  * 0 or 1 distinct answer cluster among the passages that answered  -> no conflict  -> v2 answer
  * >= 2 distinct answer clusters (fuzzy-matched, verified_vote.cluster) -> conflict -> v3 decision
A planted fact has to disagree with something to matter: when the knowledge base also holds the true
answer, the poison passage and the true passage produce two clusters, and the query is sent to v3,
whose closed-book verifier decides or abstains. Benign single-source questions (the main cause of
v3's refusals) have one cluster and keep v2's low refusal rate.
Known blind spot (stated, not hidden): a planted fact that is the ONLY answer any passage gives
(the true passage absent or silent) has no conflict and goes through v2.
"""
from __future__ import annotations

import verified_vote as vv

ENDORSED_FALLBACK = True


def route(per_passage: list[str]) -> str:
    voting = [a.strip() for a in (per_passage or []) if not vv.is_abstention(a)]
    return "v3" if len(vv.cluster(voting)) >= 2 else "v2"


def decide(question: str, per_passage: list[str], v2_record: dict, scorer) -> dict:
    """Returns {'route', 'refused', 'answer', 'v3': decision-or-None}."""
    if v2_record.get("blocked"):
        return {"route": "blocked", "refused": True, "answer": "", "v3": None}
    r = route(per_passage)
    if r == "v2":
        ref = bool(v2_record.get("abstained"))
        return {"route": "v2", "refused": ref, "answer": "" if ref else str(v2_record.get("final_response") or ""), "v3": None}
    d = vv.aggregate(question, per_passage, scorer)
    if d["abstained"] and ENDORSED_FALLBACK and d.get("verify") and sum(p >= vv.TAU for _, p in d["verify"]) >= 2:
        # every leading candidate is endorsed by the closed-book verifier: the disagreement carries no
        # sign of a false planted value (typically paraphrase / partial answers) -> keep v2's answer
        ref = bool(v2_record.get("abstained"))
        return {"route": "v3->v2", "refused": ref, "answer": "" if ref else str(v2_record.get("final_response") or ""), "v3": d}
    return {"route": "v3", "refused": d["abstained"], "answer": d["answer"] or "", "v3": d}
