"""
step_09c_routed_generator.py -- Step 9 with a per-query switch between
  v2  the deployed fused ensemble generator (step_09_generator_llm), and
  v3  verified isolate-and-vote (verified_vote.py: fuzzy per-passage vote + closed-book verifier).

How a query is routed (router.py, frozen 2026-09-25):
  1. Every query first gets step_09a's per-passage short answers (one small LLM call per passage).
  2. If the passages that answered give 0 or 1 distinct answer -> no disagreement -> v2.
     If they give >= 2 distinct answers (fuzzy-matched) -> the context disagrees -> v3.
  3. mode="safe":     v3 decides; if v3 abstains the query is refused (abstention message).
     mode="balanced": if v3 abstains ONLY because the verifier endorses >= 2 candidates
                      (no sign of a false planted value), fall back to the v2 answer.
  generator "verified" in run_full_pipeline.py = always v3 (no router).

Held-out test 2 (unseen, 40 benign / 50 plain planted-fact attacks), see router_test_results.txt:
                 plain attack success   benign refused
  v2 only             21/50                 1/40
  routed balanced      9/50                 1/40
  routed safe          5/50                 6/40
  v3 only              3/50                14/40
Blind spot: a planted fact that is the only answer any passage gives causes no disagreement and
goes through v2.
"""
from __future__ import annotations

import router
import step_09_generator_llm as s9
import step_09a_isolate_aggregate as s9a
import verified_vote as vv
from pipeline_common import PipelineState

_SCORER = None


def _scorer(model: str):
    global _SCORER
    if _SCORER is None:
        _SCORER = vv.ollama_scorer(model)
    return _SCORER


def run(state: PipelineState,
        extractor: "s9a.IsolateAggregator | None" = None,
        mode: str = "balanced",
        always_v3: bool = False,
        verifier_model: str = "mistral:7b") -> PipelineState:
    if state.blocked:
        return state
    passages = (getattr(state, "ranked_context", None) or getattr(state, "context", None) or [])
    extractor = extractor or s9a.OllamaIsolateAggregator()
    ppa = []
    for p in passages:
        try:
            ppa.append(extractor.generate_per_passage(state.prompt, p))
        except Exception as e:  # same fail-soft behaviour as step_09a
            ppa.append(f"INSUFFICIENT  (error: {e.__class__.__name__})")

    route = "v3" if always_v3 else router.route(ppa)
    info = {"per_passage_answers": ppa, "route": route, "mode": "always_v3" if always_v3 else mode,
            "n_clusters": len(vv.cluster([a for a in ppa if not vv.is_abstention(a)]))}

    if route == "v2":
        state = s9.run(state)
        state.meta["routed_generator"] = info
        state.log("step_09c_routed", route="v2", n_clusters=info["n_clusters"])
        return state

    d = vv.aggregate(state.prompt, ppa, _scorer(verifier_model))
    info.update({"v3_path": d["path"], "v3_reason": d["reason"], "verify": d.get("verify")})
    if not d["abstained"]:
        state.meta["answer"] = d["answer"]
        state.scores["disagreement"] = 0.0
    elif (not always_v3 and mode == "balanced" and d.get("verify")
          and sum(p >= vv.TAU for _, p in d["verify"]) >= 2):
        info["route"] = "v3->v2"
        state = s9.run(state)
    else:
        state.meta["answer"] = ""
        state.meta["abstention_message"] = (
            "The retrieved sources disagree and the answer could not be verified, so no answer is given. "
            f"({d['reason']})")
        state.abstain("step_09c_routed_generator", d["reason"])
    state.meta["routed_generator"] = info
    state.log("step_09c_routed", route=info["route"], v3_path=d["path"], abstained=d["abstained"])
    return state
