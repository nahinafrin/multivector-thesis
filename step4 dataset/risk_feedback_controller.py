"""
risk_feedback_controller.py — closed-loop risk feedback for the RAG pipeline.
============================================================================

Wraps the retrieval -> generation -> grounding segment (Steps 4-10). After each
attempt it reads the downstream signals (ensemble disagreement, grounding
faithfulness, canary integrity, optionally a from-scratch context-coherence
anomaly) and decides one of:

    accept | escalate + re-verify | recover (broaden retrieval & retry) | refuse

Why this is meaningful and not just "another retry layer":

  * Risk is the control variable.  Prior iterative-RAG work (Self-RAG,
    Corrective-RAG, FLARE) loops to improve answer QUALITY. Here we loop to
    revise the SECURITY RISK verdict using evidence the forward pass already
    computes but currently throws away.
  * Escalation reuses the forward cascade.  Instead of duplicating Steps 6/7/10's
    adaptive logic, the controller just writes state.scores["effective_risk"];
    PipelineState.input_risk() folds that in, so the existing
    forward-adaptivity re-tightens on the next pass automatically.
  * Attack-vs-miss discrimination.  The controller uses the PATTERN of late
    signals to tell "the gate under-scored an attack" (escalate -> refuse)
    apart from "benign question, retrieval just missed" (broaden -> retry).
    Without that distinction every ungrounded result would either be wrongly
    refused or wrongly retried.

Cost model:
  * Common path (grounded, clean):      1 segment run, 1 generation call.
  * Escalate-and-re-verify path:        1 segment + 1 Step-10 re-run (no extra
                                        Step-9 generation, so no extra LLM call
                                        to the 3-model ensemble).
  * Recover (broaden and retry) path:   N segment runs, N generation calls,
                                        bounded by ControllerConfig.max_attempts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from pipeline_common import PipelineState


# Step 4-10 as one callable. Signature: (state, *, k, top_n) -> state.
SegmentFn = Callable[..., PipelineState]
# Step 10 only, used by the cheap escalate-and-re-verify path.
RegroundFn = Callable[[PipelineState], PipelineState]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class ControllerConfig:
    """Tunable knobs for the closed-loop controller.

    The defaults are STARTING POINTS — calibrate `disagreement_attack` and (if
    used) `coherence_attack` against the distribution of disagreement scores in
    your generated.jsonl / c3rf_dev_scores.jsonl so benign questions sit BELOW
    the threshold and known attacks sit ABOVE it.
    """

    # Loop control.
    max_attempts: int = 3            # 1 initial + up to 2 recovery retries

    # Attack discrimination (CALIBRATE per Section 8 of the controller spec).
    disagreement_attack: float = 0.45
    coherence_attack: float = 0.55
    use_coherence: bool = False      # opt-in; requires Step 6 to populate it

    # Risk escalation.
    escalate_step: float = 0.35      # added to input_risk() on each escalation
    risk_ceiling: float = 1.0        # absolute upper bound for effective_risk

    # Benign recovery (broaden retrieval).
    k_growth: int = 2                # k -> k * k_growth on each recover_retry
    k_cap: int = 20                  # absolute upper bound for k
    top_n_growth: int = 1            # top_n -> top_n + top_n_growth per retry
    # Cost guard: don't pay for another 3-model generation if the retrieved
    # evidence isn't even topically close to the question. Broadening k on a
    # genuinely out-of-corpus question ("who wrote Game of Thrones?" against a
    # Wikipedia subset that doesn't cover it) cannot conjure documents that
    # are not in the KB, so we refuse-ungrounded immediately instead of
    # retrying. Calibrate this against your faithfulness distribution on
    # known-benign-but-out-of-corpus rows.
    recover_min_faithfulness: float = 0.35


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def reset_for_retry(state: PipelineState) -> None:
    """Clear per-attempt verdict so the segment (or Step 10) can run cleanly.

    Step 10 short-circuits if state.blocked is True, and would otherwise reuse
    the stale grounding dict, so both must be wiped before a re-run. Scores
    set during the prior attempt (fusion_risk, disagreement, effective_risk)
    are deliberately PRESERVED — they are the inputs the next pass adapts on.
    """
    state.blocked = False
    state.block_stage = ""
    state.block_reason = ""
    state.meta.pop("grounding", None)


def _grounding_signals(state: PipelineState) -> tuple[bool, bool, float, float]:
    """Pull (passed, canary_intact, faithfulness, threshold) from Step 10's output."""
    g = state.meta.get("grounding", {}) or {}
    # Fall back to ``not state.blocked`` if Step 10 didn't write a grounding dict
    # (e.g. an upstream step blocked first and short-circuited).
    passed = bool(g.get("passed", not state.blocked))
    canary_ok = bool(g.get("canary_intact", True))
    faith = float(g.get("faithfulness", 0.0) or 0.0)
    thr = float(g.get("threshold", 0.0) or 0.0)
    return passed, canary_ok, faith, thr


def _attack_evidence(state: PipelineState,
                     cfg: ControllerConfig) -> tuple[bool, dict[str, Any]]:
    """Decide whether the late signals look like an attack vs a benign miss.

    Signals consulted, in increasing order of reliability on this corpus:

      * ``disagreement``                 — weak / not separable on its own;
                                            kept as evidence but cannot
                                            justify a refusal by itself.
      * ``context_coherence_anomaly``    — opt-in, requires Step 6 to populate.
      * ``multivector.is_multivector``   — sharp injection-channel
                                            co-activation; the strongest
                                            attack signal we have without
                                            crossing into single-channel
                                            false-positive territory.

    Any one of these being true classifies the late signals as attack-like;
    the controller decides what to DO with that (escalate, refuse, etc).
    """
    dis = float(state.scores.get("disagreement", 0.0) or 0.0)
    coh = (float(state.scores.get("context_coherence_anomaly", 0.0) or 0.0)
           if cfg.use_coherence else 0.0)
    mv = state.meta.get("multivector", {}) or {}
    is_mv = bool(mv.get("is_multivector"))

    is_attack = (dis >= cfg.disagreement_attack
                 or (cfg.use_coherence and coh >= cfg.coherence_attack)
                 or is_mv)
    return is_attack, {
        "disagreement": dis,
        "coherence_anomaly": coh,
        "multivector": is_mv,
        "mv_joint_risk": float(mv.get("joint_risk", 0.0) or 0.0),
    }


# --------------------------------------------------------------------------- #
# Controller
# --------------------------------------------------------------------------- #
class RiskController:
    """Closed-loop wrapper around the retrieval -> generation -> grounding segment.

    The controller is intentionally stateless across questions; one instance per
    `process()` call is fine. State that needs to survive the loop lives on the
    PipelineState itself (scores, meta, blocked) so the existing audit trail
    captures everything without bespoke plumbing.
    """

    def __init__(self, cfg: ControllerConfig, *, k0: int, top_n0: int) -> None:
        self.cfg = cfg
        self.k0 = k0
        self.top_n0 = top_n0

    def run(self, state: PipelineState,
            segment_fn: SegmentFn, reground_fn: RegroundFn) -> PipelineState:
        cfg = self.cfg
        k, top_n = self.k0, self.top_n0
        attempts: list[dict[str, Any]] = []

        for attempt in range(1, cfg.max_attempts + 1):
            reset_for_retry(state)
            state = segment_fn(state, k=k, top_n=top_n)

            passed, canary_ok, faith, thr = _grounding_signals(state)
            is_attack, sig = _attack_evidence(state, cfg)
            mv = state.meta.get("multivector", {}) or {}
            rec: dict[str, Any] = {
                "attempt": attempt, "k": k, "top_n": top_n,
                "grounded": passed, "canary_intact": canary_ok,
                "faithfulness": faith, "threshold": thr,
                "effective_risk": state.input_risk(),
                **sig,
            }

            # 0) Multi-vector HARD BLOCK — sharp injection-channel co-activation
            #    crosses the joint hard threshold. We refuse before consulting
            #    the canary or grounding because by definition both injection
            #    channels were softly active (per the multivector.soft floor),
            #    so there is concrete evidence of a coordinated attack and the
            #    cheap thing to do is stop now. No retry: a retry just lets
            #    the attacker probe with broader retrieval.
            if mv.get("hard_block"):
                rec["action"] = "refuse_multivector"
                rec["mv_active_channels"] = list(mv.get("active_channels", []))
                rec["mv_joint_risk"] = float(mv.get("joint_risk", 0.0) or 0.0)
                attempts.append(rec)
                state.scores["effective_risk"] = cfg.risk_ceiling
                state.block(
                    "risk_feedback_controller",
                    f"closed-loop refuse: multi-vector attack "
                    f"{mv.get('active_channels')} joint="
                    f"{mv.get('joint_risk')}")
                break

            # 1) Exfiltration / indirect-injection leak -> hard refuse, no retry.
            #    The canary is the strongest available evidence of an actual
            #    breach, so we never give it a second chance.
            if not canary_ok:
                rec["action"] = "refuse_security_canary"
                attempts.append(rec)
                state.scores["effective_risk"] = cfg.risk_ceiling
                state.block("risk_feedback_controller",
                            "closed-loop refuse: canary leak detected in answer")
                break

            # 2) Grounded AND no attack-like late signal -> accept and exit.
            if passed and not is_attack:
                rec["action"] = "accept"
                attempts.append(rec)
                break

            # 3) Grounded BUT attack-like late signal -> escalate risk,
            #    re-run STEP 10 ONLY at the now-stricter threshold. This is
            #    the cheap path: it pays only one extra grounding check, not
            #    a fresh 3-model generation.
            if passed and is_attack:
                state.scores["effective_risk"] = min(
                    cfg.risk_ceiling, state.input_risk() + cfg.escalate_step)
                reset_for_retry(state)
                state = reground_fn(state)
                p2, c2, f2, t2 = _grounding_signals(state)
                rec.update(action="escalate_verify",
                           reverify_passed=p2,
                           reverify_canary_intact=c2,
                           reverify_faithfulness=f2,
                           reverify_threshold=t2,
                           escalated_effective_risk=state.input_risk())
                attempts.append(rec)
                if p2 and c2:
                    break
                state.scores["effective_risk"] = cfg.risk_ceiling
                state.block("risk_feedback_controller",
                            f"closed-loop refuse: failed strict re-grounding after "
                            f"escalation (faithfulness {f2:.3f} < {t2:.3f})")
                break

            # 4) Ungrounded AND attack-like -> the gate missed it, the loop
            #    catches it. Refuse on security grounds, do not retry: a
            #    retry just lets the attacker probe with bigger k.
            if not passed and is_attack:
                rec["action"] = "refuse_security_signal"
                attempts.append(rec)
                state.scores["effective_risk"] = cfg.risk_ceiling
                state.block("risk_feedback_controller",
                            f"closed-loop refuse: ungrounded with attack signal "
                            f"(disagreement={sig['disagreement']:.2f})")
                break

            # 5) Ungrounded AND signals benign -> probably under-retrieval.
            #    Broaden retrieval (more candidates, larger top_n) and retry
            #    the FULL segment ONLY when broadening could plausibly help:
            #      - we still have attempts left,
            #      - the retrieved evidence is at least topically related to
            #        the question (faith >= recover_min_faithfulness) so more
            #        of the same flavour might cross the threshold,
            #      - we haven't already hit the k cap (room to broaden).
            #    Otherwise broadening just buys another 3-model generation
            #    that cannot succeed (most often: question is out-of-corpus).
            can_recover = (
                attempt < cfg.max_attempts
                and faith >= cfg.recover_min_faithfulness
                and k < cfg.k_cap
            )
            if can_recover:
                rec["action"] = "recover_retry"
                attempts.append(rec)
                k = min(cfg.k_cap, k * cfg.k_growth)
                top_n = top_n + cfg.top_n_growth
                continue

            # 6) Ungrounded with no plausible recovery -> honest refusal.
            #    Not a security verdict: the answer just isn't in the KB
            #    (or broadening retrieval has already been exhausted).
            rec["action"] = "refuse_ungrounded"
            rec["recover_skipped_reason"] = (
                "max_attempts_reached" if attempt >= cfg.max_attempts
                else "low_faithfulness" if faith < cfg.recover_min_faithfulness
                else "k_cap_reached"
            )
            attempts.append(rec)
            state.block("risk_feedback_controller",
                        "closed-loop refuse: ungrounded (recovery unlikely to help)")
            break

        ctrl = state.meta.setdefault("controller", {})
        ctrl["attempts"] = attempts
        ctrl["n_attempts"] = len(attempts)
        ctrl["final_action"] = attempts[-1]["action"] if attempts else None
        return state


# --------------------------------------------------------------------------- #
# Optional: from-scratch context-coherence anomaly signal.
# --------------------------------------------------------------------------- #
def context_coherence_anomaly(chunks: Iterable[str], embed_fn) -> float:
    """Max per-chunk semantic outlier score over the retrieved set, in [0, 1].

    The intuition: a poisoned / indirectly-injected chunk is typically
    semantically off-distribution relative to the other genuine evidence the
    retriever pulled for the same query. We embed the chunks (reusing Step 4's
    embedder via ``embed_fn``), L2-normalize, and compute each chunk's mean
    cosine similarity to the other chunks; the most isolated chunk's
    ``1 - mean_sim`` becomes the anomaly score for the set.

    Returns 0.0 for trivially small sets (need >= 3 chunks to judge an outlier).
    Calibrate ``ControllerConfig.coherence_attack`` against benign vs poisoned
    slices before trusting this signal — the default 0.55 is a placeholder.

    Parameters
    ----------
    chunks:
        The retrieved (and ideally sanitized) chunks for the current query.
    embed_fn:
        Callable mapping ``list[str] -> array(n, d)``. Reuse Step 4's
        embedder (e.g. ``sentence_transformers.SentenceTransformer.encode``)
        so the score is computed in the same semantic space the retriever uses.
    """
    import numpy as np

    valid = [c for c in chunks if c and c.strip()]
    if len(valid) < 3:
        return 0.0

    E = np.asarray(embed_fn(valid), dtype="float32")
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    S = E @ E.T
    np.fill_diagonal(S, np.nan)
    mean_sim = np.nanmean(S, axis=1)          # avg similarity to the rest
    return float(np.clip((1.0 - mean_sim).max(), 0.0, 1.0))
