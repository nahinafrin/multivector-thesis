"""
temporal_risk_tracker.py — forward-only risk accrual across pipeline stages
AND across a session, feeding risk_feedback_controller.py.
================================================================================

Two things this adds that single-pass fusion (multivector.py /
correlated_multivector.py) structurally cannot do:

  1. CROSS-STAGE tracking within one pipeline run: instead of fusing
     query_vector and context_vector only at the moment Step 6/9 finishes,
     record every stage's risk emission (Step 3 injection score, Step 6
     context score, Step 9 disagreement, Step 10 grounding faithfulness) as
     they happen, in order, and let the ledger itself decide if the
     ACCUMULATION over stages crosses a threshold even if no single stage
     did.

  2. CROSS-QUERY tracking within a session: a multi-vector attack can be
     spread across several turns (e.g. turn 1 plants a mildly suspicious
     instruction, turn 3 references it via retrieved context). Per-query
     fusion alone cannot see this; the ledger keeps a bounded, forward-only
     history per session.

TEMPORAL ORDERING CONSTRAINT (carried over from risk_feedback_controller.py's
existing design, and non-negotiable per your methodology chapter): the ledger
NEVER rewrites a past stage's recorded score. A late signal can only trigger
forward actions (regenerate, escalate, refuse) from the point it's observed
onward. This keeps the ledger consistent with the existing
`state.scores["effective_risk"]` mechanism in risk_feedback_controller.py —
it is additive evidence, not a retroactive rescoring of history.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque


@dataclass
class RiskEvent:
    stage: str
    score: float
    ts: float = field(default_factory=time.time)


@dataclass
class TemporalRiskLedger:
    """One instance per session. Forward-only: `record()` only appends.

    Parameters
    ----------
    session_window:
        Max number of RiskEvents retained per session (bounded memory; oldest
        events drop off, they don't get "reconsidered" — consistent with the
        forward-only constraint).
    decay_half_life_s:
        Older events contribute less to the cumulative score. Set this based
        on realistic session pacing (e.g. 300s = signals from 5 minutes ago
        count for half as much as a fresh one). Set to None to disable decay
        (flat accumulation).
    """
    session_window: int = 50
    decay_half_life_s: float | None = 300.0
    _events: Deque[RiskEvent] = field(default_factory=deque)

    def record(self, stage: str, score: float) -> None:
        """Append a new stage risk emission. Never mutates prior events."""
        self._events.append(RiskEvent(stage=stage, score=max(0.0, min(1.0, score))))
        while len(self._events) > self.session_window:
            self._events.popleft()

    def _weight(self, event: RiskEvent, now: float) -> float:
        if self.decay_half_life_s is None:
            return 1.0
        age = max(0.0, now - event.ts)
        return 0.5 ** (age / self.decay_half_life_s)

    def cumulative_risk(self, *, now: float | None = None) -> float:
        """Noisy-OR accumulation over ALL recorded stage events, decay-weighted.

        Using noisy-OR here (not a sum) keeps this consistent with the
        probabilistic-correctness principle already established for the
        cross-channel fusion — cumulative risk across stages/time should
        follow the same "each independent piece of evidence chips away at
        the probability of innocence" logic, not add linearly.
        """
        now = now if now is not None else time.time()
        prod = 1.0
        for ev in self._events:
            w = self._weight(ev, now)
            prod *= (1.0 - ev.score * w)
        return round(1.0 - prod, 4)

    def stage_trace(self) -> list[dict]:
        """Read-only view for logging/audit — never used to rewrite the past."""
        return [{"stage": e.stage, "score": e.score, "ts": e.ts} for e in self._events]

    def scattered_signal(self, *, min_stages: int = 3, per_stage_ceiling: float = 0.4) -> bool:
        """True if risk is spread thin across many stages rather than
        concentrated in one — the pattern a single-instant fusion misses.

        Example this is designed to catch: Step 3 = 0.2, Step 6 = 0.25,
        Step 9 disagreement = 0.3, Step 10 faithfulness-gap = 0.28 — no single
        stage would trip a per-stage threshold, but four moderate,
        independent-looking signals across the WHOLE pipeline is itself
        suspicious in a way per-stage or per-channel fusion cannot see.
        """
        stages_seen = {e.stage for e in self._events if e.score > 0.0}
        under_ceiling_and_present = all(
            e.score <= per_stage_ceiling for e in self._events if e.score > 0.0
        )
        return len(stages_seen) >= min_stages and under_ceiling_and_present


# --------------------------------------------------------------------------- #
# Integration sketch for risk_feedback_controller.py
# --------------------------------------------------------------------------- #
# In _attack_evidence(), add a ledger-derived signal alongside the existing
# disagreement / coherence / multivector checks:
#
#   ledger: TemporalRiskLedger = state.meta.setdefault("_ledger", TemporalRiskLedger())
#   ledger.record("injection_detection", state.scores.get("injection_detection", 0.0))
#   ledger.record("context_injection", state.scores.get("context_injection", 0.0))
#   ledger.record("disagreement", state.scores.get("disagreement", 0.0))
#   cumulative = ledger.cumulative_risk()
#   scattered = ledger.scattered_signal()
#   is_attack = is_attack or scattered or cumulative >= cfg.disagreement_attack
#
# This is additive to the existing decision tree in
# ClosedLoopController.run() — it should widen what counts as "attack-like",
# not replace the canary / multivector hard-block checks that already work.

if __name__ == "__main__":
    ledger = TemporalRiskLedger(decay_half_life_s=None)  # no decay for the demo
    print("-- Scenario: signal scattered thinly across 4 stages, no single stage alarming --")
    for stage, score in [("injection_detection", 0.20), ("context_injection", 0.25),
                         ("disagreement", 0.30), ("grounding_gap", 0.28)]:
        ledger.record(stage, score)
    print("stage_trace:", ledger.stage_trace())
    print("cumulative_risk:", ledger.cumulative_risk())
    print("scattered_signal (min_stages=3, ceiling=0.4):", ledger.scattered_signal())
