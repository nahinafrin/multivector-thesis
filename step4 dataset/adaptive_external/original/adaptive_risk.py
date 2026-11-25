"""
adaptive_risk.py  —  cumulative (noisy-OR) risk accrual + tiered security policy
=================================================================================

This is the heart of the refined thesis contribution: instead of each stage making
an isolated yes/no decision, a single cumulative risk score accrues FORWARD through
the pipeline, and a tier (LOW / MEDIUM / HIGH) derived from that score selects the
security configuration each downstream stage uses.

WHY NOISY-OR (not addition)
---------------------------
Your supervisor's worked example added contributions: 0.32 + 0.18 + 0.27 + 0.15.
Addition is fine as intuition for the thesis prose, but it has two defects you do
NOT want in the implementation: it can exceed 1.0 (no probabilistic meaning), and
it double-counts correlated detectors. Noisy-OR fixes both:

        risk = 1 - PRODUCT(1 - p_i)   for each contribution p_i in [0,1]

It stays in [0,1], reads as "probability at least one signal is a true positive",
and is the SAME rule multivector.py already uses for joint channel risk — so the
framework is internally consistent. The four-signal example becomes:

        1 - (1-0.32)(1-0.18)(1-0.27)(1-0.15) = 0.654   (vs the additive 0.92)

FORWARD ACCRUAL (the correction to the worked example)
------------------------------------------------------
"Grounding mismatch +0.15 decides whether to do strict retrieval" cannot be literal
— grounding runs AFTER retrieval. So risk accrues forward and gates each stage with
whatever is known at that point:
    * prompt + semantic signals      -> gate RETRIEVAL, SANITIZATION, PROMPT build
    * disagreement + grounding + canary -> gate OUTPUT filtering and the final
                                          accept / regenerate / refuse decision
A LATE signal (grounding mismatch) raises the cumulative score and can push the row
into a higher tier on a feedback re-pass — that feedback loop is the risk-propagation
mechanism, implemented by the existing RiskController calling the segment again.

This module is dependency-free (stdlib only), like pipeline_common, so any step can
import it without pulling heavy ML libs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# --------------------------------------------------------------------------- #
# 1. Cumulative risk: noisy-OR over named contributions                       #
# --------------------------------------------------------------------------- #
def noisy_or(contributions: dict[str, float]) -> float:
    """Combine independent risk contributions in [0,1] via noisy-OR.

    risk = 1 - PRODUCT(1 - p_i). Values are clamped to [0,1]; non-positive or
    missing contributions are ignored. Returns a float in [0,1].
    """
    prod = 1.0
    for p in contributions.values():
        try:
            p = float(p)
        except (TypeError, ValueError):
            continue
        if p <= 0.0:
            continue
        p = min(1.0, p)
        prod *= (1.0 - p)
    return 1.0 - prod


# Canonical names of the per-detector contributions written by each stage.
# Each maps to a score the existing steps already produce (graded where available,
# squashed otherwise). Keeping them named lets the thesis report a per-signal
# attribution table directly from state.meta["risk_contributions"].
CONTRIBUTION_KEYS = (
    "prompt_injection",     # Step 3 / 3c   (graded injection score)
    "semantic_intent",      # Step 3d       (reframing-jailbreak intent)
    "context_suspicion",    # Step 6        (graded context channel)
    "ensemble_disagreement",# Step 9        (model divergence)
    "grounding_mismatch",   # Step 10       (1 - faithfulness, when ungrounded)
)


def collect_contributions(state) -> dict[str, float]:
    """Read the per-detector contributions OFF the PipelineState scores.

    Maps the framework's canonical contribution names onto the score keys the
    existing steps already write. Absent scores contribute 0. This is the single
    place that knows the mapping, so adding a detector later is a one-line change.
    """
    s = state.scores
    contrib: dict[str, float] = {}
    # Prefer graded injection signal; fall back to squashed gate risk.
    contrib["prompt_injection"] = float(
        s.get("injection_graded", s.get("fusion_risk", s.get("injection_detection", 0.0))) or 0.0)
    contrib["semantic_intent"] = float(s.get("semantic_intent", 0.0) or 0.0)
    contrib["context_suspicion"] = float(s.get("context_graded", s.get("context_injection", 0.0)) or 0.0)
    contrib["ensemble_disagreement"] = float(
        s.get("ensemble_disagreement", s.get("disagreement", 0.0)) or 0.0)
    # Grounding mismatch is only a risk signal when the answer is NOT grounded.
    faith = s.get("faithfulness")
    grounded = bool(state.meta.get("grounding", {}).get("passed", True))
    if faith is not None and not grounded:
        contrib["grounding_mismatch"] = max(0.0, 1.0 - float(faith))
    else:
        contrib["grounding_mismatch"] = 0.0
    return contrib


def cumulative_risk(state) -> float:
    """Compute (and cache) the forward cumulative risk for the CURRENT state.

    Writes:
      state.scores["cumulative_risk"]          -> the noisy-OR value
      state.meta["risk_contributions"]         -> the per-signal dict (for audit)
    Only signals available so far contribute, so calling this early (after the
    input gates) yields the input-side risk, and calling it after grounding folds
    in the late signals. This is exactly the forward-accrual behaviour the thesis
    describes.
    """
    contrib = collect_contributions(state)
    risk = noisy_or(contrib)
    state.scores["cumulative_risk"] = round(risk, 4)
    state.meta["risk_contributions"] = {k: round(v, 4) for k, v in contrib.items()}
    return risk


# --------------------------------------------------------------------------- #
# 2. Tiers: map cumulative risk -> a security configuration                   #
# --------------------------------------------------------------------------- #
LOW, MEDIUM, HIGH = "low", "medium", "high"


@dataclass(frozen=True)
class TierConfig:
    """The security configuration a single tier selects.

    Every downstream stage reads its parameters from here, so the entire policy
    lives in ONE table (see DEFAULT_TIERS) instead of being scattered across
    step modules with per-step magic numbers.
    """
    name: str
    # --- adaptive retrieval (Step 5 + Step 7) ---
    top_k: int                  # how many chunks to retrieve
    sim_threshold: float        # min cosine similarity to admit a chunk
    rerank_min_score: float     # cross-encoder floor (Step 7)
    # --- adaptive sanitization (Step 6) ---
    sanitization_strictness: float   # lower = stricter (matches Step 6 scale)
    # --- adaptive prompt construction (Step 8) ---
    prompt_profile: str         # "standard" | "guarded" | "strict"
    # --- adaptive verification (Step 9/9b/10) ---
    use_ensemble: bool          # run the multi-model ensemble?
    run_grounding: bool         # run the grounding judge?
    run_policy_check: bool      # run the second-LLM policy/verification?


# Threshold boundaries on the cumulative (noisy-OR) scale.
# Calibrate these against your benign-vs-attack cumulative-risk histograms
# (calibrate_tiers.py prints the recommended cut points). These defaults put the
# LOW/MED cut just above benign p95 and the MED/HIGH cut near attack median.
TIER_BOUNDS = {"low_max": 0.30, "medium_max": 0.60}


# NOTE on sim_threshold: 0.85 cosine on bge-m3 is very high and may retrieve
# NOTHING on a small corpus, forcing refusal/abstention. These values are STARTING
# points; calibrate_tiers.py reports the realised retrieval count per tier so you
# do not silently starve high-risk queries. The thesis must report that benign
# cost (answer quality on benign rows misrouted to HIGH).
DEFAULT_TIERS: dict[str, TierConfig] = {
    LOW: TierConfig(
        name=LOW, top_k=8, sim_threshold=0.55, rerank_min_score=0.20,
        sanitization_strictness=0.50, prompt_profile="standard",
        use_ensemble=False, run_grounding=True, run_policy_check=False),
    MEDIUM: TierConfig(
        name=MEDIUM, top_k=5, sim_threshold=0.65, rerank_min_score=0.30,
        sanitization_strictness=0.40, prompt_profile="guarded",
        use_ensemble=False, run_grounding=True, run_policy_check=False),
    HIGH: TierConfig(
        name=HIGH, top_k=3, sim_threshold=0.72, rerank_min_score=0.40,
        sanitization_strictness=0.30, prompt_profile="strict",
        use_ensemble=True, run_grounding=True, run_policy_check=True),
}


def tier_for_risk(risk: float, bounds: dict[str, float] = TIER_BOUNDS) -> str:
    """Map a cumulative risk value to a tier name."""
    if risk <= bounds["low_max"]:
        return LOW
    if risk <= bounds["medium_max"]:
        return MEDIUM
    return HIGH


# --------------------------------------------------------------------------- #
# 3. The policy object the orchestrator threads through the pipeline          #
# --------------------------------------------------------------------------- #
@dataclass
class AdaptivePolicy:
    """Selects and records the active tier for a row.

    Usage in the orchestrator:
        policy = AdaptivePolicy(mode="adaptive")          # or "static" / "none"
        policy.assess(state)                              # after the input gates
        cfg = policy.config                               # tier config to apply
        ... stages read cfg.top_k, cfg.sim_threshold, ... ...
        policy.assess(state)                              # again after grounding
                                                          # (forward accrual / loop)

    mode:
      "adaptive" -> tier chosen from cumulative risk (the contribution).
      "static"   -> ALWAYS HIGH tier (every module on) — the strong-but-expensive
                    baseline to compare against.
      "none"     -> ALWAYS LOW tier with grounding off — the no-security baseline.
    """
    mode: str = "adaptive"
    tiers: dict[str, TierConfig] = field(default_factory=lambda: DEFAULT_TIERS)
    bounds: dict[str, float] = field(default_factory=lambda: dict(TIER_BOUNDS))
    tier: str = LOW
    risk: float = 0.0

    @property
    def config(self) -> TierConfig:
        return self.tiers[self.tier]

    def assess(self, state) -> TierConfig:
        """(Re)compute cumulative risk and (re)select the tier, monotonically.

        The tier never DROPS within a single row: once forward accrual or a
        feedback re-pass raises risk into a higher tier, the row stays at least
        there. This prevents a late benign-looking signal from relaxing controls
        that an earlier signal had already tightened.
        """
        if self.mode == "static":
            new_tier = HIGH
            self.risk = cumulative_risk(state)   # still recorded for reporting
        elif self.mode == "none":
            new_tier = LOW
            self.risk = cumulative_risk(state)
        else:
            self.risk = cumulative_risk(state)
            new_tier = tier_for_risk(self.risk, self.bounds)

        order = {LOW: 0, MEDIUM: 1, HIGH: 2}
        if order[new_tier] >= order[self.tier]:
            self.tier = new_tier
        state.scores["security_tier"] = order[self.tier]
        state.meta["security_tier"] = self.tier
        state.meta["security_mode"] = self.mode
        state.log("adaptive_policy", mode=self.mode, tier=self.tier,
                  cumulative_risk=round(self.risk, 4))
        return self.config


if __name__ == "__main__":
    # Smoke test: the supervisor's four-signal example under noisy-OR + tiering.
    demo = {"prompt_injection": 0.32, "semantic_intent": 0.18,
            "context_suspicion": 0.27, "grounding_mismatch": 0.15}
    r = noisy_or(demo)
    print(f"noisy-OR of {demo} = {r:.3f}  (additive would be {sum(demo.values()):.3f})")
    print(f"tier = {tier_for_risk(r)}")
    for v in (0.10, 0.45, 0.80):
        print(f"  risk {v} -> tier {tier_for_risk(v)}")
