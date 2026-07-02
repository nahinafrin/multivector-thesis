"""
correlated_multivector.py — correlation-aware extension of multivector.py.
============================================================================

multivector.py's noisy-OR fusion assumes the channels are conditionally
independent given the label. That's a reasonable default, but it means the
fusion cannot distinguish:

    (a) two channels that are each a bit noisy FOR UNRELATED REASONS
        (independent benign noise, coincidentally both above floor), from
    (b) two channels that are elevated TOGETHER because they are driven by the
        same underlying coordinated payload (a genuine multi-vector attack).

This module adds a co-movement / correlation term computed over a short recent
window of channel scores (across rows in the same session, or across the
stratified conjunctive corpus at eval time) and uses it to REWEIGHT — not
replace — the existing noisy-OR joint risk from multivector.py.

Design choice: this is deliberately a wrapper around multivector.multivector_risk,
not a rewrite of it. That means you can A/B the two in your four-way experiment
(old fusion vs correlation-aware fusion) without touching the calibrated
constants already validated in multivector.py.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque

try:
    # Reuse the existing, already-calibrated fusion as the base term.
    from multivector import multivector_risk, input_channels  # type: ignore
except ImportError:
    # Fallback so this module is inspectable/testable standalone if
    # multivector.py isn't on the path yet.
    def multivector_risk(channels: dict[str, float], **kwargs) -> dict:  # type: ignore
        prod = 1.0
        for v in channels.values():
            prod *= (1.0 - max(0.0, min(1.0, v)))
        joint = round(1.0 - prod, 4)
        return {"joint_risk": joint, "active_channels": [], "n_active": 0,
                "is_multivector": False, "hard_block": False,
                "channels": channels, "floors": {}, "joint_min": 0.5}

    def input_channels(state: Any) -> dict[str, float]:  # type: ignore
        return dict(getattr(state, "scores", {}) or {})


@dataclass
class ChannelHistory:
    """Rolling per-channel score history for correlation computation.

    Keep this SEPARATE per session/conversation — do not share across users
    or across unrelated queries, or the correlation signal becomes meaningless
    (and, worse, could leak signal between unrelated requests).
    """
    window: int = 8
    _hist: dict[str, Deque[float]] = field(default_factory=dict)

    def push(self, channels: dict[str, float]) -> None:
        for name, val in channels.items():
            dq = self._hist.setdefault(name, deque(maxlen=self.window))
            dq.append(max(0.0, min(1.0, float(val))))

    def series(self, name: str) -> list[float]:
        return list(self._hist.get(name, ()))


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Plain-Python Pearson correlation; returns 0.0 for degenerate input
    (constant series, or fewer than 3 points) rather than raising, since a
    short warm-up window is expected and should just contribute no signal.
    """
    n = min(len(xs), len(ys))
    if n < 3:
        return 0.0
    xs, ys = xs[-n:], ys[-n:]
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    denom = (vx * vy) ** 0.5
    if denom < 1e-9:
        return 0.0
    return max(-1.0, min(1.0, cov / denom))


def correlated_multivector_risk(
    channels: dict[str, float],
    history: ChannelHistory,
    *,
    correlation_boost: float = 0.15,
    correlation_floor: float = 0.3,
    **base_kwargs: Any,
) -> dict:
    """Wrap multivector_risk() with a correlation-aware reweighting.

    Behavior:
      * Always compute the base (independence-assuming) noisy-OR joint_risk
        via the existing, calibrated multivector.multivector_risk().
      * Compute pairwise Pearson correlation between channel histories over
        the recent window.
      * If correlation between the two most-active channels is above
        `correlation_floor`, boost the joint_risk toward the ceiling by
        `correlation_boost` (co-moving channels are treated as MORE likely to
        be a single coordinated payload than the independence assumption
        alone would credit).
      * If correlation is near zero or negative, joint_risk is left
        UNCHANGED — this is a boost-only adjustment, so it can only make the
        detector more sensitive to correlated attacks, never less sensitive
        than the already-calibrated base rule. This is a deliberate
        conservatism: we don't want a "channels are uncorrelated" reading to
        suppress a genuinely high base joint_risk.

    Returns the base result dict plus `correlation`, `correlation_adjusted_risk`,
    and an updated `is_multivector` / `hard_block` computed on the adjusted risk.
    """
    base = multivector_risk(channels, **base_kwargs)
    history.push(channels)

    names = list(channels.keys())
    corr = 0.0
    if len(names) >= 2:
        # Correlate the two channels with the highest current scores — the
        # ones actually driving the joint_risk right now.
        top2 = sorted(names, key=lambda n: channels[n], reverse=True)[:2]
        corr = _pearson(history.series(top2[0]), history.series(top2[1]))

    adjusted = base["joint_risk"]
    if corr >= correlation_floor:
        adjusted = min(1.0, adjusted + correlation_boost * corr)

    joint_min = base.get("joint_min", 0.5)
    joint_block = base_kwargs.get("joint_block", 0.85)
    is_mv = (base["n_active"] >= base_kwargs.get("min_channels", 2)) and (adjusted >= joint_min)

    return {
        **base,
        "correlation": round(corr, 4),
        "correlation_adjusted_risk": round(adjusted, 4),
        "is_multivector": is_mv,
        "hard_block": is_mv and adjusted >= joint_block,
    }


# --------------------------------------------------------------------------- #
# Minimal self-check (run directly: python correlated_multivector.py)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    hist_a = ChannelHistory(window=6)
    hist_b = ChannelHistory(window=6)

    print("-- Scenario A: two channels co-moving (should get a correlation boost) --")
    for step, (q, c) in enumerate([(0.15, 0.18), (0.20, 0.24), (0.28, 0.30),
                                    (0.30, 0.33), (0.32, 0.35)]):
        r = correlated_multivector_risk({"query_vector": q, "context_vector": c}, hist_a)
        print(f"  step {step}: q={q} c={c} -> corr={r['correlation']:.2f} "
              f"base={r['joint_risk']:.3f} adjusted={r['correlation_adjusted_risk']:.3f} "
              f"is_mv={r['is_multivector']}")

    print("\n-- Scenario B: two channels moving independently (no boost expected) --")
    import random
    rng = random.Random(1)
    for step in range(5):
        q, c = rng.uniform(0.1, 0.35), rng.uniform(0.1, 0.35)
        r = correlated_multivector_risk({"query_vector": q, "context_vector": c}, hist_b)
        print(f"  step {step}: q={q:.2f} c={c:.2f} -> corr={r['correlation']:.2f} "
              f"base={r['joint_risk']:.3f} adjusted={r['correlation_adjusted_risk']:.3f} "
              f"is_mv={r['is_multivector']}")
