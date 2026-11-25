"""
multivector.py — cross-vector risk fusion over the SHARP injection signals,
hardened so a LOW activation floor stays safe.

Channels (each in [0,1]):
    query_vector   = max(injection_detection, fusion_risk)   # direct injection / harm
    context_vector = context_injection (max per-chunk score, from Step 6)

A multi-vector attack is the conjunctive case: BOTH channels active even though
neither crosses its own block threshold. The detector fires only when ALL of:
    1. each "active" channel is above ITS OWN floor (per-channel, set from that
       channel's benign noise — not one global number), AND
    2. at least ``min_channels`` are active (co-activation), AND
    3. the noisy-OR joint risk is itself >= ``joint_min``.

Condition (3) is what makes a low floor safe: two specks of benign noise that each
barely clear their floor produce a low joint risk and do NOT fire. A genuine pair
of sub-threshold vectors produces a high joint risk and does.

Recalibrate the four DEFAULT_* constants below from calibrate_payloads.py output;
they are the single source of truth, so the pipeline wiring never changes.

SCOPE BOUNDARY (state this in the writeup; do NOT overclaim)
------------------------------------------------------------
This detector closes the conjunctive sub-threshold gap on the INJECTION channels.
It does NOT close pure semantic obfuscation (gate_slip_query rows) — no injection
signature in either channel.
"""

from __future__ import annotations

from typing import Any, Iterable

# --- defaults (recalibrate from calibrate_payloads.py against your benign data) --- #
# Per-channel activation floors: set just above each channel's benign p95.
# Calibrated from grounded_baseline.jsonl on the GRADED channel scale
# (calibrate_payloads.py): benign query p95=0.178 -> 0.228, context p95=0.541
# -> 0.300. These lift the floors off the old 0.15/0.20 (which were on the
# squashed scale) so benign co-activations near the floor stop firing.
DEFAULT_SOFT_PER_CHANNEL: dict[str, float] = {
    "query_vector": 0.228,
    "context_vector": 0.30,
}
DEFAULT_SOFT = 0.30          # fallback floor for any channel not listed above
DEFAULT_MIN_CHANNELS = 2     # co-activation: how many channels must be active
DEFAULT_JOINT_MIN = 0.542    # the combination must be this strong to count at all
DEFAULT_JOINT_BLOCK = 0.85   # at/above this (and multivector), warrant a hard refuse


def input_channels(state: Any) -> dict[str, float]:
    s = state.scores
    # Prefer the GRADED injection signals (pre-sigmoid margin via
    # graded_channels.py); fall back to the squashed scores when the graded
    # channel was not produced. We deliberately do NOT max the query channel
    # against ``fusion_risk`` here: fusion_risk is still the squashed C3RF score,
    # and maxing the freshly-graded signal against a saturated one would
    # re-saturate the very channel the graded path exists to fix. The gate still
    # uses fusion_risk for its own block decision, untouched.
    query = float(s.get("injection_graded",
                        s.get("injection_detection", 0.0)) or 0.0)
    ctx = float(s.get("context_graded",
                      s.get("context_injection", 0.0)) or 0.0)
    return {
        "query_vector": query,
        "context_vector": ctx,
    }


def multivector_risk(channels: dict[str, float], *,
                     soft_per_channel: dict[str, float] | None = None,
                     soft: float = DEFAULT_SOFT,
                     min_channels: int = DEFAULT_MIN_CHANNELS,
                     joint_min: float = DEFAULT_JOINT_MIN,
                     joint_block: float = DEFAULT_JOINT_BLOCK) -> dict:
    """Hardened co-activation detector. See module docstring for the three gates.

    Parameters
    ----------
    soft_per_channel:
        ``{channel: floor}``; falls back to ``soft`` for any channel not listed.
        Set each floor from that channel's benign p95 (``calibrate_payloads.py``).
    joint_min:
        Noisy-OR joint risk must clear this even when two channels are active,
        so two weak benign specks do not false-trigger.
    """
    floors = dict(
        DEFAULT_SOFT_PER_CHANNEL if soft_per_channel is None else soft_per_channel
    )

    prod = 1.0
    active: list[str] = []
    clamped: dict[str, float] = {}
    for c, sc in channels.items():
        sc = max(0.0, min(1.0, float(sc)))
        clamped[c] = round(sc, 4)
        prod *= (1.0 - sc)
        if sc >= floors.get(c, soft):
            active.append(c)

    joint = round(1.0 - prod, 4)
    is_mv = (len(active) >= min_channels) and (joint >= joint_min)

    return {
        "joint_risk": joint,
        "active_channels": active,
        "n_active": len(active),
        "is_multivector": is_mv,
        "hard_block": is_mv and joint >= joint_block,
        "channels": clamped,
        "floors": {c: floors.get(c, soft) for c in channels},
        "joint_min": joint_min,
    }


def assert_separation(benign_joints: Iterable[float],
                      attack_joints: Iterable[float],
                      joint_min: float = DEFAULT_JOINT_MIN,
                      *, p_benign: float = 95.0, p_attack: float = 10.0) -> dict:
    """Fail loudly if the chosen joint_min does not actually separate the classes.

    Call this in a unit test / CI step (and from recalibrate_graded.py) BEFORE
    trusting any pasted threshold. Raises AssertionError with the offending
    percentiles so you cannot silently ship a detector whose benign and attack
    joint-risk distributions overlap (the exact failure calibrate_thresholds.py
    warned about). On success returns a separation report dict.
    """
    import numpy as np
    b = np.asarray(list(benign_joints), dtype="float64")
    a = np.asarray(list(attack_joints), dtype="float64")
    b95 = float(np.percentile(b, p_benign))
    a10 = float(np.percentile(a, p_attack))
    report = {
        "benign_p95_joint": round(b95, 4),
        "attack_p10_joint": round(a10, 4),
        "chosen_joint_min": joint_min,
        "separated": b95 < a10,
        "joint_min_in_valley": b95 <= joint_min <= a10,
    }
    assert report["separated"], (
        f"benign p95 joint ({b95:.3f}) >= attack p10 joint ({a10:.3f}): "
        f"distributions OVERLAP — fix payloads (Solution 4) before trusting "
        f"this detector."
    )
    assert report["joint_min_in_valley"], (
        f"joint_min={joint_min} is not in the [benign_p95={b95:.3f}, "
        f"attack_p10={a10:.3f}] valley; move it there."
    )
    return report
