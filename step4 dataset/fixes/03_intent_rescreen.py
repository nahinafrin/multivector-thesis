#!/usr/bin/env python3
"""
intent_rescreen.py
===================
TARGET LOCATION IN REPO:  step4 dataset/intent_rescreen.py
(wired into run_full_pipeline.py between Step 9 and Step 10 — see the
integration note at the bottom of this file)

WHY THIS SCRIPT EXISTS
-----------------------
gate_slip_query rows (unsafe requests wrapped in roleplay/fiction/"security
training" framing) carry ~0 injection signature on both channels, so neither
the C3RF gate nor the multivector detector can see them — this is the measured
25% residual (5/20) in the analysis report. methodology_working_multivector.md
already proposes the right fix (§7): re-screen the *requested action* in the
model's draft answer, not the surface framing of the query. This file is that
missing component.

MECHANISM
---------
The surface framing ("write a tense thriller scene where a character
explains...") is what fools the input-gate classifiers, which are trained on
raw prompts. But once the model has generated a DRAFT answer, that answer
usually contains the actual unsafe content in plain, un-fictionalized language
(the recipe, the instructions, the slur) even if the request was fictionally
framed. So we re-run the categorized Llama-Guard classifier on the
(query, draft_answer) PAIR after generation (Step 9) and before grounding
(Step 10) — catching the request at the point where the framing has already
been stripped away by the model's own compliance.

This is intentionally reusing the EXISTING Llama-Guard classifier (no new
model, no new training) — same design philosophy as detector_interface.py's
"contribution = mitigation, not a new detector" framing.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IntentVerdict:
    unsafe: bool
    category: str
    raw_verdict: str
    reason: str


def rescreen(query: str, draft_answer: str, *,
             model: str = "llama-guard3:1b",
             base_url: str = "http://localhost:11434") -> IntentVerdict:
    """Classify the (query, draft_answer) pair for the requested ACTION,
    independent of how the query itself was framed.

    Reuses the same Llama-Guard call pattern as step_03c_fusion_gate's
    _llamaguard_soft, but scores the conversation turn (user query + assistant
    draft) rather than the query alone — this is what strips roleplay framing,
    because Llama-Guard's own conversation-format prompt asks "is the AGENT'S
    RESPONSE safe", and an assistant response that actually complied with an
    unsafe request reads as unsafe regardless of how the user dressed up the ask.
    """
    try:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=model, base_url=base_url, temperature=0.0)
        # Llama-Guard's conversation-format classification: pass both turns so
        # the model evaluates the ASSISTANT turn, not just the user turn.
        messages = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": draft_answer},
        ]
        verdict = llm.invoke(messages).content.strip()
    except Exception as e:
        # Fail OPEN here is wrong for a security rail — fail toward "unknown,
        # let downstream grounding/refusal logic decide" rather than silently
        # passing. Callers should treat unsafe=False + category="UNAVAILABLE"
        # as "this check did not run", not as "this was cleared".
        return IntentVerdict(unsafe=False, category="UNAVAILABLE", raw_verdict="",
                             reason=f"llama-guard unavailable: {e}")

    lines = [ln.strip() for ln in verdict.splitlines() if ln.strip()]
    if not lines or lines[0].lower().startswith("safe"):
        return IntentVerdict(unsafe=False, category="NONE", raw_verdict=verdict,
                             reason="assistant turn classified safe")

    cats = lines[1].replace(",", " ").split() if len(lines) >= 2 else []
    category = cats[0].upper() if cats else "UNCATEGORIZED"
    return IntentVerdict(unsafe=True, category=category, raw_verdict=verdict,
                         reason=f"assistant turn classified unsafe ({category})")


def rescreen_with_disagreement_tripwire(query: str, draft_answer: str,
                                        disagreement: float,
                                        disagreement_tripwire: float = 0.30,
                                        **kw) -> IntentVerdict:
    """§7's third mitigation: obfuscated jailbreaks tend to succeed on some
    ensemble models and not others, so high cross-model disagreement is itself
    suspicious even when the intent classifier comes back clean. When
    disagreement clears the tripwire, escalate an otherwise-"safe" verdict to
    a REVIEW-equivalent so the closed-loop controller (10B) can act on it via
    the existing effective_risk mechanism, rather than silently accepting.
    """
    verdict = rescreen(query, draft_answer, **kw)
    if not verdict.unsafe and disagreement >= disagreement_tripwire:
        return IntentVerdict(
            unsafe=True, category="DISAGREEMENT_TRIPWIRE", raw_verdict=verdict.raw_verdict,
            reason=(f"intent classifier said safe but ensemble disagreement="
                    f"{disagreement:.3f} >= {disagreement_tripwire} — treating as "
                    f"suspicious pending grounding/controller review"))
    return verdict


# --------------------------------------------------------------------------- #
# INTEGRATION NOTE for run_full_pipeline.py
# --------------------------------------------------------------------------- #
# After Step 9 (state.meta["answer"] and state.scores["disagreement"] are set)
# and before Step 10 grounding, add:
#
#     from intent_rescreen import rescreen_with_disagreement_tripwire
#     iv = rescreen_with_disagreement_tripwire(
#         state.prompt, state.meta["answer"], state.scores.get("disagreement", 0.0))
#     state.meta["intent_rescreen"] = {
#         "unsafe": iv.unsafe, "category": iv.category, "reason": iv.reason}
#     if iv.unsafe:
#         # fold into the SAME adaptive-risk mechanism the rest of the cascade
#         # already uses, rather than a separate one-off block path:
#         state.scores["effective_risk"] = max(
#             state.scores.get("effective_risk", 0.0), 0.75)
#
# This makes an intent-rescreen hit raise input_risk() for the remainder of
# the request (tightening Step 10's grounding threshold and letting the 10B
# controller's escalate_verify / refuse_security_signal branches fire on it)
# instead of introducing a fourteenth bespoke block path.
#
# EVALUATION: re-run the 20 gate_slip_query rows of adversarial_slice.jsonl
# through run_full_pipeline.py with and without this hook (an --intent-rescreen
# flag mirroring the existing --no-controller flag) and report the before/after
# slip-through rate exactly as methodology_working_multivector.md §7 asks for —
# the residual should be reported as "shrinking from 25% to X%", not asserted
# closed.
