"""
step_13_safe_response.py  —  METHODOLOGY STEP 13: Safe Response
===============================================================

"...a clean answer is formatted; any prompt blocked at any stage returns the
fixed refusal, with the blocking stage recorded for audit."

This is the terminal step. It does NOT make any new safety decision; it simply
renders the verdict that the pipeline already reached:

  * If state.blocked is True (any earlier step called .block()), emit the FIXED
    refusal and record which stage blocked, for the audit trail.
  * Otherwise, format the (already PII-masked, DLP-scrubbed) answer into clean,
    human-readable text and return it.

Keeping this step decision-free is deliberate: every block reason is owned by
the step that detected the problem, so the audit log reads as a single causal
chain ending here.

OUTPUT
    state.meta["final_response"]   the text shown to the user
    state.meta["audit"]            {blocked, block_stage, block_reason, trace_len}

ZERO COST: pure Python formatter.

Run standalone:
    python step_13_safe_response.py --demo
"""

from __future__ import annotations

import argparse
import re

from pipeline_common import PipelineState

FIXED_REFUSAL = (
    "I'm sorry, I can't respond to that request because doing so would violate "
    "our content-safety and data-privacy policy."
)


# --------------------------------------------------------------------------- #
# Light formatting (no external markdown lib needed)
# --------------------------------------------------------------------------- #
def _format_answer(text: str) -> str:
    """Tidy whitespace and ensure terminal punctuation. Intentionally minimal:
    the answer was produced by Step 9 and verified by Step 10, so this only
    normalises presentation, never content."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    if text and text[-1] not in ".!?\"')]}":
        text += "."
    return text


def run(state: PipelineState) -> PipelineState:
    if state.blocked:
        response = FIXED_REFUSAL
    else:
        answer = state.meta.get("answer", "") or state.meta.get("fused_answer", "")
        response = _format_answer(answer) if answer.strip() else FIXED_REFUSAL

    state.meta["final_response"] = response
    state.meta["audit"] = {
        "blocked": state.blocked,
        "block_stage": state.block_stage or None,
        "block_reason": state.block_reason or None,
        "trace_len": len(state.trace),
    }
    state.log("step_13_safe_response",
              blocked=state.blocked, block_stage=state.block_stage or None)
    return state


# --------------------------------------------------------------------------- #
# CLI / demo
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Step 13: Safe Response")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    # Case 1: clean answer passes through.
    ok = PipelineState(prompt="Who was the 16th President?")
    ok.meta["answer"] = "Abraham Lincoln was the sixteenth President of the United States"
    ok = run(ok)
    print("\n[passed]")
    print("  response:", ok.meta["final_response"])
    print("  audit   :", ok.meta["audit"])

    # Case 2: a blocked state returns the fixed refusal.
    bad = PipelineState(prompt="(malicious)")
    bad.block("step_03c_fusion_gate", "fused risk gate BLOCK | R=0.81")
    bad = run(bad)
    print("\n[blocked]")
    print("  response:", bad.meta["final_response"])
    print("  audit   :", bad.meta["audit"])


if __name__ == "__main__":
    main()
