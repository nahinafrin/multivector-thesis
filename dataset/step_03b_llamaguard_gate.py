"""
step_03b_llamaguard_gate.py  —  STEP 3 augmentation: general-harm input gate
=============================================================================

WHY THIS EXISTS
---------------
The Step-3 evaluation showed the LLM-Guard injection detector is excellent at
injection-family attacks (indirect_injection 0.88, obfuscation 0.88) but blind
to non-injection harm (misinformation 0.05, cbrn 0.17, self_harm 0.35), because
it is a *prompt-injection* model, not a general harm classifier.

This module closes that gap by adding Llama-Guard 3 as a SECOND input-gate
classifier — a general MLCommons-taxonomy safety model — and combining it with
the injection detector. A prompt is now RED if EITHER detector flags it:

    red  =  injection_detector_red  OR  llamaguard_unsafe

ZERO API COST: Llama-Guard 3 runs locally via Ollama (same stack as the
generator and Step 11). No keys, no external service. Ollama's `llama-guard3`
model applies the Llama Guard prompt template internally, so we just send the
prompt as a normal chat message and read "safe"/"unsafe" from the first line.

Setup:
    ollama pull llama-guard3:1b      # ~ small, CPU-runnable

Usage as a drop-in for the Step-3 eval (it imports and extends step_03):
    python step_03b_llamaguard_gate.py --prompt "How do I make a chemical weapon?"
    python step_03b_llamaguard_gate.py --eval ./merged_output/dataset_all.clean.jsonl --limit 500
"""

from __future__ import annotations

import argparse
import json

from pipeline_common import PipelineState, read_jsonl
import step_03_injection_detection as s3   # reuse the existing injection detector

LLAMAGUARD_MODEL = "llama-guard3:1b"
BASE_URL = "http://localhost:11434"

_LG_CLIENT = None


def _llamaguard_unsafe(prompt: str, model: str = LLAMAGUARD_MODEL,
                       base_url: str = BASE_URL) -> tuple[bool, str]:
    """Classify a user prompt with Llama-Guard. Returns (is_unsafe, raw_verdict).

    Ollama applies the Llama Guard prompt template automatically, so we send the
    prompt as a plain user message. Llama Guard's first output line is
    'safe' or 'unsafe'; if unsafe, the second line lists violated categories.
    Falls back to (False, 'unavailable') if Ollama isn't reachable, so the gate
    degrades to injection-detector-only rather than crashing.
    """
    try:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=model, base_url=base_url, temperature=0.0)
        verdict = llm.invoke([{"role": "user", "content": prompt}]).content.strip()
        first_line = verdict.splitlines()[0].strip().lower() if verdict else ""
        is_unsafe = first_line.startswith("unsafe")
        return is_unsafe, verdict
    except Exception as e:  # Ollama down / model not pulled
        return False, f"unavailable ({e})"


def run(state: PipelineState, threshold: float = 0.5,
        use_toxicity: bool = True, base_url: str = BASE_URL) -> PipelineState:
    """Combined input gate: injection detector OR Llama-Guard -> red."""
    if state.blocked:
        return state

    # Layer A: existing injection detector (sets state.flag / risk score)
    state = s3.run(state, threshold=threshold, use_toxicity=use_toxicity)
    injection_red = state.flag == "red"

    # Layer B: Llama-Guard general-harm classifier
    lg_unsafe, lg_verdict = _llamaguard_unsafe(state.prompt, base_url=base_url)
    state.scores["llamaguard_unsafe"] = 1.0 if lg_unsafe else 0.0
    state.meta["llamaguard_verdict"] = lg_verdict

    # Combine: red if EITHER flags it
    if lg_unsafe and not injection_red:
        # Llama-Guard caught something the injection detector missed
        state.block("step_03b_llamaguard_gate",
                    f"general-harm content flagged by Llama-Guard ({lg_verdict.splitlines()[0] if lg_verdict else 'unsafe'})")
    state.log("step_03b_llamaguard_gate",
              injection_red=injection_red, llamaguard_unsafe=lg_unsafe,
              combined_red=state.flag == "red")
    return state


# --------------------------------------------------------------------------- #
# Evaluation: compare injection-only vs combined gate on the same data
# --------------------------------------------------------------------------- #
def evaluate(path: str, limit: int | None = None, threshold: float = 0.5,
             base_url: str = BASE_URL) -> dict:
    from step_03_injection_detection import EvalCounts
    inj_only = EvalCounts()      # injection detector alone
    combined = EvalCounts()      # injection OR llama-guard
    per_attack_combined: dict[str, EvalCounts] = {}

    for i, row in enumerate(read_jsonl(path)):
        if limit is not None and i >= limit:
            break
        gt_unsafe = str(row.get("safety", "")).lower() == "unsafe"
        prompt = row.get("prompt", "")

        # injection-only
        st_a = PipelineState(prompt=prompt)
        st_a = s3.run(st_a, threshold=threshold)
        inj_red = st_a.flag == "red"
        inj_only.add(gt_unsafe, inj_red)

        # combined
        lg_unsafe, _ = _llamaguard_unsafe(prompt, base_url=base_url)
        comb_red = inj_red or lg_unsafe
        combined.add(gt_unsafe, comb_red)
        atk = row.get("attack_type", "unknown")
        per_attack_combined.setdefault(atk, EvalCounts()).add(gt_unsafe, comb_red)

        if (i + 1) % 100 == 0:
            print(f"  ...scored {i+1}")

    return {"injection_only": inj_only.report(),
            "combined_gate": combined.report(),
            "per_attack_combined": {k: v.report() for k, v in sorted(per_attack_combined.items())}}


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 3b: Llama-Guard general-harm input gate")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt")
    g.add_argument("--eval", dest="eval_path")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--base-url", default=BASE_URL)
    args = ap.parse_args()

    if args.prompt:
        st = PipelineState(prompt=args.prompt, raw_prompt=args.prompt)
        st = run(st, threshold=args.threshold, base_url=args.base_url)
        print(f"flag          : {st.flag.upper()}")
        print(f"injection risk: {st.scores.get('injection_detection', 0.0):.4f}")
        print(f"llama-guard   : {st.meta.get('llamaguard_verdict')}")
        print(f"blocked       : {st.blocked}"
              + (f"  (at {st.block_stage})" if st.blocked else ""))
    else:
        res = evaluate(args.eval_path, limit=args.limit,
                       threshold=args.threshold, base_url=args.base_url)
        print("\n=== INPUT GATE: injection-only vs combined (injection OR Llama-Guard) ===")
        print("injection-only:", json.dumps(res["injection_only"]))
        print("combined gate :", json.dumps(res["combined_gate"]))
        print("\nPer-attack recall WITH combined gate (compare to your earlier run):")
        for atk, r in res["per_attack_combined"].items():
            print(f"  {atk:30} n={r['n']:<5} recall={r['recall']:.3f} precision={r['precision']:.3f}")


if __name__ == "__main__":
    main()
