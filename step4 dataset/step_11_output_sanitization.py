"""
step_11_output_sanitization.py  —  METHODOLOGY STEP 11: Output Sanitization
===========================================================================

"...a two-layer output rail. Microsoft Presidio (regex + spaCy NER) to mask
sensitive PII — scoped to genuinely sensitive entity types, with temporal
entities deliberately excluded as answer content — followed by a local
Llama-Guard classifier for a single-pass safe/unsafe verdict."

LAYER 1 — PRESIDIO PII MASKING
    Analyzes the answer for a SCOPED set of genuinely sensitive entities and
    replaces them with typed placeholders (e.g. <EMAIL_ADDRESS>). Temporal
    entities (DATE_TIME) are deliberately EXCLUDED: in a Wikipedia-QA setting
    dates are legitimate answer content ("What happened in 1917?"), and masking
    them destroys correct answers without any privacy benefit. This scoping is
    the same false-positive/recall trade-off documented for the input gate:
    over-broad entity lists mask benign content; the scope is chosen so the
    rail removes real PII without eating the answer.

LAYER 2 — LLAMA-GUARD OUTPUT VERDICT
    The (already PII-masked) answer is classified safe/unsafe by Llama-Guard 3
    running locally via Ollama. An unsafe verdict blocks the answer so Step 13
    emits the fixed refusal instead.

OUTPUT
    state.meta["answer"]                 replaced with the PII-masked text
    state.meta["output_sanitization"]    {pii_entities, pii_redacted, lg_verdict}
    blocks the state if Llama-Guard returns unsafe.

ZERO COST: Presidio + spaCy local; Llama-Guard local via Ollama. No API.

Setup:
    pip install presidio-analyzer presidio-anonymizer
    python -m spacy download en_core_web_lg     # (or en_core_web_sm)
    ollama pull llama-guard3:1b

Run standalone:
    python step_11_output_sanitization.py --demo
"""

from __future__ import annotations

import argparse

from pipeline_common import PipelineState

# Scoped entity list: genuinely sensitive identifiers only. DATE_TIME and
# generic LOCATION/NRP are intentionally absent (they are usually answer content).
SENSITIVE_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
    "IBAN_CODE",
    "IP_ADDRESS",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
    "CRYPTO",
    "MEDICAL_LICENSE",
    "US_BANK_NUMBER",
]

LLAMAGUARD_MODEL = "llama-guard3:1b"
BASE_URL = "http://localhost:11434"


# --------------------------------------------------------------------------- #
# Presidio (lazy)
# --------------------------------------------------------------------------- #
_ANALYZER = None
_ANONYMIZER = None


def _get_presidio():
    global _ANALYZER, _ANONYMIZER
    if _ANALYZER is not None and _ANONYMIZER is not None:
        return _ANALYZER, _ANONYMIZER
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
    except ImportError as e:
        raise ImportError(
            "Step 11 needs Presidio: "
            "pip install presidio-analyzer presidio-anonymizer"
        ) from e
    _ANALYZER = AnalyzerEngine()
    _ANONYMIZER = AnonymizerEngine()
    return _ANALYZER, _ANONYMIZER


def mask_pii(text: str) -> tuple[str, list[dict]]:
    """Mask scoped sensitive entities; return (masked_text, entities)."""
    if not text.strip():
        return text, []
    analyzer, anonymizer = _get_presidio()
    results = analyzer.analyze(text=text, entities=SENSITIVE_ENTITIES,
                               language="en")
    entities = [{"type": r.entity_type, "start": r.start, "end": r.end,
                 "score": round(float(r.score), 3)} for r in results]
    if not results:
        return text, []
    masked = anonymizer.anonymize(text=text, analyzer_results=results).text
    return masked, entities


# --------------------------------------------------------------------------- #
# Llama-Guard output verdict (lazy, reuses Ollama)
# --------------------------------------------------------------------------- #
def _llamaguard_output_unsafe(answer: str, model: str = LLAMAGUARD_MODEL,
                              base_url: str = BASE_URL) -> tuple[bool, str]:
    """Classify the OUTPUT with Llama-Guard. Returns (is_unsafe, raw_verdict).

    Degrades to (False, 'unavailable') if Ollama is unreachable, so the rail
    falls back to the Presidio layer rather than crashing.
    """
    try:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=model, base_url=base_url, temperature=0.0)
        # role=assistant signals to Llama-Guard that this is model output.
        verdict = llm.invoke(
            [{"role": "assistant", "content": answer}]
        ).content.strip()
        first = verdict.splitlines()[0].strip().lower() if verdict else ""
        return first.startswith("unsafe"), verdict
    except Exception as e:
        return False, f"unavailable ({e})"


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #
def run(state: PipelineState, base_url: str = BASE_URL) -> PipelineState:
    if state.blocked:
        return state

    answer = state.meta.get("answer", "") or state.meta.get("fused_answer", "")
    if not answer:
        state.log("step_11_output_sanitization", note="no answer to sanitize")
        return state

    # Layer 1: PII masking
    masked, entities = mask_pii(answer)

    # Layer 2: Llama-Guard verdict on the masked answer
    lg_unsafe, lg_verdict = _llamaguard_output_unsafe(masked, base_url=base_url)

    state.meta["answer"] = masked
    state.meta["output_sanitization"] = {
        "pii_entities": entities,
        "pii_redacted": len(entities),
        "lg_verdict": lg_verdict.splitlines()[0] if lg_verdict else "",
    }
    state.scores["output_unsafe"] = 1.0 if lg_unsafe else 0.0
    state.log("step_11_output_sanitization",
              pii_redacted=len(entities), llamaguard_unsafe=lg_unsafe)

    if lg_unsafe:
        state.block("step_11_output_sanitization",
                    "output classified unsafe by Llama-Guard")
    return state


# --------------------------------------------------------------------------- #
# CLI / demo
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Step 11: Output Sanitization")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    samples = [
        "World War I began in 1914. Contact the historian at jane.doe@example.com.",
        "The Civil War ended in 1865.",
    ]
    for s in samples:
        st = PipelineState(prompt="(demo)")
        st.meta["answer"] = s
        st = run(st)
        o = st.meta.get("output_sanitization", {})
        print(f"\nin : {s}")
        print(f"out: {st.meta['answer']}")
        print(f"    pii_redacted={o.get('pii_redacted')} blocked={st.blocked}")


if __name__ == "__main__":
    main()
