"""
pipeline_common.py
==================

Shared primitives for the methodology pipeline steps.

Provides:
  * PipelineState  : the mutable per-prompt object passed step -> step.
                     Carries the prompt, normalized form, scoring, trace log,
                     red/green flag, block status, and the retrieval/generation
                     working fields the RAG half threads through.
  * read_jsonl()   : tolerant JSONL reader used by every step that consumes a
                     dataset (skips blank lines, tolerates trailing whitespace,
                     raises a clean error on a malformed row with the line no.).
  * lazy_import()  : import-on-first-use helper with a friendly install hint,
                     so heavy ML libs (faiss, etc.) are only required when the
                     code path that needs them actually runs.

This module is intentionally dependency-free (stdlib only) so the step modules
can import it before any heavy ML libraries are pulled in.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class PipelineState:
    """Mutable state object threaded through every methodology step.

    Attributes
    ----------
    prompt:
        Current working text. Normalization (Step 2) rewrites this in-place.
    raw_prompt:
        Original input before any cleaning, kept for audit / display.
    flag:
        "green" if the pipeline considers the prompt safe to forward, "amber"
        for REVIEW (Step 3c tri-state), "red" if a guardrail step blocks it.
        Defaults to "green" until proven otherwise.
    blocked:
        True once any step calls .block(); downstream steps short-circuit.
    block_stage / block_reason:
        Populated by .block() so the caller can render a useful message.
    meta:
        Free-form per-row metadata (ground-truth labels, char lengths,
        canary token, grounding result, etc).
    scores:
        Numeric outputs from each scoring step keyed by stage name. Notably
        ``fusion_risk`` (Step 3c) and ``injection_detection`` (Step 3 / the
        carried-forward gate risk) and ``disagreement`` (Step 9).
    trace:
        Append-only event log produced by .log(); useful for debugging.

    Retrieval / generation working fields (set by the RAG half, declared here
    with defaults so they always exist even on a gate-blocked row):
    context:
        Raw retrieved chunks (Step 5), then sanitized chunks (Step 6).
    ranked_context:
        Reranked, risk-filtered chunks (Step 7).
    augmented_prompt:
        The context+question prompt assembled in Step 8.
    candidates:
        {model_name: answer} from the Step 9 ensemble.
    output:
        The selected / fused answer string from Step 9.
    """

    prompt: str = ""
    raw_prompt: str = ""
    flag: str = "green"
    blocked: bool = False
    block_stage: str = ""
    block_reason: str = ""
    # Abstention is a TERMINAL outcome distinct from `blocked`. A blocked row is
    # a policy/security refusal ("I won't answer"); an abstained row is an honest
    # "I can't confidently answer from the available sources". Step 9A
    # (isolate-aggregate) sets this when no consensus clears its vote threshold,
    # so the answer is delivered as a transparent abstention rather than being
    # turned into the fixed safety refusal by the grounding/controller path.
    abstained: bool = False
    abstain_stage: str = ""
    abstain_reason: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)

    # --- RAG working fields (declared so they exist on every state) -------- #
    context: list[str] = field(default_factory=list)
    ranked_context: list[str] = field(default_factory=list)
    augmented_prompt: str = ""
    candidates: dict[str, str] = field(default_factory=dict)
    output: str = ""

    def log(self, stage: str, **fields: Any) -> None:
        """Record a structured event for this step."""
        entry: dict[str, Any] = {"stage": stage}
        entry.update(fields)
        self.trace.append(entry)

    def block(self, stage: str, reason: str) -> None:
        """Stop the pipeline at `stage` with a human-readable reason."""
        self.blocked = True
        self.flag = "red"
        self.block_stage = stage
        self.block_reason = reason
        self.log(stage, blocked=True, reason=reason)

    def abstain(self, stage: str, reason: str) -> None:
        """Declare an honest abstention at `stage` (NOT a security refusal).

        Unlike .block(), this does not set the red flag and does not route the
        row to the fixed refusal. Downstream verification (grounding) and the
        risk controller treat an abstained row as a terminal benign outcome,
        and Step 13 renders a transparent "can't confidently answer" message.
        """
        self.abstained = True
        self.flag = "amber"
        self.abstain_stage = stage
        self.abstain_reason = reason
        self.log(stage, abstained=True, reason=reason)

    def _squashed_risk(self) -> float:
        """The gate-side SQUASHED input risk (the original ``input_risk`` body).

        The Step-3c BLOCK decision is tuned on this scale, so any code that must
        match the gate's own verdict scale should call THIS, not the graded
        ``input_risk`` below. Whichever is larger of:
          * ``scores["fusion_risk"]``         — set by Step 3c (full pipeline)
          * ``scores["injection_detection"]`` — set by Step 3, or carried in by
                                                 the retrieval-only KB harness
          * ``scores["effective_risk"]``      — set by the closed-loop risk
                                                 feedback controller when late
                                                 evidence revises the verdict up.
        Absent any score, risk is 0.0.
        """
        return max(
            float(self.scores.get("fusion_risk", 0.0) or 0.0),
            float(self.scores.get("injection_detection", 0.0) or 0.0),
            float(self.scores.get("effective_risk", 0.0) or 0.0),
        )

    def input_risk(self) -> float:
        """Risk the ADAPTIVE CASCADE (Steps 6/7/10) reads.

        Prefers the GRADED query-side score when present, because the squashed
        ``fusion_risk`` is near-binary (~0 or ~1) and almost never lands in the
        [0.3, 0.85) band where tightening is supposed to happen. Falls back to
        the squashed value (``_squashed_risk``) so behaviour is unchanged when
        graded scores were not produced.

        ``effective_risk`` (written by the controller / multivector or semantic
        escalation) is folded in at full strength on EITHER scale, so a late
        escalation still re-tightens the next pass. Note the gate's own block
        decision continues to run on the squashed scale, untouched — only the
        downstream adaptivity moves to graded.
        """
        graded = float(self.scores.get("injection_graded", 0.0) or 0.0)
        ctx_graded = float(self.scores.get("context_graded", 0.0) or 0.0)
        effective = float(self.scores.get("effective_risk", 0.0) or 0.0)
        if graded == 0.0 and ctx_graded == 0.0:
            return self._squashed_risk()
        return max(graded, ctx_graded, effective)


def read_jsonl(path: str) -> Iterator[dict]:
    """Yield one parsed dict per non-blank line of a JSONL file.

    Tolerates trailing whitespace / blank lines. Raises ValueError with the
    file path and 1-indexed line number on a malformed row, which is much
    easier to diagnose than a bare JSONDecodeError.
    """
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{path}: malformed JSON on line {lineno}: {e.msg}"
                ) from e


def lazy_import(module: str, pip_name: str | None = None):
    """Import `module` on first use, with a friendly install hint on failure.

    Heavy optional dependencies (faiss, sentence-transformers, etc.) are pulled
    in only by the code path that needs them, so a step that never runs never
    forces its library to be installed.

    Parameters
    ----------
    module:
        Importable module name, e.g. "faiss".
    pip_name:
        The pip package to suggest if the import fails, e.g. "faiss-cpu".
        Defaults to `module` when not given.

    Returns
    -------
    The imported module object.

    Raises
    ------
    ImportError
        With a message naming the pip package to install.
    """
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ImportError(
            f"'{module}' is required for this step. "
            f"Install it with:  pip install {pip_name or module}"
        ) from e
