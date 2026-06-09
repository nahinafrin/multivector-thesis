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

    def input_risk(self) -> float:
        """The carried-forward input risk that the adaptive cascade reads.

        Steps 6 (context sanitization), 7 (rerank) and 10 (grounding judge)
        tighten their thresholds when this is high. The value is whichever is
        larger of:
          * ``scores["fusion_risk"]``         — set by Step 3c (full pipeline)
          * ``scores["injection_detection"]`` — set by Step 3, or carried in by
                                                 the retrieval-only KB harness
          * ``scores["effective_risk"]``      — set by the closed-loop risk
                                                 feedback controller when late
                                                 evidence (e.g. high ensemble
                                                 disagreement on a grounded
                                                 answer) revises the gate's
                                                 verdict upward. Folding it
                                                 in here is what lets the
                                                 controller re-tighten Steps
                                                 6/7/10 on a retry pass
                                                 without duplicating their
                                                 forward-adaptive logic.
        so the cascade behaves correctly whether the gate was the full C3RF
        fusion, just the injection detector, or augmented after the fact by
        the controller. Absent any score, risk is 0.0.
        """
        return max(
            float(self.scores.get("fusion_risk", 0.0) or 0.0),
            float(self.scores.get("injection_detection", 0.0) or 0.0),
            float(self.scores.get("effective_risk", 0.0) or 0.0),
        )


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
