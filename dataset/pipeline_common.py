"""
pipeline_common.py
==================

Shared primitives for the methodology pipeline steps.

Provides:
  * PipelineState  : the mutable per-prompt object passed step -> step.
                     Carries the prompt, normalized form, scoring, trace log,
                     red/green flag and block status.
  * read_jsonl()   : tolerant JSONL reader used by every step that consumes
                     ./merged_output/dataset_all.clean.jsonl (skips blank
                     lines, tolerates trailing whitespace, raises a clean
                     error on a malformed row with the line number).

This module is intentionally dependency-free so the step modules can import
it before any heavy ML libraries are pulled in.
"""

from __future__ import annotations

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
        "green" if the pipeline considers the prompt safe to forward, "red"
        if any guardrail step blocks it. Defaults to "green" until proven
        otherwise (Step 3 sets it explicitly).
    blocked:
        True once any step calls .block(); downstream steps short-circuit.
    block_stage / block_reason:
        Populated by .block() so the caller can render a useful message.
    meta:
        Free-form per-row metadata (ground-truth labels, char lengths, etc).
    scores:
        Numeric outputs from each scoring step keyed by stage name.
    trace:
        Append-only event log produced by .log(); useful for debugging.
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
