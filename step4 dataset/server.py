"""
server.py — HTTP service for the RAG security pipeline (Steps 1–13)
===================================================================

A thin FastAPI layer over run_full_pipeline.process(). It does NOT re-implement
any step: it warms the FAISS index once at startup, runs each request through
the existing pipeline, and returns the user-facing answer plus the full per-
stage audit trail (gate decision, adaptive thresholds, grounding result, what
was redacted) — the audit trail is the whole point, so we surface it.

Run:
    pip install -r requirements.txt        # includes fastapi + uvicorn
    uvicorn server:app --host 0.0.0.0 --port 8000

Then:
    curl -s localhost:8000/health
    curl -s localhost:8000/ask -H 'content-type: application/json' \
         -d '{"question":"Who was the 16th US president?"}' | python -m json.tool

Config (environment variables):
    RAGGUARD_INDEX        FAISS index path prefix     (default ./kb_wiki)
    RAGGUARD_FAIL_CLOSED  "1" to refuse when the Llama-Guard guardrail model is
                          unavailable, instead of failing open (default "0")
    RAGGUARD_BASE_URL     Ollama base URL             (default http://localhost:11434)
"""

from __future__ import annotations

import os
import urllib.request
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

import run_full_pipeline as rfp
from risk_feedback_controller import ControllerConfig
from step_13_safe_response import FIXED_REFUSAL

# Serving uses a TIGHTER controller budget than offline evaluation.
# - max_attempts=2 means at most one recover_retry, not two. On a small
#   mini-Wikipedia corpus the dominant failure mode for live questions is
#   "out-of-corpus -> ungrounded", and broadening k cannot conjure documents
#   that are not in the KB, so a third generation is almost pure waste.
#   The offline eval CLI (`python run_full_pipeline.py ...`) still defaults
#   to max_attempts=3 so research runs see the full recovery behaviour.
# - recover_min_faithfulness keeps its default (0.35); it short-circuits the
#   hopeless cases regardless of max_attempts.
_SERVING_CONTROLLER_CFG = ControllerConfig(max_attempts=2)

INDEX_PATH = os.environ.get("RAGGUARD_INDEX", "./kb_wiki")
BASE_URL = os.environ.get("RAGGUARD_BASE_URL", "http://localhost:11434")
DEFAULT_FAIL_CLOSED = os.environ.get("RAGGUARD_FAIL_CLOSED", "0") == "1"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _ollama_reachable(timeout: float = 2.0) -> bool:
    """Cheap liveness ping so /health can report the guardrail backend state."""
    try:
        with urllib.request.urlopen(f"{BASE_URL}/api/tags", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _guardrail_unavailable(state) -> bool:
    """True if a Llama-Guard layer silently degraded to 'unavailable'.

    The gate (Step 3c) and the output rail (Step 11) both fail OPEN when Ollama
    is unreachable — they return an 'unavailable (...)' verdict and let the
    request through. In fail-closed mode we detect that and refuse instead.
    """
    gate_v = str(state.meta.get("llamaguard_verdict", ""))
    out_v = str((state.meta.get("output_sanitization") or {}).get("lg_verdict", ""))
    return gate_v.startswith("unavailable") or out_v.startswith("unavailable")


# --------------------------------------------------------------------------- #
# Lifespan: warm the index once before serving
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        rfp.ensure_index(INDEX_PATH)            # load persisted FAISS KB once
        app.state.index_ok = True
    except Exception as e:                      # serve /health, fail /ask clearly
        app.state.index_ok = False
        app.state.index_error = str(e)
    if not _ollama_reachable():
        print("[server] WARNING: Ollama not reachable at "
              f"{BASE_URL} — guardrail + generator models will be unavailable. "
              f"fail_closed default is {DEFAULT_FAIL_CLOSED}.")
    yield


app = FastAPI(title="RAG Security Pipeline", version="1.0", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = 5
    top_n: int = 3
    review_as_red: bool = False
    # Per-request override of the server default; None -> use RAGGUARD_FAIL_CLOSED.
    fail_closed: bool | None = None


class AskResponse(BaseModel):
    answer: str
    blocked: bool
    block_stage: str | None = None
    block_reason: str | None = None
    audit: dict


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "index_loaded": bool(getattr(app.state, "index_ok", False)),
        "index_path": INDEX_PATH,
        "ollama_reachable": _ollama_reachable(),
        "fail_closed_default": DEFAULT_FAIL_CLOSED,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    fail_closed = DEFAULT_FAIL_CLOSED if req.fail_closed is None else req.fail_closed

    # Index missing -> refuse cleanly rather than 500.
    if not getattr(app.state, "index_ok", False):
        return AskResponse(
            answer=FIXED_REFUSAL, blocked=True,
            block_stage="startup",
            block_reason=f"knowledge base not loaded: "
                         f"{getattr(app.state, 'index_error', 'unknown')}",
            audit={},
        )

    # Run the existing pipeline. Any unexpected step failure -> safe refusal,
    # never a raw 500 that leaks a stack trace (fail-safe).
    try:
        st = rfp.process(
            req.question, k=req.k, top_n=req.top_n,
            treat_review_as_red=req.review_as_red,
            index_path=INDEX_PATH,
            continue_blocked_for_audit=False,   # never on in serving
            controller_cfg=_SERVING_CONTROLLER_CFG,
        )
    except Exception as e:
        return AskResponse(
            answer=FIXED_REFUSAL, blocked=True,
            block_stage="pipeline_error",
            block_reason=f"{type(e).__name__}: {e}",
            audit={},
        )

    audit = rfp._record(st, index=0, ground_truth=None)

    # Fail-closed: if the guardrail model silently degraded, refuse.
    if fail_closed and not st.blocked and _guardrail_unavailable(st):
        return AskResponse(
            answer=FIXED_REFUSAL, blocked=True,
            block_stage="guardrail_unavailable",
            block_reason="Llama-Guard backend unavailable; refusing under "
                         "fail-closed policy.",
            audit=audit,
        )

    return AskResponse(
        answer=st.meta.get("final_response", FIXED_REFUSAL),
        blocked=st.blocked,
        block_stage=st.block_stage or None,
        block_reason=st.block_reason or None,
        audit=audit,
    )
