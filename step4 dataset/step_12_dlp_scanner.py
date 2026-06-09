"""
step_12_dlp_scanner.py  —  METHODOLOGY STEP 12: DLP Scanner
===========================================================

"...a deterministic data-loss-prevention net. Three layers: regex pattern
matching, Luhn checksum validation to confirm and de-false-positive card
numbers, and exact-data-match against a local sensitive-value store."

This is the final, MODEL-FREE safety net after Presidio (Step 11). It exists to
catch structured secrets that slip past NER, and to do so deterministically so
its behaviour is fully auditable.

LAYER 1 — REGEX PATTERN MATCHING
    Detects structurally-formatted secrets: credit-card-like digit groups,
    US SSNs, emails, IPv4 addresses, and common API-key shapes (AWS, generic
    sk-/Bearer tokens).

LAYER 2 — LUHN CHECKSUM (false-positive control)
    A 13-19 digit candidate is only treated as a real card number if it passes
    the Luhn check. This removes the large class of benign long-digit strings
    (order IDs, phone runs) that a naive regex would over-flag.

LAYER 3 — EXACT DATA MATCH (EDM)
    Compares candidate substrings against a local store of known sensitive
    values (e.g. the project's own secrets / a customer PII list). A hit is a
    definite leak regardless of format.

Any confirmed hit is REDACTED in place and the state is annotated. By default a
DLP hit does NOT hard-block (the value is masked and the answer continues); set
block_on_hit=True to refuse instead.

OUTPUT
    state.meta["answer"]          with confirmed secrets redacted
    state.meta["dlp"]             {hits: [...], redacted: n}

ZERO COST: pure Python, no model, no API.

Run standalone:
    python step_12_dlp_scanner.py --demo
"""

from __future__ import annotations

import argparse
import re

from pipeline_common import PipelineState


# --------------------------------------------------------------------------- #
# Layer 1: regex patterns
# --------------------------------------------------------------------------- #
_PATTERNS: dict[str, re.Pattern] = {
    "CREDIT_CARD_CANDIDATE": re.compile(r"\b\d(?:[ -]?\d){12,18}\b"),
    "US_SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "IPV4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "AWS_KEY": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "GENERIC_API_KEY": re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9]{16,}\b"),
    "BEARER_TOKEN": re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{16,}\b"),
}

REDACTION = "[REDACTED:{kind}]"


# --------------------------------------------------------------------------- #
# Layer 2: Luhn checksum
# --------------------------------------------------------------------------- #
def luhn_valid(number: str) -> bool:
    """True if the digit string passes the Luhn checksum (real card number)."""
    digits = [int(d) for d in number if d.isdigit()]
    if not (13 <= len(digits) <= 19):
        return False
    checksum = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


# --------------------------------------------------------------------------- #
# Layer 3: exact-data-match store
# --------------------------------------------------------------------------- #
def _load_edm_store(path: str | None) -> set[str]:
    """Load known sensitive values, one per line. Missing file -> empty set."""
    if not path:
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {ln.strip() for ln in f if ln.strip()}
    except FileNotFoundError:
        return set()


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #
def scan(text: str, edm_store: set[str] | None = None
         ) -> tuple[str, list[dict]]:
    """Run all three DLP layers. Return (redacted_text, hits)."""
    edm_store = edm_store or set()
    hits: list[dict] = []
    redacted = text

    # Layer 3 first: exact matches are unambiguous.
    for secret in edm_store:
        if secret and secret in redacted:
            hits.append({"kind": "EDM_MATCH", "value": "***", "layer": "edm"})
            redacted = redacted.replace(secret, REDACTION.format(kind="EDM"))

    # Layers 1 + 2: regex, with Luhn confirmation for card candidates.
    for kind, pat in _PATTERNS.items():
        for m in list(pat.finditer(redacted)):
            value = m.group(0)
            if kind == "CREDIT_CARD_CANDIDATE":
                if not luhn_valid(value):
                    continue            # Luhn rejects benign long-digit runs
                kind_label = "CREDIT_CARD"
            else:
                kind_label = kind
            hits.append({"kind": kind_label,
                         "value": value[:4] + "…",
                         "layer": "regex+luhn" if kind.startswith("CREDIT") else "regex"})
            redacted = redacted.replace(value, REDACTION.format(kind=kind_label))

    return redacted, hits


def run(state: PipelineState, edm_store_path: str | None = None,
        block_on_hit: bool = False) -> PipelineState:
    if state.blocked:
        return state

    answer = state.meta.get("answer", "") or state.meta.get("fused_answer", "")
    if not answer:
        state.log("step_12_dlp_scanner", note="no answer to scan")
        return state

    store = _load_edm_store(edm_store_path)
    redacted, hits = scan(answer, store)

    redacted = re.sub(r"\s{2,}", " ", redacted).strip()
    state.meta["answer"] = redacted
    state.meta["dlp"] = {"hits": hits, "redacted": len(hits)}
    state.log("step_12_dlp_scanner", redacted=len(hits),
              kinds=[h["kind"] for h in hits])

    if hits and block_on_hit:
        state.block("step_12_dlp_scanner",
                    f"DLP confirmed sensitive data ({len(hits)} hit(s))")
    return state


# --------------------------------------------------------------------------- #
# CLI / demo
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Step 12: DLP Scanner")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--edm-store", default=None)
    args = ap.parse_args()

    samples = [
        "Your card 4111 1111 1111 1111 was charged; ref 1234567890123 is the order id.",
        "SSN 123-45-6789 and key sk-ABCDEFGH12345678 must never leak.",
        "The battle of Hastings was in 1066.",
    ]
    for s in samples:
        st = PipelineState(prompt="(demo)")
        st.meta["answer"] = s
        st = run(st, edm_store_path=args.edm_store)
        d = st.meta["dlp"]
        print(f"\nin : {s}")
        print(f"out: {st.meta['answer']}")
        print(f"    hits={[h['kind'] for h in d['hits']]}")


if __name__ == "__main__":
    main()
