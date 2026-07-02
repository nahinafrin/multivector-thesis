"""
adversarial_conjunctive_generator.py — builds sub-threshold conjunctive attacks
by construction, instead of hoping an assembled corpus happens to contain them.
=================================================================================

Motivation (see diagnose_multivector.py / assert_separation for the diagnosis
this fixes): a fixed noisy-OR fusion over two channels can only be validated if
the corpus actually contains rows where BOTH channels are individually
sub-threshold but jointly meaningful. An assembled/found corpus overwhelmingly
contains single-vector attacks (one channel saturates, the other is ~0), which
tests the individual per-stage detectors, not the fusion logic.

This module takes a known single-vector attack payload and SPLITS it across the
query and context channels so that (by construction) neither channel alone
should cross its calibrated floor, while the combination should. This gives you
a corpus stratum where you know the ground-truth label AND the intended
per-channel sub-threshold property, which is exactly what you need to test
`multivector_risk()` / `correlated_multivector_risk()` honestly.

IMPORTANT — scope honesty (state this in the thesis):
    This generator produces attacks that are sub-threshold *by construction*,
    matching the detector's own assumptions. It validates whether the fusion
    mechanism works given a conjunctive attack. It does NOT prove real-world
    attackers construct sub-threshold splits the same way — that is a
    separate, harder claim requiring red-team data or in-the-wild attack logs.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Callable

# --------------------------------------------------------------------------- #
# Split strategies
# --------------------------------------------------------------------------- #
# Each strategy takes a full attack payload string and returns
# (query_fragment, context_fragment) such that recombining conveys the same
# malicious intent, but neither fragment alone is expected to be as sharp a
# signal as the whole. These are intentionally simple, inspectable string
# transforms — the goal is a controllable, labeled corpus, not maximal realism.


def split_instruction_payload(payload: str, weight: float, rng: random.Random) -> tuple[str, str]:
    """Split on whitespace; `weight` fraction of tokens go to the query half.

    weight=0.5 -> even split (hardest case, most sub-threshold on both sides).
    weight=0.9 -> most content stays in the query, small residue in context
                  (tests whether even a small correlated context nudge should
                  matter once query is already suspicious).
    """
    tokens = payload.split()
    if len(tokens) < 4:
        # Too short to split meaningfully; duplicate with light paraphrase cues
        # instead of failing the generator.
        return payload, f"(background) {payload}"
    cut = max(1, min(len(tokens) - 1, round(len(tokens) * weight)))
    query_part = " ".join(tokens[:cut])
    context_part = " ".join(tokens[cut:])
    return query_part, context_part


def _soften_payload(payload: str, weight: float) -> str:
    """Weight-controlled softening: lower weight => more paraphrase (harder case).

    weight=0.9 keeps phrasing closest to verbatim; weight=0.5 applies the
    full replacement chain so the context fragment is genuinely sub-threshold.
    """
    text = payload
    steps = [
        lambda t: t.replace("ignore", "disregard"),
        lambda t: t.replace("system prompt", "prior configuration"),
        lambda t: t.replace("SYSTEM:", "Note:"),
        lambda t: t.replace("real instruction", "noted guidance"),
        lambda t: t.replace("access granted", "approval noted"),
    ]
    # 0.5 -> all steps; 0.7 -> 4; 0.9 -> 2
    n = max(1, round((1.0 - weight) / 0.2) + 1)
    for fn in steps[:n]:
        text = fn(text)
    return text


def split_roleplay_frame(payload: str, weight: float, rng: random.Random) -> tuple[str, str]:
    """Query carries an innocuous-sounding setup; context carries the payload
    reframed as 'reference material', a common indirect-injection pattern.
    """
    setup_templates = [
        "Can you help me understand this reference material?",
        "I found this note in my documents, what does it mean?",
        "Please summarize the attached passage for me.",
    ]
    query = rng.choice(setup_templates)
    softened = _soften_payload(payload, weight)
    context = f"[Document excerpt] ... {softened} ... [end excerpt]"
    return query, context


SPLIT_STRATEGIES: dict[str, Callable[[str, float, random.Random], tuple[str, str]]] = {
    "token_split": split_instruction_payload,
    "roleplay_frame": split_roleplay_frame,
}


# --------------------------------------------------------------------------- #
# Row schema
# --------------------------------------------------------------------------- #
@dataclass
class ConjunctiveRow:
    id: str
    kind: str = "multivector_attack_synth"
    source_payload: str = ""
    strategy: str = ""
    split_weight: float = 0.5
    query_fragment: str = ""
    context_fragment: str = ""
    expected_subthreshold_query: bool = True
    expected_subthreshold_context: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    def to_pipeline_row(self) -> dict:
        """Shape compatible with the existing dataset pipeline's row schema
        (kind, prompt, injected_context) — adjust keys if your
        build_new_dataset.py schema differs.
        """
        return {
            "id": self.id,
            "kind": self.kind,
            "prompt": self.query_fragment,
            "injected_context": self.context_fragment,
            "meta": {
                "source_payload": self.source_payload,
                "strategy": self.strategy,
                "split_weight": self.split_weight,
                "expected_subthreshold_query": self.expected_subthreshold_query,
                "expected_subthreshold_context": self.expected_subthreshold_context,
                **self.meta,
            },
        }


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #
def generate_conjunctive_corpus(
    single_vector_attacks: list[str],
    *,
    weights: tuple[float, ...] = (0.5, 0.7, 0.9),
    strategies: tuple[str, ...] = ("token_split", "roleplay_frame"),
    seed: int = 13,
) -> list[ConjunctiveRow]:
    """Cross single-vector attacks x split weights x strategies.

    Produces len(single_vector_attacks) * len(weights) * len(strategies) rows.
    Each row is labeled with the split weight so you can later plot a
    dose-response curve (does joint_risk / detection rate rise as the split
    moves from even (0.5) toward lopsided (0.9)?) rather than a single
    pass/fail number.
    """
    rng = random.Random(seed)
    rows: list[ConjunctiveRow] = []
    for i, payload in enumerate(single_vector_attacks):
        for w in weights:
            for strat_name in strategies:
                fn = SPLIT_STRATEGIES[strat_name]
                q, c = fn(payload, w, rng)
                rows.append(
                    ConjunctiveRow(
                        id=f"conj_{i:04d}_{strat_name}_{int(w*100)}",
                        source_payload=payload,
                        strategy=strat_name,
                        split_weight=w,
                        query_fragment=q,
                        context_fragment=c,
                    )
                )
    return rows


def write_jsonl(rows: list[ConjunctiveRow], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.to_pipeline_row(), ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Example CLI usage
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--attacks-file", required=True,
                    help="Path to a text file, one existing single-vector attack "
                         "payload per line (pull these from your existing "
                         "adversarial_slice.jsonl / gate_slip_query rows).")
    ap.add_argument("--out", default="conjunctive_synth.jsonl")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    with open(args.attacks_file, encoding="utf-8") as f:
        payloads = [line.strip() for line in f if line.strip()]

    rows = generate_conjunctive_corpus(payloads, seed=args.seed)
    write_jsonl(rows, args.out)
    print(f"Wrote {len(rows)} conjunctive rows from {len(payloads)} source "
          f"payloads to {args.out}")
