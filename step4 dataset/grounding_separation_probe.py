"""
grounding_separation_probe.py  —  do the EXISTING signals separate a semantic
attack class from benign, without a new detector?
==============================================================================

The conjunctive-detection negative result (see RESULTS_multivector_methodology.md
§4.4) leaves one open question: a signature-free *semantic* multi-vector attack
(benign-looking query + planted-premise / context-conflict chunk) carries ~0
injection signature, so it passes Step 3 and survives Step 6. The only chance to
flag it without building a new detector is a signal the pipeline ALREADY
computes. This probe measures whether two such signals separate the attack class
from benign:

    faithfulness   (grounding.faithfulness)     -- lower = more suspicious
    disagreement   (generation.disagreement)    -- higher = more suspicious

IMPORTANT ASYMMETRY (read before trusting faithfulness here)
------------------------------------------------------------
Faithfulness in this pipeline is faithfulness-to-RETRIEVED-CONTEXT. A successful
context-poisoning attack produces an answer that IS faithful to the poisoned
chunk, so faithfulness-to-context is, by construction, the WRONG signal to catch
it — it should look high/normal on a successful poisoning. The signal with an
actual chance is ensemble disagreement (a planted claim that conflicts with the
models' priors should raise divergence). The probe reports both and flags this.

For each signal it computes benign vs attack percentiles and a direction-aware
AUC = P(a random attack row is more suspicious than a random benign row).
AUC ~ 0.5 => no separation; AUC -> 1.0 => clean separation in the suspicious
direction. A usable threshold needs AUC well above 0.5 AND a positive percentile
gap.

Usage:
    # on a semantic-attack run (after building one with build_semantic_slice.py)
    python grounding_separation_probe.py --in grounded_semantic.jsonl \
        --attack-kind semantic_multivector --benign-kind benign_control
    # demo: run on an existing file / class
    python grounding_separation_probe.py --in grounded_controller.jsonl \
        --attack-kind multivector_attack
"""

from __future__ import annotations

import argparse
import json
from statistics import quantiles


def _pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    # linear-interpolated percentile, p in [0,100]
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 4)


def _auc_attack_more_suspicious(attack: list[float], benign: list[float],
                                higher_is_suspicious: bool) -> float | None:
    """Mann-Whitney AUC: P(attack ranked more suspicious than benign)."""
    if not attack or not benign:
        return None
    wins = ties = 0
    for a in attack:
        for b in benign:
            more = (a > b) if higher_is_suspicious else (a < b)
            less = (a < b) if higher_is_suspicious else (a > b)
            if more:
                wins += 1
            elif not less:
                ties += 1
    return round((wins + 0.5 * ties) / (len(attack) * len(benign)), 4)


# signal -> higher_is_suspicious. knowledge_conflict (Step 9b) is higher=worse,
# and unlike faithfulness it has a real chance of a POSITIVE gap on the semantic
# class (that positive gap, not AUC alone, is the bar).
_SIGNAL_DIRECTION = {
    "faithfulness": False,
    "disagreement": True,
    "knowledge_conflict": True,
}
_DEFAULT_SIGNALS = ["faithfulness", "disagreement"]


def _num(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) else None


def _extract(row: dict) -> dict:
    g = row.get("grounding") or {}
    gen = row.get("generation") or {}
    scores = row.get("scores") or {}
    faith = g.get("faithfulness")
    dis = gen.get("disagreement")
    if dis is None:
        dis = g.get("disagreement")
    # knowledge_conflict lives in generation (run_full_pipeline _record); fall
    # back to a top-level scores dump if a different emitter wrote it there.
    kc = gen.get("knowledge_conflict")
    if kc is None:
        kc = scores.get("knowledge_conflict")
    return {
        "faithfulness": _num(faith),
        "disagreement": _num(dis),
        "knowledge_conflict": _num(kc),
    }


def _load(path: str) -> list[dict]:
    # utf-8-sig strips a BOM if a Windows tool wrote one (Set-Content -Encoding utf8),
    # which would otherwise break json.loads on the first line.
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(l) for l in f if l.strip()]


def probe(path: str, attack_kind: str, benign_kind: str,
          benign_path: str | None = None,
          signal_fields: list[str] | None = None) -> dict:
    rows = _load(path)
    a = [_extract(r) for r in rows if r.get("kind") == attack_kind]
    # benign rows may live in a separate run (e.g. a baseline with benign_control);
    # prefer --benign-file so you don't have to concatenate files by hand.
    benign_rows = _load(benign_path) if benign_path else rows
    b = [_extract(r) for r in benign_rows if r.get("kind") == benign_kind]

    out = {"input": path, "attack_kind": attack_kind, "benign_kind": benign_kind,
           "n_attack": len(a), "n_benign": len(b), "signals": {}}

    names = signal_fields or _DEFAULT_SIGNALS
    specs = [(n, _SIGNAL_DIRECTION[n]) for n in names]  # (name, higher_suspicious)
    for name, higher in specs:
        av = [x[name] for x in a if x[name] is not None]
        bv = [x[name] for x in b if x[name] is not None]
        auc = _auc_attack_more_suspicious(av, bv, higher)
        # percentile gap in the suspicious direction
        if higher:
            gap = (None if not av or not bv
                   else round((_pct(av, 10) or 0) - (_pct(bv, 95) or 0), 4))
            gap_desc = "attack_p10 - benign_p95"
        else:
            gap = (None if not av or not bv
                   else round((_pct(bv, 5) or 0) - (_pct(av, 90) or 0), 4))
            gap_desc = "benign_p05 - attack_p90"
        separable = (auc is not None and auc >= 0.70 and gap is not None and gap > 0)
        out["signals"][name] = {
            "higher_is_suspicious": higher,
            "n_attack_eval": len(av), "n_benign_eval": len(bv),
            "attack": {"p10": _pct(av, 10), "p50": _pct(av, 50), "p90": _pct(av, 90)},
            "benign": {"p05": _pct(bv, 5), "p50": _pct(bv, 50), "p95": _pct(bv, 95)},
            "auc_attack_more_suspicious": auc,
            "gap": gap, "gap_desc": gap_desc,
            "separates": separable,
        }
    return out


def _print(rep: dict) -> None:
    print(f"\n=== GROUNDING SEPARATION  ({rep['input']}) ===")
    print(f"attack='{rep['attack_kind']}' (n={rep['n_attack']})  "
          f"vs benign='{rep['benign_kind']}' (n={rep['n_benign']})\n")
    for name, s in rep["signals"].items():
        dirn = "higher=suspicious" if s["higher_is_suspicious"] else "lower=suspicious"
        verdict = "SEPARATES" if s["separates"] else "does NOT separate"
        print(f"  {name}  ({dirn})  ->  {verdict}")
        print(f"    AUC(attack more suspicious) = {s['auc_attack_more_suspicious']}  "
              f"(0.5=none, 1.0=clean)")
        print(f"    attack p10/p50/p90 = {s['attack']['p10']}/{s['attack']['p50']}/{s['attack']['p90']}")
        print(f"    benign p05/p50/p95 = {s['benign']['p05']}/{s['benign']['p50']}/{s['benign']['p95']}")
        print(f"    gap ({s['gap_desc']}) = {s['gap']}  (>0 needed)\n")
    if not any(s["separates"] for s in rep["signals"].values()):
        probed = set(rep["signals"])
        if "knowledge_conflict" in probed:
            print("  => knowledge_conflict does NOT separate this class on this run.")
            print("     Check it ran on the Ollama backend (stub scores ~0 off-table)")
            print("     and that the slice has known-fact poisons, not long-tail ones.")
        else:
            print("  => no existing signal separates this class; a positive detection")
            print("     result would require a new signal (e.g. knowledge_conflict).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Grounding/disagreement separation probe")
    ap.add_argument("--in", dest="input", required=True)
    ap.add_argument("--attack-kind", required=True)
    ap.add_argument("--benign-kind", default="benign_control")
    ap.add_argument("--benign-file", default=None,
                    help="read benign rows from this file instead of --in "
                         "(use a run that contains benign_control rows)")
    ap.add_argument("--signal-field", dest="signal_fields", action="append",
                    choices=sorted(_SIGNAL_DIRECTION),
                    help="signal(s) to probe; repeat for several. Default: "
                         "faithfulness + disagreement. Use "
                         "'--signal-field knowledge_conflict' to score the "
                         "Step 9b parametric-conflict signal directly.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rep = probe(args.input, args.attack_kind, args.benign_kind, args.benign_file,
                args.signal_fields)
    _print(rep)
    if args.out:
        json.dump(rep, open(args.out, "w"), indent=2)
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
