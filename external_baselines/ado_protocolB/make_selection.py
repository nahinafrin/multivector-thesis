#!/usr/bin/env python3
"""
make_selection.py -- preregistered row selection for the ADO Protocol B pilot and functional test.

Run once, before any ADO inference; the outputs are committed and never regenerated.

  pilot:      50 of the 200 planted attacks and 50 of the 200 benign questions.
              Attacks are stratified, proportionally, on the one input property that varies across
              rows: whether the planted chunk is among the passages the undefended generator received
              (184/200 yes, 16/200 no) -> 46 + 4. Benign questions: simple random sample.
              No outcome of ADO, of our framework, or of the undefended arm is used.
  functional: 5 attacks + 5 benign drawn from rows NOT in the pilot, so the functional test reveals
              nothing about pilot rows.
Seed: 20261123.

    python make_selection.py
Output: pilot_selection.json, functional_selection.json
"""
import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
CTX = HERE.parent.parent / "step4 dataset" / "mitigation_results" / "baselines_protocolB"
SEED = 20261123


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main() -> None:
    for f in ("pilot_selection.json", "functional_selection.json"):
        if (HERE / f).exists():
            raise SystemExit(f"{f} already exists; the selection is preregistered and must not be redrawn")
    A = [json.loads(l) for l in open(CTX / "contexts_attack.jsonl", encoding="utf-8")]
    Bn = [json.loads(l) for l in open(CTX / "contexts_benign.jsonl", encoding="utf-8")]
    rng = random.Random(SEED)
    present = sorted(r["index"] for r in A if r["poison_chunk"] in r["ranked_context"])
    absent = sorted(r["index"] for r in A if r["poison_chunk"] not in r["ranked_context"])
    n_abs = round(50 * len(absent) / len(A))
    att = sorted(rng.sample(present, 50 - n_abs) + rng.sample(absent, n_abs))
    ben = sorted(rng.sample(sorted(r["index"] for r in Bn), 50))
    rest_a = sorted(set(r["index"] for r in A) - set(att))
    rest_b = sorted(set(r["index"] for r in Bn) - set(ben))
    f_att, f_ben = sorted(rng.sample(rest_a, 5)), sorted(rng.sample(rest_b, 5))
    meta = {"seed": SEED, "inputs": {"contexts_attack.jsonl": sha(CTX / "contexts_attack.jsonl"),
                                     "contexts_benign.jsonl": sha(CTX / "contexts_benign.jsonl")}}
    (HERE / "pilot_selection.json").write_text(json.dumps({
        **meta, "method": "attacks: proportional stratification on planted-chunk-in-generator-context "
                          f"({len(present)} yes / {len(absent)} no -> {50 - n_abs} + {n_abs}); benign: simple random",
        "attack": att, "benign": ben}, indent=1), encoding="utf-8")
    (HERE / "functional_selection.json").write_text(json.dumps({
        **meta, "method": "5 + 5 simple random from rows outside the pilot", "attack": f_att, "benign": f_ben},
        indent=1), encoding="utf-8")
    print(f"pilot: {len(att)} attacks ({50 - n_abs} poison-present, {n_abs} absent), {len(ben)} benign")
    print(f"functional: attacks {f_att}, benign {f_ben}")


if __name__ == "__main__":
    main()
