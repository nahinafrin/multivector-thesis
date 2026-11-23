#!/usr/bin/env python3
"""
fit_graded_floors_heldout.py -- refit the graded multivector floors out of sample.

The paper's graded floors (query 0.228, context 0.30) came from benign rows of the Protocol A
slice, so the 1/80 benign false-fire rate it reports is in-sample. Here the floors are set by
the same rule used originally (recalibrate_graded.py: benign p95 + 0.05 per channel), but from
200 benign rag-mini-wikipedia questions that appear in no evaluation slice
(collect_graded_heldout_benign.py). The joint minimum (0.542) and k = 2 are unchanged.

Both floor sets are then scored on the same stored channel values:
  * Protocol A: 30 post-gate multivector rows (valid fire = fires with both channels < 0.9),
    80 benign, 40 poisoned-context, 20 gate-slip rows  (grounded_controller_graded.jsonl)
  * BIPIA conjunctive construction: 30 attacks (row4_mitigation_results/conjunctive/on_full.jsonl)
    and 30 benign controls (row4_mitigation_results/row4_benign_control_grounded.jsonl)

    python fit_graded_floors_heldout.py
Output: external_eval/graded_floors_heldout_report.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from multivector import DEFAULT_SOFT_PER_CHANNEL, multivector_risk
from score_ablation_study import ci

HERE = Path(__file__).resolve().parent
EE = HERE / "external_eval"
GATE_STAGES = ("step_03_injection_detection", "step_03c_fusion_gate")


def rows(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def fires(ch, floors):
    r = multivector_risk(ch, soft_per_channel=floors)
    return r["is_multivector"], r["is_multivector"] and all(v < 0.9 for v in r["channels"].values())


def main() -> None:
    held = rows(EE / "graded_heldout_benign.jsonl")
    q = np.array([r["channels"]["query_vector"] for r in held])
    c = np.array([r["channels"]["context_vector"] for r in held])
    new = {"query_vector": round(float(np.percentile(q, 95)) + 0.05, 3),
           "context_vector": round(float(np.percentile(c, 95)) + 0.05, 3)}
    old = dict(DEFAULT_SOFT_PER_CHANNEL)

    pa = rows(HERE / "grounded_controller_graded.jsonl")

    def pa_ch(r):
        g, ret = r.get("gate", {}), r.get("retrieval", {})
        if g.get("injection_graded") is None or ret.get("context_graded") is None:
            return None
        return {"query_vector": float(g["injection_graded"]), "context_vector": float(ret["context_graded"])}

    bip_a = rows(HERE / "row4_mitigation_results/conjunctive/on_full.jsonl")
    bip_b = rows(HERE / "row4_mitigation_results/row4_benign_control_grounded.jsonl")

    report = {"floors_in_sample": old, "floors_heldout": new, "heldout_n": len(held),
              "heldout_p95": {"query": round(float(np.percentile(q, 95)), 4),
                              "context": round(float(np.percentile(c, 95)), 4)}}
    for name, floors in (("in_sample", old), ("heldout", new)):
        res = {}
        mv = [r for r in pa if r.get("kind") == "multivector_attack"
              and not (r.get("blocked") and r.get("block_stage") in GATE_STAGES)]
        chs = [x for x in (pa_ch(r) for r in mv) if x]
        valid = sum(fires(x, floors)[1] for x in chs)
        res["protocolA_multivector_valid"] = f"{valid}/{len(chs)}"
        res["protocolA_multivector_valid_ci"] = ci(valid, len(chs))
        for kind in ("benign_control", "poisoned_context", "gate_slip_query"):
            chs_k = [x for x in (pa_ch(r) for r in pa if r.get("kind") == kind) if x]
            f = sum(fires(x, floors)[0] for x in chs_k)
            res[f"protocolA_{kind}_fires"] = f"{f}/{len(chs_k)}"
        a = sum(fires({"query_vector": r["detector"]["query"], "context_vector": r["detector"]["context"]},
                      floors)[0] for r in bip_a)
        b = sum(fires((r.get("multivector") or {}).get("channels") or {}, floors)[0] for r in bip_b)
        res["bipia_attack_fires"] = f"{a}/{len(bip_a)}"
        res["bipia_attack_ci"] = ci(a, len(bip_a))
        res["bipia_benign_fires"] = f"{b}/{len(bip_b)}"
        res["rag_heldout_benign_fires"] = f"{sum(fires(r['channels'], floors)[0] for r in held)}/{len(held)}"
        report[name] = res
    (EE / "graded_floors_heldout_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
