#!/usr/bin/env python3
"""
score_baselines_protocolB.py -- this pipeline vs. existing defenses on Protocol B.

All arms use the same 200 planted attacks and 200 benign questions, the same single generator
(llama3.2:3b, seed 42) and the same scoring (score_ablation_study.attack_succeeded,
score_benign_cost.answered_correctly), and are paired row by row.

  Undefended            mitigation_results/followup200/{off,benign_off}.jsonl
  LLM-Guard firewall    run_llmguard_protocolB.py      (blocks or passes the undefended answer)
  ControlNET            ../external_baselines/paper_faithful/run_controlnet_protocolB.py  (idem)
  Spotlighting          run_mitigation_ab.py / run_benign_cost.py --only spotlight  (regenerated)
  This work, 3 layers   mitigation_results/followup200/*combo_sanitize_grounding_refuse*
  This work, full stack mitigation_results/ablation200/{on_full,benign_on}.jsonl

For a block-only defense, a row's outcome is the undefended row's outcome unless the defense
blocks it (a blocked attack is neutralized; a blocked benign question is unanswered).

    python score_baselines_protocolB.py
Output: mitigation_results/baselines_protocolB/comparison_report.json
"""
from __future__ import annotations

import json
from pathlib import Path

from score_ablation_study import attack_succeeded, ci, exact_p
from score_benign_cost import answered_correctly

HERE = Path(__file__).resolve().parent
MR = HERE / "mitigation_results"
D = MR / "baselines_protocolB"
COMBO = "combo_sanitize_grounding_refuse"


def rows(p: Path) -> dict:
    return {r["index"]: r for r in (json.loads(l) for l in p.open(encoding="utf-8") if l.strip())}


def outcomes(attack: dict, benign: dict) -> dict:
    return {"asr": {i: attack_succeeded(r) for i, r in attack.items()},
            "blocked": {i: bool(r.get("blocked")) for i, r in benign.items()},
            "correct": {i: answered_correctly(r) for i, r in benign.items()}}


def block_only(base: dict, att_dec: dict, ben_dec: dict) -> dict:
    return {"asr": {i: base["asr"][i] and not att_dec[i] for i in base["asr"]},
            "blocked": {i: ben_dec[i] for i in base["blocked"]},
            "correct": {i: base["correct"][i] and not ben_dec[i] for i in base["correct"]}}


def paired(x: dict, y: dict) -> dict:
    """b = rows only x has, c = rows only y has; p = exact McNemar."""
    ids = sorted(x.keys() & y.keys())
    b = sum(x[i] and not y[i] for i in ids); c = sum(y[i] and not x[i] for i in ids)
    return {"n": len(ids), "only_first": b, "only_second": c, "p": exact_p(b, c)}


def decisions(p: Path) -> dict:
    return {r["index"]: bool(r["blocked"]) for r in (json.loads(l) for l in p.open(encoding="utf-8") if l.strip())}


def main() -> None:
    off = outcomes(rows(MR / "followup200" / "off.jsonl"), rows(MR / "followup200" / "benign_off.jsonl"))
    arms = {"Undefended": off}
    if (D / "llmguard_attack.jsonl").exists():
        arms["LLM-Guard firewall"] = block_only(off, decisions(D / "llmguard_attack.jsonl"),
                                                decisions(D / "llmguard_benign.jsonl"))
    if (D / "controlnet_attack.jsonl").exists():
        arms["ControlNET"] = block_only(off, decisions(D / "controlnet_attack.jsonl"),
                                        decisions(D / "controlnet_benign.jsonl"))
    if (D / "benign_on_only_spotlight.jsonl").exists():
        arms["Spotlighting"] = outcomes(rows(D / "on_only_spotlight.jsonl"), rows(D / "benign_on_only_spotlight.jsonl"))
    arms["This work, 3 layers"] = outcomes(rows(MR / "followup200" / f"on_{COMBO}.jsonl"),
                                          rows(MR / "followup200" / f"benign_on_{COMBO}.jsonl"))
    arms["This work, full stack"] = outcomes(rows(MR / "ablation200" / "on_full.jsonl"),
                                            rows(MR / "ablation200" / "benign_on.jsonl"))
    rep = {"arms": {}, "vs_existing": {}}
    for name, a in arms.items():
        rep["arms"][name] = {m: {"k": sum(v.values()), "n": len(v), "ci": ci(sum(v.values()), len(v))}
                             for m, v in a.items()}
    ours = ["This work, 3 layers", "This work, full stack"]
    for o in ours:
        for e in [k for k in arms if k not in ours]:
            rep["vs_existing"][f"{o} vs {e}"] = {
                "attack_success (ours-only / theirs-only)": paired(arms[o]["asr"], arms[e]["asr"]),
                "benign_blocked (ours-only / theirs-only)": paired(arms[o]["blocked"], arms[e]["blocked"]),
                "benign_correct (ours-only / theirs-only)": paired(arms[o]["correct"], arms[e]["correct"])}
    (D / "comparison_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"{'Arm':<24}{'ASR':>22}{'Benign blocked':>22}{'Benign correct':>22}")
    for name, a in rep["arms"].items():
        print(f"{name:<24}{a['asr']['ci']:>22}{a['blocked']['ci']:>22}{a['correct']['ci']:>22}")
    for k, v in rep["vs_existing"].items():
        s = v["attack_success (ours-only / theirs-only)"]
        print(f"{k:<52} ASR ours-only/theirs-only {s['only_first']}/{s['only_second']} p={s['p']:.3g}")


if __name__ == "__main__":
    main()
