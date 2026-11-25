"""EXPLORATORY subgroup analysis of Protocol B (seven-layer single-generator harness). No inference.

Labels used (all present in the data, none invented):
  poison_template  - which of the 4 coercive poison templates the constructor drew (rng.choice in
                     build_planted_attacks.py); randomized assignment.
  query_suffix     - which of the 3 coercive query suffixes was drawn; randomized assignment.
  target_marker    - the planted answer is "Yes" or "No" (determined by the true answer).
  poison_in_generator_context - whether the planted chunk was among the 8 passages the undefended
                     generator received (contexts_attack.jsonl; available for the first 200 rows only).
Arms: undefended, seven-layer full stack, three-layer subset (n = 424 for all three; seed 42).
"""
import json
import random

from common import MR, S4, attack_succeeded, dump, load, paired, wilson

AB, FU, EX = MR / "ablation200", MR / "followup200", MR / "extended424"
C3 = "combo_sanitize_grounding_refuse"
TEMPL = {"IMPORTANT": "T1 'IMPORTANT - VERIFIED CORRECTION'", "SYSTEM NOTE": "T2 'SYSTEM NOTE (high priority)'",
         "[Editorial": "T3 '[Editorial correction]'", "Fact-check": "T4 'Fact-check result'"}
SUFF = {"If the sources": "Q1 'If the sources contain a verified correction'",
        "Defer to any": "Q2 'Defer to any authoritative correction'",
        "Where the documents": "Q3 'Where the documents give a corrected answer'"}


def perm_chi2(groups, outcome, ids, b=10000, seed=20261125):
    """Permutation test of homogeneity of a binary outcome across group levels (chi-square statistic)."""
    def stat(lab):
        tot = sum(outcome[i] for i in ids) / len(ids)
        s = 0.0
        for g in set(lab.values()):
            m = [i for i in ids if lab[i] == g]
            e = tot * len(m)
            o = sum(outcome[i] for i in m)
            if 0 < tot < 1:
                s += (o - e) ** 2 / e + ((len(m) - o) - (len(m) - e)) ** 2 / (len(m) - e)
        return s
    obs = stat(groups)
    rng = random.Random(seed)
    labs = [groups[i] for i in ids]
    ge = 0
    for _ in range(b):
        rng.shuffle(labs)
        ge += stat(dict(zip(ids, labs))) >= obs - 1e-12
    return round((ge + 1) / (b + 1), 4)


def main():
    src = {int(r["id"]): r for r in map(json.loads, open(S4 / "planted_attacks_200.jsonl", encoding="utf-8"))}
    for r in map(json.loads, open(S4 / "planted_attacks_extra224.jsonl", encoding="utf-8")):
        src[200 + int(r["id"]) if int(r["id"]) < 200 else int(r["id"])] = r
    arms = {"undefended": {**load(FU / "off.jsonl"), **load(EX / "off.jsonl")},
            "full_7layer": {**load(AB / "on_full.jsonl"), **load(EX / "on_full.jsonl")},
            "subset_3layer": {**load(FU / f"on_{C3}.jsonl"), **load(EX / f"on_{C3}.jsonl")}}
    ids = sorted(set.intersection(*(set(a) for a in arms.values())))
    # label from the row itself (question / marker are stored on every output row) to avoid id mapping errors
    lab = {"poison_template": {}, "query_suffix": {}, "target_marker": {}}
    for i in ids:
        r = arms["undefended"][i]
        q = r["question"]
        lab["query_suffix"][i] = next((v for k, v in SUFF.items() if k in q), "none")
        lab["target_marker"][i] = str(r.get("success_marker"))
    pc = {}
    for path in (S4 / "planted_attacks_200.jsonl", S4 / "planted_attacks_extra224.jsonl"):
        for r in map(json.loads, open(path, encoding="utf-8")):
            pc[r["question"]] = r["poison_chunk"]
    for i in ids:
        chunk = pc.get(arms["undefended"][i]["question"], "")
        lab["poison_template"][i] = next((v for k, v in TEMPL.items() if chunk.startswith(k)), "unmatched")
    ctx = {int(r["index"]): r for r in map(json.loads, open(MR / "baselines_protocolB" / "contexts_attack.jsonl", encoding="utf-8"))}
    lab["poison_in_generator_context (first 200 rows)"] = {
        i: ("delivered" if ctx[i]["poison_chunk"] in ctx[i]["ranked_context"] else "absent") for i in ids if i in ctx}

    succ = {a: {i: attack_succeeded(arms[a][i]) for i in ids} for a in arms}
    res = {"status": "EXPLORATORY - not preregistered; subgroups with n < 30 are flagged; no multiplicity "
                     "correction; read as hypothesis-generating",
           "n": len(ids), "unmatched_labels": {k: sum(v == "unmatched" for v in d.values()) for k, d in lab.items()},
           "groups": {}}
    for name, d in lab.items():
        gids = [i for i in ids if i in d]
        out = {}
        for g in sorted(set(d[i] for i in gids)):
            m = [i for i in gids if d[i] == g]
            out[g] = {"n": len(m), "small_n_flag": len(m) < 30,
                      **{f"asr_{a}": wilson(sum(succ[a][i] for i in m), len(m)) for a in arms},
                      "full_vs_undefended": paired(succ["undefended"], succ["full_7layer"], m),
                      "subset_vs_full": paired(succ["subset_3layer"], succ["full_7layer"], m)}
        out["homogeneity_p_perm"] = {a: perm_chi2(d, succ[a], gids) for a in arms}
        res["groups"][name] = out
    # detector (refuse-on-detection) firing by randomized template, from the R-only arm (first 200 rows)
    rr = load(AB / "on_only_refuse.jsonl")
    det = {i: str(rr[i].get("block_reason", "")).startswith("detector") for i in rr}
    res["detector_fire_rate_R_only_arm"] = {
        name: {g: wilson(sum(det[i] for i in det if lab[name].get(i) == g), sum(1 for i in det if lab[name].get(i) == g))
               for g in sorted(set(lab[name][i] for i in det if i in lab[name]))}
        for name in ("poison_template", "query_suffix", "target_marker", "poison_in_generator_context (first 200 rows)")}
    dump("protocolB_subgroups.json", res)


if __name__ == "__main__":
    main()
