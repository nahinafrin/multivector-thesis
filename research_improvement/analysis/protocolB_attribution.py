"""Protocol B layer attribution from SAVED per-row outputs (seven-layer single-generator harness,
llama3.2:3b, seed 42). No inference. Exploratory where marked.

Factorial: the three layers S (sanitize), G (grounding gate), R (refuse-on-detection) were run in all
2^3 combinations with the other four layers off:
  {}   followup200/off            S  ablation200/on_only_sanitize   G  ablation200/on_only_grounding
  R    ablation200/on_only_refuse SG followup200/on_combo_sanitize_grounding
  SR   followup200/on_combo_sanitize_refuse    GR followup200/on_combo_grounding_refuse
  SGR  followup200/on_combo_sanitize_grounding_refuse
Full stack and leave-one-out arms: ablation200/on_full, on_no<layer>; single layers on_only_<layer>.
Benign arms: same names with benign_ prefix (followup200/benign_off for {}; ablation200/benign_on for full).
"""
from itertools import combinations

from common import MR, answered_correctly, attack_succeeded, bootstrap, dump, load, paired, wilson

AB, FU, EX = MR / "ablation200", MR / "followup200", MR / "extended424"
CELLS = {
    "": (FU / "off.jsonl", FU / "benign_off.jsonl"),
    "S": (AB / "on_only_sanitize.jsonl", AB / "benign_on_only_sanitize.jsonl"),
    "G": (AB / "on_only_grounding.jsonl", AB / "benign_on_only_grounding.jsonl"),
    "R": (AB / "on_only_refuse.jsonl", AB / "benign_on_only_refuse.jsonl"),
    "SG": (FU / "on_combo_sanitize_grounding.jsonl", FU / "benign_on_combo_sanitize_grounding.jsonl"),
    "SR": (FU / "on_combo_sanitize_refuse.jsonl", FU / "benign_on_combo_sanitize_refuse.jsonl"),
    "GR": (FU / "on_combo_grounding_refuse.jsonl", FU / "benign_on_combo_grounding_refuse.jsonl"),
    "SGR": (FU / "on_combo_sanitize_grounding_refuse.jsonl", FU / "benign_on_combo_sanitize_grounding_refuse.jsonl"),
}
LAYERS = {"sanitize": "nosanitize", "grounding": "nogrounding", "refuse": "norefuse", "tighten": "notighten",
          "prompt": "noprompt", "presidio": "nopresidio", "dlp": "nodlp"}


def main():
    A = {k: load(a) for k, (a, _) in CELLS.items()}
    Bn = {k: load(b) for k, (_, b) in CELLS.items()}
    A["full"], Bn["full"] = load(AB / "on_full.jsonl"), load(AB / "benign_on.jsonl")
    for lay, f in LAYERS.items():
        A["-" + lay], Bn["-" + lay] = load(AB / f"on_{f}.jsonl"), load(AB / f"benign_on_{f}.jsonl")
    for lay in ("tighten", "prompt", "presidio", "dlp"):
        A["only_" + lay], Bn["only_" + lay] = load(AB / f"on_only_{lay}.jsonl"), load(AB / f"benign_on_only_{lay}.jsonl")
    ia = sorted(set.intersection(*(set(v) for v in A.values())))
    ib = sorted(set.intersection(*(set(v) for v in Bn.values())))
    succ = {k: {i: attack_succeeded(A[k][i]) for i in ia} for k in A}
    blk = {k: {i: bool(Bn[k][i].get("blocked")) for i in ib} for k in Bn}
    cor = {k: {i: answered_correctly(Bn[k][i]) for i in ib} for k in Bn}
    rate = lambda d, ids: 100 * sum(d[i] for i in ids) / len(ids)  # noqa: E731
    res = {"n_attack": len(ia), "n_benign": len(ib), "note": "seven-layer single-generator harness (Protocol B), "
           "seed 42; not the complete 13-stage pipeline", "cells": {}}

    # 1. cell table
    for k in A:
        res["cells"][k or "{}"] = {"asr": wilson(sum(succ[k].values()), len(ia)),
                                    "benign_blocked": wilson(sum(blk[k].values()), len(ib)),
                                    "benign_correct": wilson(sum(cor[k].values()), len(ib))}

    # 2. Shapley values over {S,G,R} (exact, other four layers off); value v(T) = ASR({}) - ASR(T), points
    players = ["S", "G", "R"]
    key = lambda T: "".join(p for p in players if p in T)  # noqa: E731

    def shapley(metric, ids, sign=1):
        v = lambda T: sign * (rate(metric[""], ids) - rate(metric[key(T)], ids))  # noqa: E731
        out = {}
        for p in players:
            rest = [q for q in players if q != p]
            w = {0: 1 / 3, 1: 1 / 6, 2: 1 / 3}
            out[p] = sum(w[len(T)] * (v(set(T) | {p}) - v(set(T))) for r in range(3) for T in combinations(rest, r))
        return out

    sh = shapley(succ, ia)
    res["shapley_asr_reduction_pts"] = {p: {"value": round(sh[p], 1),
                                            "ci95_boot": bootstrap(lambda s, p=p: shapley(succ, s)[p], ia)}
                                        for p in players}
    res["shapley_asr_reduction_pts"]["sum_equals_SGR_reduction"] = round(rate(succ[""], ia) - rate(succ["SGR"], ia), 1)
    shb = shapley(blk, ib, sign=-1)   # value = extra benign blocking (cost)
    res["shapley_benign_blocking_cost_pts"] = {p: {"value": round(shb[p], 1),
                                                   "ci95_boot": bootstrap(lambda s, p=p: shapley(blk, s, -1)[p], ib)}
                                               for p in players}
    shc = shapley(cor, ib)            # value = loss of benign correctness (points)
    res["shapley_benign_correct_loss_pts"] = {p: {"value": round(shc[p], 1),
                                                  "ci95_boot": bootstrap(lambda s, p=p: shapley(cor, s)[p], ib)}
                                              for p in players}

    # 3. Contribution of each of S, G, R in three contexts (paired, attack success)
    ctx = {}
    for p in players:
        others = [q for q in players if q != p]
        ctx[p] = {
            "marginal_alone (X vs {})": paired(succ[""], succ[p], ia),
            "within_SGR (SGR minus X vs SGR)": paired(succ[key(others)], succ["SGR"], ia),
        }
    lay_of = {"S": "sanitize", "G": "grounding", "R": "refuse"}
    for p in players:
        ctx[p]["within_full_stack (leave-one-out vs full)"] = paired(succ["-" + lay_of[p]], succ["full"], ia)
    for lay in ("tighten", "prompt", "presidio", "dlp"):
        ctx[lay] = {"marginal_alone (X vs {})": paired(succ[""], succ["only_" + lay], ia),
                    "within_full_stack (leave-one-out vs full)": paired(succ["-" + lay], succ["full"], ia)}
    res["contribution_by_context"] = ctx
    # paper claim: "every subset beats each of its own constituent layers in paired tests"
    res["subset_vs_constituent"] = {f"{T} vs {p}": paired(succ[p], succ[T], ia)
                                    for T in ("SG", "SR", "GR", "SGR") for p in T}
    res["contribution_by_context_note"] = ("only_x = rows where the attack succeeds in the first arm only; "
                                           "diff_pts = ASR(first) - ASR(second), i.e. the reduction from adding X.")

    # 4. Interactions. (a) additive scale on the reduction D(T) = ASR({}) - ASR(T)
    D = lambda T, ids: rate(succ[""], ids) - rate(succ[T], ids)  # noqa: E731
    inter = {
        "SxG": lambda s: D("SG", s) - D("S", s) - D("G", s),
        "SxR": lambda s: D("SR", s) - D("S", s) - D("R", s),
        "GxR": lambda s: D("GR", s) - D("G", s) - D("R", s),
        "SxGxR": lambda s: D("SGR", s) - D("SG", s) - D("SR", s) - D("GR", s) + D("S", s) + D("G", s) + D("R", s),
    }
    res["interaction_additive_pts"] = {k: {"value": round(f(ia), 1), "ci95_boot": bootstrap(f, ia)} for k, f in inter.items()}
    res["interaction_additive_note"] = ("positive = more reduction than the sum of the parts; near the ASR floor "
                                        "sub-additivity is expected mechanically, so see the independence model.")
    # (b) independent-filters model: surviving fraction multiplies, ASR(XY) = ASR(X) ASR(Y) / ASR({})
    def indep(T):
        def f(s):
            base = rate(succ[""], s)
            pred = base
            for p in T:
                pred *= rate(succ[p], s) / base
            return rate(succ[T], s) - pred
        return f
    res["vs_independent_filters_pts"] = {
        T: {"observed": round(rate(succ[T], ia), 1), "predicted_if_independent": round(rate(succ[T], ia) - indep(T)(ia), 1),
            "observed_minus_predicted": round(indep(T)(ia), 1), "ci95_boot": bootstrap(indep(T), ia)}
        for T in ("SG", "SR", "GR", "SGR")}
    res["vs_independent_filters_note"] = ("observed > predicted: the layers stop overlapping rows (redundancy); "
                                          "observed < predicted: they stop more together than independence implies (synergy).")

    # 5. Row-level overlap among rows the undefended arm loses
    base_succ = [i for i in ia if succ[""][i]]
    fixed = {p: {i for i in base_succ if not succ[p][i]} for p in players}
    res["row_overlap_on_undefended_successes"] = {
        "n_undefended_successes": len(base_succ),
        "fixed_by_single_layer": {p: len(fixed[p]) for p in players},
        "fixed_by_exactly_one_single_layer": sum(1 for i in base_succ if sum(i in fixed[p] for p in players) == 1),
        "fixed_by_two_or_more_single_layers": sum(1 for i in base_succ if sum(i in fixed[p] for p in players) >= 2),
        "fixed_by_all_three_single_layers": sum(1 for i in base_succ if all(i in fixed[p] for p in players)),
        "fixed_by_no_single_layer_but_by_SGR": sum(1 for i in base_succ if not any(i in fixed[p] for p in players)
                                                   and not succ["SGR"][i]),
        "still_succeed_under_SGR": sum(succ["SGR"][i] for i in base_succ),
        "still_succeed_under_full": sum(succ["full"][i] for i in base_succ),
        "pairwise_overlap": {f"{p}&{q}": len(fixed[p] & fixed[q]) for p, q in combinations(players, 2)},
        "broken_rows (succeed under arm, not undefended)": {k: sum(succ[k][i] and not succ[""][i] for i in ia)
                                                            for k in ("S", "G", "R", "SG", "SR", "GR", "SGR", "full")},
    }

    # 6. Mechanism: how each arm stops attacks (row outcome categories)
    def cat(r):
        if r.get("blocked"):
            br = str(r.get("block_reason"))
            return "refused: detector" if br.startswith("detector") else (
                "refused: grounding" if "grounded" in br else "refused: other")
        return "delivered: attack succeeded" if attack_succeeded(r) else "delivered: attack failed"
    mech = {}
    for k in ("", "S", "G", "R", "SG", "SR", "GR", "SGR", "full"):
        c = {}
        for i in ia:
            c[cat(A[k][i])] = c.get(cat(A[k][i]), 0) + 1
        mech[k or "{}"] = dict(sorted(c.items()))
    res["outcome_mechanism_attack"] = mech
    cross = {}
    for i in ia:
        kk = ("undefended " + ("success" if succ[""][i] else "fail"), cat(A["full"][i]))
        cross[" -> ".join(kk)] = cross.get(" -> ".join(kk), 0) + 1
    res["undefended_to_full_stack_crosstab"] = dict(sorted(cross.items()))
    bm = {}
    for k in ("", "S", "G", "R", "SG", "SR", "GR", "SGR", "full"):
        c = {}
        for i in ib:
            r = Bn[k][i]
            lab = ("refused: detector" if str(r.get("block_reason")).startswith("detector") else "refused: grounding"
                   if "grounded" in str(r.get("block_reason")) else "refused: other") if r.get("blocked") else (
                "answered correctly" if cor[k][i] else "answered, not correct")
            c[lab] = c.get(lab, 0) + 1
        bm[k or "{}"] = dict(sorted(c.items()))
    res["outcome_mechanism_benign"] = bm

    # 7. Subset vs full: precise bounds (n=200 seed 42, seeds 7/1234, n=424)
    eq = {"n200_seed42": paired(succ["SGR"], succ["full"], ia)}
    for sd in ("seed7", "seed1234"):
        a1, a2 = load(MR / sd / "on_combo_sanitize_grounding_refuse.jsonl"), load(MR / sd / "on_full.jsonl")
        ids = sorted(set(a1) & set(a2))
        eq[f"n200_{sd}"] = paired({i: attack_succeeded(a1[i]) for i in ids}, {i: attack_succeeded(a2[i]) for i in ids}, ids)
    # n = 424: the 200 seed-42 rows (indices 0-199) plus the 224 extended rows (indices 200-423),
    # exactly as score_improvements.extended() combines them
    a1 = {**load(FU / "on_combo_sanitize_grounding_refuse.jsonl"), **load(EX / "on_combo_sanitize_grounding_refuse.jsonl")}
    a2 = {**load(AB / "on_full.jsonl"), **load(EX / "on_full.jsonl")}
    ids = sorted(set(a1) & set(a2))
    eq["n424"] = paired({i: attack_succeeded(a1[i]) for i in ids}, {i: attack_succeeded(a2[i]) for i in ids}, ids)
    eq["n224_new_rows_only"] = paired({i: attack_succeeded(a1[i]) for i in ids if i >= 200},
                                      {i: attack_succeeded(a2[i]) for i in ids if i >= 200}, [i for i in ids if i >= 200])
    eq["note"] = ("diff_pts = ASR(subset) - ASR(full). ci90 corresponds to a two one-sided 5% test (TOST); "
                  "no equivalence margin was prespecified, so the bound is descriptive, not a confirmatory "
                  "equivalence result. Seeds 7/1234 reuse the same 200 rows (not independent samples).")
    res["subset_vs_full"] = eq

    # 8. Run-to-run noise floor: same configuration, two runs
    r1, r2 = load(MR / "rerun200" / "off.jsonl"), load(FU / "off.jsonl")
    ids = sorted(set(r1) & set(r2))
    f1, f2 = load(MR / "rerun200" / "on_full.jsonl"), load(AB / "on_full.jsonl")
    ids2 = sorted(set(f1) & set(f2))
    res["noise_floor_same_config_reruns"] = {
        "undefended_rerun_vs_reported": paired({i: attack_succeeded(r1[i]) for i in ids}, {i: attack_succeeded(r2[i]) for i in ids}, ids),
        "full_rerun_vs_reported": paired({i: attack_succeeded(f1[i]) for i in ids2}, {i: attack_succeeded(f2[i]) for i in ids2}, ids2),
    }
    dump("protocolB_attribution.json", res)


if __name__ == "__main__":
    main()
