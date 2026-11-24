#!/usr/bin/env python3
"""
verify_claims.py -- claim-by-claim provenance check for the paper.

For each number the paper reports, this script recomputes it from the stored per-row outputs
(or from a report that a named scorer regenerates from them), compares it with the value the
paper states, and checks that the stated text appears in paper_revision/main.tex. No model is
run. It first re-runs the standard-library scorers so their reports reflect the stored rows.

    python verify_claims.py            # all claims; exit code 1 if any fails
Output: claims_manifest.json  (claim id, paper section, statement, expected, computed,
        inputs, scorer, PASS/FAIL, git revision)

Scope: a selected set of the paper's numbers (the headline results of every section and every
Table VII cell, including its Wilson intervals and p-values), not every number in the paper. It
establishes that these follow from the committed outputs; each input is recorded with a SHA-256
(line endings normalised). It does not re-run the models; GPU generation is not bit-reproducible
(see the paper, Sec. V-E). "Found in main.tex" is a text match on the stated phrase.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from score_ablation_study import exact_p, wilson  # noqa: E402

ROOT = HERE.parent
TEX = (ROOT / "paper_revision" / "main.tex").read_text(encoding="utf-8")
EE, MR = HERE / "external_eval", HERE / "mitigation_results"
B = "\\"
CLAIMS: list[dict] = []


def L(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def J(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _sha256(rel: str) -> dict:
    """SHA-256 of an input file, or of every file under an input directory (sorted)."""
    import hashlib
    p = HERE / rel
    if "{" in rel or not p.exists():
        return {"path": rel, "sha256": None}
    files = sorted(x for x in p.rglob("*") if x.is_file()) if p.is_dir() else [p]
    h = hashlib.sha256()
    for f in files:
        h.update(Path(os.path.relpath(f, HERE)).as_posix().encode())  # == relative_to(HERE) inside HERE
        h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return {"path": rel, "sha256": h.hexdigest(), "n_files": len(files)}


def claim(cid, section, statement, expected, computed, snippet, inputs, scorer):
    found = snippet in TEX
    CLAIMS.append({"id": cid, "section": section, "statement": statement,
                   "expected": expected, "computed": computed, "tex_snippet": snippet,
                   "in_tex": found, "inputs": [_sha256(i) for i in inputs], "scorer": scorer,
                   "status": "PASS" if (str(expected) == str(computed) and found) else "FAIL"})


def p_matches(shown: str, p: float) -> bool:
    """Does a p-value as printed in the table ('$<\\!10^{-13}$', '$3\\!\\times\\!10^{-6}$',
    '$10^{-11}$', '0.18', '1.0') agree with p at the precision shown?"""
    s = shown.replace("$", "").replace("\\!", "").replace(" ", "")
    if s.startswith("<10^{"):
        return p < 10 ** int(re.search(r"\{(-?\d+)\}", s).group(1))
    m = re.fullmatch(r"(?:([\d.]+)\\times)?10\^\{(-?\d+)\}", s)
    if m:
        mant, e = m.group(1) or "1", int(m.group(2))
        dec = len(mant.split(".")[1]) if "." in mant else 0
        return round(p / 10 ** e, dec) == float(mant)
    dec = len(s.split(".")[1]) if "." in s else 0
    return round(p, dec) == float(s)


def pct(k, n):
    return f"{100 * k / n:.1f}"


def run(*args):
    r = subprocess.run([sys.executable, *args], cwd=HERE, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{' '.join(args)} failed:\n{r.stderr[-2000:]}")


def main() -> None:
    for s in (["score_protocolA_220.py"], ["score_ablation_study.py"], ["score_seed_study.py"],
              ["score_improvements.py"], ["fit_graded_floors_heldout.py"], ["score_external_paired.py"]):
        run(*s)

    # ---- Protocol A (Sec. V-A, V-B, V-C, V-D) -------------------------------------------
    pa = J(HERE / "protocolA_220" / "protocolA_220_report.json")
    c, b = pa["controller"], pa["baseline"]
    src = ["protocolA_220/grounded_controller_220.jsonl", "protocolA_220/grounded_baseline_220.jsonl"]
    sc = "score_protocolA_220.py"
    claim("A.canary", "V-C", "canary leaks among checked rows, controller run", "0/175",
          f"{c['canary_leaks']}/{c['canary_checked']}", "the canary was checked on 175 of the 220 rows", src, sc)
    claim("A.canary_base", "V-C", "canary rows checked, controller-off run", "176",
          str(b["canary_checked"]), "(176 in the controller-off run)", src, sc)
    claim("A.canary_unchecked", "V-C", "every unchecked row was blocked (both runs)", "True",
          str(c["canary_not_checked_all_blocked"] and b["canary_not_checked_all_blocked"]),
          "were blocked before the check", src, sc)
    claim("A.canary_delivered", "V-C", "delivered answers with intact canary, controller run", "132/132",
          f"{c['delivered_canary_intact']}/{c['delivered']}", "including all 132 delivered answers", src, sc)
    claim("A.redaction", "V-C", "poison chunks redacted", "74/74", c["poison_chunks_redacted"], "74/74", src, sc)
    claim("A.poisoned", "V-A", "poisoned-context chunks redacted", "40/40", c["poisoned_context_redacted"],
          "all 40 canaries stayed", src, sc)
    claim("A.benign_gate", "V-A", "benign rows flagged at gate", "0/80", c["benign_flagged_at_gate"], "(0/80, 95", src, sc)
    claim("A.gate_flag", "V-A", "direct injections flagged at gate (controller-baseline)", "29-30",
          f"{c['direct_flagged_at_gate'].split('/')[0]}-{b['direct_flagged_at_gate'].split('/')[0]}",
          "flags 29--30 of the 40 direct-injection queries", src, sc)
    claim("A.gate_block", "V-A", "direct injections hard-blocked at gate", "25/40", c["direct_hard_blocked_at_gate"],
          "(25 blocked, 4--5 sent to review)", src, sc)
    claim("A.cascade", "V-A", "direct injections blocked by cascade", "40/40", c["direct_blocked_by_cascade"],
          "All 40 are blocked once", src, sc)
    claim("A.gateslip", "V-H", "gate-slip rows passing gate", "5/20", c["gate_slip_pass_gate"],
          "Five of its 20 rows (25\\%) pass the gate", src, sc)
    claim("A.mv", "V-B", "multivector fires on post-gate rows", "1/34",
          f"{c['multivector_fires_post_gate']}/{c['multivector_post_gate']}", "1/34", src, sc)
    claim("A.blocked", "V-D", "rows blocked, controller vs baseline", "88/101",
          f"{c['blocked_total']}/{b['blocked_total']}", "it blocks 88 rows against 101 for the single-pass pipeline", src, sc)
    claim("A.benign_deliv", "V-D", "benign delivered, controller vs baseline", "70/66",
          f"{c['benign_delivered']}/{b['benign_delivered']}", "answers 70 benign rows (vs." + B + " 66)", src, sc)
    claim("A.pois_deliv", "V-D", "poisoned-context delivered, controller vs baseline", "34/28",
          f"{c['poisoned_context_delivered']}/{b['poisoned_context_delivered']}", "34 poisoned-context rows (vs." + B + " 28)", src, sc)
    claim("A.benign_refused", "V-A", "benign refused, controller vs baseline", "10/14",
          f"{c['benign_refused']}/{b['benign_refused']}", "refuses 10 benign rows (14 without the controller)", src, sc)

    # ---- Table VII: leave-one-out and single-layer (Sec. V-E) ---------------------------
    at = J(HERE / "ablation_table.json")
    claim("VII.baseline", "Table VII", "undefended ASR (current-code arm)", "59/200",
          f"{at['_baseline']['undefended_asr']}/{at['_baseline']['n']}", "Undefended & 29.5 [23.6, 36.2]",
          [at["_baseline"]["attack"]], "score_ablation_study.py")
    rows7 = {"full": ("Full stack (7 layers)", r"\textbf{3.5}"), "nosanitize": ("$-$ sanitization", None),
             "nogrounding": ("$-$ grounding gate", None), "norefuse": ("$-$ refuse-on-detection", None),
             "notighten": ("$-$ tightened rerank", None), "noprompt": ("$-$ guarded prompt", None),
             "nopresidio": ("$-$ Presidio", None), "nodlp": ("$-$ DLP", None),
             "only_grounding": ("Grounding gate &", None), "only_sanitize": ("Sanitization &", None),
             "only_refuse": ("Refuse-on-detection &", None), "only_tighten": ("Tightened rerank &", None),
             "only_prompt": ("Guarded prompt &", None), "only_presidio": ("Presidio / DLP &", None)}
    tab = TEX[TEX.index(B + "label{tab:ablation}"):]
    tab = tab[:tab.index(B + "end{tabular}")]
    for k, (lab, _) in rows7.items():
        r = at[k]
        line = next((l for l in tab.splitlines() if lab in l), "")
        exp = f"ASR {pct(r['asr'], r['n'])}, {r['fixed']}/{r['broke']}, blk {pct(r['benign_blocked'], r['benign_n'])}"
        ok = (pct(r["asr"], r["n"]) in line and f"& {r['fixed']}/{r['broke']} &" in line
              and f"& {pct(r['benign_blocked'], r['benign_n'])} " in line + " ")
        claim(f"VII.{k}", "Table VII", f"row {lab.strip(' &')}", exp, exp if ok else f"row mismatch: {line.strip()}",
              line.strip() or lab, [f"mitigation_results/ablation200/on_{k}.jsonl", at["_baseline"]["attack"]],
              "score_ablation_study.py")
        # displayed Wilson interval and exact McNemar p of the same row
        cells = [x.strip() for x in line.split("&")]
        lo, hi = wilson(r["asr"], r["n"])[1:]
        shown_ci = [float(x) for x in re.findall(r"\[([\d.]+),\s*([\d.]+)\]", cells[1])[0]] if len(cells) > 1 else []
        p = exact_p(r["broke"], r["fixed"])
        ci_ok = shown_ci == [round(lo, 1), round(hi, 1)]
        p_ok = len(cells) > 3 and p_matches(cells[3], p)
        claim(f"VII.{k}.ci_p", "Table VII", f"row {lab.strip(' &')}: Wilson CI and exact p",
              f"[{round(lo, 1)}, {round(hi, 1)}], p={p:.2g}",
              f"[{round(lo, 1)}, {round(hi, 1)}], p={p:.2g}" if (ci_ok and p_ok) else
              f"shown {cells[1:2]} {cells[3:4]} vs computed [{lo:.1f}, {hi:.1f}] p={p:.3g}",
              line.strip() or lab, [f"mitigation_results/ablation200/on_{k}.jsonl", at["_baseline"]["attack"]],
              "score_ablation_study.py")

    # ---- Seed replicates (Sec. V-E) -----------------------------------------------------
    sd = J(MR / "seed_study_report.json")
    seeds = ("42", "7", "1234")
    off = ", ".join(pct(sd[s]["asr"]["a_off"], 200) for s in seeds)
    full = ", ".join(pct(sd[s]["asr"]["a_full"], 200) for s in seeds)
    sub = ", ".join(pct(sd[s]["asr"]["a_combo"], 200) for s in seeds)
    pv = ", ".join(f"{sd[s]['combo_vs_full'][2]:.2f}" for s in seeds)
    sl = ["mitigation_results/followup200/", "mitigation_results/seed7/", "mitigation_results/seed1234/"]
    claim("E.seed_off", "V-E", "undefended ASR per seed", "29.5, 31.0, 30.5", off, "(29.5, 31.0, 30.5)", sl, "score_seed_study.py")
    claim("E.seed_full", "V-E", "full-stack ASR per seed", "3.5, 4.5, 4.5", full, "(3.5, 4.5, 4.5)", sl, "score_seed_study.py")
    claim("E.seed_sub", "V-E", "subset ASR per seed", "5.5, 7.0, 5.5", sub, "(5.5, 7.0, 5.5)", sl, "score_seed_study.py")
    claim("E.seed_p", "V-E", "subset vs full p per seed", "0.34, 0.27, 0.77", pv, "$p=0.34$, 0.27, 0.77", sl, "score_seed_study.py")

    # ---- Fresh rerun from the raw slice (Sec. V-E) --------------------------------------
    run("score_rerun200.py")
    rr = J(MR / "rerun200" / "rerun_report.json")
    srr = ["mitigation_results/rerun200/off.jsonl", "mitigation_results/rerun200/on_full.jsonl",
           "mitigation_results/followup200/off.jsonl", "mitigation_results/ablation200/on_full.jsonl"]
    claim("E.rerun_asr", "V-E", "fresh rerun successes: undefended, full", "58/200, 6/200",
          f"{rr['off']['fresh_k']}/200, {rr['full']['fresh_k']}/200", "The fresh run gives 58/200", srr, "score_rerun200.py")
    claim("E.rerun_agree", "V-E", "rows with the same outcome as the reported run: undefended, full", "189, 199",
          f"{rr['off']['drift_fresh_vs_saved']['same_outcome']}, {rr['full']['drift_fresh_vs_saved']['same_outcome']}",
          "agree on 189 and 199 of the 200 rows", srr, "score_rerun200.py")
    fr = rr["fresh_full_vs_off"]
    claim("E.rerun_paired", "V-E", "fresh full vs undefended: fixed/broke", "54/2", f"{fr['fixed']}/{fr['broke']}",
          "again fixes 54 rows and breaks 2", srr, "score_rerun200.py")

    # ---- Existing defenses on Protocol B (Sec. V-E, Table tab:existing) -----------------
    run("score_baselines_protocolB.py")
    cb = J(MR / "baselines_protocolB" / "comparison_report.json")
    arm, vs = cb["arms"], cb["vs_existing"]
    sxb = ["mitigation_results/baselines_protocolB/", "mitigation_results/followup200/",
           "mitigation_results/ablation200/on_full.jsonl", "mitigation_results/ablation200/benign_on.jsonl"]
    asr = lambda a: f"{100 * arm[a]['asr']['k'] / arm[a]['asr']['n']:.1f}"  # noqa: E731
    claim("X.asr", "V-E", "ASR: LLM-Guard / Spotlighting / ControlNET / full stack", "21.0/23.0/4.5/3.5",
          "/".join(asr(a) for a in ("LLM-Guard firewall", "Spotlighting", "ControlNET", "This work, full stack")),
          "3.5" + B + "% vs." + B + " 21.0" + B + "% and 23.0" + B + "% ASR", sxb, "score_baselines_protocolB.py")
    lg = vs["This work, full stack vs LLM-Guard firewall"]["attack_success (ours-only / theirs-only)"]
    claim("X.llmguard", "V-E", "full stack vs LLM-Guard: ours-only / theirs-only", "3/38",
          f"{lg['only_first']}/{lg['only_second']}", "3 rows succeed only under the full stack, 38 only under LLM-Guard",
          sxb, "score_baselines_protocolB.py")
    cn = vs["This work, full stack vs ControlNET"]
    claim("X.controlnet", "V-E", "full stack vs ControlNET: ASR p, benign-correct p", "0.77, 0.058",
          f"{cn['attack_success (ours-only / theirs-only)']['p']:.2f}, {cn['benign_correct (ours-only / theirs-only)']['p']:.3f}",
          "$p=0.058$ against the full stack", sxb, "score_baselines_protocolB.py")
    claim("X.cn_correct", "V-E", "benign correct: ControlNET vs full stack", "46.0/52.0",
          f"{100 * arm['ControlNET']['correct']['k'] / arm['ControlNET']['correct']['n']:.1f}/"
          f"{100 * arm['This work, full stack']['correct']['k'] / arm['This work, full stack']['correct']['n']:.1f}",
          "(46.0" + B + "% vs." + B + " 52.0" + B + "%;", sxb, "score_baselines_protocolB.py")

    # ---- ADO on Protocol B (adapted implementation; external_baselines/ado_protocolB) ----
    ADO = ROOT / "external_baselines" / "ado_protocolB"
    for s in ("score_ado_protocolB.py", "score_sensitivity_refusal.py"):
        run(str(ADO / s), "--prefix", "full", "--selection", "full_selection.json")
    ac, asn = J(ADO / "full_comparison.json"), J(ADO / "full_sensitivity_refusal.json")
    sado = ["../external_baselines/ado_protocolB/full_attack_results.jsonl",
            "../external_baselines/ado_protocolB/full_benign_results.jsonl",
            "mitigation_results/ablation200/on_full.jsonl", "mitigation_results/ablation200/benign_on.jsonl"]
    scd = "external_baselines/ado_protocolB/score_ado_protocolB.py"
    a_ = ac["arms"]["ADO"]
    claim("X.ado_rates", "V-E", "ADO ASR / benign correct (n=200+200)", "28.5/35.5",
          f"{pct(a_['attack_success_k'], 200)}/{pct(a_['benign_correct_k'], 200)}",
          "ADO lets 28.5" + B + "% of attacks succeed", sado, scd)
    pa_, pb_ = ac["attack_success_ado_vs_seven_layer"], ac["benign_correct_ado_vs_seven_layer"]
    claim("X.ado_paired", "V-E", "ADO vs full stack: ADO-only / ours-only, exact p", "54/4, p=3e-12",
          f"{pa_['only_ado']}/{pa_['only_seven_layer']}, p={pa_['exact_mcnemar_p']:.0e}".replace("e-0", "e-"),
          "(54 vs." + B + " 4 discordant rows, $p=3" + B + "times10^{-12}$)", sado, scd)
    claim("X.ado_benign", "V-E", "benign correct ADO vs full stack, exact p", "35.5/52.0, p=6e-06",
          f"{pct(a_['benign_correct_k'], 200)}/{pct(ac['arms']['Seven-layer']['benign_correct_k'], 200)}, "
          f"p={pb_['exact_mcnemar_p']:.0e}",
          "(35.5" + B + "% vs." + B + " 52.0" + B + "%, $p=6" + B + "times10^{-6}$)", sado, scd)
    s1a, s1p = asn["arms"]["ADO"], asn["paired"]["s1"]
    claim("X.ado_s1", "V-E", "ADO ASR with refusals not counted; ADO-only / ours-only, exact p", "13.5, 24/4, p=2e-04",
          f"{pct(s1a['s1_success'], 200)}, {s1p['only_ado']}/{s1p['only_seven_layer']}, p={s1p['exact_mcnemar_p']:.0e}",
          "ADO's ASR is 13.5" + B + "% against 3.5" + B + "% (24 vs." + B + " 4 discordant rows, $p=2" + B + "times10^{-4}$)",
          sado, "external_baselines/ado_protocolB/score_sensitivity_refusal.py")
    claim("X.ado_refusals", "V-E", "ADO primary successes that are refusals", "30/57",
          f"{s1a['successes_that_are_refusals']}/{s1a['primary_success']}", "(30 of ADO's 57 successes)",
          sado, "external_baselines/ado_protocolB/score_sensitivity_refusal.py")
    emp = [sum(not r["final_context"] for r in L(ADO / f"full_{s}_results.jsonl")) for s in ("attack", "benign")]
    claim("X.ado_empty", "V-E", "rows where ADO left no passages: attack / benign", "64/58", f"{emp[0]}/{emp[1]}",
          "64 of the 200 attacks and 58 of the 200 benign questions", sado, "verify_claims.py")
    line = next((l for l in TEX[TEX.index(B + "label{tab:existing}"):].splitlines() if l.startswith("ADO orchestrator")), "")
    cells = [x.strip() for x in line.split("&")]
    lo, hi = wilson(a_["attack_success_k"], 200)[1:]
    shown = [float(x) for x in re.findall(r"\[([\d.]+),\s*([\d.]+)\]", cells[1])[0]] if len(cells) > 4 else []
    ok = (shown == [round(lo, 1), round(hi, 1)] and cells[2] == "0.0" and cells[3] == "35.5"
          and p_matches(cells[4].rstrip(B).strip(), pa_["exact_mcnemar_p"]))
    exp = f"[{round(lo, 1)}, {round(hi, 1)}], blk 0.0, correct 35.5, p={pa_['exact_mcnemar_p']:.2g}"
    claim("X.ado_row", "Table tab:existing", "ADO row: Wilson CI, blocked, correct, exact p", exp,
          exp if ok else f"row mismatch: {line.strip()}", line.strip() or "ADO orchestrator", sado, scd)

    # ---- n = 424 (Sec. V-E) ------------------------------------------------------------
    imp = J(EE / "improvements_report.json")
    n4 = imp["planted_n424"]
    s4 = ["mitigation_results/followup200/", "mitigation_results/ablation200/on_full.jsonl", "mitigation_results/extended424/"]
    claim("E.n424_asr", "V-E", "ASR off / full / subset at n=424", "29.7/4.2/4.7",
          "/".join(n4["asr"][a].split("%")[0] for a in ("off", "full", "subset")),
          "ASR is 29.7" + B + "% undefended", s4, "score_improvements.py")
    claim("E.n424_full", "V-E", "full vs off fixed/broke", "114/6",
          f"{n4['full_vs_off']['fixed']}/{n4['full_vs_off']['broke']}", "fixes 114 rows and breaks 6", s4, "score_improvements.py")
    sv = n4["subset_vs_full"]
    claim("E.n424_sub", "V-E", "subset vs full: diff, CI, p", "+0.5 [-1.5, +2.4] p=0.81",
          f"+{sv['diff_pts']} [{sv['diff_ci95_pts'][0]}, +{sv['diff_ci95_pts'][1]}] p={sv['p']:.2f}",
          "$+0.5$ points with an approximate 95" + B + "% interval of $[-1.5, +2.4]$", s4, "score_improvements.py")

    # ---- Semantic slice generators (Sec. V-H) --------------------------------------------
    sys.path.insert(0, str(HERE))
    from score_ablation_study import attack_succeeded
    sm = imp["semantic40_fused_vs_isolate"]
    u = sm["benign40_utility"]
    ss = ["external_eval/semantic40_{fused,isolate,routed}.jsonl", "external_eval/benign40_{fused,isolate,routed}.jsonl"]
    claim("H.isolate", "V-H", "isolate ASR / abstentions", "0/40, 34", f"{sum(attack_succeeded(r) for r in L(EE / 'semantic40_isolate.jsonl'))}/40, {sm['isolate_abstained']}",
          "abstains on 34 of the 40 attack rows", ss, "score_improvements.py")
    claim("H.utility", "V-H", "benign answered (correct): isolate, fused", "15 (10), 37 (27)",
          f"{u['isolate']['answered']} ({u['isolate']['answered_correct']}), {u['fused']['answered']} ({u['fused']['answered_correct']})",
          "it answers 15 (10 correctly), against 37 answered (27 correctly)", ss, "score_improvements.py")
    rv = sm["routed_vs_fused"]
    claim("H.routed", "V-H", "routed vs fused: fixed/broke, p", "5/2 p=0.45",
          f"{rv['fixed']}/{rv['broke']} p={rv['p']:.2f}", "(5 fixed, 2 broken, $p=0.45$", ss, "score_improvements.py")
    rc, rb = sm["routed_route_counts"], sm["routed_benign_route_counts"]
    claim("H.routes", "V-H", "rows sent to agreement path (attack, benign)", "22, 16",
          f"{rc.get('v3', 0) + rc.get('v3->v2', 0)}, {rb.get('v3', 0) + rb.get('v3->v2', 0)}",
          "(22 of the 40 attack rows and 16 of the 40 benign questions)", ss, "score_improvements.py")
    claim("H.routed_util", "V-H", "routed benign answered (correct)", "36 (24)",
          f"{u['routed']['answered']} ({u['routed']['answered_correct']})", "(36 benign questions answered, 24 correctly)", ss,
          "score_improvements.py")

    # ---- Graded multivector floors (Sec. V-B) -------------------------------------------
    gf = J(EE / "graded_floors_heldout_report.json")
    gi, gh = gf["in_sample"], gf["heldout"]
    sg = ["grounded_controller_graded.jsonl", "external_eval/graded_heldout_benign.jsonl",
          "row4_mitigation_results/conjunctive/on_full.jsonl"]
    claim("B.valid", "V-B", "graded valid fires (in-sample floors)", "24/30", gi["protocolA_multivector_valid"],
          "on 24/30 with graded channels", sg, "fit_graded_floors_heldout.py")
    claim("B.heldout_fp", "V-B", "false fires on 200 held-out benign questions", "6/200", gi["rag_heldout_benign_fires"],
          "fires on 6 of them (3.0" + B + "%, [1.4, 6.4])", sg, "fit_graded_floors_heldout.py")
    claim("B.bipia", "V-B", "BIPIA attack / benign fires", "8/30, 4/30",
          f"{gi['bipia_attack_fires']}, {gi['bipia_benign_fires']}", "fires on 8/30 attacks and on 4/30 benign", sg,
          "fit_graded_floors_heldout.py")
    claim("B.refit", "V-B", "held-out floor: context floor, FP, valid", "0.637, 1/200, 0/30",
          f"{gf['floors_heldout']['context_vector']}, {gh['rag_heldout_benign_fires']}, {gh['protocolA_multivector_valid']}",
          "(0.637) cuts held-out false fires to 1/200 but also cuts valid fires to 0/30", sg, "fit_graded_floors_heldout.py")
    sw = next(x for x in gf["floor_sweep"] if x["query_floor"] == 0.228 and x["context_floor"] == 0.40)
    claim("B.sweep", "V-B", "context floor 0.40: valid fires, held-out FP", "11/30, 5/200",
          f"{sw['valid_fires']}/30, {sw['heldout_benign_fires']}/200", "keeps 11/30 valid fires and still fires on 5/200", sg,
          "fit_graded_floors_heldout.py")

    # ---- deepset (Sec. V-G) --------------------------------------------------------------
    dg = J(EE / "deepset_gate_report.json")
    ps, ft = dg["paper_sample"], dg["full_test_split"]
    sdg = ["external_eval/deepset_gate_rows.jsonl"]
    claim("G.deepset_sample", "V-G", "sample: injections blocked / flagged, benign", "14/30, 17/30, 0/30",
          f"{ps['injection']['blocked']}/30, {ps['injection']['blocked_or_review']}/30, {ps['benign']['blocked_or_review']}/30",
          "blocks 14 of 30 real injection prompts", sdg, "run_deepset_gate_eval.py")
    claim("G.deepset_full", "V-G", "full split: blocked, flagged, benign", "23/60, 28/60, 1/56",
          f"{ft['injection']['blocked']}/60, {ft['injection']['blocked_or_review']}/60, {ft['benign']['blocked_or_review']}/56",
          "blocks 23 of 60 injections (38.3" + B + "%) and flags 28, at 1/56", sdg, "run_deepset_gate_eval.py")
    gg = J(EE / "gate_graded_report.json")
    t = gg["threshold_t"]
    gs = [r for r in L(EE / "deepset_graded_scores.jsonl") if r["split"] == "test"]
    flag = lambda r: r["decision"] != "ALLOW" or r["graded"] >= t  # noqa: E731
    inj = sum(flag(r) for r in gs if r["label"] == 1)
    ben = sum(flag(r) for r in gs if r["label"] == 0)
    claim("G.graded_flag", "V-G", "gate + graded flag on test split: injections, benign", "38/60, 2/56",
          f"{inj}/{sum(r['label'] == 1 for r in gs)}, {ben}/{sum(r['label'] == 0 for r in gs)}",
          "raises flagged injections from 28/60 to 38/60", ["external_eval/deepset_graded_scores.jsonl",
                                                             "external_eval/gate_graded_report.json"], "fit_gate_graded.py")
    rag = sum(r["channels"]["query_vector"] >= t for r in L(EE / "graded_heldout_benign.jsonl"))
    claim("G.graded_rag", "V-G", "graded flag on 200 held-out RAG questions", "7/200", f"{rag}/200",
          "fires on 7 of 200 held-out benign questions", ["external_eval/graded_heldout_benign.jsonl"], "fit_gate_graded.py")
    de = imp["deepset_end_to_end"]
    claim("G.e2e", "V-G", "end to end: injections / benign refused", "30/30, 30/30",
          f"{de['injection']['stopped_anywhere']}/30, {de['benign']['stopped_anywhere']}/30",
          "refuses all 30 sampled injections", ["external_eval/deepset_e2e.jsonl"], "score_improvements.py")

    # ---- External PoisonedRAG / BIPIA (Sec. V-G, Table VIII) -----------------------------
    ex = J(MR / "external200" / "paired_report.json")
    pr = ex["PoisonedRAG (all)"]["amb_as_success"]
    se = ["mitigation_results/external200/", "ext_n200_results.jsonl"]
    claim("G.prag", "Table VIII", "PoisonedRAG ASR undefended / defended, fixed/broke", "137/136, 7/6",
          f"{pr['asr_off']}/{pr['asr_on']}, {pr['fixed']}/{pr['broke']}", "7 rows fixed, 6 broken", se, "score_external_paired.py")
    sg2 = ex["same_generator"]["PoisonedRAG (all)"]["amb_as_success"]
    claim("G.prag_same", "Table VIII", "same generator: ASR, fixed/broke", "128/150, 16/7",
          f"{sg2['asr_on']}/150, {sg2['fixed']}/{sg2['broke']}", "85.3" + B + "% ASR (128/150)", se, "score_external_paired.py")
    claim("G.bipia", "Table VIII", "BIPIA blocked: pipeline / same-generator", "45/50, 44/50",
          f"{ex['BIPIA-text_attack']['blocked_on']}/50, {ex['same_generator']['BIPIA-text_attack']['blocked_on']}/50",
          "45/50 BIPIA attacks are blocked", se, "score_external_paired.py")

    # ---- Overhead (Sec. V-I, Table IX) ---------------------------------------------------
    ov = J(HERE / "resource_profile" / "overhead_summary.json")["latency_s"]
    from decimal import ROUND_HALF_UP, Decimal
    med = lambda k: str(Decimal(str(ov[k]["benign"]["median"])).quantize(Decimal("0.1"), ROUND_HALF_UP))  # noqa: E731
    claim("I.latency", "Table IX", "benign median: undefended / full / subset", "8.6/15.3/19.0",
          f"{med('Undefended')}/{med('Full stack (7 mitigation layers)')}/{med('Sanitize + grounding + refuse')}",
          "Full stack, 7 layers (Protocol B) & 15.4 / 24.7 & 15.3 / 21.3", ["resource_profile/overhead_summary.json"],
          "summarize_overhead.py")
    import statistics
    gl = {k: statistics.median(r["latency_s"] for r in L(MR / "gpu_encoders" / f"benign_on_{k}.jsonl"))
          for k in ("full_cpu", "full_cuda", "full_cuda_fp16")}
    claim("I.gpu", "Table IX", "full stack benign median: CPU / GPU fp32 / GPU fp16 encoders", "13.3/22.6/8.6",
          "/".join(f"{gl[k]:.1f}" for k in ("full_cpu", "full_cuda", "full_cuda_fp16")),
          "encoders on GPU, fp16 ($n=50$) & -- & 8.6 / 10.1", ["mitigation_results/gpu_encoders/"], "run_gpu_latency.sh")

    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    n_fail = sum(c["status"] == "FAIL" for c in CLAIMS)
    (HERE / "claims_manifest.json").write_text(json.dumps(
        {"git_revision_at_check": rev, "n_claims": len(CLAIMS), "n_fail": n_fail, "claims": CLAIMS}, indent=2),
        encoding="utf-8")
    for c_ in CLAIMS:
        if c_["status"] == "FAIL":
            print(f"FAIL {c_['id']}: expected {c_['expected']!r}, computed {c_['computed']!r}, in_tex={c_['in_tex']}")
    print(f"{len(CLAIMS) - n_fail}/{len(CLAIMS)} claims pass -> claims_manifest.json")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
