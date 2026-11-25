r"""PROPOSED manuscript edits (not applied). Each edit is (file, id, rationale, old, new); `old` must occur exactly
once. `trial_apply.py` applies them to SCRATCH COPIES only, compiles, and writes diffs. After approval the same
list can be applied to paper_revision/ with `trial_apply.py --apply` (not run).
"""

E = []


def edit(f, i, why, old, new):
    E.append({"file": f, "id": i, "why": why, "old": old, "new": new})


# ============================== ICCIT (6 pages, A4) ==============================
edit("main_iccit.tex", "I1-abstract-subset", "Replace the equivalence-sounding claim with a bound; add class dependence (02 §2, 05 Q4)",
     r"Ablation shows that three layers (per-chunk sanitization, grounding, and refusal on detection) are statistically indistinguishable from the seven-layer stack, with the gap bounded at 2.4 points at $n=424$.",
     r"On these instruction-bearing attacks, three layers (per-chunk sanitization, grounding, and refusal on detection) act as near-independent filters and come within 2.4 points of the seven-layer stack (95\% upper bound, $n=424$); on assertion-only attacks they do not suffice.")

edit("main_iccit.tex", "I2-contributions", "One consistent C1-C4; protocol framed as an application of standard designs (02 §6)",
     r"Our contributions are: (C1) an attribution protocol that runs every configuration on identical rows, scores each row by a machine-checkable outcome, and tests each stage with leave-one-out, add-one-in, and subset ablations under exact paired tests; (C2) a comparison with existing defenses on the same rows; (C3) a negative result, with a closed-form argument, for co-activation detection on saturated classifier scores, and a graded-margin repair; and (C4) a measured scope boundary: the defense does not extend to signature-free knowledge corruption. We do not claim a new detector; the components are known, and the contribution is measuring which of them carry the security.",
     r"Our contributions are: (C1) a fully local thirteen-stage pipeline built from known components, whose input risk score is propagated to later stages; (C2) a paired attribution protocol (leave-one-out, single-layer, and a fully crossed three-layer factorial under exact paired tests, with an independence null and a measured run-to-run noise floor), an application of standard designs~\cite{fawcett2016ablation,hutter2014fanova} released with frozen per-row outputs and a claim checker; (C3) findings that end-to-end numbers hide: three layers act as near-independent filters and suffice for instruction-bearing but not for assertion-only attacks, and protection depends on attacker wording; and (C4) a closed-form argument that co-activation detection fails on saturated scores, and a stage-by-stage account of why this defense class does not reduce PoisonedRAG knowledge corruption. We do not claim a new detector.")

edit("main_iccit.tex", "I3-related", "Position against prior art the paper omits (02 §3)",
     r"These works report point estimates without confidence intervals or paired tests, and none reports a component of its own system that fails to contribute (Table~\ref{tab:positioning}).",
     r"These works report point estimates without paired tests, and none reports a component of its own system that fails to contribute (Table~\ref{tab:positioning}). Defenses designed for knowledge corruption, such as isolate-then-aggregate generation~\cite{xiang2024robustrag} and trust-weighted multi-stage filtering~\cite{trishield2026}, target a threat our pipeline does not address.")

edit("main_iccit.tex", "I4-setup-external", "The external sets ran an adaptive-tier variant, not the complete pipeline (01 F2)",
     r"not the complete thirteen-stage pipeline with the ensemble and controller, which Protocol~A and the external sets evaluate.",
     r"not the complete thirteen-stage pipeline with the ensemble and controller, which Protocol~A evaluates. The external sets ran an adaptive-tier variant (Stages 1--10 and 13), in which input risk sets retrieval depth and generator: top-8 and a single \texttt{llama3.2:3b} at low risk, which applied to 139 of 150 PoisonedRAG rows.")

edit("main_iccit.tex", "I5-setup-k", "Scope the retrieval setting to Protocol A",
     r"retrieval uses $k=5$ and rerank top-3.",
     r"Protocol~A retrieval uses $k=5$ and rerank top-3.")

edit("main_iccit.tex", "I6-ablation-independence", "Report the factorial/independence finding (02 §2)",
     r"The layers are complementary, and the three-layer subset sanitize+ground+refuse is statistically indistinguishable from the seven-layer stack (5.5\% vs.\ 3.5\%; 7 vs.\ 3 discordant rows, $p=0.34$; benign blocking 9.5\% vs.\ 8.5\%).",
     r"The three key layers act as near-independent filters: each removes 39--49\% of the attacks that reach it, and every combination matches the product of its parts within 1.3 points (all 95\% bootstrap intervals include 0). The subset sanitize+ground+refuse leaves 5.5\% vs.\ 3.5\% (7 vs.\ 3 discordant rows, $p=0.34$; benign blocking 9.5\% vs.\ 8.5\%).")

edit("main_iccit.tex", "I7-ablation-scope", "Bound instead of 'carry all'; class dependence and wording (02 §2, 03 §2)",
     r"Three layers therefore carry essentially all the measured injection defense (Fig.~\ref{fig:asr}), although not at lower latency (Fig.~\ref{fig:tradeoff}, Section~\ref{sec:overhead}).",
     r"On these instruction-bearing attacks, three layers therefore give up at most about 2.4 points against seven (Fig.~\ref{fig:asr}), although not at lower latency (Fig.~\ref{fig:tradeoff}, Section~\ref{sec:overhead}). This does not transfer across attack classes: on the 40-row assertion-only slice (Section~\ref{sec:external}) the subset leaves 62.5\% against 35.0\% for seven layers (12 vs.\ 1 discordant rows, $p=0.003$), because the guarded prompt, harmful alone on Protocol~B, carries that defense (alone it lowers ASR from 75.0\% to 52.5\%, $p=0.035$). Protection also depends on wording (exploratory): for one of three randomized query phrasings the detector fires on every row and full-stack ASR is 0/147; for another it fires on 20\% and ASR is 9.9\% ($n=424$, homogeneity $p<0.001$).")

edit("main_iccit.tex", "I8-poisonedrag-trace", "Mechanism-level PoisonedRAG account from saved traces (04)",
     r"PoisonedRAG passages are fluent factual text without an instruction signature, Stage~6 redacts none of the 148 injected passages, and ASR is 90.7\% defended vs.\ 91.3\% undefended (7 fixed, 6 broken, $p=1.0$).",
     r"PoisonedRAG passages carry no instruction signature (gate score 0), Stage~6 redacts none of the 148, the reranker ranks the poisoned passage first in all 148, and every successful attack passes the grounding judge (median faithfulness 1.0); the true answer appears in another retrieved passage in only 11 rows, as the questions come from other corpora. ASR is 90.7\% defended vs.\ 91.3\% undefended (7 fixed, 6 broken, $p=1.0$).")

edit("main_iccit.tex", "I9-external-caption", "Accurate label for the external defended arm (01 F2)",
     r"\caption{Complete pipeline on independent external sets,",
     r"\caption{Adaptive-tier pipeline variant (Stages 1--10, 13) on independent external sets,")

edit("main_iccit.tex", "I10-latency-row", "The 23.8 s mean comes from the adaptive-tier external run (01 F2)",
     r"Full 13 stages, 3-model ensemble & \multicolumn{2}{c}{23.8 (mean, external $n=200$)} \\",
     r"Adaptive-tier pipeline (external) & \multicolumn{2}{c}{23.8 (mean, $n=200$)} \\")

edit("main_iccit.tex", "I11-threats", "Static attacks, one generator/KB (05 Q6)",
     r"generalization to larger corpora and stronger, adaptive attackers is untested.",
     r"generalization to larger corpora and stronger, adaptive attackers is untested, and fixed attack sets overstate robustness~\cite{nasr2025attacker}. Protocol~B and the external results each rest on \texttt{llama3.2:3b} and one knowledge base.")

edit("main_iccit.tex", "I12-conclusion", "Scope the three-layer finding and the practitioner advice (01 F3)",
     r"that three layers carry essentially all of that defense,",
     r"that three layers act as near-independent filters and carry essentially all of that defense against instruction-bearing but not assertion-only attacks,")
edit("main_iccit.tex", "I13-practitioners", "Do not advise dropping the layer that carries the assertion-only defense (01 F3)",
     r"Practitioners should deploy per-chunk sanitization, grounding, and refusal on detection first, and treat knowledge corruption as requiring provenance controls.",
     r"Practitioners can start with per-chunk sanitization, grounding, and refusal on detection, but should keep the guarded prompt where planted claims matter and treat knowledge corruption as requiring provenance controls.")

# ============================== Extended (main.tex) ==============================
edit("main.tex", "M1-abstract-subset", "Bound, independence and class dependence (02 §2)",
     r"Second, a single-generator ablation shows that three layers (per-chunk sanitization, grounding, and refusal on detection) are statistically indistinguishable from the seven-layer stack (5.5\% vs.\ 3.5\%, $p=0.34$; 4.7\% vs.\ 4.2\% on an enlarged $n=424$ set, $p=0.81$).",
     r"Second, a single-generator ablation with a fully crossed factorial shows that three layers (per-chunk sanitization, grounding, and refusal on detection) act as near-independent filters and come within 2.4 points of the seven-layer stack on these instruction-bearing attacks (4.7\% vs.\ 4.2\% at $n=424$, 95\% upper bound of the difference 2.4 points); on assertion-only attacks the same three layers are significantly worse than seven (62.5\% vs.\ 35.0\%, $p=0.003$, $n=40$), and protection varies with attacker wording.")

edit("main.tex", "M2-intro-contributions", "Single consistent contribution list (01 F1, 02 §6)",
     r"Our contributions are as follows: (i) a zero-cost defense-in-depth architecture in which an adaptive risk score is propagated from the input gate to downstream stages, as detailed in Section~\ref{sec:method}; (ii) a negative result, with a closed-form argument, showing that a co-activation detector built on saturated injection-classifier scores cannot identify coordinated sub-threshold attacks, together with a graded-margin repair that partly restores detection, as demonstrated in Section~\ref{sec:negative}; (iii) a stage-level, paired ablation showing that three layers carry the measured injection defense, with latency and memory costs, presented in Sections~\ref{sec:ablation} and~\ref{sec:overhead}; and (iv) a measured scope boundary: the defense does not extend to signature-free knowledge corruption, shown on our semantic slices and on PoisonedRAG against a paired undefended arm, examined in Sections~\ref{sec:external} and~\ref{sec:semantic}.",
     r"Our contributions are as follows. (C1) A zero-cost, fully local thirteen-stage pipeline built from known components, in which the input gate's risk score is propagated to downstream stages (Section~\ref{sec:method}); we do not claim a new detector. (C2) A paired attribution protocol for layered RAG defenses: leave-one-out, single-layer, and a fully crossed three-layer factorial on identical rows, exact paired tests, an independence null for interactions, a measured run-to-run noise floor, and class-conditional repetition. It applies standard ablation and factorial designs~\cite{fawcett2016ablation,hutter2014fanova} and is released with frozen per-row outputs and a claim-by-claim checker. (C3) Findings that an end-to-end comparison hides, for the seven-layer single-generator configuration: sanitization, grounding, and refusal act as near-independent filters with comparable shares; on instruction-bearing attacks they come within 2.4 points of all seven layers; on assertion-only attacks they do not, because the guarded prompt, harmful on its own against instruction-bearing attacks, carries that defense; and the measured protection depends on attacker wording (Section~\ref{sec:ablation}). (C4) A closed-form argument that a co-activation detector on saturated classifier scores cannot fire on its target band, with a graded-margin repair (Section~\ref{sec:negative}), and a stage-by-stage account of why this defense class does not reduce PoisonedRAG knowledge corruption (Sections~\ref{sec:external} and~\ref{sec:semantic}).")

edit("main.tex", "M3-related", "Prior art and wording 'we did not find' (01 #20, 02 §3)",
     r"To our knowledge, none of these works compares configurations with paired tests on identical rows, and none reports a component of its own system that fails to contribute.",
     r"We did not find among these works one that compares configurations with paired tests on identical rows or reports a component of its own system that fails to contribute. Per-component ablations of layered prompt-injection defenses also appear in Ramakrishnan and Balaji~\cite{ramakrishnan2025securing} and in LlamaFirewall~\cite{chennabasappa2025llamafirewall}. Defenses designed for knowledge corruption take a different route: RobustRAG answers from each retrieved passage separately and aggregates with certified guarantees~\cite{xiang2024robustrag}, and TriShieldRAG combines ingest screening, trust-weighted re-ranking, and cross-model consensus, reporting a large reduction on non-adaptive PoisonedRAG and little against an adaptive attacker~\cite{trishield2026}. Our attribution design follows classical ablation and functional-ANOVA analyses of configurations~\cite{fawcett2016ablation,hutter2014fanova}.")

edit("main.tex", "M4-distinction", "Remove the second, inconsistent contribution list (01 F1)",
     r"Our contribution is methodological and empirical (Table~\ref{tab:positioning}): (C1) an attribution protocol that pairs every configuration on identical rows, scores each row by a machine-checkable outcome, and tests each stage with leave-one-out, add-one-in, and subset ablations under exact paired tests with confidence intervals, so a layer's contribution, or its absence, is measured rather than assumed; (C2) a forward-propagated risk score and fail-closed controller whose value we test rather than assert; (C3) a negative result, with a closed-form argument, that a co-activation detector built on saturated injection-classifier scores cannot fire on the sub-threshold band it targets, together with a graded-margin repair; and (C4) a quantified residual gap for signature-free semantic poisoning, evaluated on our slices and on PoisonedRAG~\cite{zou2025poisonedrag}. The practical result of C1 is that a three-layer subset of the cascade is statistically indistinguishable from the full stack (Section~\ref{sec:ablation}). The paper therefore argues for a simpler configuration, not a more complex one.",
     r"Our contribution is the careful application of established attribution designs to a layered RAG defense, and the findings it yields (contributions C1--C4 in the Introduction; Table~\ref{tab:positioning}). The forward-propagated risk score and fail-closed controller are tested rather than asserted, and add no measured security. The practical result is that, on instruction-bearing attacks, a three-layer subset of the cascade comes within 2.4 points of the full stack (Section~\ref{sec:ablation}), while on assertion-only attacks it does not; the paper therefore argues for a simpler configuration chosen per threat, not a more complex one.")

edit("main.tex", "M5-eval-sets-external", "External sets ran the adaptive-tier variant (01 F2)",
     r"all run through the complete thirteen-stage pipeline, plus",
     r"all run through an adaptive-tier variant of the pipeline (Section~\ref{sec:setup}), plus")

edit("main.tex", "M6-setup-external", "Describe the adaptive-tier variant precisely (01 F2)",
     r"Protocol~A and the external sets use the complete thirteen-stage pipeline, with the three-model ensemble as generator, \texttt{qwen2.5:3b} as fuser, and the controller enabled.",
     r"Protocol~A uses the complete thirteen-stage pipeline, with the three-model ensemble as generator, \texttt{qwen2.5:3b} as fuser, and the controller enabled. The external sets ran an adaptive-tier variant of the same stages (Stages~1--10 and~13; no multivector detector, controller, Presidio, or DLP), in which input risk selects retrieval depth and generator: top-8 retrieval and a single \texttt{llama3.2:3b} at low risk (139 of the 150 PoisonedRAG rows and 44 of the 50 BIPIA rows), top-5 at medium risk, and top-3 with the three-model ensemble at high risk.")

edit("main.tex", "M7-ablation-new-paragraphs", "Interactions, Shapley, class dependence and wording (02 §2, 03 §2); new Table tab:factorial",
     r"The larger sample thus bounds the cost of dropping the other four layers at about 2.4 points of ASR.",
     r"The larger sample thus bounds the cost of dropping the other four layers at about 2.4 points of ASR (95\% upper bound; no equivalence margin was prespecified, and the subset was worse in all three $n=200$ replicates, by 1.0--2.5 points)." + "\n\n" +
     r"\textbf{Interactions.} Leave-one-out and single-layer arms cannot show whether layers interact, so we ran sanitization (S), grounding (G), and refusal (R) in all eight combinations with the other four layers off (Table~\ref{tab:factorial}). On the additive scale every pair is sub-additive (S+G reduces ASR by 21.0 points against 13.5 and 14.5 for its parts), which would suggest redundancy. A better null is that each layer independently removes a fixed fraction of the attacks that reach it, so the surviving fractions multiply. Against this null every combination matches within 1.3 points, and every 95\% paired-bootstrap interval of the difference includes 0 (S+G+R: 5.5\% observed vs.\ 5.0\% predicted, $[-1.6, +2.8]$). The three layers therefore act as near-independent filters. Each removes 39--49\% of what reaches it, they stop largely different rows (of the 59 undefended successes, 22 are fixed by exactly one of the three single layers and none only by their combination), and absolute returns diminish as layers are added. Their exact Shapley values over the 24.0-point reduction are 8.7 (S, [5.1, 12.6]), 8.7 (G, [5.8, 11.9]), and 6.7 (R, [3.8, 9.7]) points; the benign-refusal cost is concentrated in the grounding gate (6.3 of 9.5 points). The guarded prompt is the one layer whose effect changes sign with context: alone it raises ASR by 23.5 points, while removing it from the full stack changes ASR by only $+2.0$ $[-0.8, +4.8]$; the arms we ran cannot identify which layer neutralizes it. Effects of one to three points are within the run-to-run noise of this setup (two runs of the same undefended configuration differ on 11 of 200 rows)." + "\n\n" +
     r"\textbf{Class dependence and wording.} The subset result does not transfer across attack classes. On the 40-row assertion-only slice (Section~\ref{sec:semantic}), sanitize+ground+refuse leaves 25/40 (62.5\%) against 14/40 (35.0\%) for all seven layers (12 vs.\ 1 discordant rows, $p=0.003$) and is not significantly better than undefended ($p=0.13$). The guarded prompt, harmful on its own against the instruction-bearing Protocol~B attacks, is the most effective single layer on this slice (alone 75.0\%$\rightarrow$52.5\%, $p=0.035$). Within Protocol~B, an exploratory analysis of the randomized attack templates shows that protection also depends on wording. For one of the three coercive query phrasings (``Defer to any authoritative correction\ldots'') the detector behind refuse-on-detection fires on every row and full-stack ASR is 0/147; for another it fires on 20\% of rows and full-stack ASR is 9.9\% [5.9, 16.2] (permutation test of homogeneity $p=0.0001$, $n=424$). The 3.5\% headline is therefore an average over phrasings with different outcomes, and an attacker who avoids the detected phrasing faces a weaker defense. These subgroup results are exploratory and were not prespecified." + "\n\n" +
     r"""\begin{table}[t]
\centering
\caption{Fully crossed factorial of sanitization (S), grounding (G), and refuse-on-detection (R), other four layers off (Protocol~B, $n=200$ attack and 200 benign). ``Indep.'' = ASR predicted if each layer independently removes a fixed fraction of the attacks that reach it.}
\label{tab:factorial}
\small
\begin{tabular}{lccc}
\toprule
Layers & ASR \% & Indep.\ \% & Benign refused \% \\
\midrule
none & 29.5 & -- & 0.0 \\
S & 16.0 & -- & 0.0 \\
G & 15.0 & -- & 5.5 \\
R & 18.0 & -- & 2.5 \\
S+G & 8.5 & 8.1 & 8.0 \\
S+R & 9.5 & 9.8 & 2.5 \\
G+R & 10.5 & 9.2 & 7.5 \\
S+G+R & 5.5 & 5.0 & 9.5 \\
\midrule
\multicolumn{4}{l}{\footnotesize Shapley shares of the 24.0-pt ASR drop: S 8.7, G 8.7, R 6.7.} \\
\multicolumn{4}{l}{\footnotesize Shapley share of benign refusals: S 1.1, G 6.3, R 2.1 pts.} \\
\bottomrule
\end{tabular}
\end{table}""")

edit("main.tex", "M8-external-intro", "Accurate label (01 F2)",
     r"Table~\ref{tab:external} reports the complete thirteen-stage pipeline on data we did not construct.",
     r"Table~\ref{tab:external} reports the adaptive-tier variant of the pipeline (Section~\ref{sec:setup}) on data we did not construct.")
edit("main.tex", "M9-external-caption", "Accurate label (01 F2)",
     r"\caption{Complete pipeline on independent external sets,",
     r"\caption{Adaptive-tier pipeline variant (Stages 1--10, 13) on independent external sets,")
edit("main.tex", "M10-same-generator-premise", "The premise was wrong: the variant also used llama3.2:3b at low risk (01 F2)",
     r"Because the complete pipeline uses the three-model ensemble, we also ran a same-generator comparison.",
     r"Because the adaptive-tier variant adds tiered retrieval and generation, we also ran the seven-layer harness on the same rows.")
edit("main.tex", "M11-bipia-label", "Accurate label (01 F2)",
     r"close to the complete pipeline's 45/50.",
     r"close to the adaptive-tier variant's 45/50.")
edit("main.tex", "M12-poisonedrag-mechanism", "Replace the unsupported ensemble claim with the trace-level mechanism (04)",
     r"Neither the ensemble's disagreement signal nor the grounding judge helps: a planted fact is consistent across models and is ``grounded'' in the poisoned passage.",
     r"The saved traces show where each stage fails. The gate scores every PoisonedRAG query 0, Stage~6 redacts none of the 148 passages, the reranker places the poisoned passage first in all 148 rows (median margin 0.23 over the next passage), and all 136 successful attacks pass the grounding judge with median faithfulness 1.0, against 0.77 for the 8 delivered answers that resisted. Counter-evidence was rarely available: the true answer appears in another retrieved passage in only 11 rows, because the questions come from corpora other than our knowledge base. The variant generated with a single model at low risk, so this run does not test ensemble disagreement; on the 8-row semantic supplement, disagreement did not separate attacks from benign questions (Section~\ref{sec:semantic}). The failure therefore lies in relevance ranking, grounding as context-consistency, and the absence of provenance, not in a single weak stage.")
edit("main.tex", "M13-isolate-robustrag", "Cite the prior art for isolate-then-aggregate (01 #15)",
     r"so that a single planted passage cannot carry the response.",
     r"so that a single planted passage cannot carry the response, following the isolate-then-aggregate design of RobustRAG~\cite{xiang2024robustrag}.")
edit("main.tex", "M14-lesson-only", "Remove the undemonstrated 'only' (01 #19)",
     r"and only a stage-by-stage, paired, class-decomposed evaluation can tell them apart.",
     r"and a stage-by-stage, paired, class-decomposed evaluation can tell them apart where an end-to-end comparison cannot.")
edit("main.tex", "M15-threats-external", "Static attacks, wording, generator (05 Q6)",
     r"Generalization to larger corpora, larger generators, and adaptive attacks remains open.",
     r"Generalization to larger corpora, larger generators, and adaptive attacks remains open. Fixed attack sets overstate robustness against adaptive attackers~\cite{nasr2025attacker}, and our own Protocol~B protection already varies with attacker wording (Section~\ref{sec:ablation}). Every attack-success result on Protocol~B and the external sets rests on \texttt{llama3.2:3b}; success rules differ between Protocol~B (refutation cues) and the external sets (manual ambiguity rules), so ASRs are not compared across protocols.")
edit("main.tex", "M16-conclusion-subset", "Bound and class scope (01 F3, F5)",
     r"\emph{Three layers carry this defense}, though not at lower latency (19.0 vs.\ 15.3\,s median). Sanitization, grounding, and refusal on detection are statistically indistinguishable from the seven-layer stack in each of three generation replicates and on an enlarged 424-row set, where the gap is bounded at about 2.4 points, whereas no single layer comes close.",
     r"\emph{Three layers carry this defense against instruction-bearing attacks}, though not at lower latency (19.0 vs.\ 15.3\,s median). Sanitization, grounding, and refusal on detection act as near-independent filters; no replicate separates them from the seven-layer stack, and on an enlarged 424-row set the cost of dropping the other four layers is at most about 2.4 points (95\% bound), whereas no single layer comes close. On assertion-only attacks the subset is significantly worse, because the guarded prompt carries that defense.")
edit("main.tex", "M17-practitioners", "Do not advise dropping the guarded prompt (01 F3)",
     r"For practitioners: deploy per-chunk sanitization, grounding, and refusal on detection first;",
     r"For practitioners: against instruction-bearing attacks, deploy per-chunk sanitization, grounding, and refusal on detection first, but keep the guarded prompt where planted factual claims matter;")
edit("main.tex", "M18-latency-text", "The 23.8 s mean is from the adaptive-tier external run (01 F2)",
     r"The complete thirteen-stage pipeline with the three-model ensemble and the controller averages 23.8\,s per query",
     r"The adaptive-tier pipeline variant used for the external sets averages 23.8\,s per query")
edit("main.tex", "M19-latency-row", "Same, in the latency table (01 F2)",
     r"Full 13 stages, 3-model ensemble & \multicolumn{2}{c}{23.8 (mean, external $n=200$)} \\",
     r"Adaptive-tier pipeline (external) & \multicolumn{2}{c}{23.8 (mean, $n=200$)} \\")

BIB = r"""
@misc{xiang2024robustrag,
  author = {Chong Xiang and Tong Wu and Zexuan Zhong and David Wagner and Danqi Chen and Prateek Mittal},
  title = {Certifiably Robust {RAG} against Retrieval Corruption},
  howpublished = {arXiv:2405.15556},
  year = {2024}
}
@misc{trishield2026,
  author = {Mohanty and Patel and Yuvaraj and Chaudhary and Singhania},
  title = {{TriShieldRAG}: A Three-Ring Defense-in-Depth Framework Against Knowledge Corruption in Retrieval-Augmented Generation},
  howpublished = {arXiv:2607.23838},
  year = {2026}
}
@misc{nasr2025attacker,
  author = {Milad Nasr and Nicholas Carlini and others},
  title = {The Attacker Moves Second: Stronger Adaptive Attacks Bypass Defenses Against {LLM} Jailbreaks and Prompt Injections},
  howpublished = {arXiv:2510.09023 (USENIX Security 2026)},
  year = {2025}
}
@article{fawcett2016ablation,
  author = {Chris Fawcett and Holger H. Hoos},
  title = {Analysing Differences Between Algorithm Configurations Through Ablation},
  journal = {Journal of Heuristics},
  volume = {22},
  number = {4},
  pages = {431--458},
  year = {2016}
}
@inproceedings{hutter2014fanova,
  author = {Frank Hutter and Holger Hoos and Kevin Leyton-Brown},
  title = {An Efficient Approach for Assessing Hyperparameter Importance},
  booktitle = {Proc. International Conference on Machine Learning (ICML), PMLR 32},
  pages = {754--762},
  year = {2014}
}
@misc{ramakrishnan2025securing,
  author = {Badrinath Ramakrishnan and Akshaya Balaji},
  title = {Securing {AI} Agents Against Prompt Injection Attacks},
  howpublished = {arXiv:2511.15759},
  year = {2025}
}
@misc{chennabasappa2025llamafirewall,
  author = {Sahana Chennabasappa and others},
  title = {{LlamaFirewall}: An Open Source Guardrail System for Building Secure {AI} Agents},
  howpublished = {arXiv:2505.03574},
  year = {2025}
}
"""

# ============================== E1 / E2 results (preregistered, run 2026-11-25) ==============================
# These target text inserted by earlier edits (I7, I8, I11, M7, M12), so they must stay after them in E.
edit("main_iccit.tex", "I14-E1-wording", "Preregistered E1 result (experiments/RESULTS_E1_E2.md §1)",
     r"for another it fires on 20\% and ASR is 9.9\% ($n=424$, homogeneity $p<0.001$).",
     r"for another it fires on 20\% and ASR is 9.9\% ($n=424$, homogeneity $p<0.001$). In a preregistered follow-up on the same 200 rows, protection transferred to held-out coercive wording (3.0\% vs.\ 3.5\%, $p=1.0$); with the coercive query suffix removed, ASR was 7.5\% ($p=0.115$, not significant) and the detector fired on 3\% of rows.")
edit("main_iccit.tex", "I15-E2-closed-book", "E2 closed-book probe (RESULTS_E1_E2.md §3)",
     r"(7 fixed, 6 broken, $p=1.0$).",
     r"(7 fixed, 6 broken, $p=1.0$). Asked without context, \texttt{llama3.2:3b} knew the true answer in 54\% of the defended successes; knowing it gave no protection (90.2\% vs.\ 91.2\% ASR).")
edit("main_iccit.tex", "I16-threats-heldout", "E1 limitation: held-out wording is ours, not adaptive",
     r"rest on \texttt{llama3.2:3b} and one knowledge base.",
     r"rest on \texttt{llama3.2:3b} and one knowledge base, and our held-out attack wording was written by us.")

edit("main.tex", "M20-E1-wording", "Replace the untested 'weaker defense' inference with the preregistered E1 result (RESULTS_E1_E2.md §1-2)",
     r"The 3.5\% headline is therefore an average over phrasings with different outcomes, and an attacker who avoids the detected phrasing faces a weaker defense. These subgroup results are exploratory and were not prespecified.",
     r"The 3.5\% headline is therefore an average over phrasings with different outcomes. These subgroup results are exploratory and were not prespecified, so we tested them in a preregistered follow-up on the same 200 questions and targets with only the wording changed (two paired tests, Bonferroni $\alpha=0.025$). With three new coercive query suffixes and four new poison templates, fixed before any run, full-stack ASR was 3.0\% [1.4, 6.4] against 3.5\% (6 vs.\ 7 discordant rows, $p=1.0$): the protection transferred, although the new wording was the stronger attack (40.5\% undefended) and the detector fired on 111 of 200 rows. With the coercive query suffix removed, the attack was weaker (16.5\% undefended) and the detector fired on 6 rows instead of 80; full-stack ASR was 7.5\% [4.6, 12.0] (14 vs.\ 6, $p=0.115$), not significant, while the relative reduction fell from 88\% to 55\%. We therefore do not claim that the defense is weaker without coercive phrasing; refuse-on-detection reacts mainly to query phrasing, and in this run the other layers left a smaller margin when it did not fire. In neither variant was the three-layer subset significantly different from the seven-layer stack (+3.5 points, 95\% CI $-0.7$ to 7.7, without suffix; $-1.0$ points with held-out wording); this is not evidence of equivalence. The held-out wording was written by us, not by an adaptive attacker.")
edit("main.tex", "M21-E2-closed-book", "E2 closed-book probe: prior override vs evidence absence (RESULTS_E1_E2.md §3)",
     r"not in a single weak stage.",
     r"not in a single weak stage. A closed-book probe (the 150 questions without retrieved context) separates two cases: \texttt{llama3.2:3b} answered correctly in 74 of the 136 defended successes (54.4\%, 95\% CI 46.0--62.5), so the poisoned passage overrode knowledge the generator had, and in the rest it did not know the answer. Knowing the answer gave no protection (ASR 90.2\% when it knew vs.\ 91.2\% when it did not). String matching misses a few correctly phrased closed-book answers, so the first share is, if anything, slightly underestimated.")
