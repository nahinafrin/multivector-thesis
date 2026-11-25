# 02 — Novelty assessment and attribution methodology

**Bottom line.** The attribution *method* is not new. Its parts are classical:
- factorial and ablation designs;
- path ablation (Fawcett & Hoos 2016);
- fANOVA main and interaction effects (Hutter et al. 2014);
- Shapley attribution;
- exact paired tests.

Its application to a layered RAG defense, with row-level pairing, a measured noise floor and class-conditional
analysis, is careful but incremental. What the study *does* produce that a full-stack-vs-undefended comparison
would miss is a set of **empirical findings about how the layers combine**. They come from saved data (§3) and are
reportable now:
1. The three key layers act as approximately independent filters, so returns diminish multiplicatively.
2. The minimal sufficient subset depends on the attack class.
3. One layer (the guarded prompt) changes sign across attack classes.
4. Much of the protection is keyed to one attacker phrasing.

These, rather than the method, are the defensible contribution.

## 1. Formal specification of the existing attribution protocol

**Inputs.**
- A set of layers L = {l₁ … l_m}. In Protocol B: sanitize S, grounding gate G, refuse-on-detection R, tightened
  rerank T, guarded prompt P, Presidio, DLP.
- A configuration space C ⊆ 2^L.
- An attack set A and a benign set N, each row with a machine-checkable outcome.
- A fixed generator, retrieval and decoding seed.

**Per-row outcomes**, for each configuration c ∈ C and row i:
- y_i(c) ∈ {0,1}: attack success, for i ∈ A;
- b_i(c): benign refused, and k_i(c): benign answered correctly, for i ∈ N.

The same rows are used for every c (paired design).

**Estimands.** Risk differences in percentage points, with ASR(c) = mean_i y_i(c).

| Quantity | Definition | Answers | Identified from saved data for |
|---|---|---|---|
| Marginal (add-one-in) effect | ASR(∅) − ASR({l}) | Does l help on its own? | All 7 layers |
| Conditional effect in the full stack (leave-one-out) | ASR(L∖{l}) − ASR(L) | Does l add anything given all others? | All 7 layers |
| Conditional effect in a sub-stack | ASR(T∖{l}) − ASR(T) for T ⊂ L | Does l help given a specific partner set? | S, G, R within {S,G,R} |
| Shapley value over a fully crossed set K | Average conditional effect over all orderings of K (others off) | A fair split of the joint reduction of K | K = {S,G,R} only (all 2³ cells exist) |
| Interaction, additive scale | e.g. Δ(SG) − Δ(S) − Δ(G), with Δ(T) = ASR(∅) − ASR(T) | Is the joint reduction more or less than the sum? | Pairs and triple within {S,G,R} |
| Interaction vs independent filters | ASR(T) − ASR(∅)·∏_{l∈T} [ASR({l})/ASR(∅)] | Do layers stop overlapping or complementary rows beyond chance? | Pairs and triple within {S,G,R} |
| Utility cost attribution | The same quantities on b (refusals) and k (correctness) | Which layer causes the benign cost? | Same as above |
| Mechanism decomposition | Per row: refused by detector / refused by grounding / delivered and failed / delivered and succeeded | How a configuration stops attacks | All arms (block reasons are logged) |
| Class-conditional sufficiency | The comparisons above, restricted to an attack class | Is the minimal subset the same for every class? | Planted (n=424) vs semantic (n=40) |
| Noise floor | Discordance between two runs of the *same* configuration | Which differences exceed run-to-run variation? | Undefended and full (`rerun200`) |

**Assumptions** (all stated in the papers or below):
1. Rows are exchangeable units, so a paired design is valid.
2. Generation is not bit-reproducible: same-configuration reruns differ on 11/200 rows (undefended) and 1/200
   (full stack). Effects of 1–3 points are inside this floor at n=200.
3. The attack distribution is fixed and non-adaptive. All estimands are conditional on it.
4. The Shapley and interaction quantities are defined *with the other layers off*. They do not describe
   contributions inside the seven-layer stack.

**Procedure.**
1. Fix the families before testing: LOO (7 tests), single-layer (7), subsets (4).
2. Use exact McNemar tests on discordant pairs, Bonferroni within each family, and Wilson intervals for rates.
3. Report the noise floor next to every small effect.
4. Report a non-significant difference as a CI bound, never as equivalence.
5. Repeat the key comparisons per attack class.
6. Recompute every reported number from frozen per-row outputs (`verify_claims.py`).

## 2. Layer contributions and interactions (existing data; `analysis/outputs/protocolB_attribution.json`)

Protocol B, seven-layer harness, n = 200 attacks and 200 benign, seed 42.

**Cell table, the fully crossed set {S, G, R}** (other four layers off):

| Cell | ASR % [95% CI] | Benign refused % | Benign correct % |
|---|---|---|---|
| ∅ (undefended) | 29.5 [23.6, 36.2] | 0.0 | 56.5 |
| S | 16.0 [11.6, 21.7] | 0.0 | 54.5 |
| G | 15.0 [10.7, 20.6] | 5.5 | 56.0 |
| R | 18.0 [13.3, 23.9] | 2.5 | 54.5 |
| SG | 8.5 [5.4, 13.2] | 8.0 | 53.5 |
| SR | 9.5 [6.2, 14.4] | 2.5 | 53.0 |
| GR | 10.5 [7.0, 15.5] | 7.5 | 54.0 |
| SGR | 5.5 [3.1, 9.6] | 9.5 | 52.0 |
| Full (7 layers) | 3.5 [1.7, 7.0] | 8.5 | 52.0 |

**Contribution of each key layer in three contexts.** Values are the ASR reduction in points, with the
paired 95% CI and exact p:

| Layer | Alone (vs ∅) | Within SGR (remove it) | Within full stack (leave-one-out) |
|---|---|---|---|
| S | 13.5 [8.2, 18.8], p = 1.4×10⁻⁶ | 5.0 [1.4, 8.6], p = 0.013 | 7.5 [3.6, 11.4], p = 2.7×10⁻⁴ |
| G | 14.5 [9.6, 19.4], p = 3.7×10⁻⁹ | 4.0 [1.3, 6.7], p = 0.008 | 3.0 [0.6, 5.4], p = 0.031 |
| R | 11.5 [6.7, 16.3], p = 5.6×10⁻⁶ | 3.0 [0.6, 5.4], p = 0.031 | 2.5 [0.3, 4.7], p = 0.062 |
| T (tighten) | 5.0 [−1.6, 11.6], p = 0.18 | n/a | 2.5 [−0.7, 5.7], p = 0.23 |
| P (prompt) | **−23.5** [−30.3, −16.7], p = 1.8×10⁻¹⁰ (harmful) | n/a | 2.0 [−0.8, 4.8], p = 0.29 |
| Presidio / DLP | 0 / 0 | n/a | 0.5 / 0 |

**Shapley values over {S, G, R}.** These are exact, with 5,000-row paired-bootstrap CIs, and sum to the 24.0-point
reduction of SGR:

| Layer | ASR reduction (pts) | Benign-refusal cost (pts) | Benign-correctness loss (pts) |
|---|---|---|---|
| S | 8.7 [5.1, 12.6] | 1.1 [0.2, 2.2] | 2.0 [0.5, 4.0] |
| G | 8.7 [5.8, 11.9] | 6.3 [3.5, 9.8] | 0.8 [0.0, 2.0] |
| R | 6.7 [3.8, 9.7] | 2.1 [0.5, 4.1] | 1.8 [0.2, 3.7] |

The three layers contribute comparable security. Grounding accounts for about two thirds of the benign-refusal
cost; sanitization and refusal cost correctness rather than refusals.

**Interactions.**
- **Additive scale.** Every pairwise interaction is negative: S×G −7.0 [−11.5, −3.0], S×R −5.0 [−9.0, −1.0],
  G×R −7.0 [−10.5, −3.5]. Taken alone, this would suggest redundancy.
- **Independent-filters model.** Against a model in which each layer independently removes a fixed fraction of the
  surviving attacks, the observed ASR matches the prediction closely:

  | Cell | Observed | Predicted if independent | Difference [95% CI] |
  |---|---|---|---|
  | SG | 8.5 | 8.1 | +0.4 [−2.0, 3.0] |
  | SR | 9.5 | 9.8 | −0.3 [−2.4, 1.9] |
  | GR | 10.5 | 9.2 | +1.3 [−0.6, 3.3] |
  | SGR | 5.5 | 5.0 | +0.5 [−1.6, 2.8] |

  **Interpretation (supported):** S, G and R act as approximately independent filters. Each removes a roughly
  constant fraction of whatever reaches it: S 46%, G 49%, R 39% of the undefended successes. This explains three things: why every pair beats its parts, why returns diminish in absolute
  points, and why a fourth weak layer can add at most a fraction of the remaining 5.5 points. The additive
  "interactions" are a floor artifact, not real redundancy.
- **Row overlap.** Of the 59 undefended successes:

  | Fixed by | Rows |
  |---|---|
  | exactly one single layer | 22 |
  | two or more single layers | 28 |
  | all three single layers | 6 |
  | no single layer but by SGR | 0 |

  No row needs a combination to be fixed: complementarity means covering *different* rows, not joint action on the
  same row.
- **Context dependence of the guarded prompt.** Alone, it is strongly harmful (+23.5 points ASR). In the full stack,
  removing it changes ASR by 2.0 [−0.8, 4.8]. Its effect therefore depends on what else is active, but the data
  cannot identify *which* layer neutralizes the harm: no P-containing subset was run.

**What the data cannot identify:**
- interactions involving T, P, Presidio or DLP, since only their singles and leave-one-outs exist;
- any three-way structure outside {S, G, R};
- interactions inside the complete 13-stage pipeline: the ensemble, controller and multivector detector were not
  ablated factorially.

The smallest informative addition would be four arms on the same 200 + 200 rows: P+S, P+G, P+R and SGR+P. They
would identify P's interactions and test whether P is what turns SGR into the full stack. That is about 4 × 400
rows × 15 s ≈ 6.7 h. It is optional, and I don't recommend it before E1 (03 §5).

**Security-utility.**
- Every configuration with G refuses 5.5–9.5% of benign questions. Configurations without G refuse at most 2.5%.
- Benign correctness falls only from 56.5% to 52.0% across the whole range, so most of the extra refusals hit
  questions the generator answered wrongly anyway: in SGR, 104 correct + 77 wrong + 19 refused, against 113 + 87
  undefended.
- **The S+R pair is the efficient point:** 9.5% ASR at 2.5% refusals, against SGR's 5.5% at 9.5%.

**Class-conditional sufficiency** (`outputs/verify_unchecked_claims.json`, semantic slice, n = 40):

| | Planted, instruction-bearing (n=424) | Semantic, assertion-only (n=40) |
|---|---|---|
| Undefended | 29.7% | 75.0% |
| Three-layer subset SGR | 4.7% | 62.5% (vs undefended p = 0.125) |
| Full seven layers | 4.2% | 35.0% |
| Subset − full | +0.5 [−1.5, +2.4] | **+27.5 [12.0, 43.0], p = 0.0034** |
| Guarded prompt alone vs undefended | +23.5 pts (harmful), n=200 | **−22.5 pts (helpful), p = 0.035** |

The minimal sufficient subset is attack-class dependent. A configuration chosen on instruction-bearing attacks
discards the layer that carries the defense against assertion-only attacks. This is the clearest example of an
insight that a full-stack-vs-undefended comparison could not produce. It rests on n = 40 and should be reported as
exploratory.

## 3. Prior work (targeted check; original sources)

| Work | Stage-level attribution | Paired / row-level tests | Interactions | Security-utility attribution | Redundancy | Failure-mode attribution |
|---|---|---|---|---|---|---|
| Fawcett & Hoos 2016, ablation analysis (J. Heuristics 22(4)); Biedenkapp et al. AAAI 2017 | Yes, path ablation between configurations | No (performance per configuration) | Partly | n/a | n/a | n/a |
| Hutter, Hoos & Leyton-Brown 2014, fANOVA (ICML) | Yes, main effects | No | **Yes**, low-order interactions | n/a | n/a | n/a |
| Shapley-based component attribution (general; e.g. DataScope, pipeline SHAP) | Yes | No | Implicit | No | No | No |
| Spotlighting (Hines et al. 2024) | Variants compared | No | No | Partly | No | No |
| CaMeL (Debenedetti et al. 2025) | Design, not ablation | No | No | Utility reported | No | No |
| ControlNET (Yao et al. 2025) | Detector + ProNet | No (point estimates) | No | Yes | No | No |
| ADO (Pallerla et al. 2026) | Defense selection | No | No | Yes | No | No |
| Saleem / Ahmed et al. 2026, layered framework (arXiv 2606.19660) | **Yes, all 7 subsets of 3 layers** | No (point estimates) | Claim of super-additivity | Yes (FPR, latency) | No | No |
| Ramakrishnan & Balaji 2025, multi-layer RAG defense benchmark (arXiv 2511.15759) | Ablation per component (per abstract) | Not found in the abstract | Defense Bypass Rate (all-layers metric) | Yes | No | No |
| LlamaFirewall (Meta 2025) | Two guards and their combination on AgentDojo | No | Combination reported | Yes | No | No |
| RobustRAG (Xiang et al. ICML 2024) | n/a (certified aggregation) | n/a | n/a | Yes | n/a | n/a |
| TriShieldRAG (2026, arXiv 2607.23838) | Three rings against knowledge corruption | Mean ± sd over runs | n/a | Yes | n/a | **Yes**: an adaptive attacker evades it (62% vs 56% undefended) |
| Nasr et al. 2025, "The Attacker Moves Second" (USENIX Sec. 2026) | n/a | n/a | n/a | n/a | n/a | Static datasets overstate robustness (12 defenses broken) |

*Search coverage:* targeted web search on 25 Nov 2026. Search snippets were checked against the arXiv or
proceedings pages; I did not read every full PDF. Before submission, the authors should verify the "paired tests"
and "ablation" cells for Ramakrishnan & Balaji and LlamaFirewall against the full papers.

**What is not new:**
- stage-level ablation of layered prompt-injection defenses (Saleem et al., Ramakrishnan & Balaji, LlamaFirewall);
- factorial and Shapley attribution and interaction analysis (fANOVA, ablation analysis);
- isolate-and-aggregate generation (RobustRAG);
- defense-in-depth against knowledge corruption (TriShieldRAG);
- the warning that static attack sets overstate robustness (Nasr et al.).

**What we did not find in the closest RAG-defense works** (Saleem; Ramakrishnan & Balaji; ControlNET; ADO):
- exact paired tests on identical rows;
- a measured run-to-run noise floor;
- an independence (multiplicative) null for interactions;
- class-conditional sufficiency;
- reporting of a layer that is harmful alone;
- reporting of layers that contribute nothing.

These are *analysis practices*. Each is standard in experimental design; their combination in this domain is the
incremental contribution.

## 4. Narrowest defensible methodological contribution

> A fully specified, reproducible attribution protocol for layered RAG defenses: paired row-level outcomes; a
> fully crossed core subset; leave-one-out in the full stack; an independence null; noise-floor and multiplicity
> control; and class-conditional repetition. It is released with frozen per-row outputs and a claim checker. It is
> an application of established designs, not a new statistical method. Its value lies in the findings it produced
> (§2), which the conventional comparison misses.

## 5. Words to avoid and use

- **Avoid:** "first", "novel framework/methodology", "state of the art", "proves" (in the 6-page paper),
  "equivalent", "only a stage-level evaluation can…".
- **Use:** "we apply", "we did not find prior RAG-defense work that…", "bounded at", "on this attack
  distribution", "exploratory".

## 6. Proposed contributions (C1–C4)

**C1 — System.** A thirteen-stage, fully local, zero-cost defense pipeline whose input-gate risk score is
propagated to downstream stages. It is built from existing components, and we do not claim a new detector.

**C2 — Evaluation protocol** (an application of standard designs). A paired attribution protocol for layered RAG
defenses, released with frozen per-row outputs and a claim-by-claim checker:
- leave-one-out, single-layer, and a fully crossed three-layer factorial;
- exact paired tests and an independence null for interactions;
- a measured run-to-run noise floor;
- class-conditional repetition.

**C3 — Empirical findings on the seven-layer single-generator configuration** (Protocol B and the semantic slice;
not the complete 13-stage pipeline):
- (a) sanitization, grounding and refusal act as approximately independent filters with comparable Shapley shares
  (8.7 / 8.7 / 6.7 points), and the grounding gate carries most of the benign-refusal cost;
- (b) on instruction-bearing planted attacks these three layers come within 2.4 points (95% upper bound, n = 424)
  of all seven;
- (c) the minimal subset depends on the attack class: on assertion-only attacks the same three layers are
  significantly worse than seven (exploratory, n = 40), because the guarded prompt, harmful alone on
  instruction-bearing attacks, carries the assertion-only defense;
- (d) much of the measured protection is keyed to attacker phrasing (exploratory).

**C4 — Structural and boundary results.**
- (a) A closed-form argument that a co-activation detector on saturated classifier scores cannot fire on its target
  band, with a graded-margin repair that transfers poorly.
- (b) A trace-level account of why this defense class does not reduce PoisonedRAG knowledge corruption:
  - the passage is ranked first in every row;
  - it carries no instruction signature;
  - it is judged fully grounded.
  Defenses designed for knowledge corruption (e.g. TriShieldRAG) address a different mechanism.

Sources:
- [Fawcett & Hoos 2016, ablation analysis (PDF)](https://www.cs.ubc.ca/labs/beta/Projects/Ablation/papers/FawcettHoos-joh2016-ablationAnalysis.pdf)
- [Biedenkapp et al. AAAI 2017](https://ojs.aaai.org/index.php/AAAI/article/view/10657)
- [Hutter, Hoos & Leyton-Brown 2014, fANOVA](https://proceedings.mlr.press/v32/hutter14.html)
- [Saleem / Ahmed et al. 2026](https://arxiv.org/pdf/2606.19660)
- [Ramakrishnan & Balaji 2025](https://arxiv.org/abs/2511.15759v1)
- [LlamaFirewall](https://arxiv.org/pdf/2505.03574)
- [RobustRAG](https://arxiv.org/abs/2405.15556v1)
- [TriShieldRAG](https://arxiv.org/pdf/2607.23838)
- [Nasr et al., The Attacker Moves Second](https://arxiv.org/abs/2510.09023v1)
- [DataScope (Shapley over ML pipelines)](https://ar5iv.labs.arxiv.org/html/2204.11131)
