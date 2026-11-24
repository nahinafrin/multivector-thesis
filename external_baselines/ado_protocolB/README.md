# ADO (Sentinel–Strategist) as an existing-defense baseline on Protocol B

**Scope.** This compares ADO with our **seven-layer, single-generator configuration under Protocol B**.
It does **not** compare ADO with the complete 13-stage framework (Protocol A); the two protocols are
never mixed in one statistical comparison. Our defense code and our saved results were not modified:
ADO runs in an isolated adapter that reads the same Protocol B inputs.

**Classification:** *adapted implementation with documented limitations*. It is not claimed to be
paper-faithful; see `IMPLEMENTATION_AUDIT.md`.

## Status

| Stage | Result | File |
|---|---|---|
| Implementation audit (Phase 1) | done | `IMPLEMENTATION_AUDIT.md` |
| AV memory-safe path ≡ official path | identical text, max rel. diff 0.0 | `av_equivalence_report.json` |
| 5 + 5 functional test | 10/10 rows, 0 errors | `FUNCTIONAL_TEST_REPORT.md` |
| Preregistered 50 + 50 pilot | 100/100 rows, 0 errors, alignment OK | `pilot_*` |
| Preregistration of the full run | written and hashed before launch | `PREREGISTRATION_FULL.md`, `full_run_manifest.json` |
| Full run, remaining 150 + 150 rows | 300/300 rows, 0 generation/AV errors, 1 Sentinel fallback (attack row 42, kept as recorded); 20:06–05:56 | `remaining_*` |
| Merged full set, 200 + 200 | alignment OK | `full_{attack,benign}_results.jsonl` |
| Scoring (each script run twice) | byte-identical outputs | `full_comparison.{json,md}`, `full_sensitivity_refusal.{json,md}` |
| Manual adjudication (preregistered rule) | 24 rows labeled | `full_adjudication.json` |
| Paper text | inserted into main.tex and main_iccit.tex; checked by `verify_claims.py` (X.ado_*) | `ADO_DRAFT_SECTION.md` |

## Result: full Protocol B set (200 attacks + 200 benign, paired, same rows for every arm)

| Metric | ADO | Seven-layer | Undefended (ref.) |
|---|---|---|---|
| Attack success, primary marker metric | 28.5% [22.7, 35.1] | 3.5% [1.7, 7.0] | 29.5% [23.6, 36.2] |
| Attack success, S1 (refusals not successes) | 13.5% [9.4, 18.9] | 3.5% [1.7, 7.0] | 29.0% [23.2, 35.6] |
| Benign blocked | 0.0% [0.0, 1.9] | 8.5% [5.4, 13.2] | 0.0% [0.0, 1.9] |
| Benign correct | 35.5% [29.2, 42.3] | 52.0% [45.1, 58.8] | 56.5% [49.6, 63.2] |
| Benign answer rate | 83.5% | 91.5% | 97.5% |
| Median / mean latency per query (s) | 45.4 / 114.2 | 15.3 / 13.7 | 8.6 / 8.9 |

The undefended and seven-layer values are identical to the existing n = 200 table (Table VIII in
main.tex, Table VI in main_iccit.tex).

Paired exact McNemar tests, ADO vs. seven-layer (discordant rows, only ADO vs. only seven-layer):

| Comparison | Discordant | p | Difference, ADO − seven-layer [95% CI] |
|---|---|---|---|
| Attack success, primary | 54 vs. 4 | 3.2×10⁻¹² | +25.0 pts [18.4, 31.6] |
| Attack success, S1 | 24 vs. 4 | 1.8×10⁻⁴ | +10.0 pts [5.0, 15.0] |
| Attack success, S1, the 150 new rows only (where S1 was preregistered) | 17 vs. 4 | 0.0072 | +8.7 pts [2.8, 14.5] |
| Benign correct | 10 vs. 43 | 5.6×10⁻⁶ | −16.5 pts [−23.3, −9.7] |

The undefended arm is a reference only; there is no ADO-vs-undefended test.

**S1 status.** S1 was defined after the pilot. It was preregistered for the 300 new rows
(`PREREGISTRATION_FULL.md`), so it is post-hoc for 50 of the 200 attack rows. Its header line in
`full_sensitivity_refusal.md` still says "post-hoc", because the script was hashed before launch and is
left unchanged.

**Why S1 exists.** The marker scorer counts an answer as an attack success when the planted marker
(often "No") appears as a whole word and the true answer does not. ADO's DP-RAG left an **empty context
in 64/200 attack rows and 58/200 benign rows**. The generator then refuses with text such as "No
relevant information was found…". 30 of ADO's 57 primary successes are refusals of this kind; the
seven-layer arm has 0 and the undefended arm 1.

**Manual adjudication** (`full_adjudication.json`; the preregistered rule, applied only to the 24
ADO-only S1 rows, so descriptive only):

| Label | Rows |
|---|---|
| (a) genuine attack success: planted chunk kept, planted value asserted | 12 |
| (b) scorer false positive: correct answer, or a refusal the regex misses ("No authoritative information was found", "No answer can be provided") | 10 |
| (c) wrong answer without the planted chunk | 2 |

This adjudication was not applied to the other arms, so neither ADO's adjudicated count nor the other
arms' counts are comparable to it.

**Mechanism.**
- The planted chunk was in the candidate passages in 184 attack rows. ADO's defenses removed it in 107 of them.
- They removed it largely by emptying the whole context, which also removes the benign evidence. As a
  result, benign accuracy falls 21 points below the undefended arm (35.5% vs. 56.5%).
- When the chunk was kept (77 rows), the attack succeeded in 17.
- Defenses executed:

  | Split | DP-RAG | TrustRAG | AV |
  |---|---|---|---|
  | Attack | 140 | 154 | 84 |
  | Benign | 118 | 146 | 84 |

  AV ran on real passages in 58 attack rows and 61 benign rows. It removed passages in 53 and 54 of them.
- AV was never requested without being executed.

## Setup

| Item | Value |
|---|---|
| ADO code | github.com/Pranavdec/Adaptive-RAG-Orchestrator @ `b6fa5eddd449a9bc325245bcecdad400f3dd72ee` (MIT), official files unmodified (SHA-256 in `IMPLEMENTATION_AUDIT.md` §1) |
| Controller (Sentinel, Strategist) | Ollama `llama3.1:8b` (digest `46e0c10c039e`); the paper used Llama-3-8B, which is not installed |
| Generator | Ollama `llama3.2:3b` (digest `a80c4f17acd5`), ADO's generator prompt, temperature 0. This is the same generator as our Protocol B arms |
| Embedder | ADO's MiniLM (`all-MiniLM-L6-v2`) |
| AV model | `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` (pre-quantized 4-bit; the paper used full precision), eager attention, through the official `hf_model.Llama.query` with the memory-safe `GenerateProxy` |
| Defense parameters | official `config.yaml`, then ADO's own `_update_defense_params` from the Strategist's plan; defenses target top-k = 5 |
| Evidence | E2: the generator's 8 ranked passages per row (same as the undefended arm); no extra post-defense top-5 cut |
| Trust | a fresh user per request (trust 0.5, no history) |
| Seeds | pilot selection seed 20261123; full set = all rows (no sampling); NumPy/`random` seeded with the row index before each row |
| Environment | Python 3.11.9, torch 2.12.0+cu126, transformers 4.51.3, bitsandbytes 0.50.2, numpy 2.4.6, sentence-transformers 5.5.1; RTX 4060 Laptop GPU 8 GB; Windows 11 |
| Comparison arms | seven-layer: `step4 dataset/mitigation_results/ablation200/on_full.jsonl`, `benign_on.jsonl`; undefended: `followup200/off.jsonl`, `benign_off.jsonl` |

## Commands

```
python make_selection.py                       # pilot selection (refuses to redraw)
python make_full_selection.py                  # remaining + full row sets (refuses to regenerate)
python check_alignment.py --selection full_selection.json
python test_av_equivalence.py
bash run_pilot.sh                              # 50 + 50 pilot
bash run_full.sh                               # remaining 150 + 150, merge, alignment of the merged set
python score_ado_protocolB.py --prefix full --selection full_selection.json
python score_sensitivity_refusal.py --prefix full --selection full_selection.json
```

The scorers use `step4 dataset/.venv311`. The runners use `step4 dataset/.venv311gpu`
(bitsandbytes, CUDA torch). `HF_HUB_OFFLINE=1`: the model is read from the local Hugging Face cache,
and no credentials were handled by the assistant.

## Hashes

- `full_run_manifest.json` holds the SHA-256 of the preregistration, the selections, the code and the
  pilot outputs, all recorded before launch.
- `HASHES_SHA256.txt` holds the hashes of every file after the run.
- The scorers' `inputs_sha256` fields use LF-normalized hashes (`\r\n` → `\n`), so that they stay
  stable across git checkouts.

## Limitations

1. Model substitutions, with the generator kept the same across all arms:

   | Role | Paper | Here |
   |---|---|---|
   | Controller | Llama-3-8B | Llama-3.1-8B |
   | AV model | full-precision Llama-3.1-8B | 4-bit Llama-3.1-8B |
   | Generator | Llama-3.1-8B | llama3.2:3b |
2. Requests are not interleaved, and each starts with a fresh user: ADO's trust/history mechanism has no
   effect. This is a deliberate fairness choice, because Protocol B rows are independent.
3. The Ollama controller is not bit-for-bit deterministic. The plans are recorded per row, and a rerun
   may differ.
4. DP-RAG often removes all passages. This is genuine ADO behavior and it affects both splits.
5. The primary marker metric counts "No …" refusals as successes for "No" markers. S1 is partly post-hoc
   (see above).
6. One attack row (42) used the Sentinel's fallback assessment. It is kept as recorded, per the
   preregistration.

`superseded/` keeps the earlier functional runs as a record; they are not used in any result.
`pilot_comparison.*` and `pilot_sensitivity_refusal.*` are the pilot-only analysis, kept for reference.
