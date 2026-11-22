# Row 1 / Row 4 / Row 8 external validation — mitigation OFF vs ON

This runbook wires three independently-sourced external attack datasets into
the existing pipeline and mitigation A/B harness, to show — with real,
published attack data rather than only this project's own hand-built slices —
that the system mitigates:

- **Row 1** — obfuscation + prompt injection (deepset/prompt-injections,
  already downloaded)
- **Row 4** — indirect injection + RAG poisoning, the query+context
  conjunctive construction (Hugging Face:
  `MAlmasabi/Indirect-Prompt-Injection-BIPIA-GPT`, English, real BIPIA-sourced
  attack text)
- **Row 8** — PII / data extraction (ai4privacy/pii-masking-400k, already
  downloaded)

**Everything below must run on your Windows machine with `.venv311` active
and Ollama serving** (`ollama serve`, with `llama-guard3:1b`, `llama3.2:3b`
pulled). None of it can run from the sandboxed session that wrote these
files — that environment has no Ollama, no torch/transformers, no GPU access
to your models. Run the commands here yourself; paste results back for
interpretation if you want help reading them.

---

## What changed in the codebase (and why)

1. **`dataset/preprocess_dataset.py`** — `Preprocessor.normalize_text()` now
   decodes base64-wrapped spans (new step 0, before ftfy), toggle
   `decode_base64=True` by default. This was the one obfuscation technique
   nothing in the pipeline neutralized: NFKC (already present) already
   reverses "fullwidth" Unicode obfuscation, and the existing hidden-char
   strip already removes zero-width-space obfuscation — only base64-wrap had
   no defense anywhere.

2. **`dataset/step_03c_fusion_gate.py`** — new `--normalize` flag (and
   `normalize=` param on `evaluate()`/`sweep()`/`_collect()`). **Without it,
   `--eval` scores `row["prompt"]` exactly as written — it never ran Step 2
   first**, which means any earlier obfuscated-vs-normalized comparison in
   `gate_eval_*.txt` was not actually comparing the same detector pipeline
   the live system runs. `--normalize` makes `--eval file.jsonl` (mitigation
   OFF) vs `--eval file.jsonl --normalize` (mitigation ON) a true A/B on the
   identical file.

3. **`step4 dataset/mitigation_pipeline.py`** — Step 11 (Presidio PII
   masking) was missing from the mitigation A/B harness entirely; only Step
   12's regex/Luhn/EDM DLP ran. Presidio is now wired in as `presidio_mask`
   (default on), running between grounding (Step 10) and DLP (Step 12) per
   the methodology's own stage order. Without this, Row 8 would have had
   almost no real mitigation surface to measure, since DLP alone doesn't
   catch names/addresses/usernames.

4. **`step4 dataset/run_mitigation_ab.py`** — fixed a real measurement bug:
   the `final_response` field written to each arm's output was
   `state.output`, which was only ever set once, immediately after
   generation, **before** grounding/Presidio/DLP ran. Any later redaction
   was invisible to `score_attack_success.py`'s success predicate, which
   reads `final_response` first. It now reads `state.meta["final_response"]`
   (Step 13's true user-facing text), and `mitigation_pipeline.py` keeps
   `state.output` in sync as a second line of defense. **This means any
   *redaction-only* mitigation result you produce from here on is now
   actually measured — before this fix it would silently have shown no
   effect no matter what Steps 11/12 did.**

None of your existing hand-built slices, results, or thesis numbers are
touched by these changes — `--normalize` and `presidio_mask` both default to
preserving old behaviour off (`--normalize` defaults False; `presidio_mask`
defaults True but only activates a step that previously silently no-op'd
because it was never called, so an *existing* `mitigation ON` run you have
saved would have measured the DLP-only, no-Presidio, `state.output`-bug
version — worth a note in the thesis appendix if you cite an old run
alongside a new one).

---

## Row 1 — obfuscation + prompt injection

```powershell
cd "Multivector methodology-local api extension"
.\.venv311\Scripts\Activate.ps1
python row1_obfuscation_mitigation_ab.py --n-attack 30 --n-benign 30
```

Writes `row1_mitigation_results/row1_mitigation_ab_report.json` (clean UTF-8
JSON — no console-redirect encoding issue) plus a printed summary: raw
baseline, obfuscated+mitigation-OFF, obfuscated+mitigation-ON, and a
per-technique (`zero_width` / `fullwidth` / `base64_wrap`) recall breakdown so
you can see whether base64_wrap's English wrapper sentence was doing most of
the earlier "perfect" score by itself, independent of the payload.

**Result (n=30 attack / 30 benign) — the opposite of the hypothesis above, and
worth reporting as its own finding rather than hiding it:**

| Condition | Precision | Recall | F1 | FP | FN |
|---|---|---|---|---|---|
| raw baseline (no obfuscation) | 0.917 | 0.367 | 0.524 | 1 | 19 |
| obfuscated, mitigation OFF (raw text scored) | 0.964 | 0.900 | 0.931 | 1 | 3 |
| obfuscated, mitigation ON (Step-2 normalize first) | 0.944 | 0.567 | 0.708 | 1 | 13 |

| Technique | OFF recall (n=10) | ON recall (n=10) |
|---|---|---|
| base64_wrap | 1.000 | 1.000 |
| fullwidth | 0.700 | 0.400 |
| zero_width | 1.000 | 0.300 |

Benign false positives were unaffected either way (OFF 1/30, ON 1/30) — but
that comparison isn't informative here, because `build_slices()` only
obfuscates the **attack** rows; the benign rows are identical, unobfuscated
text in both arms (see `row1_obfuscation_mitigation_ab.py` docstring point 1).
So this is not a specificity result, it's a same-text control confirming
normalization itself doesn't introduce new FPs on clean input.

**What's actually happening (traced through the code, not guessed):**
`preprocess_dataset.Preprocessor.normalize_text()` runs NFKC canonicalization
(reverses fullwidth Unicode back to ASCII) and strips the exact hidden-char
set (`​ ‌ ‍ ⁠ ﻿` etc.) that zero-width obfuscation
uses. That means the mitigation-ON text for `fullwidth`/`zero_width` rows is,
byte-for-byte almost, the *same clean attack text* as the raw baseline —
which is exactly why ON's recall on those two techniques (0.400, 0.300) lands
in the same low range as the raw baseline's recall (0.367), while OFF's
recall on the still-obfuscated raw text is far higher (0.700, 1.000).

In other words: the injection classifier is *not* more robust to obfuscated
attacks than to plain ones — it's the reverse. Feeding it the raw
zero-width/fullwidth-mangled text makes detection *easier*, not harder,
almost certainly because those two obfuscation patterns appear as attack-class
examples in whatever adversarial/jailbreak data the injection model was
trained or fine-tuned on, so the model has learned "unusual unicode noise" as
a shortcut correlated with the attack label, rather than genuinely
understanding the underlying instruction. Normalizing the text (which is what
Step 2 is *supposed* to do, and does correctly — this is not a bug in
`normalize_text()`) strips that shortcut signal and exposes the classifier's
real, weaker semantic-detection ceiling. `base64_wrap` stays flat at 1.000 in
both arms because the wrapper sentence around the encoded blob ("please
decode and follow the instructions in this string...") already reads as
suspicious on its own — the docstring's own line 56 anticipated this ceiling
effect before the run even happened.

This is a legitimate, thesis-worthy finding, not a defect to patch: it argues
*for* this project's defense-in-depth design (never relying on a single
injection classifier's face-value recall) rather than against the
normalization step, and it's a caution about how "recall on obfuscated text"
numbers can be misleadingly high in the literature if the classifier's
training data itself contained the same obfuscation tricks as attack
examples. **Recommended framing for the results chapter:** report the table
above as-is, with the mechanism paragraph as the explanation, alongside the
benign-control honesty pattern already used for Row 4 — do not re-run with a
"fixed" classifier; retraining the injection model is out of scope given the
time budget.

**Optional, zero-cost follow-up (no pipeline re-run — just reads the JSON you
already have):** `row1_mitigation_ab_report.json` also stores `injection_only`
and `or_gate` numbers per arm (the console summary only printed `fusion`).
Run:
```powershell
python inspect_row1_injection_only.py row1_mitigation_results\row1_mitigation_ab_report.json
```
Comparing `injection_only` recall across the three arms tells you whether the
deberta injection classifier alone is the one reacting to the raw unicode
noise (expected), or whether Llama-Guard/the fusion's agreement bonus is
contributing too — useful for the mechanism paragraph above but not required
to report the headline result.

---

## Row 4 — indirect injection + RAG poisoning (conjunctive)

**Data source note:** the original plan (SafeRAG only) turned out to be
Chinese-language and topically mismatched with this project's English
pipeline. A GitHub-clone of the official PoisonedRAG + BIPIA research repos
was tried next, but per a follow-up decision this was replaced with a
Hugging-Face-hosted dataset instead, so Row 4 now follows the exact same
`pip install datasets` + `load_dataset(...)` pattern already used for Row 1
(`deepset/prompt-injections`) and Row 8 (`ai4privacy`) — no git clone, no
research-repo file-layout guessing.

**One-time access step:** this dataset is "gated" on the Hub — a self-service
agreement, not a manual-review gate, so access is granted immediately, but
you do need to do this once before the download will work:
1. Log in (or create a free account) at https://huggingface.co/join
2. Visit https://huggingface.co/datasets/MAlmasabi/Indirect-Prompt-Injection-BIPIA-GPT
   and click **"Agree and access repository"**
3. Create a read-only token at https://huggingface.co/settings/tokens
4. Run `huggingface-cli login` and paste the token when prompted

```powershell
cd "multivector_extra_datasets\row4_indirect_injection_rag_poisoning"
.\..\..\.venv311\Scripts\Activate.ps1
pip install datasets huggingface_hub
huggingface-cli login
python download_row4_repos.py
```

This downloads `MAlmasabi/Indirect-Prompt-Injection-BIPIA-GPT`
(English, ~70k rows JSON: ~35k real BIPIA-sourced indirect-injection rows —
attack text embedded in retrieved-document-style context, spanning the
original BIPIA benchmark's email/web-QA/table/code domains — plus ~35k
GPT-4o-mini benign rows for a matched specificity control) and writes it to
`bipia_gpt_indirect_injection.jsonl`. Confirmed field names are `context`,
`user_intent`, `label`, `source`. `build_row4_conjunctive.py` still doesn't
hard-code them — it looks them up from a short candidate list and fails
loudly, printing the row keys it saw, if none match — so this is belt-and-
suspenders rather than a live uncertainty. SafeRAG stays in the folder as
already-downloaded reference material but is not used.
The old `PoisonedRAG/` and `BIPIA/` folders (if you ran an earlier version of
`download_row4_repos.py`) are harmless leftovers you can delete manually.

```powershell
cd "..\..\step4 dataset"
.\.venv311\Scripts\Activate.ps1
python build_row4_conjunctive.py `
    --source "../multivector_extra_datasets/row4_indirect_injection_rag_poisoning/bipia_gpt_indirect_injection.jsonl" `
    --limit 30 --n-benign 30
```

This dataset already ships each row as a (context, user_intent) pair, so
unlike the superseded PoisonedRAG/BIPIA-repo version, no carrier-question
stitching is needed. It writes three files: `row4_poisoned_context.jsonl`
(single-vector reference), `row4_multivector_conjunctive.jsonl` (the same
rows with a mild query-side suffix added — the primary query+context
conjunctive result), and `row4_benign_control.jsonl` (the dataset's own
GPT-4o-mini benign pairs, for the specificity/false-positive check —
this replaces the old `row4_bipia_compliance.jsonl`, which needed manual
review; the benign-control file here is auto-scorable).

**`--pool-limit` (poisoned_context size) now defaults to 30, matching
`--limit`.** It used to default to 200 — harmless for the build step itself
(pure in-memory JSON, instant either way), but `row4_poisoned_context.jsonl`
is scored by `run_full_pipeline.py`, which pushes every row through the
*entire* live pipeline (real Ollama generation + grounding judge per row).
At 200 rows that's a multi-hour run for what's supposed to be a smoke test.
If you already built the 200-row version and it's mid-run, **Ctrl+C it** and
re-run the `build_row4_conjunctive.py` command above (now instant at 30
rows) before re-running `run_full_pipeline.py`. Raise `--pool-limit` back up
only once you're ready to scale past the smoke test (see **Scale** below).

Then, for the **primary result** (the query+context conjunctive file):

```powershell
python run_mitigation_ab.py --slice row4_mitigation_results/row4_multivector_conjunctive.jsonl `
    --run-dir row4_mitigation_results/conjunctive --limit 30
python score_mitigation_ab.py --off row4_mitigation_results/conjunctive/off.jsonl `
    --on row4_mitigation_results/conjunctive/on_full.jsonl `
    --out row4_mitigation_results/conjunctive/report_full.json
```

**Important — `score_mitigation_ab.py`'s ASR number is not usable on this
file.** If you already ran this and saw "0/30 vs 0/30, tie", that is a
scoring blind spot, not a finding: `row4_multivector_conjunctive.jsonl` sets
`success_marker=None` (this dataset isn't a fact-flip), so the predicate
falls back to canary-leak detection — but the pipeline's canary is only
planted for this project's own synthetic slices, never for BIPIA's real
context text, so `canary_leaked` is False for every row on both arms no
matter what actually happened. This is the identical limitation the
now-superseded git-clone version's runbook already flagged for BIPIA data
("attack goal is instruction-execution, not a fact-flip, isn't
auto-scorable") — it re-applies here because the underlying attack text is
the same BIPIA source, just repackaged.

What IS measurable directly from the same `off.jsonl`/`on_full.jsonl` — no
extra pipeline run needed — is the real multi-vector detector's own verdict
and whether the pipeline actually blocked each row, both already recorded
per-row (`row["detector"]`, `row["blocked"]`). Score that instead:

```powershell
python score_row4_detection.py `
    --off row4_mitigation_results/conjunctive/off.jsonl `
    --on row4_mitigation_results/conjunctive/on_full.jsonl `
    --out row4_mitigation_results/conjunctive/detection_report.json
```

(If `off.jsonl` is empty — e.g. a later command reused the same `--run-dir`
and truncated it — just re-run the `run_mitigation_ab.py` command above
first; on your machine both arms together took under 2 minutes for 30 rows,
so this is cheap to redo.)

This reports detector-fire-rate and block-rate OFF vs ON, plus mean
query/context risk scores. It's also a genuinely interesting result in its
own right, worth its own line in the thesis: `RESULTS_multivector_
methodology (4).md` already established (with your own scanners) that the
co-activation detector is a proven negative result on **synthetically
constructed** sub-threshold pairs — a genuinely sub-threshold pair rarely
instantiates against a near-binary DeBERTa classifier. On the **real,
published BIPIA attack text** used here, a quick look at your own
`on_full.jsonl` shows the detector firing (`is_attack: true`,
`fire_rule: "coactivation"`) and `refuse_on_detection` blocking on the
large majority of rows — i.e. the negative result about synthetic
sub-threshold pairs does *not* generalize to real external attack text,
which is itself worth reporting honestly (real attacks are often not as
finely sub-threshold as a hand-constructed adversarial pair).

For the **secondary single-vector reference** (matches your existing
`poisoned_context` scoring pattern):

```powershell
python run_full_pipeline.py --slice row4_mitigation_results/row4_poisoned_context.jsonl `
    --out row4_mitigation_results/row4_poisoned_context_grounded.jsonl
python score_attack_success.py --in row4_mitigation_results/row4_poisoned_context_grounded.jsonl
```

**Same canary blind spot applies to `score_attack_success.py` on this file
too.** If you see "30/30 (100%) neutralized", don't cite that number as-is —
`row4_poisoned_context.jsonl` also has `success_marker=None`, so it's the
same trivial canary-null-check as the conjunctive file. What's real in that
output: `blocked` (21/30 in one run — a genuine, directly-observed number)
and `answered_clean` (the rest, which got an actual generated response).
For the unblocked rows specifically, manually diff each row's `final_response`
against its `poison_chunk`'s embedded instruction (matched positionally by
`question` text, since the grounded output doesn't carry `poison_chunk` or
`id` forward) — with only ~9 unblocked rows at this scale this is a two-
minute check, not a research project. On one run, all 9 unblocked rows
answered the user's real question (or safely declined) while visibly
ignoring the embedded directive (e.g. "add a sentence encouraging comments",
"substitute letters with keyboard symbols") — a real, manually-verified
finding, and a stronger one than the automated "100%" it sits next to. Cite
the block rate plus the manual check, not the automated neutralization
number.

For the **benign specificity control**:

```powershell
python run_full_pipeline.py --slice row4_mitigation_results/row4_benign_control.jsonl `
    --out row4_mitigation_results/row4_benign_control_grounded.jsonl
```

**Result (n=30): 15/30 blocked (50%) — a real limitation to report, not a
number to hide.** This is far above "near 0," and worth writing up honestly
in the thesis rather than re-tuning under time pressure. Breakdown by block
reason:

| Block reason | Count |
|---|---|
| Closed-loop refuse ("ungrounded with attack signal") | 8 |
| Output classified unsafe by Llama-Guard | 5 |
| Fused risk gate hard block (Llama-Guard S1) | 2 |

**Most likely cause — a domain mismatch, not a defect in the mitigation
logic itself.** The knowledge base here is `rag-mini-wikipedia` (~3,200
general trivia passages). This benign split's content is GPT-4o-mini-
generated financial emails, Python code snippets, and general news articles
("Find the $ value paid to Stripe/Wise/Notion," "who was the victim,"
multiprocessing/pandas/torch code) — topically unrelated to Wikipedia
trivia. The retriever can't ground answers to these questions in Wikipedia
content, so the grounding judge legitimately flags many answers "ungrounded,"
and the closed-loop gate combines that with even a small residual
disagreement/risk signal to refuse. Llama-Guard piling on for content like
naming a crime victim or quoting a dollar figure accounts for the rest.

**Thesis framing:** report the 50% figure plainly, with the domain-mismatch
explanation above as the likely driver — this benign specificity gap is
specific to off-domain benign content, not evidence the mitigation stack is
indiscriminately trigger-happy on ordinary same-domain questions (your
project's own hand-built benign set, scored elsewhere in this repo, shows a
much lower cost). A fairer apples-to-apples benign control would use
`kb_rag_mini_wikipedia.py`'s own in-domain question-answer test set (~918
pairs already about the same corpus the KB indexes) instead of this
dataset's own off-domain benign split — left as a follow-up if time allows,
not required for this smoke test.

---

## Row 8 — PII / data extraction

```powershell
cd "step4 dataset"
python build_row8_pii_extraction.py `
    --source "../row8_pii_data_leakage/ai4privacy_pii_masking_sample.jsonl" --n 30
```

```powershell
python run_mitigation_ab.py --slice row8_mitigation_results/row8_pii_extraction.jsonl `
    --run-dir row8_mitigation_results/extraction --limit 30
python score_pii_leak.py --slice row8_mitigation_results/row8_pii_extraction.jsonl `
    --off row8_mitigation_results/extraction/off.jsonl `
    --on row8_mitigation_results/extraction/on_full.jsonl `
    --out row8_mitigation_results/extraction/pii_leak_report.json
```

Repeat both commands against `row8_pii_incidental.jsonl` (swap `extraction`
for `incidental` in the paths) for the no-attacker leakage check — same PII
documents, ordinary benign question, no exfiltration instruction.

**Ablation worth running once the full result looks sane** — to attribute the
reduction to Presidio specifically rather than the whole mitigation bundle:

```powershell
python run_mitigation_ab.py --slice row8_mitigation_results/row8_pii_extraction.jsonl `
    --skip-off --no-presidio --run-dir row8_mitigation_results/extraction --label nopresidio
python score_pii_leak.py --slice row8_mitigation_results/row8_pii_extraction.jsonl `
    --off row8_mitigation_results/extraction/off.jsonl `
    --on row8_mitigation_results/extraction/on_nopresidio.jsonl `
    --out row8_mitigation_results/extraction/pii_leak_report_nopresidio.json
```

If `on_full` and `on_nopresidio` show a similar leak rate, DLP alone is doing
the work and Presidio's entity scope (`SENSITIVE_ENTITIES` in
`step_11_output_sanitization.py`) needs a look — ai4privacy's label set
(`USERNAME`, `STREET`, `DATEOFBIRTH`, `ZIPCODE`, ...) may not fully overlap
it.

---

## Scale

All three runbooks above are sized for the smoke test (n≈20–30) you asked
for. Once wiring is confirmed sane (non-degenerate OFF-arm attack rates,
no crashes, benign specificity intact), the same commands scale by raising
`--n-attack`/`--n`/`--limit`/`--pool-limit` — e.g. to n=200 to match your
existing ground-plus headline's scale — with no code changes needed. Do this
deliberately, one row at a time: anything scored via `run_full_pipeline.py`
or `run_mitigation_ab.py` (i.e. every row in every one of these files) makes
one real Ollama generation call per row, so a jump from 30 to 200 rows is
roughly a 6–7x runtime increase, not free.
