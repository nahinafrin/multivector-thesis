> **2026-09-23 update:** the CWG real-dataset validation results below (§6) have now
> been written up into the project docs as `multivector-thesis-analysis.md` Addendum 7
> and `novelty-contributions-draft.md` §4 — those are the citable, deadline-day versions.
> This file remains the working notes; treat the project docs as authoritative for the
> submission.

# Integration & paper-writing notes

Four files, each addressing one specific gap identified against SPARK's methodology.
None of these were written against your live repository (no device link is active in
this session), so they are clean-room reference implementations built strictly from
the interfaces and behaviour already documented in your own project files
(`multivector-thesis-analysis.md`, `novelty-contributions-draft.md`). Treat them as
drop-in-ready scaffolding you adapt to your real function/field names, not as a
verified patch — run each one's built-in self-test/demo first (instructions below),
then wire in your real data.

---

## 1. `calibrate_threshold_valonly.py` — fixes the test-set leakage gap

**The problem it fixes:** your τ_block=0.45 recommendation (Addendum 3) was chosen by
evaluating candidates directly on the same 1,193-row test set the final recall/F1 is
reported on. SPARK never does this — every statistic is computed on train/val only.

**What to do:**
1. Add one line to `step_03c_fusion_gate.py`'s evaluation path that dumps a per-row
   `{"id":..., "score": R, "label": 0/1}` JSONL for the **validation** split (you
   likely already compute `R` per row; this just persists it to a file).
2. Do the same for the **test** split if you don't already have `test_scores.jsonl`.
3. Run:
   ```
   python calibrate_threshold_valonly.py --val val_scores.jsonl --test test_scores.jsonl \
       --min-precision 0.90
   ```
4. Use the printed VAL-selected τ_block and the TEST-only metrics as your new,
   leakage-free headline number. Cite it with the sentence the script prints.

Already tested against synthetic data in this session — runs correctly out of the box.

---

## 2. `tgec_controller.py` — the missing "retry vs. triad" control

**The problem it fixes:** you cannot currently tell whether TGEC's benefit comes from
the **triad signal** (d, f, c) or from the **retry mechanism** itself (a second
generation attempt). SPARK never lets an effect stand until a control has isolated it
(SCARF-2x, BE-SCARF, Random KAN); this is your missing fourth control.

**What to do:**
1. Read the class — it's a faithful implementation of the TGEC state machine your
   own docs already formally specify, with one new constructor flag: `allow_retry`.
2. Run `python tgec_controller.py` — it reproduces Addendum 4's three real rows
   (hard-veto, escalate-and-pass) exactly, plus a fourth synthetic row showing the
   new ablation branch (`allow_retry=False` → immediate REFUSE instead of retry).
3. Lowest-risk integration: don't touch your existing controller. Add ONE new arm,
   `triad_no_retry`, to `run_mitigation_ab.py`'s existing arm list, using this class
   configured with `allow_retry=False` for that arm only. Run it on the same 200-row
   slice as your other five arms.
4. Add the new arm as a sixth bar in Fig 5.5 (the ASR chart). The gap between `full`
   and `triad_no_retry` is the retry mechanism's own contribution — write this up
   explicitly in Chapter 5 as answering the SPARK-style attribution question your
   thesis currently leaves open.

---

## 3. Scaling the multivector slice past n=30 (no new code needed)

Addendum 6 already diagnosed the exact blocker: PowerShell's argument tokenization
corrupts the `scale_and_merge_slice.py` wrapper's nested `--builder-args "..."`
string. The underlying builder works fine called directly. From `step4 dataset`,
with `.venv311` active:

```
python build_adversarial_slice.py --n-multivector 100 --out multivector_new.jsonl
python -c "
import json
seen = set(json.loads(l)['prompt'] for l in open('row8_mitigation_results/../adversarial_slice.jsonl') if l.strip())
with open('multivector_new.jsonl') as f, open('multivector_scaled.jsonl','w') as out:
    for line in f:
        row = json.loads(line)
        if row['prompt'] not in seen:
            out.write(line)
"
```

No nested quoting, no wrapper — this sidesteps the tooling issue entirely rather than
fighting PowerShell's quoting again. Merge the result into your existing 30-row
`multivector_attack` slice and re-run `run_mitigation_ab.py` / your fire-rate check
on the combined set. This is the single cheapest rigor upgrade available to you.

---

## 4. `margin_preserving_fusion.py` — Novel Algorithm 1 (Margin-Preserving Conjunctive Fusion)

Formalises your already-validated fix (reading the pre-sigmoid margin instead of the
squashed probability) as a small, **learned** logistic model with an explicit
conjunctive interaction term — see the equations in the file's docstring (Eq. 1–6).
This is the direct analogue of SPARK's move from a fixed scalar weight to a learnable
spline: instead of hand-combining the two margins, you fit the combination.

**What to do:**
1. Extract `z_q`, `z_c` (the pre-sigmoid margins you already read for the current
   fix) for a batch of labelled benign + multivector_attack rows into
   `calib.jsonl` — your existing 210-row slice (or the scaled version from step 3)
   is a fine calibration set.
2. `python margin_preserving_fusion.py fit --data calib.jsonl --out mpcf_params.json`
3. Report the fitted `gamma` (the interaction weight) in the paper. Also fit with
   `gamma` fixed at 0 (comment out that term, or just note it in the ablation table)
   and compare fire rates — this is your depth-ablation-style sanity check showing
   the conjunctive term is pulling real weight, not decoration.
4. Use `score_mpcf` in place of your current hand-set combination function, with
   thresholds calibrated the same val-only way as file 1.

Already tested against synthetic data in this session — the fit and scoring both run
correctly.

---

## 5. `corroboration_weighted_groundedness.py` — Novel Algorithm 2 (Corroboration-Weighted Groundedness)

Directly targets your one fully-open limitation (semantic-plausibility blindness /
root cause II). See the file's docstring for the full derivation (Eq. 1–5): it adds a
trust-weighted, noisy-OR corroboration requirement on top of your existing
faithfulness gate, so a single fabricated-but-fluent chunk from an untrusted source
can no longer pass on fluency alone.

**What to do:**
1. Assign a static trust prior per source in your corpus (e.g. 1.0 for the verified
   primary KB, lower for anything less verified) — this is the "external
   provenance" signal your own docs already say is the correct fix, now made precise.
2. Run `python corroboration_weighted_groundedness.py demo` — it shows the intended
   behaviour on three cases modelled directly on your row-201 finding (a single
   fabricated, untrusted chunk gets blocked; a genuine trusted chunk and a
   two-source corroborated case both pass).
3. Calibrate `phi_min`, `theta_t`, `m_min` on a labelled validation set (planted
   misinformation vs. benign rows) with `grid-search`, subject to a benign
   false-block ceiling — never on test.
4. Wire `cwg_passed(...)` into `step_10_grounding_judge.py` as an additional gate
   alongside (not replacing) your existing `passed = max(lexical, model) >= threshold`
   check, then re-run the 8-row misinformation slice to see whether the 4/8 blind
   spot narrows. Even a partial improvement moves limitation #2 from "diagnosed,
   untouched" to "attempted with a real result" — which is the credit SPARK gets for
   testing every control it names.

---

## 6. Real-dataset train/calibrate + held-out smoke test for CWG

Answers the specific ask: calibrate CWG on one real, citable dataset, then smoke-test
it unchanged on a **different** real dataset, so the reported number isn't calibrated
and validated on the same data.

**Files:**
- `phi_proxy.py` — a TF-IDF cosine-similarity stand-in for your real cross-encoder
  faithfulness judge (`model_faithfulness`). Stateless, dependency-light (sklearn
  only), swap-compatible: replace `phi_score(claim, evidence)` with a call into your
  real judge and nothing downstream changes.
- `train_calibrate_cwg_climate_fever.py` — downloads **Climate-FEVER** (Diggelmann
  et al. 2020, arXiv:2012.00614; 1,535 real climate claims, each with 5 human-annotated
  Wikipedia evidence sentences), builds per-claim `(phis, trusts, lexical_overlap,
  label)` features, grid-searches `(phi_min, theta_t, m_min)` on an 80% TRAIN split
  only, then reports the result ONCE on the 20% held-out VAL split, then saves the
  winning params to `cwg_calibrated_params.json`.
- `smoke_test_cwg_liar_plus.py` — downloads **LIAR-PLUS** (Alhindi et al. 2018, EMNLP
  FEVER workshop; PolitiFact-checked US political statements, one justification
  passage per statement), loads the frozen Climate-FEVER params, and evaluates
  baseline-vs-CWG on this completely different dataset **without re-tuning anything**.
  The script refuses to run if `cwg_calibrated_params.json` is missing, specifically
  so nobody accidentally calibrates on the "held-out" set.

**Note on data source:** this environment's network policy blocks both
`huggingface.co` and `kaggle.com` outright (confirmed via the proxy status endpoint —
a policy denial, not a transient failure). Both datasets were instead pulled from
their own original GitHub-hosted releases (`tdiggelm/climate-fever-dataset`,
`Tariq60/LIAR-PLUS`) — identical data and citation, just a different host. On your own
machine, if it can reach `huggingface.co`, `datasets.load_dataset("tdiggelm/climate_fever")`
gives the same rows and you can skip the download step.

**Actual results obtained this session (report these numbers as-is, with the caveats
below — do not round up):**

*Climate-FEVER, held-out VAL split (n=307), calibrated on TRAIN only:*

| | Baseline | CWG |
|---|---|---|
| should-block rows blocked | 12/186 (6.5%) | 36/186 (19.4%) |
| benign rows wrongly blocked | 4/121 (3.3%) | 14/121 (11.6%) |

*LIAR-PLUS, held-out smoke test (n=1,258), Climate-FEVER params frozen/unchanged:*

| | Baseline | CWG |
|---|---|---|
| should-block rows blocked | 393/812 (48.4%) | 635/812 (78.2%) |
| benign rows wrongly blocked | 205/446 (46.0%) | 329/446 (73.8%) |

**Honest mechanistic explanation for the paper's limitations section (this matters —
don't just report the LIAR-PLUS numbers as "CWG improves catch rate 30pp," because the
benign false-block rate rose by almost exactly the same amount, which is a different
and much less flattering story than the Climate-FEVER result):**

1. LIAR-PLUS gives CWG only **k=1** context chunk (one justification passage per
   statement), never k≥2. With trust uniform at 1.0 and a single chunk, the noisy-OR
   trust-weighted-support term `T` reduces algebraically to `phi` itself (proven and
   already flagged in `test_cwg_smoke.py`'s Case 3/4 discussion), and the
   `n_corr >= m_min` branch is structurally unreachable when `m_min=2` but only one
   chunk ever exists. So on this dataset CWG collapses to: `baseline AND phi >= 0.30`
   — i.e. it is not doing corroboration-weighting at all here, it is simply raising
   the effective faithfulness bar from `theta_f=0.08` to `theta_t=0.30`.
2. That collapsed rule inflates *both* the catch rate and the false-block rate by
   roughly the same amount, because it isn't discriminating attacks from benign
   rows on a new axis — it's just moving one threshold up on the same TF-IDF
   proxy score. That the two numbers rose together (not the catch rate alone) is
   itself the tell.
3. Separately, the phi-score distribution on LIAR-PLUS (median 0.06, 25th
   percentile exactly 0.0) is shaped very differently from Climate-FEVER's — political
   justification paragraphs discuss context, quotes, and other people's statements
   rather than restating the claim's own vocabulary, so lexical overlap is
   structurally lower and noisier here regardless of truth value.

**What to write in the thesis, honestly:** CWG's validated, real improvement is the
Climate-FEVER multi-chunk result (6.5%→19.4% catch rate at a 3.3%→11.6% false-block
cost) — cite that as primary evidence. Then report the LIAR-PLUS smoke test as a
genuine, informative negative/boundary result: CWG's corroboration mechanism requires
k≥2 retrieved chunks to do anything beyond a single-threshold shift, and thresholds
calibrated on one domain/context-shape do not transfer cleanly to a domain with a
different chunk count and a different lexical relationship between claim and evidence.
This is exactly the kind of stated, tested boundary-of-validity SPARK gets credit for
— do not omit it or only report the flattering half.

**Two things to do before trusting these numbers further:** (1) swap `phi_score` in
`phi_proxy.py` for your real cross-encoder judge — TF-IDF cosine is a lexical proxy
and under-detects paraphrased contradictions; (2) if you want a fairer LIAR-PLUS test
of the corroboration branch specifically (not just the threshold-shift artifact above),
retrieve k≥2 chunks per statement there too (e.g. treat `context` + `justification` as
two separate chunks) rather than using the single `justification` field alone.

---

## Where each of these goes in the paper

- New Methodology subsections: "4.x Margin-Preserving Conjunctive Fusion" and
  "4.y Corroboration-Weighted Groundedness," each with its own numbered equations
  (copy the derivations from the two files' docstrings — they're written in
  paper-ready prose already).
- New Results subsections: the `triad_no_retry` ablation bar (Fig 5.5, sixth bar),
  the leakage-free τ_block number (replaces the Addendum 3 table), and — if you run
  the CWG re-test — a before/after on the 4/8 misinformation catch rate.
- Update the Conclusion's "Limitations and Boundaries of Validity": limitation #2 (if
  CWG improves it, even partially) and the test-set leakage item move from "open" to
  "addressed," and add the retry-vs-triad attribution as a newly-answered question.
