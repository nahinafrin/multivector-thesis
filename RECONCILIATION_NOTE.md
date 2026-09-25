# Reconciliation note — one coherent story for two things that used to read as two stories

Written to close out limitation #10 (two internal result narratives never merged) and
the §3-vs-Addendum recall discrepancy, both flagged in `multivector-thesis-analysis.md`.
This is a standalone note, not a regenerated report — `UNIFIED_RESULTS.md` is
script-generated and already carries the hand-verified headline/caveat text for
Protocol B; this file adds the missing connective narrative and the recall-convention
side-by-side, without touching that file's auto-generated JSON blocks.

## 1. The two ASR narratives, reconciled

**Narrative A ("proven negative"), `RESULTS_multivector_methodology.md`:** the conjunctive
multi-vector detector fires validly on 0/34 attack rows — a structural, proven-not-just-
observed failure (Saturation Impossibility). Zero canary leaks / 74/74 poison chunks
redacted regardless, via defense-in-depth (sanitization + canary + grounding).

**Narrative B ("the fix works"), `mitigation_run/` + `UNIFIED_RESULTS.md`:** after the
graded-margin fix, the multivector detector's fire rate is 24/30 (80%); separately, the
full pipeline's ASR on the planted200 slice drops 30.5%→16.5% (45.9% relative, McNemar
p=2.46e-7, using the ground-plus arm — the corrected headline after the marker-matching
scorer bug fix).

**These are not in tension — they measure different things, at different times, and
that's the whole point of citing both:**
- Narrative A is the *pre-fix* baseline result on the *canary-based* protocol (does the
  multivector detector itself fire?). It stays true and citable as the founding negative
  result that motivated the graded-margin fix.
- Narrative B is the *post-fix* result, on a *different* protocol (does the full
  cascade's end-to-end attack-success rate improve?), evaluated on a different slice
  (planted200, marker-based, not the 30/34-row canary slice).
- The correct thesis framing is chronological, not competing: "the detector was proven
  incapable (Narrative A) → the mechanism was fixed and revalidated on the real slice
  (0/30→24/30) → the fixed mechanism's contribution to the full cascade's ASR was
  separately measured on a larger, independent slice (Narrative B)." Present them in that
  order, explicitly dated, rather than as alternative accounts of the same claim.

## 2. Input-gate recall: report both conventions, don't pick one

| | BLOCK-only (Addenda 3/5 convention) | REVIEW-as-positive (`--review-as-red`) | §3's original figure |
|---|---|---|---|
| Precision | 0.940 | 0.905 | 0.937 |
| Recall | 0.636 | 0.753 | 0.771 |
| F1 | 0.758 | 0.822 | 0.846 |

State plainly in the thesis: §3's figure used a REVIEW-inclusive evaluation convention
(every REVIEW-flagged row counted as caught), while the deployed-gate figures in
Addenda 3/5 use a stricter BLOCK-only convention matching what the gate actually returns
to callers. REVIEW-as-positive (0.753) closes ~85% of the gap to §3's number — a
well-evidenced, mechanistically-explained match, not an exact byte-for-byte
reconciliation (a small residual 0.018 recall gap is unaccounted for and should be
stated as such). Report both conventions side by side in the Results table; do not
present either alone as "the" recall number.

## 3. What NOT to do (a real near-miss from this session, worth recording)

`build_unified_results_report.py`'s own header says "Generated ... Do not hand-edit;
re-run the script after any results file changes" — but the committed
`UNIFIED_RESULTS.md` actually carries a hand-written headline/caveat paragraph for
Protocol B (the ground-plus vs. full arm distinction, and the terse-polar-answer
provisional-cost note) that the script itself does NOT generate. Re-running the script
silently deletes that paragraph, because it only knows how to emit raw JSON blocks, not
prose narrative. If this file is ever regenerated, the Protocol B headline/caveat
paragraph must be manually re-added afterward, or (better) migrated into the script
itself as a templated section so `Do not hand-edit` becomes true again. Filed here so
the next person (or future you) doesn't lose it the way this session almost did.
