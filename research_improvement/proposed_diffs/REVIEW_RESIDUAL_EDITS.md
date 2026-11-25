# Follow-up wording edits — review copy (NOT applied)

Source: `residual_edits.py`. Raw diffs: `residual_main.tex.diff` and `residual_response_to_reviewers.md.diff`.

**Trial results:**
- `main.tex` compiles to 17 pages with no warnings.
- `verify_claims.py` passes 86/86 on the edited text.

`main_iccit.tex` needs no change.

## R1-subset-n200 — `main.tex`
*Why:* 'statistically indistinguishable' reads as equivalence (brief: do not describe an insignificant difference as equivalence)

**Before**
```latex
the three-layer subset sanitize+ground+refuse is statistically indistinguishable from the seven-layer stack: 5.5\% vs.\ 3.5\% ASR,
```
**After**
```latex
the three-layer subset sanitize+ground+refuse is not significantly different from the seven-layer stack: 5.5\% vs.\ 3.5\% ASR,
```

## R2-not-equivalent — `main.tex`
*Why:* Say what we do claim (a bound), not a softer synonym for equivalence

**Before**
```latex
so we call them indistinguishable, not equivalent.
```
**After**
```latex
so we do not claim equivalence; the enlarged sample below bounds the gap instead.
```

## R3-carry-all — `main.tex`
*Why:* Unscoped: the class-dependence paragraph shows it does not hold for assertion-only attacks

**Before**
```latex
The defensible conclusion is that three layers carry essentially all the measured injection defense.
```
**After**
```latex
The defensible conclusion is that, on these instruction-bearing attacks, three layers carry most of the measured injection defense; they do not suffice for assertion-only attacks (see \emph{Class dependence and wording} below).
```

## R4-controlnet — `main.tex`
*Why:* Same equivalence wording for the ControlNET comparison

**Before**
```latex
its 4.5\% ASR is statistically indistinguishable from the full stack's ($p=0.77$) and from the three-layer subset's 5.5\% ($p=0.80$).
```
**After**
```latex
its 4.5\% ASR is not significantly different from the full stack's ($p=0.77$) or the three-layer subset's 5.5\% ($p=0.80$).
```

## R5-sanitize-semantic — `main.tex`
*Why:* Same wording, sanitization-only control on the semantic slice

**Before**
```latex
(77.5\% ASR, indistinguishable from undefended)
```
**After**
```latex
(77.5\% ASR, not significantly different from undefended)
```

## L1-summary-recommend — `response_to_reviewers.md`
*Why:* Letter recommended the subset without the class limitation now in both papers

**Before**
```markdown
  * A three-layer subset (sanitization, grounding, refusal on detection) is statistically
    indistinguishable from the seven-layer stack, and we now recommend it.
```
**After**
```markdown
  * On instruction-bearing attacks, a three-layer subset (sanitization, grounding, refusal on detection)
    is not significantly different from the seven-layer stack; the gap is at most 2.4 points at n = 424.
    It does not suffice on assertion-only attacks, so we recommend it as a starting point and keep the
    guarded prompt where planted factual claims matter. The layers act as near-independent filters
    (new factorial analysis).
```

## L2-controlnet-table — `response_to_reviewers.md`
*Why:* Same equivalence wording in the letter's table

**Before**
```markdown
| p = 0.77 (statistically indistinguishable) |
```
**After**
```markdown
| p = 0.77 (not significantly different) |
```

## L3-stats — `response_to_reviewers.md`
*Why:* Match the papers' wording: a bound, not 'indistinguishable'

**Before**
```markdown
  three-layer subset "statistically indistinguishable", not "equivalent".
```
**After**
```markdown
  three-layer subset "not significantly different" and report an upper bound on the gap, not "equivalent".
```
