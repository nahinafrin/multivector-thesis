"""
supersession_cue.py — a NEW, dependency-free per-chunk detector targeting the
"temporal-supersession" poisoning style: a planted false fact framed as a
correction, update, or revision of earlier/original information.

WHY THIS EXISTS (read before touching the regex)
-------------------------------------------------
The fresh-attack evaluation (2026-09-24, LLM-paraphrased attacks unseen by
every system) found that the deployed multi-vector detector's context channel
(context_injection / context_graded, both derived from a PROMPT-INJECTION
classifier — see step_06_context_sanitization.py, graded_channels.py) fires
on only ~39% of fresh poisoned rows, down from ~100% on the old templated
slice. Root cause: the fresh poison chunks contain no injection signature at
all ("ignore previous instructions", "SYSTEM:", etc.) — they read as a plain,
confident statement of fact. A prompt-injection classifier is simply the
wrong tool for that; no amount of threshold recalibration on it closes this
gap (see the project's own analysis addenda on why re-tuning an
injection-style channel on a differently-styled attack does not generalize).

WHAT THIS DETECTS INSTEAD
--------------------------
Inspecting the 9 fresh multivector rows where the planted fact actually
leaked into the final answer showed a consistent, DIFFERENT rhetorical
pattern across all 9: the poison chunk does not try to look like an
instruction, it tries to look like a CORRECTION — "Corrected educational
content...", "Further research has revealed...", "UPDATE: ... replacing the
previously accepted response", "Historical records indicate ... was
indeed...", "the correct answer is now confirmed to be...". The generated
answers echo this framing almost verbatim ("the most recent/up-to-date
information indicates...", "however, further research has revealed...").

This is a real, learnable, ATTACK-INTRINSIC signature: a chunk that plants a
false fact needs some rhetorical device to make a reader/model prefer it over
the (also-present) correct chunk, and "I am the newer, corrected version"
is a natural, cheap way to do that — independent of whether the false content
itself is worded like an instruction.

VALIDATION (do not skip this section when re-tuning the regex)
----------------------------------------------------------------
- Fresh slice (106 rows, LLM-paraphrased, nothing tuned on it): matches the
  poison chunk in 9/9 of the rows that actually leaked, and 21/27 fresh
  multivector poison chunks overall. False-positive rate: 6/40 benign rows
  have >=1 raw chunk that matches, but in every one of those 6 rows the
  matched chunk does NOT hold the gold answer (hand-checked against
  ground_truth) — so dropping it costs nothing in practice on this slice.
- Old templated slice (mv130, 149 poison rows, "editorial note" /
  "authoritative sources" / "SYSTEM: disregard" style — a DIFFERENT template
  family this regex was not designed from): matches only 6/149 poison chunks
  (that style is already caught by the existing injection classifier instead)
  but DOES catch row 201 ("Footnote: the corrected figure is..."), the one
  confirmed real leak on that slice. Benign false-positive rate: 2/80.
- This is therefore a genuinely ADDITIVE, orthogonal channel, not a
  replacement for the injection classifier: the two together cover more
  framings than either alone, and neither was tuned on the other's slice.

HONEST SCOPE BOUNDARY — state this in the writeup, do not overclaim
---------------------------------------------------------------------
This catches poison that frames itself as a correction/update/revision. It
will NOT catch a planted false fact stated as a plain, unframed assertion
with no supersession language at all (e.g. just "Ducks filter water using
microscopic filaments." with no "corrected"/"updated" wrapper) — that
residual case is not closed by this module, and remains the same open
"claim-level corroboration" problem already named as future work (CWG,
Addendum 12). Re-validate this cue list (the same way: fresh unseen attacks,
check the false-positive rate on benign chunks, check it isn't cherry-picked
from the one slice that motivated it) before trusting a future edit to it.
"""
from __future__ import annotations

import re

# Matches: "correct(ed/ing/ion)", "update(d/s)", "revis(ed/ion)", "supersede*",
# "most/more recent", "further research", "re-evaluat*", "latest
# information/data/update", "previously reported/cited/accepted/stated",
# "new(ly) confirmed/discovered", "accurate explanation/answer/information",
# "historical records indicate/show", "now confirmed", "has (been) revealed",
# "amended", "clarif(y/ied/ication)".
SUPERSESSION_CUES = re.compile(
    r"\b(correct(ed|ing|ion)?|update[ds]?|revis(ed|ion)|supersed\w*|"
    r"most recent|more recent|further research|re-?evaluat\w*|"
    r"latest (information|data|update)|"
    r"previously (reported|cited|accepted|stated)|"
    r"new(ly)? (confirmed|discovered)|"
    r"accurate (explanation|answer|information)|"
    r"historical records (indicate|show)|"
    r"now confirmed|has (been )?revealed|amended|clarif(y|ied|ication))\b",
    re.IGNORECASE,
)


def supersession_score(text: str) -> float:
    """1.0 if the chunk contains a temporal-supersession cue, else 0.0.

    Boolean-shaped on purpose: unlike the injection classifier's probability,
    there is no calibrated intermediate meaning to a "partial" match here, and
    keeping it boolean makes the false-positive rate in the docstring above
    directly auditable (it IS the rate reported).
    """
    return 1.0 if SUPERSESSION_CUES.search(text or "") else 0.0


def has_supersession_cue(text: str) -> bool:
    return bool(SUPERSESSION_CUES.search(text or ""))
