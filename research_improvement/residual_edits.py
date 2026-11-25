r"""PROPOSED follow-up edits (not applied): older phrases left after the approved edit set (manuscript_edits.py) that
still read as equivalence or as an unscoped recommendation. Same format as manuscript_edits.py.

    python residual_edits.py <scratch_dir> <tectonic.exe>            # trial on scratch copies; writes proposed_diffs/residual_*
    python residual_edits.py --apply                                  # only after the author approves
"""
import difflib
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent / "paper_revision"
R = []


def edit(f, i, why, old, new):
    R.append({"file": f, "id": i, "why": why, "old": old, "new": new})


edit("main.tex", "R1-subset-n200", "'statistically indistinguishable' reads as equivalence (brief: do not describe an insignificant difference as equivalence)",
     r"the three-layer subset sanitize+ground+refuse is statistically indistinguishable from the seven-layer stack: 5.5\% vs.\ 3.5\% ASR,",
     r"the three-layer subset sanitize+ground+refuse is not significantly different from the seven-layer stack: 5.5\% vs.\ 3.5\% ASR,")
edit("main.tex", "R2-not-equivalent", "Say what we do claim (a bound), not a softer synonym for equivalence",
     r"so we call them indistinguishable, not equivalent.",
     r"so we do not claim equivalence; the enlarged sample below bounds the gap instead.")
edit("main.tex", "R3-carry-all", "Unscoped: the class-dependence paragraph shows it does not hold for assertion-only attacks",
     r"The defensible conclusion is that three layers carry essentially all the measured injection defense.",
     r"The defensible conclusion is that, on these instruction-bearing attacks, three layers carry most of the measured injection defense; they do not suffice for assertion-only attacks (see \emph{Class dependence and wording} below).")
edit("main.tex", "R4-controlnet", "Same equivalence wording for the ControlNET comparison",
     r"its 4.5\% ASR is statistically indistinguishable from the full stack's ($p=0.77$) and from the three-layer subset's 5.5\% ($p=0.80$).",
     r"its 4.5\% ASR is not significantly different from the full stack's ($p=0.77$) or the three-layer subset's 5.5\% ($p=0.80$).")
edit("main.tex", "R5-sanitize-semantic", "Same wording, sanitization-only control on the semantic slice",
     r"(77.5\% ASR, indistinguishable from undefended)",
     r"(77.5\% ASR, not significantly different from undefended)")

edit("response_to_reviewers.md", "L1-summary-recommend", "Letter recommended the subset without the class limitation now in both papers",
     "  * A three-layer subset (sanitization, grounding, refusal on detection) is statistically\n    indistinguishable from the seven-layer stack, and we now recommend it.",
     "  * On instruction-bearing attacks, a three-layer subset (sanitization, grounding, refusal on detection)\n    is not significantly different from the seven-layer stack; the gap is at most 2.4 points at n = 424.\n    It does not suffice on assertion-only attacks, so we recommend it as a starting point and keep the\n    guarded prompt where planted factual claims matter. The layers act as near-independent filters\n    (new factorial analysis).")
edit("response_to_reviewers.md", "L2-controlnet-table", "Same equivalence wording in the letter's table",
     "| p = 0.77 (statistically indistinguishable) |",
     "| p = 0.77 (not significantly different) |")
edit("response_to_reviewers.md", "L3-stats", "Match the papers' wording: a bound, not 'indistinguishable'",
     "  three-layer subset \"statistically indistinguishable\", not \"equivalent\".",
     "  three-layer subset \"not significantly different\" and report an upper bound on the gap, not \"equivalent\".")


def _apply(text, name):
    nl = "\r\n" in text
    t = text.replace("\r\n", "\n")
    for e in (x for x in R if x["file"] == name):
        n = t.count(e["old"])
        if n != 1:
            raise SystemExit(f"{e['id']}: old text found {n} times in {name}; nothing written")
        t = t.replace(e["old"], e["new"])
    return t.replace("\n", "\r\n") if nl else t


def trial(scratch, tect):
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    for f in PAPER.iterdir():
        if f.is_file() and f.suffix in (".tex", ".bib", ".md"):
            shutil.copy2(f, scratch / f.name)
    out = HERE / "proposed_diffs"
    for name in ("main.tex", "response_to_reviewers.md"):
        orig = (scratch / name).read_bytes().decode("utf-8")
        new = _apply(orig, name)
        (scratch / name).write_bytes(new.encode("utf-8"))
        d = difflib.unified_diff(orig.replace("\r\n", "\n").splitlines(True), new.replace("\r\n", "\n").splitlines(True),
                                 fromfile=f"paper_revision/{name}", tofile=f"paper_revision/{name} (proposed)", n=0)
        (out / f"residual_{name}.diff").write_text("".join(d), encoding="utf-8")
    r = subprocess.run([tect, "-X", "compile", "main.tex"], cwd=scratch, capture_output=True, text=True)
    bad = [l for l in (r.stdout + r.stderr).splitlines() if l.startswith("error") or "Overfull" in l or "undefined" in l.lower()]
    import pymupdf
    print(f"main.tex rc={r.returncode} pages={pymupdf.open(scratch / 'main.pdf').page_count} issues={bad[:5]}")


def apply():
    new = {n: _apply((PAPER / n).read_bytes().decode("utf-8"), n) for n in ("main.tex", "response_to_reviewers.md")}
    for n, t in new.items():
        (PAPER / n).write_bytes(t.encode("utf-8"))
    print("applied", len(R), "edits")


if __name__ == "__main__":
    apply() if sys.argv[1:] == ["--apply"] else trial(Path(sys.argv[1]), sys.argv[2])
