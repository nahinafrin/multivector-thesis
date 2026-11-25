"""Apply the PROPOSED edits (manuscript_edits.py) to scratch copies, compile, and write unified diffs.

    python trial_apply.py <scratch_dir> <tectonic.exe>      # trial only; paper_revision/ is not touched
The --apply mode is intentionally not implemented: applying to paper_revision/ needs the author's approval.
"""
import difflib
import shutil
import subprocess
import sys
from pathlib import Path

from manuscript_edits import BIB, E

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent / "paper_revision"


def main():
    scratch, tect = Path(sys.argv[1]), sys.argv[2]
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    for f in PAPER.iterdir():
        if f.is_file() and f.suffix in (".tex", ".bib", ".png", ".pdf", ".svg", ".cls", ".bst") and f.name != "main.pdf":
            shutil.copy2(f, scratch / f.name)
    diffs = HERE / "proposed_diffs"
    diffs.mkdir(exist_ok=True)
    report = []
    for name in ("main_iccit.tex", "main.tex"):
        text = (scratch / name).read_text(encoding="utf-8")
        orig = text
        for e in (x for x in E if x["file"] == name):
            n = text.count(e["old"])
            if n != 1:
                raise SystemExit(f"{e['id']}: old text found {n} times in {name}")
            text = text.replace(e["old"], e["new"])
        (scratch / name).write_bytes(text.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8"))
        d = difflib.unified_diff(orig.splitlines(keepends=True), text.splitlines(keepends=True),
                                 fromfile=f"paper_revision/{name}", tofile=f"paper_revision/{name} (proposed)", n=0)
        (diffs / f"{name}.diff").write_text("".join(d), encoding="utf-8")
    (scratch / "refs.bib").write_text((scratch / "refs.bib").read_text(encoding="utf-8") + BIB, encoding="utf-8")
    (diffs / "refs.bib.additions").write_text(BIB, encoding="utf-8")
    for name in ("main_iccit.tex", "main.tex"):
        r = subprocess.run([tect, "-X", "compile", name], cwd=scratch, capture_output=True, text=True)
        log = (r.stdout + r.stderr)
        errs = [l for l in log.splitlines() if l.startswith("error") or "undefined" in l.lower()]
        import pymupdf
        pdf = scratch / name.replace(".tex", ".pdf")
        pages = pymupdf.open(pdf).page_count if pdf.exists() else None
        report.append(f"{name}: rc={r.returncode} pages={pages} issues={errs[:8]}")
    print("\n".join(report))
    (diffs / "trial_compile.txt").write_text("\n".join(report) + "\n", encoding="utf-8")


def apply():
    """Approved by the author on 2026-11-25. Applies E to paper_revision/ and appends BIB to refs.bib.
    All-or-nothing: every `old` must occur exactly once (in sequence) before anything is written."""
    new = {}
    for name in ("main_iccit.tex", "main.tex"):
        raw = (PAPER / name).read_bytes().decode("utf-8")
        text = raw.replace("\r\n", "\n")
        for e in (x for x in E if x["file"] == name):
            n = text.count(e["old"].replace("\r\n", "\n"))
            if n != 1:
                raise SystemExit(f"{e['id']}: old text found {n} times in {name}; nothing written")
            text = text.replace(e["old"], e["new"])
        new[name] = text.replace("\n", "\r\n") if "\r\n" in raw else text
    bib = (PAPER / "refs.bib").read_text(encoding="utf-8")
    if "xiang2024robustrag" in bib:
        raise SystemExit("refs.bib already has the new entries; nothing written")
    for name, text in new.items():
        (PAPER / name).write_bytes(text.encode("utf-8"))
    (PAPER / "refs.bib").write_text(bib.rstrip("\n") + "\n" + BIB, encoding="utf-8", newline="\n")
    print("applied", len(E), "edits and", BIB.count("@"), "bib entries")


if __name__ == "__main__":
    apply() if sys.argv[1:] == ["--apply"] else main()
