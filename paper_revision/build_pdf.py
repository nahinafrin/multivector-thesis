#!/usr/bin/env python3
"""
build_pdf.py -- render main.tex to a PDF without a LaTeX installation.

Converts this paper's LaTeX (IEEEtran subset: sections, tables, equations, citations,
the four TikZ/pgfplots figures) to a two-column HTML page, typesets math with KaTeX, and
prints it to PDF with headless Microsoft Edge (or Chrome). The figures are redrawn as SVG
from the same values as the pgfplots code. Overleaf remains the reference build.

    python build_pdf.py            # writes main.html and main.pdf next to this script
"""
from __future__ import annotations

import html
import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = HERE / "main.tex"
BIB = HERE / "refs.bib"
OUT_HTML = HERE / "main.html"
OUT_PDF = HERE / "main.pdf"

ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]


# ---------------------------------------------------------------- brace helpers
def match_brace(s: str, i: int) -> int:
    """s[i] == '{'; return index of the matching '}'."""
    depth = 0
    for j in range(i, len(s)):
        c = s[j]
        if c == "\\":
            continue
        if c == "{" and (j == 0 or s[j - 1] != "\\"):
            depth += 1
        elif c == "}" and (j == 0 or s[j - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return j
    raise ValueError("unbalanced braces near: " + s[i:i + 60])


def args(s: str, i: int, n: int):
    """Read n brace arguments starting at/after i. Returns (list, end_index)."""
    out = []
    for _ in range(n):
        while s[i] in " \n":
            i += 1
        assert s[i] == "{", s[i:i + 40]
        j = match_brace(s, i)
        out.append(s[i + 1:j])
        i = j + 1
    return out, i


def replace_cmd(s: str, name: str, fn, nargs: int = 1) -> str:
    pat = re.compile(r"\\" + name + r"(?![A-Za-z])\s*(?=\{)")
    while True:
        m = pat.search(s)
        if not m:
            return s
        a, end = args(s, m.end(), nargs)
        s = s[:m.start()] + fn(*a) + s[end:]


def split_top(s: str, sep: str):
    """Split on sep at brace depth 0, ignoring escaped separators."""
    parts, depth, cur, i = [], 0, [], 0
    while i < len(s):
        if s[i] == "\\" and s.startswith(sep, i) and sep == "\\\\":
            if depth == 0:
                parts.append("".join(cur)); cur = []; i += 2
                # swallow optional [2pt]
                m = re.match(r"\[[^\]]*\]", s[i:])
                if m:
                    i += m.end()
                continue
            cur.append(s[i:i + 2]); i += 2; continue
        if s[i] == "\\":
            cur.append(s[i:i + 2]); i += 2; continue
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
        if depth == 0 and s.startswith(sep, i):
            parts.append("".join(cur)); cur = []; i += len(sep); continue
        cur.append(s[i]); i += 1
    parts.append("".join(cur))
    return parts


# ---------------------------------------------------------------- bibliography
def parse_bib(text: str) -> dict:
    entries = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+),", text):
        typ, key = m.group(1).lower(), m.group(2).strip()
        start = text.index("{", m.start())
        body = text[m.end():match_brace(text, start)]
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*\{", body):
            j = match_brace(body, fm.end() - 1)
            fields[fm.group(1).lower()] = body[fm.end():j]
        entries[key] = (typ, fields)
    return entries


def fmt_authors(a: str) -> str:
    names = [x.strip() for x in a.split(" and ")]
    out, etal = [], False
    for n in names:
        if n == "others":
            etal = True
            continue
        if n.startswith("{") and n.endswith("}"):
            out.append(n[1:-1]); continue
        if "," in n:
            last, first = [x.strip() for x in n.split(",", 1)]
        else:
            parts = n.split(); last, first = parts[-1], " ".join(parts[:-1])
        initials = " ".join(p[0] + "." for p in re.split(r"[\s~]+", first) if p)
        out.append(f"{initials} {last}".strip())
    if etal:
        return out[0] + " <i>et al.</i>"
    if len(out) > 6:
        return out[0] + " <i>et al.</i>"
    if len(out) <= 2:
        return " and ".join(out)
    return ", ".join(out[:-1]) + ", and " + out[-1]


def bib_item(typ: str, f: dict) -> str:
    t = lambda k: inline(f.get(k, ""))
    parts = [fmt_authors(inline(f["author"])) if "author" in f else ""]
    title = f"&ldquo;{t('title')},&rdquo;"
    if typ == "inproceedings":
        s = f"{parts[0]}, {title} in <i>{t('booktitle')}</i>, {t('year')}"
    elif typ == "article":
        s = f"{parts[0]}, {title} <i>{t('journal')}</i>, vol. {t('volume')}, no. {t('number')}, pp. {t('pages')}, {t('year')}"
    else:
        hp = (", " + t("howpublished")) if "howpublished" in f else ""
        s = f"{parts[0]}, {title}{hp}, {t('year')}"
    if "doi" in f:
        s += f", doi: {t('doi')}"
    return s + "."


# ---------------------------------------------------------------- inline text
MATH_STORE: list = []


def stash_math(s: str) -> str:
    def disp(m):
        MATH_STORE.append((m.group(1), False))
        return f"\x00M{len(MATH_STORE) - 1}\x00"
    return re.sub(r"(?<!\\)\$(.+?)(?<!\\)\$", disp, s, flags=re.S)


def unstash(s: str) -> str:
    def put(m):
        tex, display = MATH_STORE[int(m.group(1))]
        cls = "mathd" if display else "math"
        return f'<span class="{cls}">{html.escape(tex)}</span>'
    return re.sub(r"\x00M(\d+)\x00", put, s)


CITE_ORDER: dict = {}
LABELS: dict = {}


def inline(s: str) -> str:
    s = s.replace("\\%", "\x01PCT").replace("\\&", "\x01AMP").replace("\\_", "_").replace("\\#", "#")
    s = s.replace("{,}", ",")
    s = html.escape(s, quote=False)
    s = s.replace("\x01PCT", "%").replace("\x01AMP", "&amp;")
    s = replace_cmd(s, "textbf", lambda a: f"<b>{a}</b>")
    s = replace_cmd(s, "emph", lambda a: f"<i>{a}</i>")
    s = replace_cmd(s, "textit", lambda a: f"<i>{a}</i>")
    s = replace_cmd(s, "texttt", lambda a: f"<code>{a}</code>")
    s = replace_cmd(s, "url", lambda a: f"<code>{a}</code>")
    s = replace_cmd(s, "makecell", lambda a: a.replace("\\\\", "<br>"))
    s = replace_cmd(s, "footnotesize", lambda a: f'<span class="fn">{a}</span>')
    s = replace_cmd(s, "label", lambda a: "")
    s = replace_cmd(s, "IEEEauthorblockN", lambda a: a)

    def cite(a):
        nums = []
        for k in a.split(","):
            k = k.strip()
            CITE_ORDER.setdefault(k, len(CITE_ORDER) + 1)
            nums.append(CITE_ORDER[k])
        return "[" + ", ".join(str(n) for n in nums) + "]"
    s = replace_cmd(s, "cite", cite)
    s = replace_cmd(s, "ref", lambda a: LABELS.get(a, "??"))
    s = re.sub(r"\\footnotesize\s*", "", s)
    s = re.sub(r"\\(small|centering|noindent)\b\s*", "", s)
    s = s.replace("\\checkmark{}", "&#10003;").replace("\\checkmark", "&#10003;")
    s = s.replace("\\\\", "<br>")
    s = s.replace("\\,", "&#8201;").replace("\\ ", " ").replace("\\quad", "&emsp;")
    s = s.replace("---", "&mdash;").replace("--", "&ndash;")
    s = s.replace("``", "&ldquo;").replace("''", "&rdquo;").replace("`", "&lsquo;")
    s = s.replace("~", "&nbsp;")
    s = s.replace("{", "").replace("}", "")
    return s


# ---------------------------------------------------------------- tables
def render_tabular(spec: str, body: str, wide: bool) -> str:
    cols = re.findall(r"p\{[^}]*\}|[lcr]", spec)
    align = []
    for c in cols:
        if c.startswith("p"):
            w = re.search(r"\{([^}]*)\}", c).group(1)
            align.append(("left", w))
        else:
            align.append(({"l": "left", "c": "center", "r": "right"}[c], None))
    rows_html, border_next, prev_tr = [], "", None
    for raw in split_top(body, "\\\\"):
        r = raw.strip()
        cls = []
        while True:
            m = re.match(r"\\(toprule|midrule|bottomrule|hline)\s*", r)
            if not m:
                break
            kind = m.group(1)
            if kind == "bottomrule" and rows_html:
                rows_html[-1] = rows_html[-1].replace("<tr", '<tr class="bot"', 1) if 'class=' not in rows_html[-1][:12] else rows_html[-1].replace('class="', 'class="bot ', 1)
            else:
                cls.append({"toprule": "top", "midrule": "mid", "hline": "mid"}[kind])
            r = r[m.end():].strip()
        if not r:
            border_next = " ".join(cls) or border_next
            continue
        allcls = " ".join(filter(None, [border_next] + cls))
        border_next = ""
        cells, tds = split_top(r, "&"), []
        ci = 0
        for cell in cells:
            cell = cell.strip()
            m = re.match(r"\\multicolumn\s*", cell)
            if m:
                (n, al, content), _ = args(cell, m.end(), 3)
                a = {"l": "left", "c": "center", "r": "right"}.get(al.strip()[0], "left")
                tds.append(f'<td colspan="{n}" style="text-align:{a}">{unstash(inline(content))}</td>')
                ci += int(n)
                continue
            a, w = align[ci] if ci < len(align) else ("left", None)
            st = f"text-align:{a}" + (f";width:{w}" if w else ";white-space:nowrap")
            tds.append(f'<td style="{st}">{unstash(inline(cell))}</td>')
            ci += 1
        cattr = f' class="{allcls}"' if allcls else ""
        rows_html.append(f"<tr{cattr}>{''.join(tds)}</tr>")
    return f'<table class="tab{" wide" if wide else ""}">{"".join(rows_html)}</table>'


# ---------------------------------------------------------------- figures (SVG)
def svg_hbar(labels, series, xmax, xlabel, label_w, height_per=22, legend=None, value_labels=False, width=340):
    n = len(labels)
    top, bottom = 8, 36
    plot_w = width - label_w - 12
    H = top + n * height_per + bottom
    sx = lambda v: label_w + plot_w * v / xmax
    out = [f'<svg viewBox="0 0 {width} {H}" xmlns="http://www.w3.org/2000/svg" font-family="Times New Roman, serif" font-size="9">']
    step = 10 if xmax <= 70 else 20
    for t in range(0, int(xmax) + 1, step):
        x = sx(t)
        out.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + n * height_per}" stroke="#ddd" stroke-width="0.6"/>')
        out.append(f'<text x="{x:.1f}" y="{top + n * height_per + 11}" text-anchor="middle">{t}</text>')
    out.append(f'<rect x="{label_w}" y="{top}" width="{plot_w}" height="{n * height_per}" fill="none" stroke="#333" stroke-width="0.7"/>')
    k = len(series)
    bw = (height_per * 0.72) / k
    for i, lab in enumerate(labels):
        yc = top + i * height_per + height_per / 2
        lines = lab if isinstance(lab, list) else [lab]
        for li, line in enumerate(lines):
            dy = (li - (len(lines) - 1) / 2) * 10
            out.append(f'<text x="{label_w - 4}" y="{yc + 3 + dy:.1f}" text-anchor="end">{html.escape(line)}</text>')
        for si, (vals, fill, stroke) in enumerate(series):
            v, plus, minus = vals[i][:3]
            y = yc - (k * bw) / 2 + si * bw
            out.append(f'<rect x="{label_w}" y="{y:.1f}" width="{max(sx(v) - label_w, 0):.1f}" height="{bw - 1:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="0.6"/>')
            ym = y + (bw - 1) / 2
            x0, x1 = sx(max(v - minus, 0)), sx(min(v + plus, xmax))
            out.append(f'<line x1="{x0:.1f}" y1="{ym:.1f}" x2="{x1:.1f}" y2="{ym:.1f}" stroke="#111" stroke-width="0.7"/>')
            for xx in (x0, x1):
                out.append(f'<line x1="{xx:.1f}" y1="{ym - 2.5:.1f}" x2="{xx:.1f}" y2="{ym + 2.5:.1f}" stroke="#111" stroke-width="0.7"/>')
            if value_labels:
                out.append(f'<text x="{x1 + 3:.1f}" y="{ym + 3:.1f}">{vals[i][3]}</text>')
    out.append(f'<text x="{label_w + plot_w / 2:.1f}" y="{H - 6}" text-anchor="middle">{xlabel}</text>')
    if legend:
        lx, ly = label_w + plot_w - 92, top + n * height_per - 30
        out.append(f'<rect x="{lx - 4}" y="{ly - 6}" width="94" height="30" fill="white" fill-opacity="0.9" stroke="#bbb" stroke-width="0.5"/>')
        for li, (name, fill, stroke) in enumerate(legend):
            out.append(f'<rect x="{lx}" y="{ly + li * 12 - 3}" width="10" height="7" fill="{fill}" stroke="{stroke}" stroke-width="0.6"/>')
            out.append(f'<text x="{lx + 14}" y="{ly + li * 12 + 3}">{name}</text>')
    out.append("</svg>")
    return "".join(out)


def fig_asr() -> str:
    labels = ["Full stack", "San+Gnd+Ref", "San+Gnd", "San+Ref", "Gnd+Ref", "Grounding only", "Sanitize only",
              "Refuse only", "Rerank only", "Presidio/DLP only", "Prompt only", "Undefended"]
    asr = [(3.5, 3.5, 1.8), (5.5, 4.1, 2.4), (8.5, 4.7, 3.1), (9.5, 4.9, 3.3), (10.5, 5.0, 3.5), (15.0, 5.6, 4.3),
           (16.0, 5.7, 4.4), (18.0, 5.9, 4.7), (24.5, 6.4, 5.4), (29.5, 6.7, 5.9), (53.0, 6.8, 6.9), (29.5, 6.7, 5.9)]
    ben = [(8.5, 4.7, 3.1), (9.5, 4.9, 3.3), (8.0, 4.6, 3.0), (2.5, 3.2, 1.4), (7.5, 4.5, 2.9), (5.5, 4.1, 2.4),
           (0.0, 1.9, 0), (2.5, 3.2, 1.4), (0.0, 1.9, 0), (0.0, 1.9, 0), (0.0, 1.9, 0), (0.0, 1.9, 0)]
    blue, orange = ("#6f8fd6", "#2f4f9e"), ("#f2a65a", "#b0641f")
    return svg_hbar(labels, [(asr, *blue), (ben, *orange)], 62, "Rate (%) with 95% Wilson CI", 78,
                    legend=[("Attack success", *blue), ("Benign blocked", *orange)])


def fig_headline() -> str:
    rows = [(["Canary leaks", "(A, 0/220)"], 0.0, 1.7, 0, "0.0"),
            (["Poison chunks redacted", "(A, 74/74)"], 100.0, 0, 4.9, "100"),
            (["ASR undefended", "(B, 59/200)"], 29.5, 6.7, 5.9, "29.5"),
            (["ASR full stack", "(B, 7/200)"], 3.5, 3.5, 1.8, "3.5"),
            (["Valid conj. fires,", "saturated (0/34)"], 0.0, 10.2, 0, "0.0"),
            (["Valid conj. fires,", "graded (24/30)"], 80.0, 10.5, 17.3, "80.0"),
            (["BIPIA blocked", "(45/50)"], 90.0, 5.7, 11.4, "90.0"),
            (["Semantic ASR,", "full stack (14/40)"], 35.0, 15.5, 12.9, "35.0"),
            (["PoisonedRAG ASR,", "undefended (137/150)"], 91.3, 3.6, 5.6, "91.3"),
            (["PoisonedRAG ASR,", "defended (136/150)"], 90.7, 3.7, 5.8, "90.7")]
    vals = [(v, p, m, t) for _, v, p, m, t in rows]
    return svg_hbar([r[0] for r in rows], [(vals, "#5fb3a9", "#2b756c")], 112,
                    "Measured rate (%) with 95% Wilson CI", 96, height_per=26, value_labels=True)


def fig_tradeoff() -> str:
    pts = [(8.6, 29.5, "Undefended"), (8.3, 18.0, "Refuse"), (10.6, 16.0, "Sanitize"), (16.4, 15.0, "Grounding"),
           (10.7, 9.5, "San+Ref"), (19.0, 5.5, "San+Gnd+Ref"), (15.3, 3.5, "Full stack")]
    W, H, L, B, T, R = 340, 200, 34, 30, 8, 10
    pw, ph = W - L - R, H - B - T
    sx = lambda x: L + pw * (x - 7) / 14
    sy = lambda y: T + ph * (1 - y / 33)
    o = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="Times New Roman, serif" font-size="9">']
    for x in range(8, 22, 2):
        o.append(f'<line x1="{sx(x):.1f}" y1="{T}" x2="{sx(x):.1f}" y2="{T + ph}" stroke="#ddd" stroke-width="0.6"/>')
        o.append(f'<text x="{sx(x):.1f}" y="{T + ph + 11}" text-anchor="middle">{x}</text>')
    for y in range(0, 31, 5):
        o.append(f'<line x1="{L}" y1="{sy(y):.1f}" x2="{L + pw}" y2="{sy(y):.1f}" stroke="#ddd" stroke-width="0.6"/>')
        o.append(f'<text x="{L - 4}" y="{sy(y) + 3:.1f}" text-anchor="end">{y}</text>')
    o.append(f'<rect x="{L}" y="{T}" width="{pw}" height="{ph}" fill="none" stroke="#333" stroke-width="0.7"/>')
    place = {"San+Gnd+Ref": ("end", -5, -6), "Full stack": ("end", -5, 3)}
    for x, y, lab in pts:
        o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.8" fill="#2f4f9e"/>')
        anchor, dx, dy = place.get(lab, ("start", 5, 3))
        o.append(f'<text x="{sx(x) + dx:.1f}" y="{sy(y) + dy:.1f}" text-anchor="{anchor}">{lab}</text>')
    o.append(f'<text x="{L + pw / 2:.1f}" y="{H - 4}" text-anchor="middle">Median latency per benign query (s)</text>')
    o.append(f'<text transform="translate(10,{T + ph / 2:.1f}) rotate(-90)" text-anchor="middle">Attack success (%)</text>')
    o.append("</svg>")
    return "".join(o)


def fig_workflow(src: str) -> str:
    boxes = []
    for m in re.finditer(r"\\node\[wfstep, fill=(\w+)![^\]]*\]\s*\(\w+\)\s*\{(.*?)\};", src):
        color = {"blue": "#e8eefb", "green": "#e6f5e6", "orange": "#fdeedd"}[m.group(1)]
        txt = m.group(2).replace("\\\\[2pt]", "\\\\")
        boxes.append(f'<div class="wfbox" style="background:{color}">{unstash(inline(stash_math(txt)))}</div>')
    return '<div class="wf">' + '<div class="wfarr">&#10148;</div>'.join(boxes) + "</div>"


# ---------------------------------------------------------------- main conversion
def main() -> None:
    tex = TEX.read_text(encoding="utf-8")
    tex = re.sub(r"(?<!\\)%.*", "", tex)
    body = tex[tex.index("\\begin{document}") + len("\\begin{document}"):tex.index("\\end{document}")]

    # ---- pass 1: numbering of labels
    sec = sub = tab = fig = eq = 0
    for m in re.finditer(r"\\(section|subsection)\{|\\begin\{(table\*?|figure\*?|equation)\}|\\label\{([^}]*)\}", body):
        if m.group(1) == "section":
            sec += 1; sub = 0; cur = ROMAN[sec]
        elif m.group(1) == "subsection":
            sub += 1; cur = f"{ROMAN[sec]}-{chr(64 + sub)}"
        elif m.group(2):
            env = m.group(2).rstrip("*")
            if env == "table":
                tab += 1; cur = ROMAN[tab]
            elif env == "figure":
                fig += 1; cur = str(fig)
            else:
                eq += 1; cur = str(eq)
        elif m.group(3):
            LABELS[m.group(3)] = cur

    title = inline(args(body, body.index("\\title") + 6, 1)[0][0])

    # ---- pass 2: blocks
    blocks = []
    pos = 0
    env_re = re.compile(r"\\begin\{(abstract|IEEEkeywords|table\*?|figure\*?|equation|enumerate)\}")
    sec = sub = tab = fig = eq = 0
    text_chunks = []

    def flush_text(t: str):
        t = re.sub(r"\\(maketitle|bibliographystyle\{[^}]*\}|bibliography\{[^}]*\})", "", t)
        t = re.sub(r"\\title\{.*?\}\s*\n", "", t, count=1)
        t = re.sub(r"\\author\{.*\}\s*\n", "", t, count=1)
        nonlocal sec, sub
        parts = re.split(r"(\\(?:section|subsection)\*?\{)", t)
        i = 0
        out = []
        while i < len(parts):
            p = parts[i]
            if p in ("\\section{", "\\subsection{", "\\section*{", "\\subsection*{"):
                rest = parts[i + 1]
                j = match_brace("{" + rest, 0) - 1
                head, after = rest[:j], rest[j + 1:]
                head = inline(stash_math(head))
                if p.endswith("*{"):
                    out.append(f'<h2>{unstash(head)}</h2>')
                elif p == "\\section{":
                    sec += 1; sub = 0
                    out.append(f'<h2>{ROMAN[sec]}. {unstash(head)}</h2>')
                else:
                    sub += 1
                    out.append(f'<h3>{chr(64 + sub)}. {unstash(head)}</h3>')
                parts[i + 1] = after
                i += 1
                continue
            for para in re.split(r"\n\s*\n", p):
                para = para.strip()
                if not para:
                    continue
                # \emph{..} paragraphs separated by single newlines in this file start new paragraphs
                for line in re.split(r"\n(?=\\emph\{|\\textbf\{)", para):
                    line = re.sub(r"\\label\{[^}]*\}", "", line).strip()
                    if line:
                        out.append(f"<p>{unstash(inline(stash_math(line)))}</p>")
            i += 1
        blocks.append("".join(out))

    for m in env_re.finditer(body):
        if m.start() < pos:
            continue
        flush_text(body[pos:m.start()])
        env = m.group(1)
        end_tag = "\\end{" + env + "}"
        end = body.index(end_tag, m.end())
        content = body[m.end():end]
        pos = end + len(end_tag)
        if env == "abstract":
            blocks.append(f'<div class="abstract"><b><i>Abstract</i></b>&mdash;{unstash(inline(stash_math(content.strip())))}</div>')
        elif env == "IEEEkeywords":
            blocks.append(f'<div class="abstract"><b><i>Index Terms</i></b>&mdash;{unstash(inline(stash_math(content.strip())))}</div>')
        elif env == "equation":
            eq += 1
            tex_eq = re.sub(r"\\label\{[^}]*\}", "", content).strip()
            MATH_STORE.append((tex_eq, True))
            blocks.append(f'<div class="eq"><span class="mathd">{html.escape(tex_eq)}</span><span class="eqn">({eq})</span></div>')
        elif env == "enumerate":
            items = re.split(r"\\item", content)[1:]
            lis = []
            for it in items:
                mm = re.match(r"\s*\[([^\]]*)\]", it)
                lab = mm.group(1) if mm else ""
                it = it[mm.end():] if mm else it
                lis.append(f'<li><span class="lab">{lab}</span> {unstash(inline(stash_math(it.strip())))}</li>')
            blocks.append(f'<ul class="enum">{"".join(lis)}</ul>')
        elif env.startswith("table"):
            tab += 1
            wide = env.endswith("*")
            cap = args(content, content.index("\\caption") + 8, 1)[0][0]
            tm = re.search(r"\\begin\{tabular\}", content)
            spec, after = args(content, tm.end(), 1)
            tb = content[after:content.index("\\end{tabular}")]
            small = "\\small" in content
            table = render_tabular(spec[0], stash_math(tb), wide)
            blocks.append(f'<div class="float{" span" if wide else ""}{" small" if small else ""}">'
                          f'<div class="tcap">TABLE {ROMAN[tab]}<br><span class="sc">{unstash(inline(stash_math(cap)))}</span></div>{table}</div>')
        else:
            fig += 1
            wide = env.endswith("*")
            cap = args(content, content.index("\\caption") + 8, 1)[0][0]
            lab = re.search(r"\\label\{([^}]*)\}", content).group(1)
            svg = {"fig:workflow": lambda: fig_workflow(content), "fig:asr": fig_asr,
                   "fig:tradeoff": fig_tradeoff, "fig:headline": fig_headline}[lab]()
            blocks.append(f'<div class="float fig{" span" if wide else ""}">{svg}'
                          f'<div class="fcap">Fig. {fig}. {unstash(inline(stash_math(cap)))}</div></div>')
    flush_text(body[pos:])

    # ---- references
    bib = parse_bib(BIB.read_text(encoding="utf-8"))
    refs = []
    for key, n in sorted(CITE_ORDER.items(), key=lambda kv: kv[1]):
        if key in bib:
            refs.append(f'<li><span class="rn">[{n}]</span> {unstash(bib_item(*bib[key]))}</li>')
        else:
            refs.append(f'<li><span class="rn">[{n}]</span> ?? missing bib entry {key}</li>')
    blocks.append('<h2 class="refs">References</h2><ol class="refs">' + "".join(refs) + "</ol>")

    doc = TEMPLATE.replace("%TITLE%", title).replace("%BODY%", "\n".join(blocks))
    OUT_HTML.write_text(doc, encoding="utf-8")
    print(f"[html] {OUT_HTML}  ({len(CITE_ORDER)} citations, {tab} tables, {fig} figures, {eq} equations)")
    missing = [k for k in CITE_ORDER if k not in bib]
    if missing:
        print("[warn] missing bib entries:", missing)

    browser = next((p for p in [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                                r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                                shutil.which("msedge") or "", shutil.which("chrome") or ""] if p and Path(p).exists()), None)
    if not browser:
        print("[warn] no Edge/Chrome found; open main.html and print to PDF")
        return
    import tempfile
    profile = tempfile.mkdtemp(prefix="pdfprofile_")
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--no-first-run",
                    f"--user-data-dir={profile}",
                    "--virtual-time-budget=20000", f"--print-to-pdf={OUT_PDF}", OUT_HTML.as_uri()],
                   check=True, capture_output=True, timeout=180)
    shutil.rmtree(profile, ignore_errors=True)
    if not OUT_PDF.exists():
        raise SystemExit("[fail] browser did not write the PDF")
    print(f"[pdf ] {OUT_PDF}")


TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>%TITLE%</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<style>
@page { size: letter; margin: 0.75in 0.62in 0.9in 0.62in; }
html { font-family: "Times New Roman", Times, serif; font-size: 10pt; line-height: 1.18; color: #000; background: #fff; }
body { margin: 0; }
h1 { font-size: 22pt; font-weight: normal; text-align: center; line-height: 1.15; margin: 0 0 10pt; }
.author { text-align: center; font-size: 11pt; margin-bottom: 14pt; }
.cols { column-count: 2; column-gap: 0.24in; text-align: justify; hyphens: auto; }
p { margin: 0; text-indent: 1em; }
h2 + p, h3 + p, .abstract + p, .eq + p { text-indent: 0; }
h2 { font-size: 10pt; font-weight: normal; font-variant: small-caps; text-align: center; margin: 10pt 0 5pt; letter-spacing: 0.02em; break-after: avoid; }
h3 { font-size: 10pt; font-weight: normal; font-style: italic; margin: 7pt 0 3pt; break-after: avoid; }
.abstract { font-size: 9pt; font-weight: bold; margin-bottom: 6pt; text-align: justify; }
.abstract b { font-weight: bold; }
code { font-family: "Courier New", monospace; font-size: 0.92em; }
.eq { display: flex; align-items: center; justify-content: center; margin: 5pt 0; position: relative; break-inside: avoid; }
.eqn { position: absolute; right: 0; }
.enum { list-style: none; padding-left: 0.6em; margin: 3pt 0; }
.enum li { margin: 2pt 0; } .enum .lab { margin-right: 0.3em; }
.float { break-inside: avoid; margin: 8pt 0; }
.float.span { column-span: all; margin: 10pt 0; }
.tcap { text-align: center; font-size: 8pt; margin-bottom: 4pt; line-height: 1.25; }
.tcap .sc { font-variant: small-caps; }
.fcap { font-size: 8pt; margin-top: 4pt; text-align: justify; }
table.tab { border-collapse: collapse; margin: 0 auto; font-size: 8pt; line-height: 1.15; }
.float.small table.tab { font-size: 7.4pt; }
table.tab td[colspan] { white-space: normal; }
table.tab td { padding: 1.2pt 3pt; vertical-align: top; text-align: left; hyphens: none; }
table.tab tr.top td { border-top: 1pt solid #000; }
table.tab tr.mid td { border-top: 0.5pt solid #000; }
table.tab tr.bot td { border-bottom: 1pt solid #000; }
table.tab tr:first-child td { border-top: 1pt solid #000; }
.fn { font-size: 7pt; }
.fig svg { width: 100%; height: auto; display: block; }
.wf { display: flex; align-items: center; justify-content: center; gap: 2pt; }
.wfbox { border: 0.8pt solid #333; border-radius: 3pt; padding: 4pt; width: 15%; min-height: 62pt; font-size: 8pt; text-align: center; line-height: 1.2; }
.wfarr { font-size: 11pt; }
h2.refs { margin-top: 12pt; }
ol.refs { list-style: none; padding: 0; margin: 0; font-size: 8pt; text-align: left; }
ol.refs li { position: relative; padding-left: 2.6em; margin-bottom: 2.5pt; }
ol.refs .rn { position: absolute; left: 0; }
.katex { font-size: 1.0em; }
</style></head><body>
<h1>%TITLE%</h1>
<div class="author">Anonymous Authors</div>
<div class="cols">
%BODY%
</div>
<script>
document.querySelectorAll('.math, .mathd').forEach(function (el) {
  try { katex.render(el.textContent, el, {throwOnError: false, displayMode: el.classList.contains('mathd')}); }
  catch (e) { el.style.color = 'red'; }
});
</script>
</body></html>
"""

if __name__ == "__main__":
    main()
