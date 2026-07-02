#!/usr/bin/env python3
"""Generate the adaptive framework architecture diagram as a standalone SVG.

Shows: the linear pipeline, the forward-accruing cumulative risk rail beneath it,
the tier branch (low/medium/high) into the adaptive stages, and the grounding
feedback loop back to the controller. Output is a self-contained .svg suitable for
embedding in the thesis (vector, scales cleanly in print)."""

W, H = 1180, 760
BLUE, BLUE_D = "#E6F1FB", "#0C447C"
TEAL, TEAL_D = "#E1F5EE", "#0F6E56"
AMBER, AMBER_D = "#FAEEDA", "#854F0B"
RED, RED_D = "#FCEBEB", "#A32D2D"
GRAY, GRAY_D = "#F1EFE8", "#444441"
PURPLE, PURPLE_D = "#EEEDFE", "#3C3489"
INK = "#2C2C2A"
MUT = "#5F5E5A"

def box(x, y, w, h, fill, stroke, title, sub="", tsize=15, rx=8):
    s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
    if sub:
        s += f'<text x="{x+w/2}" y="{y+h/2-4}" text-anchor="middle" font-family="Arial" font-size="{tsize}" font-weight="500" fill="{stroke}">{title}</text>'
        s += f'<text x="{x+w/2}" y="{y+h/2+15}" text-anchor="middle" font-family="Arial" font-size="11.5" fill="{stroke}">{sub}</text>'
    else:
        s += f'<text x="{x+w/2}" y="{y+h/2+5}" text-anchor="middle" font-family="Arial" font-size="{tsize}" font-weight="500" fill="{stroke}">{title}</text>'
    return s

def arrow(x1, y1, x2, y2, color=INK, w=1.6, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{w}"{d} marker-end="url(#ah)"/>'

def label(x, y, t, size=12, color=MUT, weight="400", anchor="middle"):
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Arial" font-size="{size}" font-weight="{weight}" fill="{color}">{t}</text>'

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Arial">']
svg.append(f'<defs><marker id="ah" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L7,3 L0,6 Z" fill="{INK}"/></marker>'
           f'<marker id="ahr" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L7,3 L0,6 Z" fill="{RED_D}"/></marker></defs>')
svg.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>')

svg.append(label(W/2, 34, "Adaptive Risk-Aware Security Framework for RAG", 21, INK, "500"))
svg.append(label(W/2, 56, "forward cumulative-risk accrual with a grounding feedback loop", 13, MUT))

# --- Input block (top row) ------------------------------------------------- #
y0 = 86
svg.append(label(70, y0-6, "INPUT ANALYSIS", 11, BLUE_D, "500", "start"))
xs = 40
for i,(t,sub) in enumerate([("User input",""),("Normalization",""),("Injection\ndetection","Step 3"),("Semantic\nintent","Step 3d")]):
    bx = xs + i*200
    if "\n" in t:
        a,b = t.split("\n")
        svg.append(box(bx, y0, 150, 56, BLUE, BLUE_D, a, b))
    else:
        svg.append(box(bx, y0, 150, 56, BLUE, BLUE_D, t))
    if i>0:
        svg.append(arrow(bx-50, y0+28, bx, y0+28))
# arrow to controller
svg.append(arrow(xs+3*200+150, y0+28, xs+3*200+150+40, y0+28))

# --- Adaptive Security Controller ------------------------------------------ #
cx, cy, cw, ch = 870, y0-6, 270, 78
svg.append(box(cx, cy, cw, ch, PURPLE, PURPLE_D, "Adaptive Security", "Controller  ·  selects tier", 16))

# --- Cumulative risk rail -------------------------------------------------- #
ry = 200
svg.append(f'<rect x="40" y="{ry}" width="1100" height="46" rx="8" fill="{GRAY}" stroke="{GRAY_D}" stroke-width="1.2"/>')
svg.append(label(60, ry+19, "Cumulative risk  (noisy-OR)", 12.5, GRAY_D, "500", "start"))
svg.append(label(60, ry+37, "risk = 1 \u2212 \u220f(1 \u2212 pi)   accrues forward", 11, MUT, "400", "start"))
# contribution chips feeding the rail
chips = [("prompt +", 470),("semantic +", 600),("context +", 730),("disagree +", 860),("grounding +", 990)]
for t,x in chips:
    svg.append(f'<rect x="{x}" y="{ry+9}" width="115" height="28" rx="6" fill="#FFFFFF" stroke="{GRAY_D}" stroke-width="1"/>')
    svg.append(label(x+57, ry+27, t, 11, GRAY_D))
# controller reads rail
svg.append(arrow(cx+cw/2, cy+ch, cx+cw/2, ry, PURPLE_D, 1.6, "4 3"))
svg.append(label(cx+cw/2+86, (cy+ch+ry)/2+4, "reads / writes risk", 10.5, PURPLE_D))

# --- Tier branch ----------------------------------------------------------- #
ty = 290
svg.append(arrow(cx+cw/2, ry+46, cx+cw/2, ty-4, PURPLE_D))
tiers = [("LOW", "K=8 · sim 0.55\nstd prompt · 1 LLM\ngrounding on", TEAL, TEAL_D, 120),
         ("MEDIUM", "K=5 · sim 0.65\nguarded prompt · 1 LLM\ngrounding on", AMBER, AMBER_D, 470),
         ("HIGH", "K=3 · sim 0.72 · strict prompt\nensemble + policy check\ngrounding on", RED, RED_D, 820)]
for name, body, fill, stroke, bx in tiers:
    svg.append(box(bx, ty, 300, 30, fill, stroke, name+" tier", "", 14))
    lines = body.split("\n")
    svg.append(f'<rect x="{bx}" y="{ty+34}" width="300" height="70" rx="8" fill="#FFFFFF" stroke="{stroke}" stroke-width="1.2"/>')
    for j,ln in enumerate(lines):
        svg.append(label(bx+150, ty+54+j*19, ln, 12, stroke))
    svg.append(arrow(cx+cw/2 if bx==470 else (bx+150), ty-4, bx+150, ty, PURPLE_D, 1.3))

# --- Adaptive pipeline stages (shared, parameterized by tier) -------------- #
py = 440
svg.append(label(70, py-8, "ADAPTIVE PIPELINE  (parameters set by active tier)", 11, TEAL_D, "500", "start"))
stages = [("Trust-aware\nretrieval","sim \u00d7 trust"),("Context\nsanitization","strictness"),
          ("Prompt\nconstruction","profile"),("Generation","1 / ensemble"),
          ("Grounding\nverification","faithfulness"),("Output\nfiltering / DLP","redact")]
sx = 40
for i,(t,sub) in enumerate(stages):
    bx = sx + i*188
    title = t.replace("\n", " ")
    svg.append(box(bx, py, 158, 64, TEAL, TEAL_D, title, sub, 13))
    if i>0:
        svg.append(arrow(bx-30, py+32, bx, py+32))
# tier -> pipeline
svg.append(arrow(cx+cw/2, ty+104, cx+cw/2, py-2, PURPLE_D, 1.3, "4 3"))
svg.append(label(cx+cw/2+96, (ty+104+py)/2, "applies tier config", 10.5, PURPLE_D))

# --- Final response + feedback loop ---------------------------------------- #
fy = 580
svg.append(box(40, fy, 200, 56, GRAY, GRAY_D, "Final safe", "response / refuse", 15))
svg.append(arrow(sx+5*188+79, py+64, sx+5*188+79, fy+10, INK))  # output -> down
svg.append(arrow(sx+5*188+79, fy+10, 240, fy+28, INK))           # -> final (left)

# grounding feedback loop (red, dashed) back up to controller
gx = sx + 4*188 + 79   # grounding stage center x
svg.append(f'<path d="M {gx} {py+64} C {gx} {fy+30}, {cx+cw/2} {fy+90}, {cx+cw/2} {fy+90} '
           f'L 1150 {fy+90} L 1150 {cy+ch-10} L {cx+cw} {cy+ch-10}" fill="none" stroke="{RED_D}" stroke-width="1.8" stroke-dasharray="6 4" marker-end="url(#ahr)"/>')
svg.append(label(1158, fy+50, "grounding mismatch \u2192 raise risk \u2192 re-tier / regenerate / refuse", 12, RED_D, "500", "end"))

# legend
ly = fy+110
svg.append(label(40, ly, "Feedback loop:", 12, RED_D, "500", "start"))
svg.append(label(135, ly, "a LATE signal (grounding) re-enters the controller, can push the row to a higher tier and trigger a guarded re-pass or refusal.", 12, MUT, "400", "start"))

svg.append('</svg>')

open("/mnt/user-data/outputs/adaptive-framework/architecture_diagram.svg","w").write("\n".join(svg))
print("written architecture_diagram.svg")
