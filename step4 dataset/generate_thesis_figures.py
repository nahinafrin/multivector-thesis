#!/usr/bin/env python3
"""Generate thesis figures from existing JSON result artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "thesis_figures"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def svg_text(x: float, y: float, text: str, size: int = 14, anchor: str = "start", weight: str = "400") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="#203040">{escape(str(text))}</text>'


def svg_start(title: str, width: int = 1100, height: int = 650) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        svg_text(width / 2, 42, title, 24, "middle", "700"),
    ]


def save(name: str, lines: list[str]) -> None:
    lines.append("</svg>")
    (OUT / name).write_text("\n".join(lines), encoding="utf-8")


def bar_chart(name: str, title: str, labels: list[str], series: list[tuple[str, list[float], str]], ymax: float, suffix: str = "%") -> None:
    width, height = 1100, 650
    left, right, top, bottom = 100, 40, 90, 105
    plot_w, plot_h = width - left - right, height - top - bottom
    lines = svg_start(title, width, height)
    for tick in range(0, int(ymax) + 1, 10):
        y = top + plot_h - plot_h * tick / ymax
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#dbe4ec"/>')
        lines.append(svg_text(left - 12, y + 5, f"{tick}{suffix}", 12, "end"))
    group_w = plot_w / len(labels)
    bar_w = min(42, group_w / (len(series) + 1))
    for i, label in enumerate(labels):
        cx = left + group_w * (i + 0.5)
        lines.append(svg_text(cx, height - 55, label, 12, "middle"))
        for j, (_, values, color) in enumerate(series):
            value = values[i]
            x = cx + (j - (len(series)-1)/2) * (bar_w + 5) - bar_w/2
            h = plot_h * value / ymax
            y = top + plot_h - h
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="3" fill="{color}"/>')
            lines.append(svg_text(x + bar_w/2, y - 7, f"{value:g}", 11, "middle", "700"))
    for j, (label, _, color) in enumerate(series):
        x = left + 20 + j * 190
        lines.append(f'<rect x="{x}" y="{height-28}" width="14" height="14" fill="{color}"/>')
        lines.append(svg_text(x + 21, height - 16, label, 12))
    save(name, lines)


def scatter(name: str, rows: list[tuple[str, float, float, str]]) -> None:
    width, height = 1000, 680
    left, right, top, bottom = 100, 60, 90, 100
    plot_w, plot_h = width-left-right, height-top-bottom
    lines = svg_start("Security gain versus benign cost", width, height)
    for tick in range(0, 51, 10):
        x = left + plot_w * tick / 50
        y = top + plot_h - plot_h * tick / 50
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" stroke="#dbe4ec"/>')
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#dbe4ec"/>')
        lines.append(svg_text(x, top+plot_h+25, f"{tick}", 12, "middle"))
        lines.append(svg_text(left-12, y+5, f"{tick}", 12, "end"))
    lines.append(svg_text(left+plot_w/2, height-25, "Benign block cost (percentage points)", 14, "middle", "700"))
    lines.append(svg_text(25, top+plot_h/2, "ASR reduction (percentage points)", 14, "middle", "700"))
    for label, gain, cost, color in rows:
        x = left + plot_w * cost / 50
        y = top + plot_h - plot_h * gain / 50
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{color}" stroke="#203040" stroke-width="2"/>')
        lines.append(svg_text(x+13, y-10, label, 13, "start", "700"))
    save(name, lines)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    result_dir = HERE / "mitigation_results" / "planted200"
    reports = {
        "Ground-plus": read_json(result_dir / "report_groundplus.json"),
        "Full": read_json(result_dir / "report_full.json"),
        "Ground-only": read_json(result_dir / "report_groundonly.json"),
        "No-refuse": read_json(result_dir / "report_norefuse.json"),
        "Refuse-only": read_json(result_dir / "report_refuseonly.json"),
    }
    benign = {
        "Ground-plus": read_json(HERE / "mitigation_results" / "planted200_groundplus_recheck" / "benign_cost_groundplus.json"),
        "Full": read_json(HERE / "mitigation_results" / "planted200_recheck" / "benign_cost_full.json"),
    }

    labels = list(reports)
    off = [reports[k]["asr_off"]["rate_pct"] for k in labels]
    on = [reports[k]["asr_on"]["rate_pct"] for k in labels]
    bar_chart("five_arm_asr.svg", "Five-arm n=200 attack success comparison", labels,
              [("OFF", off, "#94a3b8"), ("ON", on, "#0f766e")], 35)

    trade_rows = []
    colors = {"Ground-plus": "#0f766e", "Full": "#b45309"}
    for label in ("Ground-plus", "Full"):
        trade_rows.append((label, reports[label]["absolute_reduction_pct"], benign[label]["false_positives_added_pct"], colors[label]))
    scatter("security_gain_benign_cost.svg", trade_rows)

    bar_chart("scorer_correction.svg", "Marker scorer correction: original versus corrected", ["OFF", "ON"],
              [("Original", [31.0, 18.5], "#94a3b8"), ("Corrected", [30.5, 16.5], "#0f766e")], 35)

    ablation = read_json(HERE / "ablation_results.json")["ablation"]
    ab_labels = [row["scorer"].replace("_", " ") for row in ablation]
    ab_values = [row["pass_rate"] * 100 for row in ablation]
    bar_chart("grounding_scorer_ablation.svg", "Grounding scorer ablation", ab_labels,
              [("Pass rate", ab_values, "#2563eb")], 100)

    build = read_json(ROOT / "dataset" / "merged_output" / "build_report.json")
    source_counts = build["source_ingested_counts"]
    source_labels = list(source_counts)
    source_values = [source_counts[k] for k in source_labels]
    bar_chart("corpus_source_composition.svg", "Corpus source composition", source_labels,
              [("Rows", source_values, "#475569")], max(source_values) * 1.15, "")

    architecture = [
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1400\" height=\"760\" viewBox=\"0 0 1400 760\">",
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        svg_text(700, 45, "Thirteen-stage adaptive RAG defense pipeline", 26, "middle", "700"),
    ]
    stages = ["1 Input", "2 Normalize", "3 Detect", "3c Fuse", "4 Retrieve", "5 Search", "6 Sanitize", "7 Rank", "8 Guarded prompt", "9 Generate", "10 Ground", "11 Sanitize output", "12 DLP / 13 Safe response"]
    for i, stage in enumerate(stages):
        x = 50 + (i % 4) * 330
        y = 95 + (i // 4) * 170
        fill = "#ccfbf1" if i in (2, 3, 6, 9, 10) else "#e2e8f0"
        architecture.append(f'<rect x="{x}" y="{y}" width="270" height="80" rx="12" fill="{fill}" stroke="#0f766e" stroke-width="2"/>')
        architecture.append(svg_text(x+135, y+47, stage, 16, "middle", "700"))
        if i < len(stages)-1:
            nx = 50 + ((i+1) % 4) * 330 + 135
            ny = 95 + ((i+1) // 4) * 170 + 40
            architecture.append(f'<line x1="{x+135}" y1="{y+80}" x2="{nx}" y2="{ny}" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)"/>')
    architecture.insert(1, '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#64748b"/></marker></defs>')
    architecture.append("</svg>")
    (OUT / "pipeline_architecture.svg").write_text("\n".join(architecture), encoding="utf-8")

    print(f"wrote {len(list(OUT.glob('*.svg')))} SVG figures to {OUT}")


if __name__ == "__main__":
    main()
