#!/usr/bin/env python3
"""Generate thesis figures from existing JSON result artifacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "thesis_figures"
sys.path.insert(0, str(HERE))

from score_ablation_study import wilson
from score_followup_study import (
    AB,
    FU,
    NY,
    NY_LABELS,
    attack_succeeded,
    answered_correctly,
    blocked,
    flags,
    load_all,
    paired,
    rows,
)


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


def horizontal_bars(
    name: str,
    title: str,
    items: list[tuple[str, float, float, float, str]],
    max_value: float = 100,
    tick_step: int = 20,
    unit: str = "%",
    note: str = "",
    show_legend: bool = True,
) -> None:
    width, row_height = 1420, 34
    left, right, top, bottom = 390, 85, 92, 70
    height = top + row_height * len(items) + bottom
    plot_w = width - left - right
    plot_h = row_height * len(items)
    lines = svg_start(title, width, height)
    if show_legend:
        legend = [
            ("Undefended", "#475569"),
            ("Full stack", "#0f766e"),
            ("Leave-one-out", "#2563eb"),
            ("Single layer", "#d97706"),
            ("Combination", "#8b5cf6"),
        ]
        for index, (label, color) in enumerate(legend):
            x = left + index * 190
            lines.append(f'<rect x="{x}" y="64" width="12" height="12" rx="2" fill="{color}"/>')
            lines.append(svg_text(x + 18, 75, label, 11))
    for tick in range(0, int(max_value) + 1, tick_step):
        x = left + plot_w * tick / max_value
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" stroke="#dbe4ec"/>')
        lines.append(svg_text(x, top + plot_h + 22, f"{tick}{unit}", 12, "middle"))
    for index, (label, value, low, high, color) in enumerate(items):
        center_y = top + row_height * (index + 0.5)
        lines.append(svg_text(left - 14, center_y + 5, label, 13, "end"))
        bar_w = plot_w * value / max_value
        lines.append(f'<rect x="{left}" y="{center_y-8:.1f}" width="{bar_w:.1f}" height="16" rx="3" fill="{color}"/>')
        low_x = left + plot_w * low / max_value
        high_x = left + plot_w * high / max_value
        lines.append(f'<line x1="{low_x:.1f}" y1="{center_y:.1f}" x2="{high_x:.1f}" y2="{center_y:.1f}" stroke="#203040" stroke-width="2"/>')
        lines.append(f'<line x1="{low_x:.1f}" y1="{center_y-5:.1f}" x2="{low_x:.1f}" y2="{center_y+5:.1f}" stroke="#203040"/>')
        lines.append(f'<line x1="{high_x:.1f}" y1="{center_y-5:.1f}" x2="{high_x:.1f}" y2="{center_y+5:.1f}" stroke="#203040"/>')
        lines.append(svg_text(left + max(bar_w, high_x-left) + 9, center_y + 5, f"{value:.1f}{unit}", 12, "start", "700"))
    if note:
        lines.append(svg_text(left, height - 18, note, 11))
    save(name, lines)


def scatter_tradeoff(
    name: str,
    points: list[tuple[str, float, float, str]],
    note: str,
) -> None:
    width, height = 1180, 780
    left, right, top, bottom = 100, 80, 95, 110
    plot_w, plot_h = width - left - right, height - top - bottom
    x_max, y_max = 60, 25
    lines = svg_start("Current-code attack success versus benign blocking", width, height)
    legend = [
        ("Undefended", "#475569"),
        ("Full stack", "#0f766e"),
        ("Leave-one-out", "#2563eb"),
        ("Single layer", "#d97706"),
        ("Combination", "#8b5cf6"),
    ]
    for index, (label, color) in enumerate(legend):
        x = 110 + index * 205
        lines.append(f'<circle cx="{x}" cy="72" r="6" fill="{color}" stroke="#203040" stroke-width="1"/>')
        lines.append(svg_text(x + 11, 76, label, 11))
    for tick in range(0, x_max + 1, 10):
        x = left + plot_w * tick / x_max
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" stroke="#dbe4ec"/>')
        lines.append(svg_text(x, top + plot_h + 23, f"{tick}%", 12, "middle"))
    for tick in range(0, y_max + 1, 5):
        y = top + plot_h - plot_h * tick / y_max
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#dbe4ec"/>')
        lines.append(svg_text(left - 12, y + 5, f"{tick}%", 12, "end"))
    lines.append(svg_text(left + plot_w / 2, height - 42, "Attack success rate", 14, "middle", "700"))
    lines.append(
        f'<text x="0" y="0" transform="translate(28 {top + plot_h / 2:.1f}) rotate(-90)" '
        'font-family="Arial, sans-serif" font-size="14" text-anchor="middle" '
        'font-weight="700" fill="#203040">Benign rows blocked</text>'
    )
    for label, attack_rate, blocked_rate, color in points:
        x = left + plot_w * attack_rate / x_max
        y = top + plot_h - plot_h * blocked_rate / y_max
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" stroke="#203040" stroke-width="1"/>')
        if label in {"Undefended (fresh)", "all seven layers", "all minus sanitization", "guarded prompt only"}:
            lines.append(svg_text(x + 9, y - 8, label, 11, "start", "700"))
    lines.append(svg_text(left, height - 15, note, 11))
    save(name, lines)


def paired_heatmap(
    name: str,
    title: str,
    rows_to_plot: list[tuple[str, int, int]],
    note: str,
    first_label: str = "Fixed / baseline-only",
    second_label: str = "Broke / configuration-only",
) -> None:
    width, row_height = 1120, 34
    left, top = 360, 105
    cell_w, cell_h = 210, 25
    height = top + row_height * len(rows_to_plot) + 75
    max_count = max((max(a, b) for _, a, b in rows_to_plot), default=1) or 1
    lines = svg_start(title, width, height)
    lines.append(svg_text(left + cell_w / 2, top - 18, first_label, 13, "middle", "700"))
    lines.append(svg_text(left + cell_w * 1.5, top - 18, second_label, 13, "middle", "700"))
    for index, (label, first, second) in enumerate(rows_to_plot):
        y = top + index * row_height
        lines.append(svg_text(left - 14, y + cell_h / 2 + 5, label, 12, "end"))
        for column, value in enumerate((first, second)):
            intensity = value / max_count
            fill = "#bfdbfe" if intensity < 0.35 else "#60a5fa" if intensity < 0.7 else "#2563eb"
            x = left + column * cell_w
            lines.append(f'<rect x="{x}" y="{y}" width="{cell_w-10}" height="{cell_h}" rx="3" fill="{fill}"/>')
            text_color = "#ffffff" if intensity >= 0.7 else "#203040"
            lines.append(f'<text x="{x + (cell_w-10)/2}" y="{y + cell_h/2 + 5}" font-family="Arial, sans-serif" font-size="12" text-anchor="middle" fill="{text_color}">{value}</text>')
    lines.append(svg_text(left, height - 18, note, 11))
    save(name, lines)


def share(flags_by_id: dict[str, bool]) -> tuple[int, int, float, float, float]:
    count, total = sum(flags_by_id.values()), len(flags_by_id)
    rate, low, high = wilson(count, total)
    return count, total, rate, low, high


def confusion_counts(
    attack_successes: dict[str, bool],
    benign_blocks: dict[str, bool],
) -> tuple[int, int, int, int, int, int]:
    attack_n = len(attack_successes)
    benign_n = len(benign_blocks)
    true_blocked = attack_n - sum(attack_successes.values())
    missed_attacks = sum(attack_successes.values())
    benign_blocked = sum(benign_blocks.values())
    benign_allowed = benign_n - benign_blocked
    return true_blocked, missed_attacks, benign_blocked, benign_allowed, attack_n, benign_n


def confusion_matrix_figure(
    name: str,
    title: str,
    entries: list[tuple[str, tuple[int, int, int, int, int, int]]],
    columns: int,
    note: str,
) -> None:
    panel_w, panel_h = (340, 230) if columns > 1 else (780, 410)
    rows_count = (len(entries) + columns - 1) // columns
    width = columns * panel_w + 50
    height = 112 + rows_count * panel_h + 62
    lines = svg_start(title, width, height)
    for index, (label, values) in enumerate(entries):
        x = 25 + (index % columns) * panel_w
        y = 90 + (index // columns) * panel_h
        tp, fn, fp, tn, attack_n, benign_n = values
        lines.append(svg_text(x + panel_w / 2, y + 20, label, 15, "middle", "700"))
        if columns == 1:
            matrix_x, matrix_y, cell_w, cell_h = x + 300, y + 95, 190, 90
            row_label_x = x + 105
            header_y = y + 83
            lines.append(svg_text(matrix_x + cell_w, y + 49, "Predicted action", 13, "middle", "700"))
            lines.append(svg_text(x + 75, matrix_y + cell_h, "Expected class", 13, "middle", "700"))
        else:
            matrix_x, matrix_y, cell_w, cell_h = x + 120, y + 70, 96, 48
            row_label_x = x + 12
            header_y = y + 64
            lines.append(svg_text(x + 185, y + 44, "Predicted action", 11, "middle", "700"))
        lines.append(svg_text(matrix_x + cell_w / 2, header_y, "Blocked", 12 if columns == 1 else 10, "middle"))
        lines.append(svg_text(matrix_x + cell_w * 1.5, header_y, "Allowed", 12 if columns == 1 else 10, "middle"))
        lines.append(svg_text(row_label_x, matrix_y + cell_h / 2 + 5, "Attack", 13 if columns == 1 else 11, "end" if columns == 1 else "start"))
        lines.append(svg_text(row_label_x, matrix_y + cell_h * 1.5 + 5, "Benign", 13 if columns == 1 else 11, "end" if columns == 1 else "start"))
        for row_index, (first, second) in enumerate(((tp, fn), (fp, tn))):
            for col_index, count in enumerate((first, second)):
                cell_x = matrix_x + col_index * cell_w
                cell_y = matrix_y + row_index * cell_h
                correct = row_index == col_index
                fill = "#a7f3d0" if correct else "#fed7aa"
                lines.append(f'<rect x="{cell_x}" y="{cell_y}" width="{cell_w-3}" height="{cell_h-3}" rx="4" fill="{fill}" stroke="#cbd5e1"/>')
                lines.append(svg_text(cell_x + (cell_w-3)/2, cell_y + cell_h / 2 + 6, str(count), 24 if columns == 1 else 17, "middle", "700"))
        attack_rate = 100 * fn / attack_n if attack_n else 0
        benign_block_rate = 100 * fp / benign_n if benign_n else 0
        lines.append(svg_text(x + panel_w / 2, matrix_y + cell_h * 2 + (25 if columns == 1 else 19),
                              f"ASR {attack_rate:.1f}% | benign blocked {benign_block_rate:.1f}%",
                              12 if columns == 1 else 10, "middle"))
    lines.append(f'<rect x="30" y="{height-45}" width="14" height="14" rx="2" fill="#a7f3d0"/>')
    lines.append(svg_text(50, height - 33, "Correct handling", 11))
    lines.append(f'<rect x="175" y="{height-45}" width="14" height="14" rx="2" fill="#fed7aa"/>')
    lines.append(svg_text(195, height - 33, "Incorrect handling", 11))
    lines.append(svg_text(360, height - 33, note, 11))
    save(name, lines)


def plot_color(label: str) -> str:
    if label == "full":
        return "#0f766e"
    if label.startswith("combo_"):
        return "#8b5cf6"
    if label.startswith("only_"):
        return "#d97706"
    if label == "fresh undefended":
        return "#475569"
    return "#2563eb"


def add_current_followup_figures() -> int:
    fresh_attack = rows(FU / "off.jsonl")
    fresh_benign = rows(FU / "benign_off.jsonl")
    if fresh_attack is None or fresh_benign is None:
        raise FileNotFoundError("Current-code follow-up baseline is incomplete; run the follow-up study first.")
    base_attack = flags(fresh_attack, attack_succeeded)
    base_blocked = flags(fresh_benign, blocked)
    base_correct = flags(fresh_benign, answered_correctly)
    configs = load_all()

    metrics: dict[str, dict[str, object]] = {}
    for label, (description, attack_rows, benign_rows) in configs.items():
        result: dict[str, object] = {"description": description}
        if attack_rows is not None:
            values = flags(attack_rows, attack_succeeded)
            aligned = {key: values[key] for key in sorted(set(base_attack) & set(values))}
            count, total, rate, low, high = share(aligned)
            fixed, broke, _, _ = paired(base_attack, aligned)
            result.update(attack=(count, total, rate, low, high), attack_pair=(fixed, broke))
        if benign_rows is not None:
            values = flags(benign_rows, blocked)
            aligned = {key: values[key] for key in sorted(set(base_blocked) & set(values))}
            count, total, rate, low, high = share(aligned)
            correct_values = flags(benign_rows, answered_correctly)
            aligned_correct = {key: correct_values[key] for key in sorted(set(base_correct) & set(correct_values))}
            correct_count, correct_total, correct_rate, correct_low, correct_high = share(aligned_correct)
            fixed, broke, _, _ = paired(base_blocked, aligned)
            result.update(blocked=(count, total, rate, low, high), correct=(correct_count, correct_total, correct_rate, correct_low, correct_high), benign_pair=(fixed, broke))
        metrics[label] = result

    matrix_entries = [
        ("Fresh undefended", confusion_counts(base_attack, base_blocked)),
    ]
    for label, (description, attack_rows, benign_rows) in configs.items():
        if attack_rows is None or benign_rows is None:
            continue
        matrix_entries.append(
            (
                description,
                confusion_counts(flags(attack_rows, attack_succeeded), flags(benign_rows, blocked)),
            )
        )
    confusion_matrix_figure(
        "followup_confusion_matrix_full.svg",
        "Full-stack operational confusion matrix",
        [entry for entry in matrix_entries if entry[0] == "all seven layers"],
        1,
        "Expected: block attacks, allow benign. Top row = attack; bottom row = benign.",
    )
    confusion_matrix_figure(
        "followup_confusion_matrices_all.svg",
        "Operational confusion matrices across current-code configurations",
        matrix_entries,
        4,
        "Each arm contains planted attacks and benign controls; green = correct handling, orange = missed attack or blocked benign.",
    )

    labels = list(metrics)
    baseline_attack = share(base_attack)
    baseline_blocked = share(base_blocked)
    baseline_correct = share(base_correct)

    def metric_items(key: str, include_baseline: bool = True) -> list[tuple[str, float, float, float, str]]:
        items = []
        if include_baseline:
            count, total, rate, low, high = {
                "attack": baseline_attack,
                "blocked": baseline_blocked,
                "correct": baseline_correct,
            }[key]
            items.append(("Undefended (fresh)", rate, low, high, plot_color("fresh undefended")))
        for label in labels:
            value = metrics[label].get(key)
            if value is not None:
                _, _, rate, low, high = value
                items.append((metrics[label]["description"], rate, low, high, plot_color(label)))
        return items

    note = "Wilson 95% intervals; current-code follow-up, n=200 per arm."
    horizontal_bars("followup_attack_success_all.svg", "Attack success by current-code configuration", metric_items("attack"), note=note)
    horizontal_bars("followup_benign_blocking_all.svg", "Benign block rate by current-code configuration", metric_items("blocked"), note=note)
    horizontal_bars("followup_benign_correct_all.svg", "Benign answer correctness by current-code configuration", metric_items("correct"), note=note)

    tradeoff = [("Undefended (fresh)", baseline_attack[2], baseline_blocked[2], plot_color("fresh undefended"))]
    for label in labels:
        attack = metrics[label].get("attack")
        block = metrics[label].get("blocked")
        if attack and block:
            tradeoff.append((metrics[label]["description"], attack[2], block[2], plot_color(label)))
    scatter_tradeoff("followup_security_utility_tradeoff.svg", tradeoff, "Each point is one configuration; all arms are scored on the matched n=200 follow-up set.")

    leave_labels = ["full", "noprompt", "norefuse", "nogrounding", "nosanitize", "notighten", "nopresidio", "nodlp"]
    leave_items = [("Undefended (fresh)", *baseline_attack[2:5], plot_color("fresh undefended"))]
    for label in leave_labels:
        count, total, rate, low, high = metrics[label]["attack"]
        leave_items.append((metrics[label]["description"], rate, low, high, plot_color(label)))
    horizontal_bars("followup_leave_one_out_attack.svg", "Full stack and leave-one-out attack success", leave_items, note=note)
    paired_heatmap(
        "followup_leave_one_out_transitions.svg",
        "Paired attack transitions versus undefended baseline",
        [(metrics[label]["description"], *metrics[label]["attack_pair"]) for label in leave_labels[1:]],
        "Fixed = attack succeeded only when undefended; broke = succeeded only with this configuration (n=200).",
    )

    single_labels = ["only_refuse", "only_grounding", "only_sanitize", "only_tighten", "only_prompt", "only_presidio", "only_dlp"]
    single_items = [("Undefended (fresh)", *baseline_attack[2:5], plot_color("fresh undefended"))]
    for label in single_labels:
        count, total, rate, low, high = metrics[label]["attack"]
        single_items.append((metrics[label]["description"], rate, low, high, plot_color(label)))
    horizontal_bars("followup_single_layer_attack.svg", "Single-layer attack success versus current baseline", single_items, note=note)

    combo_labels = ["combo_sanitize_grounding_refuse", "combo_grounding_refuse", "combo_sanitize_grounding", "combo_sanitize_refuse"]
    combo_items = [("Full stack", *metrics["full"]["attack"][2:5], plot_color("full"))]
    for label in combo_labels:
        count, total, rate, low, high = metrics[label]["attack"]
        combo_items.append((metrics[label]["description"], rate, low, high, plot_color(label)))
    horizontal_bars("followup_combinations_attack.svg", "Combination configurations versus full stack", combo_items, note=note)

    attack_pairs = []
    for label in labels:
        fixed, broke = metrics[label].get("attack_pair", (0, 0))
        attack_pairs.append((metrics[label]["description"], fixed, broke))
    paired_heatmap(
        "followup_all_paired_attack_transitions.svg",
        "Paired attack transitions versus fresh undefended baseline",
        attack_pairs,
        "Fixed = baseline-only successes; broke = configuration-only successes; paired exact comparisons, n=200.",
    )
    benign_pairs = []
    for label in labels:
        fixed, broke = metrics[label].get("benign_pair", (0, 0))
        benign_pairs.append((metrics[label]["description"], fixed, broke))
    paired_heatmap(
        "followup_benign_block_transitions.svg",
        "Paired benign-block transitions versus fresh undefended baseline",
        benign_pairs,
        "Fixed = baseline-only blocks; broke = configuration-only blocks; paired rows, n=200.",
        first_label="Baseline-only blocks",
        second_label="Configuration-only blocks",
    )

    combo_vs_single = []
    for layers, _description in [
        ("sanitize,grounding,refuse", "sanitize + grounding + refuse"),
        ("grounding,refuse", "grounding + refuse"),
        ("sanitize,grounding", "sanitize + grounding"),
        ("sanitize,refuse", "sanitize + refuse"),
    ]:
        combo = "combo_" + "_".join(layers.split(","))
        for single in layers.split(","):
            combo_vs_single.append(
                (f"{metrics[combo]['description']} vs {metrics[f'only_{single}']['description']}",
                 *paired(flags(configs[combo][1], attack_succeeded), flags(configs[f"only_{single}"][1], attack_succeeded))[:2])
            )
    paired_heatmap(
        "followup_combinations_vs_components.svg",
        "Paired attack outcomes: combinations versus their single layers",
        combo_vs_single,
        "Left = success only under combination; right = success only under single layer (n=200).",
        first_label="Combination-only",
        second_label="Single-layer-only",
    )

    _, _, fresh_attack_rate, _, _ = baseline_attack
    _, _, fresh_correct_rate, _, _ = baseline_correct
    old_attack = rows(AB / "off.jsonl")
    old_benign = rows(AB / "benign_off.jsonl")
    if old_attack is not None and old_benign is not None:
        old_a = share(flags(old_attack, attack_succeeded))
        old_c = share(flags(old_benign, answered_correctly))
        bar_chart(
            "followup_baseline_drift.svg",
            "Undefended baseline drift: July versus current code",
            ["Attack success", "Benign correct"],
            [("July", [old_a[2], old_c[2]], "#94a3b8"),
             ("Current", [fresh_attack_rate, fresh_correct_rate], "#0f766e")],
            100,
        )

    nonyesno_base = rows(NY / "off.jsonl")
    if nonyesno_base is not None:
        nonyesno_attack = flags(nonyesno_base, attack_succeeded)
        nonyesno_full = None
        nonyesno_metrics = []
        nonyesno_pairs = []
        for label, description in NY_LABELS:
            arm = rows(NY / f"on_{label}.jsonl")
            if arm is None:
                continue
            attack = flags(arm, attack_succeeded)
            blocked_flags = flags(arm, blocked)
            count, total, rate, low, high = share(attack)
            block_count, block_total, block_rate, block_low, block_high = share(blocked_flags)
            color = plot_color(label)
            nonyesno_metrics.append((description, rate, low, high, color))
            nonyesno_pairs.append((description, *paired(nonyesno_attack, attack)[:2]))
            if label == "full":
                nonyesno_full = attack
        base_summary = share(nonyesno_attack)
        nonyesno_metrics.insert(0, ("Undefended", *base_summary[2:5], plot_color("fresh undefended")))
        horizontal_bars(
            "followup_nonyesno_attack_success.svg",
            "Attack success on non-yes/no planted-fact slice",
            nonyesno_metrics,
            note="Wilson 95% intervals; n=40, free-text target values. Small-sample estimates are uncertain.",
        )
        nonyesno_block_items = []
        baseline_blocked = share(flags(nonyesno_base, blocked))
        nonyesno_block_items.append(("Undefended", *baseline_blocked[2:5], plot_color("fresh undefended")))
        for label, description in NY_LABELS:
            arm = rows(NY / f"on_{label}.jsonl")
            if arm is None:
                continue
            block_count, block_total, block_rate, block_low, block_high = share(flags(arm, blocked))
            nonyesno_block_items.append((description, block_rate, block_low, block_high, plot_color(label)))
        horizontal_bars(
            "followup_nonyesno_blocking.svg",
            "Attack queries blocked on non-yes/no slice",
            nonyesno_block_items,
            note="Wilson 95% intervals; n=40. Slice construction excludes rows blocked by scanner/redaction.",
        )
        paired_heatmap(
            "followup_nonyesno_transitions.svg",
            "Paired attack transitions versus undefended non-yes/no baseline",
            nonyesno_pairs,
            "Fixed = baseline-only attack successes; broke = configuration-only successes (n=40).",
        )
        if nonyesno_full is not None:
            full_label_flags = nonyesno_full
            full_pairs = []
            for label, description in NY_LABELS:
                if label == "full":
                    continue
                arm = rows(NY / f"on_{label}.jsonl")
                if arm is not None:
                    full_pairs.append((description, *paired(flags(arm, attack_succeeded), full_label_flags)[:2]))
            paired_heatmap(
                "followup_nonyesno_vs_full.svg",
                "Non-yes/no attack transitions versus full stack",
                full_pairs,
                "Left = success only without full stack; right = success only under full stack (n=40).",
                first_label="Without-full-only",
                second_label="Full-stack-only",
            )

    pii_items = []
    for label, directory in [("Extraction", "extraction"), ("Incidental", "incidental")]:
        report = read_json(HERE / "row8_mitigation_results" / directory / "pii_leak_report.json")
        for arm, key in [("OFF", "pii_leak_rate_off"), ("ON", "pii_leak_rate_on")]:
            metric = report[key]
            rate, low, high = metric["rate_pct"], metric["ci"][0], metric["ci"][1]
            pii_items.append((f"{label} / {arm} (n={report['n']})", rate, low, high, "#64748b" if arm == "OFF" else "#0f766e"))
    horizontal_bars(
        "followup_pii_leak_rates.svg",
        "PII leakage rates in extraction and incidental slices",
        pii_items,
        max_value=20,
        tick_step=5,
        note="Wilson 95% intervals. Both slices measured 0 leaks; upper bounds reflect the small n=30 samples.",
        show_legend=False,
    )

    return len(list(OUT.glob("followup_*.svg")))


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

    new_count = add_current_followup_figures()
    print(f"wrote {len(list(OUT.glob('*.svg')))} SVG figures to {OUT} ({new_count} current follow-up figures)")


if __name__ == "__main__":
    main()
