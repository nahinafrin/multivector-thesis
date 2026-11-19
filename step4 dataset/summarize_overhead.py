#!/usr/bin/env python3
"""
summarize_overhead.py -- per-query latency and memory overhead for the paper revision.

Latency comes from the `latency_s` field every arm of the mitigation A/B harness
already writes (wall-clock seconds per row, end to end). Memory comes from the
1 Hz sampler in resource_profile/sample_resources.ps1, recorded while the full
13-stage pipeline (run_full_pipeline.py) processed resource_profile/profile_slice_20.jsonl.

Usage (from the `step4 dataset` directory):
    python summarize_overhead.py            # prints tables, writes resource_profile/overhead_summary.json
"""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (label, attack arm, benign arm), all from the current-code ablation (n=200 each)
ARMS = [
    ("Undefended", "ablation200/off.jsonl", "ablation200/benign_off.jsonl"),
    ("Full stack (7 mitigation layers)", "ablation200/on_full.jsonl", "ablation200/benign_on.jsonl"),
    ("Sanitize + grounding + refuse", "followup200/on_combo_sanitize_grounding_refuse.jsonl",
     "followup200/benign_on_combo_sanitize_grounding_refuse.jsonl"),
    ("Sanitize + refuse", "followup200/on_combo_sanitize_refuse.jsonl",
     "followup200/benign_on_combo_sanitize_refuse.jsonl"),
    ("Refuse-on-detection only", "ablation200/on_only_refuse.jsonl", "ablation200/benign_on_only_refuse.jsonl"),
    ("Sanitization only", "ablation200/on_only_sanitize.jsonl", "ablation200/benign_on_only_sanitize.jsonl"),
    ("Grounding gate only", "ablation200/on_only_grounding.jsonl", "ablation200/benign_on_only_grounding.jsonl"),
]


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    i = (len(xs) - 1) * q
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def lat_stats(rows: list[dict]) -> dict:
    lat = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
    return {"n": len(lat), "median": round(st.median(lat), 2), "mean": round(st.mean(lat), 2),
            "p95": round(pct(lat, 0.95), 2), "max": round(max(lat), 2)}


def memory_stats(csv_path: Path) -> dict | None:
    if not csv_path.exists():
        return None
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return None
    col = lambda k: [float(r[k]) for r in rows if r[k] not in ("", None)]
    gpu, py, ol = col("gpu_mem_mib"), col("python_rss_mib"), col("ollama_rss_mib")
    tot = [a + b for a, b in zip(py, ol)]
    return {"samples": len(rows),
            "gpu_mem_mib": {"median": st.median(gpu), "peak": max(gpu)},
            "gpu_util_pct": {"median": st.median(col("gpu_util_pct")), "peak": max(col("gpu_util_pct"))},
            "python_rss_mib": {"median": st.median(py), "peak": max(py)},
            "ollama_rss_mib": {"median": st.median(ol), "peak": max(ol)},
            "host_ram_total_mib": {"median": st.median(tot), "peak": max(tot)}}


def main() -> None:
    base = HERE / "mitigation_results"
    out: dict = {"latency_s": {}}
    print(f"{'Configuration':36s} {'attack med/p95':>16s} {'benign med/p95':>16s}")
    for label, a, b in ARMS:
        sa, sb = lat_stats(load(base / a)), lat_stats(load(base / b))
        out["latency_s"][label] = {"attack": sa, "benign": sb}
        print(f"{label:36s} {sa['median']:7.2f}/{sa['p95']:6.2f}   {sb['median']:7.2f}/{sb['p95']:6.2f}")

    # full 13-stage pipeline on the external n=200 set (ensemble generator, controller on)
    log = HERE / "run_n200_log.txt"
    if log.exists():
        txt = log.read_text(encoding="utf-16", errors="ignore")
        import re
        m = re.findall(r"\[200/200\].*?elapsed=([\d.]+)s", txt)
        if m:
            out["full_pipeline_external_n200_s_per_query"] = round(float(m[-1]) / 200, 2)
            print(f"\nFull 13-stage pipeline, external n=200: {float(m[-1]) / 200:.2f} s/query (mean)")

    prof = HERE / "resource_profile"
    rows = load(prof / "profile_full_pipeline_20.jsonl") if (prof / "profile_full_pipeline_20.jsonl").exists() else []
    if rows and (prof / "run_start.txt").exists() and (prof / "run_end.txt").exists():
        wall = int((prof / "run_end.txt").read_text()) - int((prof / "run_start.txt").read_text())
        out["profile_run"] = {"rows": len(rows), "wall_s": wall, "s_per_query_incl_model_load": round(wall / len(rows), 2)}
        print(f"Profiled run: {len(rows)} rows, {wall} s wall ({wall / len(rows):.1f} s/query incl. model load)")
    mem = memory_stats(prof / "resources_full_pipeline.csv")
    if mem:
        out["memory"] = mem
        print("Memory:", json.dumps(mem))
    (prof / "overhead_summary.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
