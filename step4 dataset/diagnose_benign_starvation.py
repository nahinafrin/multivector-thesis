#!/usr/bin/env python3
"""Compare retrieval/evidence stats for benign queries: full vs grounding-only."""
from __future__ import annotations
import argparse, json, random, statistics as stats

from mitigation_pipeline import process_with_mitigation, MitigationConfig
from detector_interface import get_detector
from run_mitigation_ab import build_on_config
from run_full_pipeline import ensure_index


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def run_sample(qa_path, n, seed, cfg, label):
    ensure_index()
    det = get_detector("existing")
    rows = [r for r in read_jsonl(qa_path) if r.get("question")]
    random.Random(seed).shuffle(rows)
    rows = rows[:n]
    out = []
    for r in rows:
        st = process_with_mitigation(r["question"], mitigation=cfg, detector=det)
        g = st.meta.get("grounding") or {}
        out.append({
            "label": label,
            "blocked": st.blocked,
            "retrieved_k": len(st.context or []),
            "ranked_k": len(st.ranked_context or []),
            "sanitization_dropped": st.meta.get("sanitization_dropped"),
            "rerank_min_score": st.meta.get("rerank_min_score"),
            "grounding_passed": g.get("passed"),
            "faithfulness": g.get("faithfulness"),
            "threshold": g.get("threshold"),
        })
    return out


def summarize(rows, label):
    blocked = [r for r in rows if r["label"] == label and r["blocked"]]
    all_r = [r for r in rows if r["label"] == label]
    def mean(key):
        vals = [r[key] for r in all_r if r.get(key) is not None]
        return round(stats.mean(vals), 2) if vals else None
    print(f"\n=== {label} (n={len(all_r)}) ===")
    print(f"  blocked: {len(blocked)} ({100*len(blocked)/len(all_r):.1f}%)")
    print(f"  mean retrieved_k: {mean('retrieved_k')}  ranked_k: {mean('ranked_k')}")
    if blocked:
        print(f"  blocked mean ranked_k: {round(stats.mean([r['ranked_k'] for r in blocked]),2)}")
        print(f"  blocked mean faithfulness: {round(stats.mean([r['faithfulness'] for r in blocked if r['faithfulness'] is not None]),3)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="data/question-answer/test.jsonl")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    full = MitigationConfig()
    ground = MitigationConfig(enabled=True, sanitize=False, tighten_retrieval=False,
                              guarded_prompt=False, grounding_gate=True, dlp=False,
                              refuse_on_detection=False)
    rows = run_sample(args.qa, args.n, args.seed, full, "full")
    rows += run_sample(args.qa, args.n, args.seed, ground, "groundonly")
    summarize(rows, "full")
    summarize(rows, "groundonly")


if __name__ == "__main__":
    main()
