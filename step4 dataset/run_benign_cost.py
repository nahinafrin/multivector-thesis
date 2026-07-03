#!/usr/bin/env python3
"""
run_benign_cost.py  —  the measurement that keeps the mitigation result honest
==============================================================================

A mitigation ASR reduction only means something ALONGSIDE its benign cost. A pipeline
that refuses everything gets 0% ASR and is useless. This harness runs benign queries
through the SAME mitigation-ON configuration and measures what the mitigation costs
legitimate users:

  * benign block rate      — benign queries wrongly refused (false positives)
  * benign degradation     — benign queries answered but no longer grounded / correct
  * attribution            — which mitigation layer caused each benign block

Run it in TWO modes so the cost is a clean delta, mirroring the ASR A/B:
  off : mitigation disabled (baseline benign behaviour)
  on  : full mitigation (same config as the ASR "on" arm)

USAGE (from `step4 dataset`):
    python run_benign_cost.py --qa data/question-answer/test.jsonl --n 200
    python score_benign_cost.py --off benign_off.jsonl --on benign_on.jsonl
"""
from __future__ import annotations
import argparse, json, time, random

from mitigation_pipeline import process_with_mitigation, MitigationConfig
from detector_interface import get_detector

try:
    from run_full_pipeline import ensure_index
except Exception:
    def ensure_index(*a, **k):  # type: ignore
        pass


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def run_arm(qa_path, arm, out_path, detector, *, n, seed, base_url):
    cfg = MitigationConfig() if arm == "on" else MitigationConfig.off()
    ensure_index()
    rows = [r for r in read_jsonl(qa_path) if (r.get("question") or "").strip()]
    random.Random(seed).shuffle(rows)
    rows = rows[:n]
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            q = r["question"]
            t0 = time.perf_counter()
            st = process_with_mitigation(q, mitigation=cfg, detector=detector,
                                         base_url=base_url)
            out = {
                "index": r.get("id", i),
                "kind": "benign_control",
                "arm": arm,
                "question": q,
                "true_answer": r.get("answer"),
                "blocked": st.blocked,
                "block_reason": st.meta.get("block_reason"),
                "detector": st.meta.get("detector"),
                "mitigation_applied": st.meta.get("mitigation_applied", []),
                "final_response": st.output,
                "generation": {"answer": st.meta.get("answer", st.output)},
                "grounding": st.meta.get("grounding", {}),
                "latency_s": round(time.perf_counter() - t0, 4),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1
    print(f"[{arm}] wrote {written} benign rows -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Benign cost of mitigation (FP + degradation)")
    ap.add_argument("--qa", default="data/question-answer/test.jsonl")
    ap.add_argument("--detector", default="existing")
    ap.add_argument("--out-off", default="benign_off.jsonl")
    ap.add_argument("--out-on", default="benign_on.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base-url", default="http://localhost:11434")
    args = ap.parse_args()

    detector = get_detector(args.detector)
    run_arm(args.qa, "off", args.out_off, detector, n=args.n, seed=args.seed, base_url=args.base_url)
    run_arm(args.qa, "on", args.out_on, detector, n=args.n, seed=args.seed, base_url=args.base_url)


if __name__ == "__main__":
    main()
