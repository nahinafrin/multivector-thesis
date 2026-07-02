#!/usr/bin/env python3
"""
run_three_way.py  —  the experiment that carries the thesis
============================================================

Runs the SAME evaluation slice under three security modes and writes one results
JSONL per mode, so the scorer can compute security / quality / efficiency side by
side:

    none      : no security    (LOW tier, grounding off)   -> attack ceiling
    static    : full security   (HIGH tier always)          -> strong but expensive
    adaptive  : tiered by risk   (the contribution)         -> match static, cheaper

The precise claim under test:
    adaptive matches static SECURITY (ASR) while spending less COMPUTE and
    blocking fewer BENIGN queries.

Three publishable outcomes:
    * adaptive == static ASR at lower cost                 -> the win
    * adaptive slightly worse ASR, much cheaper            -> a trade-off curve
    * adaptive worse on every axis                         -> an honest negative

USAGE (from `step4 dataset`, after copying the adaptive_* modules in):
    python run_three_way.py --slice adversarial_slice.jsonl --out-prefix three_way
    # writes three_way_none.jsonl, three_way_static.jsonl, three_way_adaptive.jsonl
    python score_three_way.py --prefix three_way
"""
from __future__ import annotations
import argparse, json, time
from pipeline_common import read_jsonl


def run_mode(slice_path: str, mode: str, out_path: str, *, limit: int | None) -> None:
    """Run the whole slice through process_adaptive() in one security mode."""
    # Imported here so a missing optional dep doesn't break --help.
    from run_full_pipeline_adaptive import process_adaptive, ensure_index
    ensure_index()
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in read_jsonl(slice_path):
            if limit and n >= limit:
                break
            q = row.get("question") or row.get("prompt") or ""
            t0 = time.perf_counter()
            st = process_adaptive(q, mode=mode,
                                  poison_chunk=row.get("poison_chunk"),
                                  query_suffix=row.get("query_suffix"))
            latency = time.perf_counter() - t0
            out = {
                "index": row.get("id", n),
                "kind": row.get("kind"),
                "question": q,
                "success_marker": row.get("success_marker", ""),
                "blocked": st.blocked,
                "abstained": getattr(st, "abstained", False),
                "security_tier": st.meta.get("security_tier"),
                "cumulative_risk": st.scores.get("cumulative_risk"),
                "risk_contributions": st.meta.get("risk_contributions"),
                "generation_mode": st.meta.get("generation_mode"),
                "verify_cost_units": st.meta.get("verify_cost_units"),
                "verify_timing": st.meta.get("verify_timing"),
                "end_to_end_latency": round(latency, 4),
                "generation": {"answer": st.meta.get("answer", st.output)},
                "grounding": st.meta.get("grounding", {}),
                "retrieval": {
                    "top_k": st.meta.get("active_top_k"),
                    "sim_threshold": st.meta.get("active_sim_threshold"),
                    "rerank_min_score": st.meta.get("rerank_min_score"),
                    "sanitization_strictness": st.meta.get("active_sanitization"),
                    "poison_redacted": st.meta.get("poison_redacted"),
                    "canary": st.meta.get("canary"),
                },
                "controller": st.meta.get("controller", {}),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
    print(f"[{mode}] wrote {n} rows -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Three-way static-vs-adaptive experiment")
    ap.add_argument("--slice", required=True)
    ap.add_argument("--out-prefix", default="three_way")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--modes", nargs="+", default=["none", "static", "adaptive"])
    args = ap.parse_args()
    for mode in args.modes:
        run_mode(args.slice, mode, f"{args.out_prefix}_{mode}.jsonl", limit=args.limit)


if __name__ == "__main__":
    main()
