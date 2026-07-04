#!/usr/bin/env python3
"""
run_benign_cost.py  —  the measurement that keeps the mitigation result honest
==============================================================================

Runs benign queries through mitigation OFF vs ON (ablatable, same flags as
run_mitigation_ab.py) and measures false-positive block rate + utility loss.

USAGE (from `step4 dataset`):
    python run_benign_cost.py --qa data/question-answer/test.jsonl --n 200 --run-dir mitigation_results/planted30v2
    python run_benign_cost.py --qa data/question-answer/test.jsonl --n 200 --skip-off --no-refuse --run-dir mitigation_results/planted30v2 --label norefuse
"""
from __future__ import annotations
import argparse, json, time, random
from pathlib import Path

from mitigation_pipeline import process_with_mitigation, MitigationConfig
from detector_interface import get_detector
from run_mitigation_ab import build_on_config, auto_label, _cfg_dict

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


def run_arm(qa_path, arm, out_path, detector, *, n, seed, base_url, on_cfg=None):
    if arm == "on":
        cfg = on_cfg or MitigationConfig()
    else:
        cfg = MitigationConfig.off()
    ensure_index()
    rows = [r for r in read_jsonl(qa_path) if (r.get("question") or "").strip()]
    random.Random(seed).shuffle(rows)
    rows = rows[:n]
    written = 0
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
                "block_reason": st.block_reason or st.meta.get("block_reason"),
                "detector": st.meta.get("detector"),
                "mitigation_applied": st.meta.get("mitigation_applied", []),
                "mitigation_config": _cfg_dict(cfg) if arm == "on" else None,
                "final_response": st.output,
                "generation": {"answer": st.meta.get("answer", st.output)},
                "grounding": st.meta.get("grounding", {}),
                "latency_s": round(time.perf_counter() - t0, 4),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1
    print(f"[{arm}] wrote {written} benign rows -> {out_path}")


def resolve_outputs(args, label: str) -> tuple[str, str]:
    if args.run_dir:
        base = Path(args.run_dir)
        base.mkdir(parents=True, exist_ok=True)
        off = str(base / "benign_off.jsonl")
        on_name = "benign_on.jsonl" if label == "full" else f"benign_on_{label}.jsonl"
        return off, str(base / on_name)
    return args.out_off, args.out_on


def main():
    ap = argparse.ArgumentParser(description="Benign cost of mitigation (FP + degradation)")
    ap.add_argument("--qa", default="data/question-answer/test.jsonl")
    ap.add_argument("--detector", default="existing")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out-off", default="benign_off.jsonl")
    ap.add_argument("--out-on", default="benign_on.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--skip-off", action="store_true",
                    help="reuse existing benign_off.jsonl; only run ON arm")
    ap.add_argument("--no-refuse", action="store_true")
    ap.add_argument("--no-sanitize", action="store_true")
    ap.add_argument("--no-tighten", action="store_true")
    ap.add_argument("--no-prompt", action="store_true")
    ap.add_argument("--no-grounding", action="store_true")
    ap.add_argument("--no-dlp", action="store_true")
    ap.add_argument("--only-grounding", action="store_true")
    args = ap.parse_args()

    label = auto_label(args)
    on_cfg = build_on_config(args)
    active = [k for k, v in _cfg_dict(on_cfg).items() if v and k != "enabled"]
    print(f"[config] label={label!r}  ON-arm active layers: {active or '(none)'}")

    out_off, out_on = resolve_outputs(args, label)
    detector = get_detector(args.detector)

    if not args.skip_off:
        run_arm(args.qa, "off", out_off, detector, n=args.n, seed=args.seed,
                base_url=args.base_url)
    else:
        print(f"[off] skipped (reusing {out_off})")
        if not Path(out_off).is_file():
            raise SystemExit(f"--skip-off but missing {out_off}")

    run_arm(args.qa, "on", out_on, detector, n=args.n, seed=args.seed,
            base_url=args.base_url, on_cfg=on_cfg)


if __name__ == "__main__":
    main()
