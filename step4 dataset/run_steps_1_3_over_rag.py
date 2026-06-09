"""
run_steps_1_3_over_rag.py — populate gate.risk_score over the 918 RAG questions.

Reads results.jsonl (the Step 4-8 retrieval output), threads each question
through the front-end pipeline implemented in ../dataset/:

    Step 1  user input
    Step 2  normalization  (ftfy + NFKC + hidden-char strip + spaCy)
    Step 3  injection detection  (LLM-Guard: deberta-v3 + Toxicity)

and writes gate_scores.jsonl with one record per question:

    {
      "index": 1,
      "question": "...",
      "gate": {
        "risk_score": 0.04,
        "flag": "green",
        "blocked": false,
        "injection_score": 0.02,
        "toxicity_score": 0.04,
        "length_ok": true,
        "chars_raw": 56,
        "chars_clean": 56
      }
    }

This closes the "risk_in = 0.0 across all 918 rows" gap from the evaluation
memo: after this runs, the gate.risk_score field is empirical, not a stub.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Both folders ship their own `pipeline_common.py`. Step 1-3 want the version
# in ../dataset, so prepend that path BEFORE the local one.
_HERE = Path(__file__).resolve().parent
_DATASET_DIR = (_HERE.parent / "dataset").resolve()
sys.path.insert(0, str(_DATASET_DIR))

# Now import Steps 1-3 from ../dataset (uses dataset/pipeline_common.py).
from pipeline_common import PipelineState  # noqa: E402
import step_01_user_input as s1            # noqa: E402
import step_02_normalization as s2         # noqa: E402
import step_03_injection_detection as s3   # noqa: E402


def _extract_per_scanner(state: PipelineState) -> dict[str, float]:
    """Pull per-scanner scores back out of Step 3's trace entry."""
    out = {"injection_score": 0.0, "toxicity_score": 0.0}
    for entry in reversed(state.trace):
        if entry.get("stage") == "step_03_injection_detection":
            detail = entry.get("detail") or {}
            inj = detail.get("PromptInjection")
            tox = detail.get("Toxicity")
            if inj:
                out["injection_score"] = float(inj.get("score", 0.0))
            if tox:
                out["toxicity_score"] = float(tox.get("score", 0.0))
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Steps 1-3 over RAG questions.")
    ap.add_argument("--in", dest="in_path", default="./results.jsonl")
    ap.add_argument("--out", default="./gate_scores.jsonl")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--no-spacy", action="store_true",
                    help="Skip spaCy in Step 2 (still does ftfy + NFKC + hidden-char strip).")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    s2.set_config(use_spacy=not args.no_spacy)

    src = Path(args.in_path)
    rows = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    bands = {"<0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, ">=0.8": 0}
    n_red = 0
    n_length_flag = 0

    with open(out, "w", encoding="utf-8") as fout:
        for i, row in enumerate(rows, start=1):
            question = row.get("question", "")
            st = PipelineState(prompt=question, raw_prompt=question)
            st = s1.run(st)
            st = s2.run(st, drop_short=False)
            if not st.blocked:
                st = s3.run(st, threshold=args.threshold)

            risk = float(st.scores.get("injection_detection", 0.0))
            blocked = bool(st.blocked)
            flag = "red" if blocked else "green"
            per = _extract_per_scanner(st)
            length_ok = bool(st.meta.get("length_ok", True))

            if blocked:
                n_red += 1
            if not length_ok:
                n_length_flag += 1
            if risk < 0.2: bands["<0.2"] += 1
            elif risk < 0.4: bands["0.2-0.4"] += 1
            elif risk < 0.6: bands["0.4-0.6"] += 1
            elif risk < 0.8: bands["0.6-0.8"] += 1
            else: bands[">=0.8"] += 1

            record = {
                "index": int(row.get("index", i)),
                "question": question,
                "gate": {
                    "risk_score": round(risk, 4),
                    "flag": flag,
                    "blocked": blocked,
                    "injection_score": round(per["injection_score"], 4),
                    "toxicity_score":  round(per["toxicity_score"], 4),
                    "length_ok": length_ok,
                    "chars_raw": len(question),
                    "chars_clean": len(st.prompt),
                },
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            if i % 100 == 0:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0.0
                eta = (len(rows) - i) / rate if rate > 0 else 0.0
                print(f"  scored {i}/{len(rows)}  "
                      f"elapsed={elapsed:.0f}s rate={rate:.2f}/s eta={eta:.0f}s "
                      f"red={n_red} length_flag={n_length_flag}")

    print(f"\n[gate] wrote {len(rows)} rows -> {out}")
    print(f"[gate] red-flagged: {n_red}/{len(rows)}")
    print(f"[gate] length-flagged: {n_length_flag}/{len(rows)}  "
          f"(flagged but not blocked, per methodology)")
    print("[gate] risk_score distribution:")
    for band, count in bands.items():
        print(f"  {band:8s} : {count}")


if __name__ == "__main__":
    main()
