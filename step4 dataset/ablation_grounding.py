"""
ablation_grounding.py — Step 10 scorer ablation (B in the eval memo).

Reads grounded.jsonl (already populated by step_10_grounding_judge.py) and
re-applies the SAME per-row dynamic threshold under three scoring strategies:

  - lexical_only  : grounded if grounding.lexical_score >= threshold_used
  - model_only    : grounded if grounding.model_score   >= threshold_used
  - hybrid_max    : grounded if max(lex, model)         >= threshold_used
                    (this is the production scorer; matches grounded.passed)

The point is to justify the hybrid design empirically: if either single
component matched the hybrid, the hybrid wouldn't be earning its keep.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def summarise(label: str, scores: list[float], passes: list[bool]) -> dict:
    n = len(scores)
    pass_count = sum(passes)
    return {
        "scorer": label,
        "n": n,
        "mean_score": round(statistics.fmean(scores), 4) if n else 0.0,
        "min": round(min(scores), 4) if n else 0.0,
        "max": round(max(scores), 4) if n else 0.0,
        "passed": pass_count,
        "pass_rate": round(pass_count / n, 4) if n else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 10 scorer ablation.")
    ap.add_argument("--in", dest="in_path", default="./grounded.jsonl")
    ap.add_argument("--out", default="./ablation_results.json",
                    help="Where to write the JSON summary (default: ./ablation_results.json).")
    args = ap.parse_args()

    src = Path(args.in_path)
    rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]

    lex_scores: list[float] = []
    mdl_scores: list[float] = []
    hyb_scores: list[float] = []
    lex_pass: list[bool] = []
    mdl_pass: list[bool] = []
    hyb_pass: list[bool] = []

    # How often did each lone scorer disagree with the hybrid?
    only_lex_won = 0  # hybrid passed, model-only would have failed
    only_mdl_won = 0  # hybrid passed, lexical-only would have failed
    both_needed_to_lose = 0  # hybrid failed, both components failed independently

    for row in rows:
        g = row.get("grounding", {})
        if not g:
            continue
        thr = float(g.get("threshold_used", 0.70))
        lex = float(g.get("lexical_score", 0.0))
        mdl = float(g.get("model_score", 0.0))
        hyb = float(g.get("faithfulness", max(lex, mdl)))

        lex_scores.append(lex); mdl_scores.append(mdl); hyb_scores.append(hyb)
        lp = lex >= thr; mp = mdl >= thr; hp = hyb >= thr
        lex_pass.append(lp); mdl_pass.append(mp); hyb_pass.append(hp)

        if hp:
            if not mp:
                only_lex_won += 1
            if not lp:
                only_mdl_won += 1
        else:
            if not lp and not mp:
                both_needed_to_lose += 1

    out = {
        "ablation": [
            summarise("lexical_only", lex_scores, lex_pass),
            summarise("model_only",   mdl_scores, mdl_pass),
            summarise("hybrid_max",   hyb_scores, hyb_pass),
        ],
        "complementarity": {
            "hybrid_passes_rescued_by_lexical": only_lex_won,
            "hybrid_passes_rescued_by_model":   only_mdl_won,
            "hybrid_failures_both_components_failed": both_needed_to_lose,
            "n_rows": len(rows),
        },
    }

    print("\n=== Step 10 scorer ablation (per-row dynamic threshold) ===\n")
    print(f"{'Scorer':<14}{'Mean':>10}{'Min':>10}{'Max':>10}{'Pass-rate':>12}{'Passed':>10}")
    for r in out["ablation"]:
        print(f"{r['scorer']:<14}"
              f"{r['mean_score']:>10.3f}"
              f"{r['min']:>10.3f}"
              f"{r['max']:>10.3f}"
              f"{r['pass_rate']:>12.3f}"
              f"{r['passed']:>10d}")

    c = out["complementarity"]
    print("\n=== Hybrid complementarity ===")
    print(f"hybrid passed because LEXICAL pulled it through (model would have failed): "
          f"{c['hybrid_passes_rescued_by_lexical']}")
    print(f"hybrid passed because MODEL pulled it through (lexical would have failed): "
          f"{c['hybrid_passes_rescued_by_model']}")
    print(f"hybrid failed AND both components failed independently: "
          f"{c['hybrid_failures_both_components_failed']}")

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[ablation] wrote {args.out}")


if __name__ == "__main__":
    main()
