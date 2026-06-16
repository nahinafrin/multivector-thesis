"""
score_multivector_fp.py - inspect benign rows that trip the multivector detector.

Usage:
    python score_multivector_fp.py --in grounded_baseline.jsonl
    python score_multivector_fp.py --in grounded_controller.jsonl
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect benign multivector false positives")
    ap.add_argument("--in", dest="input", required=True)
    ap.add_argument("--kind", default="benign_control")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    benign = [r for r in rows if r.get("kind") == args.kind]
    hits = [r for r in benign if (r.get("multivector") or {}).get("is_multivector")]

    print(f"=== BENIGN MULTIVECTOR FALSE POSITIVES ({args.input}) ===")
    print(f"{args.kind}: {len(hits)}/{len(benign)} tripped multivector")
    for r in hits:
        mv = r.get("multivector") or {}
        print(
            f"  index={r.get('index')} joint={float(mv.get('joint_risk', 0.0)):.4f} "
            f"channels={mv.get('channels')} action="
            f"{(r.get('controller') or {}).get('final_action')}"
        )


if __name__ == "__main__":
    main()
