"""
diagnose_multivector.py — evidence-only diagnosis of the multi-vector path.

Reads a finished run JSONL (default grounded_controller.jsonl) where every row
carries gate / multivector / block_stage / kind, and reports:

  1. per-class detection: gate decisions, final-blocked, and block-stage source
  2. multivector firing vs dormancy, per class, with channel-value distributions
  3. whether the multivector signal actually changed the outcome
  4. channel separation (benign vs attack) — the calibration-failure diagnosis

No model calls, no mutation; pure read of the artifacts.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter, defaultdict

# Which kinds are ground-truth attacks (should be caught) vs benign.
ATTACK_KINDS = {"adversarial_query", "poisoned_context", "multivector_attack",
                "gate_slip_query"}
BENIGN_KINDS = {"benign_control"}


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 4)


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def diagnose(path):
    rows = load(path)
    by_kind = defaultdict(list)
    for r in rows:
        by_kind[r.get("kind", "?")].append(r)

    print(f"\n{'='*72}\nFILE: {path}   ({len(rows)} rows)\n{'='*72}")

    # ---- 1 & 2: per-class detection + multivector firing ------------------- #
    print(f"\n{'kind':18} {'n':>3}  {'ALLOW':>5} {'REVIEW':>6} {'BLOCK':>5}  "
          f"{'final_blk':>9}  {'mv_fired':>8} {'mv_block':>8}")
    for kind in sorted(by_kind):
        items = by_kind[kind]
        dec = Counter((r.get("gate") or {}).get("decision", "?") for r in items)
        final_blocked = sum(1 for r in items if r.get("blocked"))
        mv_fired = sum(1 for r in items
                       if (r.get("multivector") or {}).get("is_multivector"))
        mv_block = sum(1 for r in items
                       if (r.get("block_stage") == "risk_feedback_controller"
                           and "multi-vector" in (r.get("block_reason") or "").lower()))
        print(f"{kind:18} {len(items):>3}  {dec.get('ALLOW',0):>5} "
              f"{dec.get('REVIEW',0):>6} {dec.get('BLOCK',0):>5}  "
              f"{final_blocked:>9}  {mv_fired:>8} {mv_block:>8}")

    # ---- block-stage attribution (where the catch actually happens) -------- #
    print("\n-- block_stage attribution (final blocked rows only) --")
    stage = Counter(r.get("block_stage") for r in rows if r.get("blocked"))
    for s, n in stage.most_common():
        print(f"   {str(s):28} {n}")

    # ---- 3: did multivector change the outcome? ---------------------------- #
    print("\n-- multivector firing detail (per attack class) --")
    for kind in ["multivector_attack", "gate_slip_query", "adversarial_query",
                 "poisoned_context"]:
        items = by_kind.get(kind, [])
        if not items:
            continue
        jr = [(r.get("multivector") or {}).get("joint_risk", 0.0) for r in items]
        qv = [(r.get("multivector") or {}).get("channels", {}).get("query_vector", 0.0)
              for r in items]
        cv = [(r.get("multivector") or {}).get("channels", {}).get("context_vector", 0.0)
              for r in items]
        fired = sum(1 for r in items
                    if (r.get("multivector") or {}).get("is_multivector"))
        hard = sum(1 for r in items
                   if (r.get("multivector") or {}).get("hard_block"))
        print(f"\n   {kind}  (n={len(items)})")
        print(f"     is_multivector fired : {fired}/{len(items)}   hard_block: {hard}")
        print(f"     joint_risk      min/med/max : "
              f"{pct(jr,0)} / {pct(jr,50)} / {pct(jr,100)}")
        print(f"     query_vector    min/med/max : "
              f"{pct(qv,0)} / {pct(qv,50)} / {pct(qv,100)}")
        print(f"     context_vector  min/med/max : "
              f"{pct(cv,0)} / {pct(cv,50)} / {pct(cv,100)}")
        zero_both = sum(1 for q, c in zip(qv, cv) if q == 0.0 and c == 0.0)
        print(f"     rows with BOTH channels == 0 : {zero_both}/{len(items)}")

    # ---- 4: channel separation (calibration-failure diagnosis) ------------- #
    print("\n-- channel separation: benign vs attack (raw, as scored) --")
    for ch in ["query_vector", "context_vector"]:
        benign = [(r.get("multivector") or {}).get("channels", {}).get(ch, 0.0)
                  for r in rows if r.get("kind") in BENIGN_KINDS]
        attack = [(r.get("multivector") or {}).get("channels", {}).get(ch, 0.0)
                  for r in rows if r.get("kind") in ATTACK_KINDS]
        sat_b = sum(1 for v in benign if v < 0.01 or v > 0.99) / max(len(benign), 1)
        sat_a = sum(1 for v in attack if v < 0.01 or v > 0.99) / max(len(attack), 1)
        b95 = pct(benign, 95)
        a10 = pct(attack, 10)
        sep = round(a10 - b95, 4) if (a10 is not None and b95 is not None) else None
        print(f"\n   {ch}")
        print(f"     benign  n={len(benign):>3} p50={pct(benign,50)} p95={b95} "
              f"saturated={sat_b:.2f}")
        print(f"     attack  n={len(attack):>3} p10={a10} p50={pct(attack,50)} "
              f"saturated={sat_a:.2f}")
        floor = (rows[0].get("multivector") or {}).get("floors", {}).get(ch)
        jmin = (rows[0].get("multivector") or {}).get("joint_min")
        print(f"     floor={floor}  joint_min={jmin}  "
              f"separation(attack_p10 - benign_p95)={sep}  "
              f"{'OVERLAP (bad)' if sep is not None and sep <= 0 else 'separated'}")
        in_band = sum(1 for v in attack
                      if floor is not None and jmin is not None
                      and floor <= v < jmin)
        print(f"     attack rows in [floor, joint_min) band : {in_band}/{len(attack)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="paths", nargs="+",
                    default=["grounded_controller.jsonl"])
    args = ap.parse_args()
    for p in args.paths:
        diagnose(p)


if __name__ == "__main__":
    main()
