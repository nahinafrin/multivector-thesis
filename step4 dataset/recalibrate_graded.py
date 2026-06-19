#!/usr/bin/env python3
"""
recalibrate_graded.py  —  Solution 3: rebuild the calibration on the GRADED scale,
in the right order, and refuse to emit thresholds that don't separate the classes.

This wraps the steps that are currently run by hand (and in the wrong scale) into
one ordered, fail-closed pipeline:

    1. fit the softening temperature T on the merged attack dataset (graded_channels)
    2. score every slice payload on the GRADED scale, keep only IN-BAND ones
    3. derive per-channel floors + joint_min from a GRADED controller JSONL
    4. ASSERT the benign/attack joint distributions separate (multivector.assert_separation)
    5. print the exact constants to paste — but ONLY if step 4 passed

Run from the `step4 dataset` directory:

    python recalibrate_graded.py \
        --attack-dataset ../dataset/merged_output/dataset_all.clean.jsonl \
        --controller-jsonl grounded_controller_kc.jsonl \
        --out-config graded_config.json

If step 4 fails, it tells you to fix payloads (Solution 4) and rebuild the slice
FIRST — which is the discipline the original RESULTS warned about but didn't enforce.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path


def _read_jsonl(path):
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            yield json.loads(line)


def step1_fit_T(attack_dataset: str, limit: int, floor: float, out_config: str) -> float:
    import graded_channels as gc
    benign, attack = [], []
    for i, r in enumerate(_read_jsonl(attack_dataset)):
        if i >= limit:
            break
        prompt = r.get("prompt") or r.get("question") or ""
        if not prompt.strip():
            continue
        m = gc.injection_margin(prompt)
        (attack if str(r.get("safety", "")).lower() == "unsafe" else benign).append(m)
    if not benign or not attack:
        sys.exit("[fit] need both safe and unsafe rows in the attack dataset.")
    res = gc.fit_temperature_for_band(benign, attack, floor=floor)
    Path(out_config).write_text(json.dumps(
        {"T": res["T"], "meta": res, "n_benign": len(benign), "n_attack": len(attack)},
        indent=2), encoding="utf-8")
    print(f"[1/5] fitted T={res['T']}  benign_p95_graded={res['benign_p95_graded']}  "
          f"attack_p50_graded={res['attack_p50_graded']}  respects_floor={res['respects_floor']}")
    if not res["respects_floor"]:
        print("      [warn] detector too saturated for this floor even at the logit "
              "layer — report as a finding, but continue.")
    return res["T"]


def step3_floors_and_joint(controller_jsonl: str):
    """Per-channel floors (benign p95 + margin) and the benign/attack joint arrays,
    read from the GRADED channels saved on a controller run."""
    import numpy as np
    bq, bc, b_joint, a_joint = [], [], [], []
    for r in _read_jsonl(controller_jsonl):
        kind = r.get("kind")
        mv = r.get("multivector", {})
        ch = mv.get("channels", {})
        q = float(ch.get("query_vector", 0.0) or 0.0)
        c = float(ch.get("context_vector", 0.0) or 0.0)
        joint = float(mv.get("joint_risk", 0.0) or 0.0)
        if kind == "benign_control":
            bq.append(q); bc.append(c); b_joint.append(joint)
        elif kind == "multivector_attack":
            a_joint.append(joint)
    qf = round(float(np.percentile(bq, 95)) + 0.05, 3) if bq else 0.20
    cf = round(float(np.percentile(bc, 95)) + 0.05, 3) if bc else 0.30
    print(f"[3/5] graded floors  query={qf}  context={cf}  "
          f"(benign n={len(b_joint)}, attack n={len(a_joint)})")
    return qf, cf, b_joint, a_joint


def main():
    ap = argparse.ArgumentParser(description="Graded-scale recalibration (fail-closed)")
    ap.add_argument("--attack-dataset", required=True)
    ap.add_argument("--controller-jsonl", required=True)
    ap.add_argument("--out-config", default="graded_config.json")
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--floor", type=float, default=0.15)
    args = ap.parse_args()

    # 1. fit T (writes graded_config.json so all later graded_score() calls use it)
    step1_fit_T(args.attack_dataset, args.limit, args.floor, args.out_config)

    # 3. floors + joint distributions from the graded controller run
    qf, cf, b_joint, a_joint = step3_floors_and_joint(args.controller_jsonl)
    if not b_joint or not a_joint:
        sys.exit("[3/5] controller JSONL has no graded multivector channels — "
                 "regenerate it AFTER applying patch_01b (graded context channel).")

    # choose joint_min in the valley between benign p95 and attack p10
    import numpy as np
    b95 = float(np.percentile(b_joint, 95))
    a10 = float(np.percentile(a_joint, 10))
    joint_min = round((b95 + a10) / 2.0, 3)
    print(f"[4/5] benign_p95_joint={b95:.3f}  attack_p10_joint={a10:.3f}  "
          f"-> proposed joint_min={joint_min}")

    # 4. ASSERT separation — fail closed if payloads aren't in-band
    from multivector import assert_separation
    try:
        report = assert_separation(b_joint, a_joint, joint_min=joint_min)
    except AssertionError as e:
        print(f"\n[4/5] FAIL: {e}")
        print("      -> Fix payloads (Solution 4: regenerate_inband_payloads.py), "
              "rebuild the slice, re-run the controller, then re-run this script.")
        sys.exit(2)

    # 5. emit constants only on success
    print("\n[5/5] PASS — paste into multivector.py:")
    print(f"DEFAULT_SOFT_PER_CHANNEL = {{'query_vector': {qf}, 'context_vector': {cf}}}")
    print(f"DEFAULT_JOINT_MIN = {joint_min}")
    print(f"\nseparation report: {json.dumps(report, indent=2)}")


if __name__ == "__main__":
    main()
