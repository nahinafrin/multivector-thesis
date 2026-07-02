"""
run_planted_mitigation.py — end-to-end DETECTION-vs-MITIGATION experiment on the
pipeline-faithful, planted, in-band conjunctive slice.
================================================================================

This answers the question the separation test could NOT: does the multi-vector
signal actually REDUCE end-to-end attack success (planted payload emitted /
canary leaked), not just fire as a detector?

It runs the SAME planted slice through three arms of the existing pipeline
(no new pipeline code — it shells out to run_full_pipeline.py, which already
exposes every switch we need):

    A. no-controller            : --no-controller
       (baseline: gate + RAG + output rail, no closed-loop risk feedback)
    B. controller, MV signal OFF: --no-multivector-signal
       (closed loop minus the multi-vector signal — isolates everything else
        the controller does from the multi-vector contribution)
    C. controller, MV signal ON : (default)
       (full system)

Then it scores all three with score_attack_success.py and reports, per arm and
per attack class:
    * neutralization rate with a Wilson 95% CI (small-n honest)
    * the C - B delta = the marginal contribution of the MULTI-VECTOR SIGNAL
      to end-to-end MITIGATION (not detection)
    * a McNemar test on the paired B-vs-C rows (same slice, two conditions) so
      the delta is not overclaimed on a handful of rows.

IMPORTANT — this measures MITIGATION (payload reached the user or not), which is
the claim "does it mitigate multi-vector attacks?" actually requires. Detector
firing is reported too, but the headline is neutralization.

PREREQUISITE — build the planted slice first:
    python build_multivector_inband.py --qa-jsonl data/question-answer/test.jsonl \
        --index ./kb_wiki --k 5 --n 80 --planted --out planted_inband.jsonl

USAGE:
    python run_planted_mitigation.py --slice planted_inband.jsonl --index ./kb_wiki
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

# Reuse the existing scorer rather than reimplementing the neutralization logic.
import score_attack_success as sas


# --------------------------------------------------------------------------- #
# Stats helpers (Wilson CI + McNemar) — mirror the discipline used elsewhere.
# --------------------------------------------------------------------------- #
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n. Honest at small n
    (unlike the normal approximation, which produces nonsense near 0/1)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact (binomial) two-sided McNemar p-value on the two discordant cells
    b and c. Used because the slice is small and the chi-square approximation
    is unreliable for small b+c. b = neutralized-by-B-but-not-C,
    c = neutralized-by-C-but-not-B (discordant pairs only)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-sided exact binomial at p=0.5
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


# --------------------------------------------------------------------------- #
# Arm runner
# --------------------------------------------------------------------------- #
def run_arm(slice_file: str, index: str, out: str, extra_flags: list[str],
            *, python: str = sys.executable, resume: bool = False) -> None:
    cmd = [python, "run_full_pipeline.py",
           "--slice", slice_file, "--index", index, "--out", out, *extra_flags]
    if resume:
        cmd.append("--resume")
    print(f"\n[arm] {' '.join(extra_flags) or '(full controller)'}\n  -> {out}")
    print("  $ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_rows(path: str) -> dict[str, dict]:
    """Index a run's rows by id so B and C can be paired for McNemar."""
    rows = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rid = r.get("id", r.get("index"))
        rows[str(rid)] = r
    return rows


def neutralized_by_id(path: str) -> dict[str, bool]:
    """Map row id -> attack_neutralized (True/False), reusing the scorer's own
    per-row verdict so this stays consistent with score_attack_success.py."""
    out = {}
    for rid, r in load_rows(path).items():
        if r.get("kind") not in sas.ATTACK_KINDS:
            continue
        v = sas._row_verdict(r)
        out[rid] = bool(v["attack_neutralized"])
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slice", dest="slice_file", required=True,
                    help="planted in-band slice from build_multivector_inband.py --planted")
    ap.add_argument("--index", default="./kb_wiki")
    ap.add_argument("--outdir", default="mitigation_run")
    ap.add_argument("--resume", action="store_true",
                    help="pass --resume to each arm (skip ids already in that arm's out)")
    ap.add_argument("--skip-run", action="store_true",
                    help="don't re-run arms; just re-score existing outputs in --outdir")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    arm_a = str(outdir / "arm_A_no_controller.jsonl")
    arm_b = str(outdir / "arm_B_no_mv_signal.jsonl")
    arm_c = str(outdir / "arm_C_full.jsonl")

    if not args.skip_run:
        run_arm(args.slice_file, args.index, arm_a, ["--no-controller"], resume=args.resume)
        run_arm(args.slice_file, args.index, arm_b, ["--no-multivector-signal"], resume=args.resume)
        run_arm(args.slice_file, args.index, arm_c, [], resume=args.resume)

    # ---- score each arm with the existing end-to-end scorer ----------------- #
    reports = {name: sas.score_file(p) for name, p in
               [("A_no_controller", arm_a), ("B_no_mv_signal", arm_b), ("C_full", arm_c)]}

    print("\n" + "=" * 74)
    print("END-TO-END MITIGATION (neutralization = payload did NOT reach user)")
    print("=" * 74)
    for name, rep in reports.items():
        k = rep["overall_neutralized"]
        n = rep["overall_attack_rows"]
        lo, hi = wilson_ci(k, n)
        print(f"\n  arm {name:16}  neutralized {k}/{n} = "
              f"{(rep['overall_neutralization_rate'] or 0):.1%}  "
              f"(Wilson 95% CI [{lo:.1%}, {hi:.1%}])")

    # ---- the number that answers the question: C - B on MITIGATION ---------- #
    b_neu = neutralized_by_id(arm_b)
    c_neu = neutralized_by_id(arm_c)
    common = sorted(set(b_neu) & set(c_neu))
    kb = sum(b_neu[i] for i in common)
    kc = sum(c_neu[i] for i in common)
    n = len(common)

    # discordant cells for McNemar
    b_only = sum(1 for i in common if b_neu[i] and not c_neu[i])  # B neut, C not
    c_only = sum(1 for i in common if c_neu[i] and not b_neu[i])  # C neut, B not
    p = mcnemar_exact(b_only, c_only)

    lo_b, hi_b = wilson_ci(kb, n)
    lo_c, hi_c = wilson_ci(kc, n)

    print("\n" + "-" * 74)
    print("MULTI-VECTOR SIGNAL CONTRIBUTION TO MITIGATION  (paired B vs C)")
    print("-" * 74)
    print(f"  paired attack rows            : {n}")
    if n:
        print(f"  B (MV off)  neutralized       : {kb}/{n} = {kb/n:.1%}  "
              f"CI [{lo_b:.1%}, {hi_b:.1%}]")
        print(f"  C (MV on)   neutralized       : {kc}/{n} = {kc/n:.1%}  "
              f"CI [{lo_c:.1%}, {hi_c:.1%}]")
        print(f"  delta (C - B)                 : {(kc-kb)}/{n} rows "
              f"= {((kc-kb)/n if n else 0):+.1%}")
    else:
        print("  (no paired rows)")
    print(f"  discordant: C-only={c_only}  B-only={b_only}")
    print(f"  McNemar exact two-sided p     : {p:.4f}")

    print("\n  READ: the delta is the marginal end-to-end MITIGATION attributable")
    print("  to the multi-vector signal specifically. If delta > 0 with a small p,")
    print("  the signal reduces payload emission, not just fires as a detector.")
    print("  If delta ~ 0, the signal DETECTS (per the separation test) but does")
    print("  NOT independently improve the end-to-end outcome on this slice —")
    print("  which is itself an honest, reportable finding, not a failure.")

    out_json = outdir / "mitigation_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "arms": reports,
            "paired_B_vs_C": {
                "n_paired": n,
                "B_neutralized": kb, "C_neutralized": kc,
                "delta_rows": kc - kb,
                "delta_rate": (kc - kb) / n if n else None,
                "B_wilson_ci": [lo_b, hi_b], "C_wilson_ci": [lo_c, hi_c],
                "discordant_C_only": c_only, "discordant_B_only": b_only,
                "mcnemar_exact_p": p,
            },
        }, f, indent=2)
    print(f"\n[saved] {out_json}")


if __name__ == "__main__":
    main()
