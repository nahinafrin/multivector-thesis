"""
calibration.py — temperature / Platt calibration for the near-binary soft scores
=================================================================================

WHY THIS EXISTS
---------------
The injection scanner (deberta-v3) and the per-chunk context scorer are
NEAR-BINARY: benign text scores ~0.0 and injection ~0.9-1.0, with almost nothing
in between (see calibrate_payloads.py). That sparsity is fatal for the
multi-vector premise: a conjunctive sub-threshold attack needs each channel to
land in the [floor, block) band, and a saturated 0/1 score never does. It is
also why "0 in-band payloads" keeps showing up — you cannot hand-author strings
that hit a band the detector refuses to emit.

Calibration fixes this at the score level instead of the payload level. We map
the raw probability through its logit and rescale, so the confident-but-not-
identical raw scores (0.92 vs 0.97 vs 0.999 -> logits 2.4, 3.5, 6.9) spread back
into a usable graded band that is monotonic in the original ranking.

Two transforms, both monotonic and invertible:

  temperature : p_cal = sigmoid(logit(p_raw) / T)          (unsupervised; T>1
                softens an over-confident detector toward the middle)
  platt       : p_cal = sigmoid(a * logit(p_raw) + b)      (supervised; fit a,b
                on labelled (score, is_attack) pairs by logistic regression)

The transforms operate on logit(clip(p, eps)) so saturated 0.0 / 1.0 inputs are
handled gracefully (clipped to eps / 1-eps) rather than producing +/-inf.

WHERE IT PLUGS IN
-----------------
Calibration is applied to the CHANNELS that feed multivector.multivector_risk
(query_vector, context_vector) — NOT to the C3RF gate's s_I (whose thresholds
are tuned on the raw scale) and NOT to Step-6's drop threshold. This localizes
the change to exactly the path that needs a graded signal. Because
calibrate_payloads.py derives the per-channel floors + joint_min from the
channel values written to each row, calibrating the channels means the floors
are automatically re-derived on the calibrated scale — the loop stays
self-consistent: calibrate channels -> re-fit floors -> re-run the slice.

RUN
---
    # Fit query_vector from the labelled attack dataset (scores prompts itself).
    python calibration.py fit-dataset \
        --dataset ./merged_output/dataset_all.clean.jsonl --limit 800 \
        --method platt --channel query_vector

    # Fit any channel from a precomputed (score, label) JSONL.
    python calibration.py fit \
        --scores my_scores.jsonl \
        --score-field gate.injection_score --label-field safety \
        --label-positive unsafe --method platt --channel query_vector

    # Inspect the saved calibration and a before/after transform table.
    python calibration.py show

Writes calibration.json (per-channel params). load_calibration() returns an
identity CalibrationSet when the file is absent, so the pipeline runs unchanged
until you have actually fit something.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

DEFAULT_PATH = "calibration.json"
DEFAULT_EPS = 1e-4


# --------------------------------------------------------------------------- #
# Numeric helpers
# --------------------------------------------------------------------------- #
def _logit(p: np.ndarray, eps: float = DEFAULT_EPS) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable logistic sigmoid.
    out = np.empty_like(z, dtype="float64")
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _nll(p_cal: np.ndarray, y: np.ndarray, eps: float = DEFAULT_EPS) -> float:
    p = np.clip(p_cal, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _brier(p_cal: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p_cal - y) ** 2))


# --------------------------------------------------------------------------- #
# Calibrator
# --------------------------------------------------------------------------- #
@dataclass
class Calibrator:
    """A single monotonic score transform p_raw -> p_cal.

    method == "identity"    : returns the input unchanged.
    method == "temperature" : p_cal = sigmoid(logit(p_raw) / T).
    method == "platt"       : p_cal = sigmoid(a * logit(p_raw) + b).
    """

    method: str = "identity"
    a: float = 1.0          # platt slope
    b: float = 0.0          # platt intercept
    temperature: float = 1.0
    eps: float = DEFAULT_EPS

    def transform(self, p: Any) -> Any:
        arr = np.atleast_1d(np.asarray(p, dtype="float64"))
        if self.method == "identity":
            out = np.clip(arr, 0.0, 1.0)
        elif self.method == "temperature":
            out = _sigmoid(_logit(arr, self.eps) / max(self.temperature, 1e-6))
        elif self.method == "platt":
            out = _sigmoid(self.a * _logit(arr, self.eps) + self.b)
        else:
            raise ValueError(f"unknown calibration method: {self.method!r}")
        out = np.clip(out, 0.0, 1.0)
        return float(out[0]) if np.isscalar(p) or np.ndim(p) == 0 else out

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "a": self.a,
            "b": self.b,
            "temperature": self.temperature,
            "eps": self.eps,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Calibrator":
        return cls(
            method=d.get("method", "identity"),
            a=float(d.get("a", 1.0)),
            b=float(d.get("b", 0.0)),
            temperature=float(d.get("temperature", 1.0)),
            eps=float(d.get("eps", DEFAULT_EPS)),
        )


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #
def fit_temperature(scores: np.ndarray, labels: np.ndarray,
                    eps: float = DEFAULT_EPS) -> Calibrator:
    """Fit a single temperature T>0 minimising NLL of sigmoid(logit(p)/T).

    Golden-section search over log-temperature in [-4, 4] (T in ~[0.018, 55]);
    no SciPy dependency.
    """
    x = _logit(scores, eps)
    y = labels.astype("float64")

    def loss(log_t: float) -> float:
        t = math.exp(log_t)
        return _nll(_sigmoid(x / t), y, eps)

    lo, hi = -4.0, 4.0
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)
    fc, fd = loss(c), loss(d)
    for _ in range(80):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo)
            fc = loss(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo)
            fd = loss(d)
    log_t = (lo + hi) / 2.0
    return Calibrator(method="temperature", temperature=math.exp(log_t), eps=eps)


def fit_platt(scores: np.ndarray, labels: np.ndarray, eps: float = DEFAULT_EPS,
              l2: float = 1.0, iters: int = 100) -> Calibrator:
    """Fit p_cal = sigmoid(a*logit(p)+b) by ridge-regularised IRLS on (a, b).

    The feature logit(clip(p)) is STANDARDISED before fitting so the L2 ridge has
    a consistent scale and so near-separable, near-binary inputs (which drive the
    unregularised MLE to infinity) produce a finite, sane slope instead of a
    1e9 blow-up. The fitted weights are then mapped back to the raw-logit scale.
    Falls back to identity when the feature has no spread (all scores equal).
    """
    x = _logit(scores, eps)
    y = labels.astype("float64")
    m, s = float(np.mean(x)), float(np.std(x))
    if s < 1e-9:                              # no resolution to calibrate
        return Calibrator(method="identity", eps=eps)

    xs = (x - m) / s
    n = len(xs)
    X = np.column_stack([xs, np.ones(n)])     # design matrix for [w1, w0]
    w = np.zeros(2, dtype="float64")
    ridge = l2 * np.eye(2)
    for _ in range(iters):
        p = _sigmoid(X @ w)
        grad = X.T @ (p - y) + l2 * w
        S = p * (1.0 - p)
        H = X.T @ (X * S[:, None]) + ridge
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        w_new = w - step
        if not np.all(np.isfinite(w_new)):
            break
        if np.max(np.abs(w_new - w)) < 1e-9:
            w = w_new
            break
        w = w_new

    # Map standardized weights back to the raw-logit scale: a*x + b.
    a = float(w[0] / s)
    b = float(w[1] - w[0] * m / s)
    return Calibrator(method="platt", a=a, b=b, eps=eps)


def fit(scores: Iterable[float], labels: Iterable[int], *, method: str = "platt",
        eps: float = DEFAULT_EPS) -> tuple[Calibrator, dict]:
    """Fit a calibrator and return (calibrator, diagnostics)."""
    s = np.asarray(list(scores), dtype="float64")
    y = np.asarray(list(labels), dtype="float64")
    if len(s) != len(y) or len(s) == 0:
        raise ValueError("scores and labels must be non-empty and equal length")

    if method == "temperature":
        cal = fit_temperature(s, y, eps)
    elif method == "platt":
        cal = fit_platt(s, y, eps)
    elif method == "identity":
        cal = Calibrator(method="identity", eps=eps)
    else:
        raise ValueError(f"unknown method {method!r}")

    p_cal = np.atleast_1d(cal.transform(s))
    diag = _diagnostics(s, p_cal, y)
    diag["method"] = method
    diag["n"] = int(len(s))
    diag["params"] = cal.to_dict()
    return cal, diag


def _percentile(xs: np.ndarray, p: float) -> float:
    return float(np.percentile(xs, p)) if len(xs) else 0.0


def _diagnostics(raw: np.ndarray, cal: np.ndarray, y: np.ndarray) -> dict:
    """Before/after spread + separation. The headline is how much the benign
    p95 / attack p10 SEPARATE after calibration — that separation is what makes
    a graded floor meaningful."""
    pos_raw, neg_raw = raw[y == 1], raw[y == 0]
    pos_cal, neg_cal = cal[y == 1], cal[y == 0]
    return {
        "raw": {
            "benign_p50": round(_percentile(neg_raw, 50), 4),
            "benign_p95": round(_percentile(neg_raw, 95), 4),
            "attack_p10": round(_percentile(pos_raw, 10), 4),
            "attack_p50": round(_percentile(pos_raw, 50), 4),
            "nll": round(_nll(raw, y), 4),
            "brier": round(_brier(raw, y), 4),
            # fraction of scores stuck within 0.01 of {0,1}: the saturation rate
            "saturated_frac": round(float(np.mean((raw < 0.01) | (raw > 0.99))), 4),
        },
        "calibrated": {
            "benign_p50": round(_percentile(neg_cal, 50), 4),
            "benign_p95": round(_percentile(neg_cal, 95), 4),
            "attack_p10": round(_percentile(pos_cal, 10), 4),
            "attack_p50": round(_percentile(pos_cal, 50), 4),
            "nll": round(_nll(cal, y), 4),
            "brier": round(_brier(cal, y), 4),
            "saturated_frac": round(float(np.mean((cal < 0.01) | (cal > 0.99))), 4),
        },
        # margin between benign p95 and attack p10 (positive = clean separation)
        "separation_raw": round(_percentile(pos_raw, 10) - _percentile(neg_raw, 95), 4),
        "separation_calibrated": round(
            _percentile(pos_cal, 10) - _percentile(neg_cal, 95), 4),
    }


# --------------------------------------------------------------------------- #
# Per-channel calibration set (the thing the pipeline loads)
# --------------------------------------------------------------------------- #
@dataclass
class CalibrationSet:
    """A registry of per-channel calibrators. Unknown channels are identity."""

    channels: dict[str, Calibrator] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def get(self, channel: str) -> Calibrator:
        return self.channels.get(channel, Calibrator(method="identity"))

    def transform(self, channel: str, value: float) -> float:
        return self.get(channel).transform(value)

    def apply_channels(self, channels: dict[str, float]) -> dict[str, float]:
        """Calibrate a {channel: score} dict; channels with no fitted calibrator
        pass through unchanged."""
        return {c: self.transform(c, v) for c, v in channels.items()}

    def is_identity(self) -> bool:
        return all(c.method == "identity" for c in self.channels.values())

    def to_dict(self) -> dict:
        return {
            "channels": {c: cal.to_dict() for c, cal in self.channels.items()},
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationSet":
        chans = {c: Calibrator.from_dict(v)
                 for c, v in (d.get("channels") or {}).items()}
        return cls(channels=chans, meta=d.get("meta", {}))

    def save(self, path: str = DEFAULT_PATH) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8")


def load_calibration(path: str = DEFAULT_PATH) -> CalibrationSet:
    """Load calibration.json, or an identity set if it does not exist.

    The pipeline calls this once at import; an absent file means the
    multi-vector channels are passed through raw (current behaviour), so adding
    this module changes nothing until you actually fit a calibrator.
    """
    p = Path(path)
    if not p.exists():
        return CalibrationSet()
    try:
        return CalibrationSet.from_dict(json.loads(p.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return CalibrationSet()


# --------------------------------------------------------------------------- #
# CLI data loading
# --------------------------------------------------------------------------- #
def _dig(row: dict, dotted: str) -> Any:
    cur: Any = row
    for key in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _load_score_label_jsonl(path: str, score_field: str, label_field: str,
                            label_positive: str) -> tuple[list[float], list[int]]:
    scores: list[float] = []
    labels: list[int] = []
    skipped = 0
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        raw = _dig(row, score_field)
        lab = _dig(row, label_field)
        if raw is None or lab is None:
            skipped += 1
            continue
        try:
            raw_f = float(raw)
        except (TypeError, ValueError):
            skipped += 1
            continue
        if raw_f < 0:            # sentinel "not scored" (e.g. gate injection_score = -1)
            skipped += 1
            continue
        scores.append(min(1.0, max(0.0, raw_f)))
        labels.append(1 if str(lab).lower() == label_positive.lower() else 0)
    if skipped:
        print(f"[fit] skipped {skipped} rows (missing/sentinel score or label)")
    return scores, labels


def _score_dataset(path: str, limit: Optional[int], threshold: float,
                   cache: Optional[str] = None) -> tuple[list[float], list[int]]:
    """Score prompts through the real injection scanner, reusing the Step-3c
    path so the calibrated channel matches what the pipeline actually emits.

    If ``cache`` is given and exists, load the (score, label) pairs from it
    instead of re-running the scanner (scoring is the slow part). If it is given
    but missing, score once and write the cache for instant re-fits.
    """
    if cache and Path(cache).exists():
        print(f"[fit] loading cached scores from {cache}")
        scores: list[float] = []
        labels: list[int] = []
        for line in open(cache, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            scores.append(float(d["score"]))
            labels.append(int(d["label"]))
        return scores, labels

    # The gate modules (Step 3c + its injection scorer) live in ../dataset, the
    # same place run_full_pipeline.py loads them from. Mirror that path insert so
    # this scorer matches the pipeline exactly.
    import sys
    from pathlib import Path as _Path
    _dataset_dir = (_Path(__file__).resolve().parent.parent / "dataset").resolve()
    if str(_dataset_dir) not in sys.path:
        sys.path.insert(1, str(_dataset_dir))

    import step_03c_fusion_gate as gate
    from pipeline_common import read_jsonl

    scores = []
    labels = []
    for i, row in enumerate(read_jsonl(path)):
        if limit is not None and i >= limit:
            break
        prompt = row.get("prompt") or row.get("question") or ""
        if not prompt.strip():
            continue
        s_i, _detail = gate._injection_score(prompt, threshold=threshold)
        scores.append(float(s_i))
        labels.append(1 if str(row.get("safety", "")).lower() == "unsafe" else 0)
        if (i + 1) % 100 == 0:
            print(f"  ...scored {i + 1}")

    if cache:
        with open(cache, "w", encoding="utf-8") as f:
            for sc, lb in zip(scores, labels):
                f.write(json.dumps({"score": sc, "label": lb}) + "\n")
        print(f"[fit] wrote score cache -> {cache}")
    return scores, labels


# --------------------------------------------------------------------------- #
# CLI reporting
# --------------------------------------------------------------------------- #
def _print_diag(channel: str, diag: dict) -> None:
    r, c = diag["raw"], diag["calibrated"]
    print(f"\n=== calibration fit: channel '{channel}' "
          f"({diag['method']}, n={diag['n']}) ===")
    print(f"  params: {diag['params']}")
    print(f"  {'':14}{'benign_p50':>11}{'benign_p95':>11}"
          f"{'attack_p10':>11}{'attack_p50':>11}{'nll':>8}{'sat':>7}")
    print(f"  {'raw':14}{r['benign_p50']:>11}{r['benign_p95']:>11}"
          f"{r['attack_p10']:>11}{r['attack_p50']:>11}{r['nll']:>8}{r['saturated_frac']:>7}")
    print(f"  {'calibrated':14}{c['benign_p50']:>11}{c['benign_p95']:>11}"
          f"{c['attack_p10']:>11}{c['attack_p50']:>11}{c['nll']:>8}{c['saturated_frac']:>7}")
    print(f"  separation (attack_p10 - benign_p95): "
          f"raw={diag['separation_raw']}  calibrated={diag['separation_calibrated']}")
    if _calibration_helped(diag):
        print("  -> calibration de-saturated the score and kept/widened class "
              "separation. Re-derive floors with calibrate_payloads.py on a fresh "
              "controller run.")
    else:
        print("  -> calibration did NOT help: the raw signal is saturated at "
              "0/1 and/or the classes already overlap (attack_p10 <= benign_p95), "
              "so no monotone transform can recover a graded sub-threshold band. "
              "This is a FINDING about the detector, not a tuning failure.")


def _calibration_helped(diag: dict) -> bool:
    """A calibrator is only worth saving if it actually de-saturates the score
    (meaningfully fewer values stuck at 0/1) without losing class separation."""
    raw, cal = diag["raw"], diag["calibrated"]
    de_saturated = cal["saturated_frac"] <= raw["saturated_frac"] - 0.05
    kept_separation = diag["separation_calibrated"] >= diag["separation_raw"] - 1e-9
    return de_saturated and kept_separation


def _cmd_fit(args: argparse.Namespace) -> None:
    if args.mode == "fit-dataset":
        scores, labels = _score_dataset(args.dataset, args.limit, args.threshold,
                                        cache=args.cache)
    else:
        scores, labels = _load_score_label_jsonl(
            args.scores, args.score_field, args.label_field, args.label_positive)

    n_pos = sum(labels)
    print(f"[fit] {len(scores)} rows, {n_pos} positive / {len(scores) - n_pos} negative")
    if n_pos == 0 or n_pos == len(scores):
        print("[fit] ERROR: need both positive and negative labels to calibrate. "
              "Point --dataset at the labelled attack set (with safety=unsafe rows).")
        return

    cal, diag = fit(scores, labels, method=args.method)
    _print_diag(args.channel, diag)

    helped = _calibration_helped(diag)
    if not helped and not args.force:
        print(f"\n[skip] calibration for '{args.channel}' did not de-saturate the "
              f"signal; leaving it as identity (raw pass-through). Re-run with "
              f"--force to save the fitted params anyway.")
        cal = Calibrator(method="identity")
        saved_method = "identity"
    else:
        saved_method = args.method

    cs = load_calibration(args.out)
    cs.channels[args.channel] = cal
    cs.meta[args.channel] = {
        "method": saved_method,
        "requested_method": args.method,
        "helped": bool(helped),
        "forced": bool(args.force and not helped),
        "n": diag["n"],
        "source": args.dataset if args.mode == "fit-dataset" else args.scores,
        "saturated_frac_raw": diag["raw"]["saturated_frac"],
        "saturated_frac_calibrated": diag["calibrated"]["saturated_frac"],
        "separation_raw": diag["separation_raw"],
        "separation_calibrated": diag["separation_calibrated"],
    }
    cs.save(args.out)
    print(f"\n[saved] {args.out}  channel '{args.channel}' -> {saved_method} "
          f"(all channels: {sorted(cs.channels)})")


def _cmd_show(args: argparse.Namespace) -> None:
    cs = load_calibration(args.out)
    if cs.is_identity() and not cs.channels:
        print(f"No calibration at {args.out} (pipeline runs with identity / raw "
              f"channels).")
        return
    print(f"=== {args.out} ===")
    print(json.dumps(cs.to_dict(), indent=2))
    print("\n--- demo transform (raw -> calibrated) ---")
    demo = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0]
    header = "  raw   " + "".join(f"{c:>16}" for c in sorted(cs.channels))
    print(header)
    for v in demo:
        cells = "".join(f"{cs.transform(c, v):>16.4f}" for c in sorted(cs.channels))
        print(f"  {v:<6.2f}{cells}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate near-binary soft scores")
    sub = ap.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default=DEFAULT_PATH, help="calibration JSON path")
    common.add_argument("--channel", default="query_vector",
                        help="which multivector channel this calibrator serves")
    common.add_argument("--method", default="platt",
                        choices=["platt", "temperature", "identity"])
    common.add_argument("--force", action="store_true",
                        help="save the fitted calibrator even if it does not "
                             "de-saturate the signal (default: fall back to identity)")

    fd = sub.add_parser("fit-dataset", parents=[common],
                        help="score a labelled attack dataset and fit a channel")
    fd.add_argument("--dataset", required=True,
                    help="JSONL with prompt + safety(unsafe/safe) labels")
    fd.add_argument("--limit", type=int, default=None)
    fd.add_argument("--threshold", type=float, default=0.5,
                    help="injection-detector internal cutoff (matches Step 3c)")
    fd.add_argument("--cache", default=None,
                    help="path to load/save scored (score,label) pairs so re-fits "
                         "skip the slow scanner pass")
    fd.set_defaults(func=_cmd_fit)

    fs = sub.add_parser("fit", parents=[common],
                        help="fit from a precomputed (score,label) JSONL")
    fs.add_argument("--scores", required=True)
    fs.add_argument("--score-field", default="gate.injection_score",
                    help="dotted path to the raw score in each row")
    fs.add_argument("--label-field", default="safety",
                    help="dotted path to the label in each row")
    fs.add_argument("--label-positive", default="unsafe",
                    help="label value that counts as the positive (attack) class")
    fs.set_defaults(func=_cmd_fit)

    sh = sub.add_parser("show", parents=[common], help="print saved calibration")
    sh.set_defaults(func=_cmd_show)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
