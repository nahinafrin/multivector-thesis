"""
graded_channels.py  (v2) — read the injection detector's PRE-SIGMOID margin so
the multi-vector channels arrive GRADED instead of saturated at 0/1.

WHY (one line): LLM-Guard returns a softmax PROBABILITY that saturates (~0/~1);
the magnitude needed for a sub-threshold band survives only in the logits, as
margin = logit(INJECTION) - logit(SAFE). We read that margin and squash it with a
deliberately SOFTENING temperature T>1 so confident scores spread across [0,1].

This is ADDITIVE: Step 3 / Step 6 block decisions (tuned on the squashed scale)
are untouched; only the multi-vector channels read the graded score.

The softening temperature T is loaded from graded_config.json if present (fit it
once with `python graded_channels.py fit ...`); otherwise DEFAULT_T is used, so
the module works out of the box and improves once you fit T on your data.
"""
from __future__ import annotations
import json, math
from pathlib import Path
from typing import Iterable

MODEL_NAME = "protectai/deberta-v3-base-prompt-injection-v2"
DEFAULT_T = 3.0
CONFIG_PATH = "graded_config.json"

_BUNDLE = None          # (tokenizer, model, inj_idx)

def _load_T() -> float:
    p = Path(CONFIG_PATH)
    if p.exists():
        try:
            return float(json.loads(p.read_text(encoding="utf-8")).get("T", DEFAULT_T))
        except Exception:
            return DEFAULT_T
    return DEFAULT_T

_T = _load_T()

def _load_model():
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE
    import torch  # noqa
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    id2label = getattr(model.config, "id2label", {0: "SAFE", 1: "INJECTION"})
    inj_idx = next((i for i, lbl in id2label.items() if "INJ" in str(lbl).upper()), 1)
    _BUNDLE = (tok, model, int(inj_idx))
    return _BUNDLE

def injection_margin(text: str) -> float:
    """Unbounded logit margin = logit(INJECTION) - logit(SAFE)."""
    import torch
    tok, model, inj_idx = _load_model()
    enc = tok(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**enc).logits[0]
    safe_idx = 1 - inj_idx if logits.shape[0] == 2 else int(logits.argmin())
    return float(logits[inj_idx] - logits[safe_idx])

def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))

def graded_from_margin(margin: float, T: float | None = None) -> float:
    return _sigmoid(margin / max(T if T is not None else _T, 1e-6))

def graded_score(text: str, T: float | None = None) -> float:
    """sigmoid(margin / T) in [0,1]; uses the fitted/config T by default."""
    return graded_from_margin(injection_margin(text), T)

def fit_temperature_for_band(benign_margins: Iterable[float],
                             attack_margins: Iterable[float],
                             *, floor: float = 0.15, p_benign: float = 95.0) -> dict:
    """Largest softening T s.t. benign p95 margin maps below `floor`."""
    import numpy as np
    bm = np.asarray(list(benign_margins), dtype="float64")
    am = np.asarray(list(attack_margins), dtype="float64")
    b95 = float(np.percentile(bm, p_benign))
    L = math.log(floor / (1.0 - floor))            # logit(floor) < 0
    T = max(1.0, b95 / L) if (b95 < 0 < (b95 / L)) else DEFAULT_T
    return {
        "T": round(T, 4), "floor": floor,
        "benign_p95_margin": round(b95, 4),
        "benign_p95_graded": round(_sigmoid(b95 / T), 4),
        "attack_p10_graded": round(_sigmoid(float(np.percentile(am, 10)) / T), 4),
        "attack_p50_graded": round(_sigmoid(float(np.percentile(am, 50)) / T), 4),
        "respects_floor": _sigmoid(b95 / T) <= floor,
    }

# --------------------------------------------------------------------------- #
def _read_jsonl(path):
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            yield json.loads(line)

def _cmd_fit(args):
    benign, attack = [], []
    cache = Path(args.cache) if args.cache else None
    if cache and cache.exists():
        for d in _read_jsonl(cache):
            (attack if d["label"] == 1 else benign).append(d["margin"])
        print(f"[fit] loaded {len(benign)+len(attack)} cached margins")
    else:
        pairs = []
        for i, r in enumerate(_read_jsonl(args.dataset)):
            if args.limit and i >= args.limit:
                break
            prompt = r.get("prompt") or r.get("question") or ""
            if not prompt.strip():
                continue
            m = injection_margin(prompt)
            lab = 1 if str(r.get("safety", "")).lower() == "unsafe" else 0
            (attack if lab else benign).append(m)
            pairs.append({"margin": m, "label": lab})
            if (i + 1) % 100 == 0:
                print(f"  ...scored {i+1}")
        if cache:
            with open(cache, "w", encoding="utf-8") as f:
                for p in pairs:
                    f.write(json.dumps(p) + "\n")
            print(f"[fit] wrote margin cache -> {cache}")
    if not benign or not attack:
        print("[fit] need both safe and unsafe rows; aborting.")
        return
    res = fit_temperature_for_band(benign, attack, floor=args.floor)
    Path(args.out).write_text(json.dumps({"T": res["T"], "meta": res,
        "n_benign": len(benign), "n_attack": len(attack)}, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    print(f"[saved] {args.out}  (T={res['T']})")
    if not res["respects_floor"]:
        print("[warn] benign p95 graded > floor: signal too saturated even at "
              "the logit layer for this floor — report as a detector finding.")

def _cmd_show(args):
    print(f"active T = {_T}  (config: {CONFIG_PATH})")
    print(f"{'margin':>8}{'prob':>10}{'graded':>10}")
    for m in (-10, -6, 0.4, 1, 2, 3, 5, 8):
        print(f"{m:>8.1f}{_sigmoid(m):>10.4f}{graded_from_margin(m):>10.4f}")

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Graded injection channel via logit margin")
    sub = ap.add_subparsers(dest="mode", required=True)
    fp = sub.add_parser("fit", help="fit softening T from a labelled dataset")
    fp.add_argument("--dataset", required=True)
    fp.add_argument("--limit", type=int, default=800)
    fp.add_argument("--floor", type=float, default=0.15)
    fp.add_argument("--cache", default=None)
    fp.add_argument("--out", default=CONFIG_PATH)
    fp.set_defaults(func=_cmd_fit)
    sp = sub.add_parser("show", help="print active T and a transform table")
    sp.set_defaults(func=_cmd_show)
    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
