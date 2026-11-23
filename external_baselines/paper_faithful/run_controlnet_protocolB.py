#!/usr/bin/env python3
"""
run_controlnet_protocolB.py -- the paper-faithful ControlNET detector (controlnet_faithful.py)
applied to Protocol B: the 200 planted attacks and the 200 benign questions.

Training is identical to controlnet_faithful.py: layer-1 last-token activations of the
reduced Llama-3.1-8B GGUF, Activation Shift Index against the 200 benign anchors, decision tree
on ASI with depth chosen by validation F1 (train/val/anchor rows from contexts_bge.jsonl).
The trained tree then scores each Protocol B row, in the paper's "Knowledge / System Prompt /
Query" format. By default (--context generator) the knowledge is exactly the 8 reranked passages
the undefended generator received; --context top5_plus_poison reproduces the earlier variant. Passages come from
step4 dataset/mitigation_results/baselines_protocolB/contexts_{attack,benign}.jsonl
(dump_undefended_contexts.py).

ControlNET only blocks or passes, so a guarded deployment delivers the undefended pipeline's
stored answer on rows it does not block; the comparison is scored by score_baselines_protocolB.py.

    python run_controlnet_protocolB.py --gguf-dir <dir with llama3.1-8b_blk0.gguf>
Output: step4 dataset/mitigation_results/baselines_protocolB/controlnet_{attack,benign}.jsonl,
        controlnet_protocolB_report.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import controlnet_faithful as cf  # noqa: E402
from controlnet_faithful import Layer1, acts, asi, fmt  # noqa: E402

# controlnet_faithful.py passes dtype=..., which transformers < 4.56 calls torch_dtype=...
# Translate it here so the original runner stays unmodified; the loaded weights are the same.
_from_pretrained = cf.AutoModel.from_pretrained


def _compat_from_pretrained(*a, **k):
    if "dtype" in k:
        try:
            return _from_pretrained(*a, **k)
        except TypeError:
            k["torch_dtype"] = k.pop("dtype")
    return _from_pretrained(*a, **k)


cf.AutoModel.from_pretrained = _compat_from_pretrained

D = HERE.parent.parent / "step4 dataset" / "mitigation_results" / "baselines_protocolB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf-dir", required=True)
    ap.add_argument("--gguf-file", default="llama3.1-8b_blk0.gguf")
    ap.add_argument("--tag", default="llama3.1-8b")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--context", choices=["generator", "top5_plus_poison"], default="generator",
                    help="generator: exactly the 8 reranked passages the undefended generator received "
                         "(the planted chunk only where it survived reranking; 184/200 rows). "
                         "top5_plus_poison: controlnet_faithful.py's Protocol A convention, top-5 "
                         "retrieved plus the planted chunk on every attack row (earlier variant).")
    args = ap.parse_args()
    sfx = "" if args.context == "generator" else "_top5_plus_poison"
    cdir = HERE / "results" / "cache"
    cdir.mkdir(parents=True, exist_ok=True)

    C = [json.loads(l) for l in open(HERE / "contexts_bge.jsonl", encoding="utf-8")]
    anchors = [r for r in C if r["split"] == "anchor"]
    train = [r for r in C if r["split"] == "train"]
    val = [r for r in C if r["split"] == "val"]
    rows = {s: [json.loads(l) for l in open(D / f"contexts_{s}.jsonl", encoding="utf-8")] for s in ("attack", "benign")}

    def docs(r):
        if args.context == "generator":
            return r["ranked_context"]
        poison = r.get("poison_chunk")
        d = [c for c in r["retrieved_top8"] if c != poison][:5]
        return d + [poison] if poison else d

    model = Layer1(args.gguf_dir, args.gguf_file, args.max_len)
    A = acts(model, [fmt(r["question"], r["chunks_fresh"]) for r in anchors], f"{cdir}/{args.tag}_anchor.npz")
    T = acts(model, [fmt(r["question"], r["chunks_fresh"]) for r in train], f"{cdir}/{args.tag}_train.npz")
    V = acts(model, [fmt(r["question"], r["chunks_fresh"]) for r in val], f"{cdir}/{args.tag}_val_{len(val)}.npz")
    P = {s: acts(model, [fmt(r["question"], docs(r)) for r in rs], f"{cdir}/{args.tag}_protocolB_{s}{sfx}.npz")
         for s, rs in rows.items()}
    ytr = np.array([r["label"] for r in train]); yva = np.array([r["label"] for r in val])

    report = {"model": args.tag, "layer": 1, "n_anchor": len(anchors), "n_train": len(train), "n_val": len(val)}
    out = {s: [{"index": r["index"]} for r in rs] for s, rs in rows.items()}
    for pool, k in (("last", 0), ("mean", 1)):
        a_tr, a_va = asi(T[k], A[k]), asi(V[k], A[k])
        best = None
        for depth in (1, 2, 3, 4):
            dt = DecisionTreeClassifier(max_depth=depth, class_weight="balanced", random_state=0).fit(a_tr[:, None], ytr)
            f1 = f1_score(yva, dt.predict(a_va[:, None]))
            if best is None or f1 > best[0]:
                best = (f1, depth, dt)
        f1, depth, dt = best
        sc = {s: asi(P[s][k], A[k]) for s in rows}
        y = np.r_[np.ones(len(sc["attack"])), np.zeros(len(sc["benign"]))]
        report[pool] = {"val_f1": float(f1), "tree_depth": depth,
                        "protocolB_auroc_asi_attack_vs_benign": float(roc_auc_score(y, np.r_[sc["attack"], sc["benign"]]))}
        for s in rows:
            pred = dt.predict(sc[s][:, None])
            for o, x, p in zip(out[s], sc[s], pred):
                o[f"asi_{pool}"] = float(x); o[f"blocked_{pool}"] = bool(p)
    for s in rows:
        with open(D / f"controlnet_{s}{sfx}.jsonl", "w", encoding="utf-8") as fh:
            for o in out[s]:
                o["blocked"] = o["blocked_last"]          # headline read-out, as in controlnet_faithful.py
                fh.write(json.dumps(o) + "\n")
    report["blocked_attack"] = sum(o["blocked"] for o in out["attack"])
    report["blocked_benign"] = sum(o["blocked"] for o in out["benign"])
    report["context"] = args.context
    json.dump(report, open(D / f"controlnet_protocolB_report{sfx}.json", "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
