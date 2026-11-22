"""
Reimplementation of a ControlNET-style RAG firewall baseline: a SINGLE
activation-shift score computed from a local model's hidden states, with NO
separate query/context channels and NO retry loop (single-shot
classify-then-block) -- faithful to the architecture already characterized
in this project's own novelty-positioning-comparison.md ("single activation-
shift score, no separate query+context channels"; "single-shot classify-
then-block"; "sole signal is activation-shift divergence").

This is a REIMPLEMENTATION from the paper's described architecture, not the
authors' original (unpublished) code. Label it as such in the thesis.

Requires a local HF model with exposed hidden states. Ollama does not expose
these, so this deliberately uses a small `transformers`-loaded model instead
of one of the three Ollama models in the generation ensemble -- that's fine,
this baseline probes a DIFFERENT model's internals than the generation
ensemble, it doesn't need to share a backbone with it.

Usage:
    python baseline_controlnet.py \
        --slice adversarial_slice.jsonl \
        --train train.jsonl --val val.jsonl \
        --model sentence-transformers/all-MiniLM-L6-v2 \
        --out grounded_controlnet.jsonl
"""
import argparse
import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from transformers import AutoModel, AutoTokenizer


def get_activation(text, tokenizer, model, layer=-4, device="cpu"):
    """Mean-pooled hidden state at a mid-to-late layer -- the standard probe
    point for this class of activation-shift defense."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    h = out.hidden_states[layer][0]  # (seq_len, hidden_dim)
    return h.mean(dim=0).cpu().numpy()


def build_full_text(row):
    """ControlNET's own design has NO separate query/context channels --
    concatenate query and retrieved context into a single input. This is
    the point of the baseline: it shows exactly what is lost by not
    separating the two channels the way this project's own multivector.py
    (query_vector / context_vector, each with its own floor) does."""
    ctx = row.get("retrieved_text") or row.get("context") or row.get("poison_injected") or ""
    query = row.get("prompt", row.get("query", ""))
    return f"{query}\n\n{ctx}".strip()


def calibrate(train_rows, val_rows, tok, model):
    X_train = np.stack([get_activation(build_full_text(r), tok, model) for r in train_rows])
    y_train = np.array([0 if r.get("safety") == "safe" else 1 for r in train_rows])

    # A single fused score: logistic-regression probability over the raw
    # activation vector. This IS "the" activation-shift score once there is
    # no separate-channel architecture left to fuse -- one number, one
    # threshold, classify-then-block.
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_train, y_train)

    X_val = np.stack([get_activation(build_full_text(r), tok, model) for r in val_rows])
    y_val = np.array([0 if r.get("safety") == "safe" else 1 for r in val_rows])
    probs_val = clf.predict_proba(X_val)[:, 1]

    best_tau, best_f1 = 0.5, -1.0
    for tau in np.linspace(0.1, 0.9, 33):
        f1 = f1_score(y_val, (probs_val >= tau).astype(int))
        if f1 > best_f1:
            best_f1, best_tau = f1, float(tau)
    print(f"[calibration] best_tau={best_tau:.3f} val_f1={best_f1:.3f}")
    return clf, best_tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.eval()

    train_rows = [json.loads(line) for line in open(args.train, encoding="utf-8")]
    val_rows = [json.loads(line) for line in open(args.val, encoding="utf-8")]
    clf, tau = calibrate(train_rows, val_rows, tok, model)

    n_rows = 0
    with open(args.slice, encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as out:
        for line in f:
            row = json.loads(line)
            act = get_activation(build_full_text(row), tok, model)
            score = float(clf.predict_proba(act.reshape(1, -1))[0, 1])
            blocked = score >= tau
            row["baseline"] = "controlnet_reimpl"
            row["blocked"] = bool(blocked)
            row["block_stage"] = "controlnet_activation_shift" if blocked else None
            row["baseline_score"] = score
            out.write(json.dumps(row) + "\n")
            n_rows += 1

    print(f"[done] scored {n_rows} rows -> {args.out}")


if __name__ == "__main__":
    main()
