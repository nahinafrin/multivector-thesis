"""
build_contexts_bge.py -- run on the Windows PC (thesis .venv python, read-only use).

Uses the THESIS pipeline's own retrieval (kb_wiki.faiss + BAAI/bge-m3, 512-d
Matryoshka, identical to step_04/step_05) to produce retrieved context for:
  * all 210 slice rows (fresh top-5; the pipeline-logged chunks are kept too)
  * the ControlNET anchor / train / val prompts
Nothing is written to the thesis folder. Output: contexts_bge.jsonl
"""
import json, os, random, sys
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

THESIS = os.environ.get("THESIS_DIR", r"C:\Users\sarke\Desktop\Multivector methodology-local api extension")
HERE = os.path.dirname(os.path.abspath(__file__))
K, SEED = 5, 0
N_ANCHOR, N_TRAIN = 200, 1500


def main():
    idx = faiss.read_index(os.path.join(THESIS, "step4 dataset", "kb_wiki.faiss"))
    texts = [l.rstrip("\n") for l in open(os.path.join(THESIS, "step4 dataset", "kb_wiki.texts"), encoding="utf-8") if l.strip()]
    assert idx.ntotal == len(texts) and idx.d == 512
    model = SentenceTransformer("BAAI/bge-m3")

    def search(qs):
        v = model.encode(qs, normalize_embeddings=True, batch_size=16, show_progress_bar=True)[:, :512]
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        _, I = idx.search(v.astype("float32"), K)
        return [[texts[i] for i in row] for row in I]

    rows = []
    slice_rows = [json.loads(l) for l in open(os.path.join(THESIS, "step4 dataset", "adversarial_slice.jsonl"), encoding="utf-8")]
    full = {r["index"]: r for r in map(json.loads, open(os.path.join(THESIS, "step4 dataset", "baseline_comparison", "grounded_full.jsonl"), encoding="utf-8"))}
    fresh = search([r["question"] for r in slice_rows])
    agree = n_logged = 0
    for r, fr in zip(slice_rows, fresh):
        poison = r.get("poison_chunk")
        raw = ((full.get(r["id"]) or {}).get("retrieval") or {}).get("raw_chunks")
        logged = [c for c in raw if c != poison][:K] if raw else None
        if logged:
            n_logged += 1
            agree += int(logged[:3] == fr[:3])
        rows.append({"split": "slice", "id": r["id"], "kind": r["kind"], "question": r["question"],
                     "chunks_fresh": fr, "chunks_logged": logged, "poison": poison})

    train = [json.loads(l) for l in open(os.path.join(THESIS, "dataset", "merged_output", "train.jsonl"), encoding="utf-8")]
    val = [json.loads(l) for l in open(os.path.join(THESIS, "dataset", "merged_output", "val.jsonl"), encoding="utf-8")]
    rng = random.Random(SEED)
    safe = [r for r in train if r["safety"] == "safe"]
    rng.shuffle(safe)
    anchors = safe[:N_ANCHOR]
    rest = [r for r in train if r not in anchors]
    rng.shuffle(rest)
    tr = rest[:N_TRAIN]
    for split, group in [("anchor", anchors), ("train", tr), ("val", val)]:
        ch = search([r["prompt"] for r in group])
        for r, c in zip(group, ch):
            rows.append({"split": split, "question": r["prompt"], "label": int(r["safety"] != "safe"),
                         "attack_type": r.get("attack_type"), "chunks_fresh": c})

    out = os.path.join(HERE, "contexts_bge.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[contexts] wrote {len(rows)} rows -> {out}")
    print(f"[sanity] fresh FAISS top-3 == pipeline-logged top-3 on {agree}/{n_logged} slice rows")


if __name__ == "__main__":
    main()
