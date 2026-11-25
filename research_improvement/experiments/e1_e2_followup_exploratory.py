"""Exploratory follow-up to E1/E2 (written AFTER results; not preregistered, descriptive only).

1. E1-NS: full-stack ASR by the ORIGINAL suffix the row carried (was the protection lost where the detector used to fire?).
2. E1-HO: full-stack ASR by held-out poison template.
3. E2: sample of llama3.2:3b closed-book answers labelled "other", to judge string-match undercounting.

    python e1_e2_followup_exploratory.py  ->  e1_e2_followup_exploratory.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "analysis"))
from common import MR, attack_succeeded, load, wilson  # noqa: E402
from score_e1_e2 import wb  # noqa: E402

orig_full = load(MR / "ablation200" / "on_full.jsonl")
det = lambda r: str(r.get("block_reason", "")).startswith("detector")  # noqa: E731
out = {}

ns_src = {int(r["id"]): r for r in map(json.loads, open(HERE / "e1_attacks_NS.jsonl", encoding="utf-8"))}
ns_full = load(HERE / "E1_NS" / "on_full.jsonl")
by = {}
for suf in sorted({r["e1_original_suffix"] for r in ns_src.values()}):
    ids = [i for i in ns_full if ns_src[i]["e1_original_suffix"] == suf]
    by[suf] = {"n": len(ids),
               "full_asr_NS": wilson(sum(attack_succeeded(ns_full[i]) for i in ids), len(ids)),
               "full_asr_original": wilson(sum(attack_succeeded(orig_full[i]) for i in ids), len(ids)),
               "detector_fires_NS": sum(det(ns_full[i]) for i in ids),
               "detector_fires_original": sum(det(orig_full[i]) for i in ids)}
out["NS_by_original_suffix"] = by

ho_src = {int(r["id"]): r for r in map(json.loads, open(HERE / "e1_attacks_HO.jsonl", encoding="utf-8"))}
ho = {a: load(HERE / "E1_HO" / f"{a}.jsonl") for a in ("off", "on_full")}
by = {}
for t in sorted({r["e1_template"] for r in ho_src.values()}):
    ids = [i for i in ho["on_full"] if ho_src[i]["e1_template"] == t]
    by[f"P{t}"] = {"n": len(ids),
                   "full_asr": wilson(sum(attack_succeeded(ho["on_full"][i]) for i in ids), len(ids)),
                   "undefended_asr": wilson(sum(attack_succeeded(ho["off"][i]) for i in ids), len(ids)),
                   "detector_fires_full": sum(det(ho["on_full"][i]) for i in ids)}
out["HO_by_template"] = by

cb = [json.loads(l) for l in open(HERE / "e2_closed_book.jsonl", encoding="utf-8")]
other = [r for r in cb if r["model"] == "llama3.2:3b"
         and not (wb(r["true_answer"], r["answer"]) ^ wb(r["target"], r["answer"]))]
out["E2_llama_other_sample"] = [{k: r[k] for k in ("pos", "question", "true_answer", "target", "answer")}
                               for r in other[:: max(1, len(other) // 15)][:15]]
(HERE / "e1_e2_followup_exploratory.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k != "E2_llama_other_sample"}, indent=1, ensure_ascii=False))
for r in out["E2_llama_other_sample"]:
    print(f"- T={r['true_answer']!r} | tgt={r['target']!r} | A={r['answer'][:140]!r}")
