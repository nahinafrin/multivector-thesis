#!/usr/bin/env python3
"""
check_alignment.py -- fail (exit 1) unless ADO's Protocol B inputs and outputs line up with the
compared arms.

Checks, for the selected rows of each split:
  1. the context rows exist and their question equals the question of the undefended arm and of the
     seven-layer arm (followup200/off.jsonl, ablation200/on_full.jsonl and their benign files), which
     include the attack query suffix;
  2. attack rows' poison_chunk and success_marker match planted_attacks_200.jsonl;
  3. every ADO output row (if --outputs given) has the same index and question as its context row,
     was generated from no more than those passages (final_context is a subset of ranked_context),
     and carries no field derived from evaluation data before generation (the scoring fields are
     present and equal to the reference files);
  4. no duplicate or missing indices.

    python check_alignment.py --selection pilot_selection.json [--outputs pilot]
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
S4 = HERE.parent.parent / "step4 dataset"
MR = S4 / "mitigation_results"
CTX = MR / "baselines_protocolB"


def rows(p: Path) -> dict:
    return {r["index"]: r for r in (json.loads(l) for l in open(p, encoding="utf-8") if l.strip())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", required=True)
    ap.add_argument("--outputs", default=None, help="prefix of {prefix}_{attack,benign}_results.jsonl")
    a = ap.parse_args()
    sel = json.loads((HERE / a.selection).read_text(encoding="utf-8"))
    planted = {r.get("id", i): r for i, r in enumerate(json.loads(l) for l in open(S4 / "planted_attacks_200.jsonl", encoding="utf-8") if l.strip())}
    errs = []
    for split, off_f, ours_f in (("attack", "followup200/off.jsonl", "ablation200/on_full.jsonl"),
                                 ("benign", "followup200/benign_off.jsonl", "ablation200/benign_on.jsonl")):
        ids = sel[split]
        if len(ids) != len(set(ids)):
            errs.append(f"{split}: duplicate ids in selection")
        ctx, off, ours = rows(CTX / f"contexts_{split}.jsonl"), rows(MR / off_f), rows(MR / ours_f)
        for i in ids:
            for name, src in (("contexts", ctx), ("undefended", off), ("seven-layer", ours)):
                if i not in src:
                    errs.append(f"{split} {i}: missing in {name}")
            if all(i in s for s in (ctx, off, ours)) and \
                    len({ctx[i]["question"], off[i]["question"], ours[i]["question"]}) != 1:
                errs.append(f"{split} {i}: question differs across contexts / undefended / seven-layer")
            if split == "attack" and i in ctx and i in planted:
                if ctx[i].get("poison_chunk") != planted[i].get("poison_chunk"):
                    errs.append(f"attack {i}: poison_chunk differs from planted_attacks_200")
        if a.outputs:
            p = HERE / f"{a.outputs}_{split}_results.jsonl"
            if not p.exists():
                errs.append(f"{split}: missing {p.name}"); continue
            out = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            got = [r["index"] for r in out]
            if len(got) != len(set(got)):
                errs.append(f"{split}: duplicate rows in {p.name}")
            if set(got) - set(ids):
                errs.append(f"{split}: rows outside the selection: {sorted(set(got) - set(ids))}")
            for r in out:
                i = r["index"]
                if r["question"] != ctx[i]["question"]:
                    errs.append(f"{split} {i}: output question differs from context question")
                if not set(r.get("final_context") or []) <= set(ctx[i]["ranked_context"]):
                    errs.append(f"{split} {i}: ADO used passages outside the generator's context")
                if split == "attack" and r.get("success_marker") != planted[i].get("success_marker"):
                    errs.append(f"attack {i}: success_marker differs from planted_attacks_200")
                if split == "benign" and r.get("true_answer") != off[i].get("true_answer"):
                    errs.append(f"benign {i}: true_answer differs from the undefended arm")
    if errs:
        print("ALIGNMENT FAILED:"); [print("  " + e) for e in errs]
        sys.exit(1)
    print(f"alignment OK: {len(sel['attack'])} attack + {len(sel['benign'])} benign rows"
          + (f"; outputs '{a.outputs}' checked" if a.outputs else ""))


if __name__ == "__main__":
    main()
