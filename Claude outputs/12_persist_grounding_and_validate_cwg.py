#!/usr/bin/env python3
"""
12_persist_grounding_and_validate_cwg.py
==========================================
Closes limitation #2 in-system (or reports honestly that it doesn't), by
fixing the ONE thing blocking it: step_10_grounding_judge.py computes
per-chunk faithfulness scores and then throws them away, only keeping the
final pass/fail boolean. This is the exact "grounding: {}" cache gap
Addendum 6 diagnosed for the terse-polar-answer replay, and Addendum 7
found blocks CWG's in-system validation too. Fixing it once unblocks both.

This file is NOT a blind patch. Per this project's established discipline
(see fixes 01/06/08/09's prompts), read step_10_grounding_judge.py's real
current source first and adapt the call site below to match -- the function
names and call site are confirmed as of Addendum 6:

    lexical_overlap(answer, chunks, question="")
    model_faithfulness(answer, chunks, question="")
    passed = (max(lexical_overlap(...), model_faithfulness(...)) >= threshold) and canary_intact

If any of those three names/signatures have since changed, adapt
`persist_grounding_detail()` below accordingly before wiring it in.

WHAT THIS ADDS
--------------
A `persist_grounding_detail()` helper that computes the SAME per-chunk
faithfulness scores the judge already computes internally (it does not
change the pass/fail decision at all -- zero risk of regressing any
existing result), and writes them into `state.meta["grounding_detail"]`
so they get serialized into the row's output JSON alongside the existing
pass/fail boolean. This is the only change needed to unblock both the
Addendum 6 grounding-replay item and the Addendum 7 CWG validation.

USAGE (after wiring into step_10_grounding_judge.py -- see VS Code prompt)
---------------------------------------------------------------------------
1. Re-run the 8-row misinformation slice with persistence now enabled:
       python run_full_pipeline.py --slice semantic_slice.jsonl \\
           --out semantic_slice_with_grounding.jsonl --index kb_wiki

2. Get a benign comparison set the same way -- the 8-row slice is attack-only,
   so CWG's false-block cost can't be measured from it alone. Take ~20-30 rows
   from the existing benign `question-answer` split (the same clean-QA source
   used for specificity controls in Addendum 1/2) and run them through the
   same pipeline/grounding step to get their own `grounding_detail`:
       python run_full_pipeline.py --slice benign_qa_sample.jsonl \\
           --out benign_qa_with_grounding.jsonl --index kb_wiki

3. Convert both output files to replay_cwg_on_semantic_slice.py's expected
   schema with `build_cwg_replay_input()` below, then run the existing
   replay script:
       python 12_persist_grounding_and_validate_cwg.py convert \\
           --attack semantic_slice_with_grounding.jsonl \\
           --benign benign_qa_with_grounding.jsonl \\
           --out cwg_insystem_input.jsonl
       python replay_cwg_on_semantic_slice.py --data cwg_insystem_input.jsonl \\
           --theta-f <your existing grounding threshold> \\
           --theta-t <starting point: try 0.5-0.65, this project's real
                       cross-encoder scores are not TF-IDF cosine, so the
                       Climate-FEVER-calibrated 0.88 does NOT transfer --
                       re-grid-search on this data if time allows, or report
                       the result at a stated, reasonable starting value and
                       flag it as not independently calibrated> \\
           --phi-min 0.5 --m-min 2

4. Report the printed before/after table exactly as-is in Addendum 8 /
   the thesis's CWG subsection -- whatever it says. If it's a clear
   improvement, that's the first real in-system result. If it's null or
   ambiguous at n=8 (very possible at this sample size), say so plainly,
   the same way Addendum 1 item 9 reported one catch + one regression
   rather than rounding up to "bigger model wins."
"""
from __future__ import annotations
import argparse
import json


def persist_grounding_detail_SNIPPET() -> str:
    """
    Not runnable -- this is the snippet to insert into
    step_10_grounding_judge.py at the real call site, once confirmed
    against the actual current source. Returned as a string so this file
    stays a single self-contained artifact; paste the snippet in by hand
    (or have your VS Code assistant do it per the prompt in
    multivector-thesis-vscode-fix-prompts.md item 12).
    """
    return '''
    # --- BEGIN: persist per-chunk grounding detail (Addendum 7/8 fix) ---
    # Insert immediately around the existing line:
    #   passed = (max(lexical_overlap(...), model_faithfulness(...)) >= threshold) and canary_intact
    # `chunks` here must be the same post-sanitization/post-rerank chunk list
    # already passed into lexical_overlap()/model_faithfulness() above --
    # reuse that exact variable, do not re-retrieve.
    _lex = lexical_overlap(answer, chunks, question=state.prompt)
    _phis = [model_faithfulness(answer, c, question=state.prompt) for c in chunks]
    _sources = [getattr(c, "source", "kb_wiki") if not isinstance(c, dict)
                else c.get("source", "kb_wiki") for c in chunks]
    # Single-source rag-mini-wikipedia KB -> uniform trust unless
    # trust_aware_retrieval.py's per-source trust map is already wired in
    # (see multivector-thesis-vscode-fix-prompts.md item 2); if it is,
    # replace the line below with a real per-source lookup instead of 1.0.
    _trusts = [1.0 for _ in chunks]
    state.meta["grounding_detail"] = {
        "lexical_overlap": _lex,
        "chunk_phis": _phis,
        "chunk_trusts": _trusts,
        "chunk_sources": _sources,
    }
    # --- END: persist per-chunk grounding detail ---
    '''


def build_cwg_replay_input(attack_path: str, benign_path: str | None, out_path: str) -> None:
    """
    Converts real pipeline output rows (which now carry
    row["meta"]["grounding_detail"] per the snippet above, or
    row["grounding_detail"] depending on how your output serializer flattens
    `state.meta`) into replay_cwg_on_semantic_slice.py's expected schema:
        {"id":..., "label": 0/1, "lexical_overlap":..., "chunk_phis":[...], "chunk_trusts":[...]}
    label=1 for attack rows (should be blocked), label=0 for benign rows.
    Adjust the `detail = row.get(...)` line below if your real output nests
    `grounding_detail` somewhere else -- check one row with
    `python -c "import json; print(json.dumps(json.loads(open('semantic_slice_with_grounding.jsonl').readline()), indent=2))"`
    first if unsure.
    """
    def _rows(path: str, label: int):
        out = []
        for i, line in enumerate(open(path, encoding="utf-8")):
            if not line.strip():
                continue
            row = json.loads(line)
            detail = row.get("grounding_detail") or row.get("meta", {}).get("grounding_detail")
            if not detail:
                print(f"WARNING: row {i} in {path} has no grounding_detail -- "
                      f"the persistence patch may not have run for this row, skipping.")
                continue
            out.append({
                "id": row.get("id", i),
                "label": label,
                "lexical_overlap": detail["lexical_overlap"],
                "chunk_phis": detail["chunk_phis"],
                "chunk_trusts": detail.get("chunk_trusts") or [1.0] * len(detail["chunk_phis"]),
            })
        return out

    rows = _rows(attack_path, label=1)
    if benign_path:
        rows += _rows(benign_path, label=0)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_attack = sum(1 for r in rows if r["label"] == 1)
    n_benign = sum(1 for r in rows if r["label"] == 0)
    print(f"Wrote {len(rows)} rows ({n_attack} attack, {n_benign} benign) to {out_path}.")
    if n_benign == 0:
        print("NOTE: no benign rows included -- pass --benign to also measure "
              "CWG's false-block cost, not just its catch rate. Without it you "
              "can only report the attack-side number, which is half the story "
              "(see Addendum 7's Climate-FEVER-vs-LIAR-PLUS finding for why the "
              "false-block side matters just as much).")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--attack", required=True)
    c.add_argument("--benign", default=None)
    c.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "convert":
        build_cwg_replay_input(args.attack, args.benign, args.out)


if __name__ == "__main__":
    main()
