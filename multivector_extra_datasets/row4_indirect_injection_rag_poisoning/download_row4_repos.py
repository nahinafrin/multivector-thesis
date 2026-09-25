#!/usr/bin/env python3
"""
download_row4_repos.py -- fetch Row 4's dataset from Hugging Face (English)

===========================================================================
REPLACES the earlier GitHub-clone approach (PoisonedRAG + microsoft/BIPIA
repos). Per an explicit follow-up request, Row 4 now sources from Hugging
Face instead, using the SAME `datasets` library pattern already used for
Row 1 (deepset/prompt-injections) and Row 8 (ai4privacy, TAB) -- one
`pip install datasets` + `load_dataset(...)` call, no git clone, no research-
repo file-layout guessing.
===========================================================================

DATASET: MAlmasabi/Indirect-Prompt-Injection-BIPIA-GPT
https://huggingface.co/datasets/MAlmasabi/Indirect-Prompt-Injection-BIPIA-GPT

  - English, ~70,000 rows, JSON.
  - ~35,000 MALICIOUS rows sourced from the original BIPIA benchmark (Yi et
    al., Microsoft Research, KDD 2025) spanning email/web-QA/table/code
    domains -- real indirect-injection instructions embedded in retrieved-
    document-style context.
  - ~35,000 BENIGN rows synthetically generated (GPT-4o-mini) for a matched
    specificity/false-positive control -- useful as-is, no separate benign
    source needed for Row 4.
  - Reported fields per row: `context` (the external content that may carry
    a malicious instruction -- this is the poison_chunk / context-vector
    side), `user_intent` (the legitimate user request -- this is the clean
    query side), `label` (0=benign, 1=malicious), `source` ("BIPIA" or
    "GPT-4o-mini"). This script does NOT hard-code those exact spellings for
    the build step -- see build_row4_conjunctive.py's candidate-field
    lookup, following this project's own stated discipline (multivector_
    extra_datasets/README.md: "read each file's first row before writing an
    adapter, don't assume field names") in case the live schema differs
    slightly from what the dataset card describes.

This dataset already ships as a (context, user_intent) PAIR per row, which is
a closer match to the RAG-poisoning threat model than PoisonedRAG's
knowledge-corruption pairs were -- no separate carrier-question stitching
needed. SafeRAG (already git-cloned into this same folder) is left in place
as reference material but is not used: its nctd_datasets/ content is
Chinese-language and would not integrate with this project's English
pipeline. The previous PoisonedRAG/BIPIA git-clone path is superseded by
this script; if you already ran the old version, the PoisonedRAG/ and
BIPIA/ folders it created are harmless leftovers you can delete manually.

===========================================================================
GATED DATASET -- one-time, self-service, no manual review
===========================================================================
This dataset is "gated" on the Hub: it requires a logged-in HF account that
has clicked "Agree and access repository" on the dataset page, plus a local
login so `load_dataset()` can send that account's token. This is a
self-service agreement (share contact info), NOT a manual-approval gate --
access is granted immediately, no waiting on the dataset author. One-time
setup:
    1. Create a free account at https://huggingface.co/join if you don't
       have one, then log in.
    2. Visit https://huggingface.co/datasets/MAlmasabi/Indirect-Prompt-Injection-BIPIA-GPT
       and click "Agree and access repository".
    3. Create an access token (read-only is enough) at
       https://huggingface.co/settings/tokens
    4. In your terminal (.venv311 active):
           pip install huggingface_hub
           huggingface-cli login
       and paste the token when prompted.
    5. Re-run this script.

USAGE (run on your Windows machine, .venv311 active, real internet access):
    cd multivector_extra_datasets/row4_indirect_injection_rag_poisoning
    pip install datasets huggingface_hub
    huggingface-cli login   # one-time, after accepting the dataset's terms
    python download_row4_repos.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_PATH = HERE / "bipia_gpt_indirect_injection.jsonl"


def main() -> None:
    from datasets import load_dataset

    print("[row4] MAlmasabi/Indirect-Prompt-Injection-BIPIA-GPT ...")
    try:
        ds = load_dataset("MAlmasabi/Indirect-Prompt-Injection-BIPIA-GPT")
    except Exception as e:
        msg = str(e)
        print(f"  FAILED: {msg}")
        if "gated" in msg.lower() or "access" in msg.lower():
            print(
                "\n  This dataset requires a one-time, self-service access "
                "step (not manual review):\n"
                "    1. Log in at https://huggingface.co/join (or sign in)\n"
                "    2. Visit https://huggingface.co/datasets/MAlmasabi/"
                "Indirect-Prompt-Injection-BIPIA-GPT and click "
                '"Agree and access repository"\n'
                "    3. Create a read-only token at "
                "https://huggingface.co/settings/tokens\n"
                "    4. Run: huggingface-cli login   (paste the token)\n"
                "    5. Re-run this script.\n"
            )
        raise SystemExit(1)

    split = list(ds.keys())[0]
    rows = ds[split]
    print(f"  split={split!r}  n={len(rows)}  columns={rows.column_names}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        n = 0
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            n += 1
    print(f"  wrote {n} rows -> {OUT_PATH.relative_to(HERE)}")

    print("\n[verify] first row (confirm real field names before building the slice):")
    print(json.dumps(dict(rows[0]), ensure_ascii=False, indent=2)[:1500])

    print("\nNext:")
    print('  cd "..\\..\\step4 dataset"')
    print("  python build_row4_conjunctive.py "
          f'--source "../multivector_extra_datasets/row4_indirect_injection_rag_poisoning/{OUT_PATH.name}" '
          "--limit 30")


if __name__ == "__main__":
    main()
