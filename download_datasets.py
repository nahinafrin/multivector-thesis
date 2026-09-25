#!/usr/bin/env python3
"""
download_datasets.py

Single-command downloader for the three external validation datasets discussed
for the multivector-thesis project (rows 4, 1, 8 of the multi-vector attack
comparison table). Run this once from inside this folder:

    python download_datasets.py

Requirements (installed automatically if missing):
    pip install datasets huggingface_hub

Gated dataset note:
    Mindgard/evaded-prompt-injection-and-jailbreak-samples requires a free
    Hugging Face account that has clicked "Agree" on the dataset's terms page,
    plus a login token. If you have one, set it before running:

        # Windows PowerShell:
        $env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"
        # macOS/Linux:
        export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"

    Without a token this one dataset is skipped (a warning is printed) and
    everything else still downloads normally.

Output layout (JSONL, one row per line, UTF-8):
    row4_indirect_injection_rag_poisoning/
        bipia_indirect_injection.jsonl
    row1_obfuscation_prompt_injection/
        deepset_prompt_injections.jsonl
        mindgard_evaded_injection_jailbreak.jsonl      (only if HF_TOKEN set)
    row8_pii_data_leakage/
        tab_text_anonymization_train.jsonl
        tab_text_anonymization_val_test.jsonl
        ai4privacy_pii_masking_sample.jsonl            (first 2000 rows only —
                                                          see --ai4privacy-limit)
    SafeRAG/                                            (git-cloned separately,
                                                          see README.md)
"""

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _write_jsonl(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"  wrote {n} rows -> {out_path.relative_to(HERE)}")
    return n


def download_bipia():
    print("[row4] BIPIA (Indirect-Prompt-Injection-BIPIA-GPT) ...")
    from datasets import load_dataset

    try:
        ds = load_dataset("MAlmasabi/Indirect-Prompt-Injection-BIPIA-GPT")
    except Exception as e:
        print(f"  FAILED: {e}")
        return
    split = list(ds.keys())[0]
    out = HERE / "row4_indirect_injection_rag_poisoning" / "bipia_indirect_injection.jsonl"
    _write_jsonl(ds[split], out)


def download_deepset_prompt_injections():
    print("[row1] deepset/prompt-injections ...")
    from datasets import load_dataset

    try:
        ds = load_dataset("deepset/prompt-injections")
    except Exception as e:
        print(f"  FAILED: {e}")
        return
    for split in ds.keys():
        out = HERE / "row1_obfuscation_prompt_injection" / f"deepset_prompt_injections_{split}.jsonl"
        _write_jsonl(ds[split], out)


def download_mindgard():
    print("[row1] Mindgard/evaded-prompt-injection-and-jailbreak-samples ...")
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("  SKIPPED: this dataset is gated. Set HF_TOKEN (see top of this file) and re-run "
              "to include it. Everything else will still download.")
        return
    from datasets import load_dataset

    try:
        ds = load_dataset(
            "Mindgard/evaded-prompt-injection-and-jailbreak-samples", token=token
        )
    except Exception as e:
        print(f"  FAILED (check your token has accepted the dataset's terms on huggingface.co): {e}")
        return
    split = list(ds.keys())[0]
    out = HERE / "row1_obfuscation_prompt_injection" / "mindgard_evaded_injection_jailbreak.jsonl"
    _write_jsonl(ds[split], out)


def download_tab():
    print("[row8] Text Anonymization Benchmark (TAB) ...")
    from datasets import load_dataset

    for hf_name, out_name in [
        ("mattmdjaga/text-anonymization-benchmark-train", "tab_text_anonymization_train.jsonl"),
        ("mattmdjaga/text-anonymization-benchmark-val-test", "tab_text_anonymization_val_test.jsonl"),
    ]:
        try:
            ds = load_dataset(hf_name)
        except Exception as e:
            print(f"  FAILED ({hf_name}): {e}")
            continue
        split = list(ds.keys())[0]
        out = HERE / "row8_pii_data_leakage" / out_name
        _write_jsonl(ds[split], out)


def download_ai4privacy(limit: int):
    print(f"[row8] ai4privacy/pii-masking-400k (first {limit} rows as a held-out sample) ...")
    from datasets import load_dataset

    try:
        ds = load_dataset("ai4privacy/pii-masking-400k", split=f"train[:{limit}]")
    except Exception as e:
        print(f"  FAILED: {e}")
        return
    out = HERE / "row8_pii_data_leakage" / "ai4privacy_pii_masking_sample.jsonl"
    _write_jsonl(ds, out)


def clone_saferag():
    print("[row4] SafeRAG (git clone, not on the HF datasets hub) ...")
    target = HERE / "row4_indirect_injection_rag_poisoning" / "SafeRAG"
    if target.exists():
        print(f"  already present at {target.relative_to(HERE)}, skipping (delete it to re-clone)")
        return
    rc = os.system(f'git clone --depth 1 https://github.com/IAAR-Shanghai/SafeRAG "{target}"')
    if rc != 0:
        print("  FAILED: git clone did not succeed. If this machine has no internet access, "
              "download the repo as a zip from https://github.com/IAAR-Shanghai/SafeRAG "
              "and extract it into that folder manually.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ai4privacy-limit", type=int, default=2000,
                     help="Rows to pull from ai4privacy/pii-masking-400k (default 2000; it has 400k total).")
    ap.add_argument("--skip-saferag", action="store_true", help="Skip the SafeRAG git clone.")
    args = ap.parse_args()

    print("Downloading external validation datasets for row 4 / row 1 / row 8 ...\n")
    download_bipia()
    download_deepset_prompt_injections()
    download_mindgard()
    download_tab()
    download_ai4privacy(args.ai4privacy_limit)
    if not args.skip_saferag:
        clone_saferag()

    print("\nDone. See README.md in this folder for schema notes and how each file maps "
          "to your existing adversarial_slice.jsonl categories.")


if __name__ == "__main__":
    main()
