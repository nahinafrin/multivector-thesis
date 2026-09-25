# External validation datasets — rows 4, 1, 8

Companion datasets for testing whether `multivector-thesis` actually defends against
the three multi-vector attack rows discussed: (4) Indirect Injection + RAG Poisoning,
(1) Obfuscation + Prompt Injection, (8) Prompt attack + Data extraction (scoped here
to PII exposure via the RAG context, not foundation-model training-data extraction).

## Important: run the download script on YOUR machine, not from this folder as-is

This folder was assembled in a sandboxed cloud environment whose network policy
blocks huggingface.co outright (GitHub is allowed, which is why `SafeRAG/` below
already contains real cloned data). Everything hosted on Hugging Face still needs
to be pulled — that will work fine on your own machine, which has normal internet
access. One command:

```powershell
cd multivector_extra_datasets
pip install datasets huggingface_hub
python download_datasets.py
```

That's it — re-run it any time to refresh. It's idempotent (SafeRAG is skipped if
already present; HF files are simply overwritten).

### The one dataset that needs an extra step: Mindgard (row 1)

`Mindgard/evaded-prompt-injection-and-jailbreak-samples` is gated behind a free
Hugging Face account that has clicked "Agree" on the dataset's own terms page.
Without that, the script skips it (everything else still downloads). To include it:

1. Create a free account at huggingface.co if you don't have one.
2. Open https://huggingface.co/datasets/Mindgard/evaded-prompt-injection-and-jailbreak-samples
   and click through to accept its terms (CC-BY-NC-4.0 — fine for academic/thesis use).
3. Get a token: https://huggingface.co/settings/tokens (a "Read" token is enough).
4. Before running the script:
   ```powershell
   $env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"
   python download_datasets.py
   ```

## Folder layout after running the script

```
multivector_extra_datasets/
├── download_datasets.py
├── README.md   (this file)
├── row4_indirect_injection_rag_poisoning/
│   ├── bipia_indirect_injection.jsonl
│   └── SafeRAG/                              <- already populated (real clone)
├── row1_obfuscation_prompt_injection/
│   ├── deepset_prompt_injections_train.jsonl  (name depends on the dataset's split)
│   └── mindgard_evaded_injection_jailbreak.jsonl   (only if HF_TOKEN set)
└── row8_pii_data_leakage/
    ├── tab_text_anonymization_train.jsonl
    ├── tab_text_anonymization_val_test.jsonl
    └── ai4privacy_pii_masking_sample.jsonl    (first 2000 of 400k — see
                                                  --ai4privacy-limit to pull more)
```

## What's already in `SafeRAG/` (real data, already cloned)

This is the official IAAR-Shanghai/SafeRAG repo (ACL 2025) — a benchmark built
specifically to test RAG systems against poisoning/noise injected into retrieved
documents, the closest existing academic match to your `poisoned_context` /
`multivector_attack` threat model. Inside it:

- `nctd_datasets/` — the actual attack/noise construction data
- `knowledge_base/` — the retrieval corpora the attacks are tested against
- `evaluator.py`, `quick_start_nctd.py` — their own evaluation harness (useful as
  a reference implementation, not something you need to run directly — you want
  to feed *their attack constructions* through *your* pipeline, not run their
  evaluator against their own system)
- `configs/`, `retrievers/`, `llms/`, `prompts/`, `metric/` — their pipeline's
  own scaffolding; skim `README.md` inside that folder for their data format
  before writing an adapter

## How each file maps onto your existing pipeline

Your `build_adversarial_slice.py` / `adversarial_slice.jsonl` already use a
`{prompt, safety, attack_type}`-style row schema (confirmed in
`multivector-thesis-analysis.md` §2). None of these external datasets will
already match that schema exactly — **read each file's first row before writing
an adapter, don't assume field names** (this project's own working discipline,
established across Addenda 1–6: several of its own bugs came from guessing
field names instead of checking them). Concretely:

```powershell
# after running download_datasets.py, inspect a real row from each file:
Get-Content row4_indirect_injection_rag_poisoning\bipia_indirect_injection.jsonl -TotalCount 1
Get-Content row1_obfuscation_prompt_injection\deepset_prompt_injections_train.jsonl -TotalCount 1
Get-Content row8_pii_data_leakage\tab_text_anonymization_train.jsonl -TotalCount 1
```

**Row 4 (BIPIA + SafeRAG)**: map their poisoned-context field into a new row with
`attack_type="indirect_injection"` (or a new `attack_type="external_rag_poisoning"`
if you want to keep it distinguishable from your own hand-built poison chunks),
run through `run_full_pipeline.py` as-is, and compare the redaction/canary rate
against your existing 40/40 result on your own `poisoned_context` rows.

**Row 1 (deepset + Mindgard)**: deepset's rows are your single-vector baseline
(`attack_type="prompt_injection"`). Mindgard's paired original/modified rows let
you measure recall on the *same underlying prompt* before and after obfuscation —
tag modified rows `attack_type="obfuscation"` and join them back to their
`attack_name` field to see which specific obfuscation technique (Base64 emoji
smuggling, Unicode injection, etc.) most reduces your gate's recall.

**Row 8 (TAB + ai4privacy)**: plant TAB's real annotated PII spans into retrieved
documents the same way your `poisoned_context` rows are constructed, then check
(a) input-gate `pii_leakage` recall against this independently-sourced PII versus
your existing 31.7% figure, and (b) whether the documented Presidio SSN/DATE_TIME
collision (see `thesis_overview.pdf` §4.4) reproduces against TAB's real-world PII
formatting, which is less templated than synthetic generators.

## Scope note

None of this is required for the SJRC abstract — it's follow-up validation work
for the fuller thesis manuscript, extending limitation #7 (small evaluation
slices) and limitation #4 (data-starved categories) with independently-sourced,
peer-reviewed benchmarks rather than growing your own synthetic data further.
