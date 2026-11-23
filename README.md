# Attributing Security in a Local, Zero-Cost Defense-in-Depth Pipeline for RAG

Code, data slices, frozen per-row outputs, and scoring scripts for the paper
*Attributing Security in a Local, Zero-Cost Defense-in-Depth Pipeline for Retrieval-Augmented
Generation: Instruction Injection Is Contained, Knowledge Corruption Is Not*
(manuscript and reviewer response in [`paper_revision/`](paper_revision/)).

The pipeline is a thirteen-stage prompt-injection defense for RAG that runs entirely on one
laptop with open-weight models and no API calls. The paper attributes its security stage by
stage, with paired tests on identical rows. In short:

* **Instruction injection is contained.** Attack success on delivered answers drops from
  30.3 ± 0.8 % to 4.2 ± 0.6 % (three generation seeds, n = 200 each), and 45/50 BIPIA attacks
  are blocked.
* **Three cheap layers do most of the work.** Per-chunk sanitization, grounding, and refusal on
  detection are not significantly different from the full stack (6.0 ± 0.9 % ASR).
* **Knowledge corruption is not contained.** On PoisonedRAG the defended pipeline is no better
  than undefended RAG (90.7 % vs. 91.3 %, p = 1.0).

## Repository layout

| Path | Contents |
|---|---|
| [`step4 dataset/`](step4%20dataset/) | The pipeline (`step_02_…` to `step_13_…`), evaluation runners, slices, frozen outputs, and scoring scripts. **Everything needed to reproduce the paper lives here.** |
| [`external_baselines/`](external_baselines/) | ControlNET and Sentinel–Strategist (ADO) baseline code and their comparison results. |
| [`dataset/`](dataset/) | Builds the merged prompt-injection / jailbreak / toxicity data used to calibrate the input gate. |
| [`paper_revision/`](paper_revision/) | `main_iccit.tex` (ICCIT 2026 camera-ready version, within the 6-page limit), `main.tex` (extended version with all tables, figures, and analyses), `refs.bib`, the reviewer response, and `build_pdf.py` (renders a PDF without LaTeX). |
| [`docs/`](docs/) | Methodology notes, consolidated results, and the reconciliation notes written during the study. |
| [`download_datasets.py`](download_datasets.py) | Re-downloads the third-party datasets that are not stored in the repository. |

## Environment

All results were produced on one laptop: AMD Ryzen 5 7640HS, 16 GB RAM, NVIDIA RTX 4060
Laptop (8 GB), Windows 11, Python 3.11.9, Ollama 0.40.16. Full versions are in Table V of the
paper and in [`step4 dataset/requirements_lock_venv311.txt`](step4%20dataset/requirements_lock_venv311.txt)
(the exact `pip freeze` of the environment used).

```powershell
cd "step4 dataset"
py -3.11 -m venv .venv311
.venv311\Scripts\Activate.ps1
pip install -r requirements_lock_venv311.txt
python -m spacy download en_core_web_lg

# Local LLMs (Q4_K_M generators, Q8_0 guard), served by Ollama
ollama pull llama3.2:3b
ollama pull mistral:7b
ollama pull qwen2.5:3b
ollama pull llama-guard3:1b
```

The rag-mini-wikipedia knowledge base and its FAISS index are committed
(`step4 dataset/data/`, `step4 dataset/kb_wiki.faiss`).

## Reproducing the paper

There are two ways to check a number.

1. **Re-score the frozen outputs (minutes, no GPU).** Every run's per-row outputs are committed as
   JSONL. The scoring scripts read them with the standard library and print the paper's numbers.
2. **Re-run the models (hours, GPU).** The runner scripts regenerate those JSONL files. Ollama GPU
   generation is not bit-reproducible even at a fixed seed, so re-runs match the paper
   statistically, not row for row. The paper measures this spread (Sec. V-E, "Replicates").

All commands below run from `step4 dataset/` with the venv active.

**Claim-by-claim check.** `python verify_claims.py` re-runs the standard-library scorers and checks
79 selected claims: the headline numbers of each results section and every Table VII cell, including
its Wilson interval and exact p-value. Each is recomputed from the committed per-row outputs, compared
with the value the paper states, and matched against the text of `paper_revision/main.tex`. It writes
`claims_manifest.json` with, per claim, the paper section, expected and computed values, input files
with SHA-256 hashes, the scorer, and PASS/FAIL, plus the git revision; it exits non-zero on any
failure. It covers selected claims, not every number in the paper, and it does not re-run the models.

| Paper result | Run (regenerates outputs) | Score (reads frozen outputs) | Frozen outputs |
|---|---|---|---|
| Protocol A slice, specificity, redaction, canaries, controller vs. baseline (Sec. V-A, V-C, V-D; Tables IV, X) | `build_adversarial_slice.py`, `run_full_pipeline.py` | `score_protocolA_220.py` | `protocolA_220/` (the 220-row runs, restored from git `27bf190`; see its README) |
| Conjunctive detector, saturated vs. graded (Sec. V-B, Table VI) | `revalidate_multivector_graded.py` | same script | `multivector_graded_revalidation.json`, `grounded_controller_graded.jsonl` |
| Graded detector on external BIPIA conjunctive rows (Table VI) | `build_row4_conjunctive.py`¹ | `score_row4_detection.py` | `row4_mitigation_results/` |
| Ablation: leave-one-out and single-layer arms (Table VII, Figs. 2–3) | `run_ablation_study.py` | `score_ablation_study.py` (pairs every arm with the current-code undefended arm in `followup200/`; `--baseline july` gives the older July arm, not used in the paper) | `mitigation_results/ablation200/`, `ablation_table.json` |
| Ablation: layer subsets (Table VII) | `run_followup_study.py` | `score_followup_study.py` | `mitigation_results/followup200/` |
| Seed replicates (Sec. V-E, "Replicates") | `run_seed_study.py` | `score_seed_study.py` | `mitigation_results/seed7/`, `seed1234/`, `seed_study_report.json` |
| Fresh rerun of the undefended and full-stack arms from the raw slice (Sec. V-E, "Replicates") | `run_mitigation_ab.py --slice planted_attacks_200.jsonl --run-dir mitigation_results/rerun200` (`--off-only`, then `--skip-off --label full`) | `score_rerun200.py` | `mitigation_results/rerun200/` (with `code_revision.txt`, `rerun_report.json`) |
| LLM-Guard detector baseline (Sec. V-E) | `run_external_guard_baseline.py` | same script | `baseline_comparison/llmguard_detector_baseline.json` |
| ControlNET and ADO baselines (Sec. V-E) | `../external_baselines/paper_faithful/controlnet_faithful.py` (paper-faithful ControlNET), `../external_baselines/run_ado_official_on_slice.py` | `../external_baselines/compare_paper_faithful.py` | `../external_baselines/*_results.txt` |
| Existing defenses on Protocol B: Spotlighting, LLM-Guard firewall, ControlNET (Sec. V-E, Table VIII) | `run_mitigation_ab.py --only spotlight` and `run_benign_cost.py --only spotlight`; `dump_undefended_contexts.py`, `run_llmguard_protocolB.py`; `../external_baselines/paper_faithful/run_controlnet_protocolB.py --gguf-dir <reduced Llama-3.1-8B GGUF>` (default `--context generator`: the exact passages the undefended generator received; `--context top5_plus_poison` gives the earlier variant) | `score_baselines_protocolB.py` | `mitigation_results/baselines_protocolB/` |
| PoisonedRAG and BIPIA, complete pipeline (Table VIII) | `convert_external_datasets.py`, `run_full_pipeline.py` | `final_n200_summary.py` | `external_validation_n200.jsonl`, `ext_n200_results.jsonl` |
| Same rows undefended and same-generator (Table VIII) | `run_mitigation_ab.py --slice external_validation_n200.harness.jsonl` | `score_external_paired.py` | `mitigation_results/external200/` |
| 40-row semantic slice (Sec. V-H) | `build_semantic_slice.py`, `run_followup_study.py` | `score_followup_study.py` | `semantic_slice_non_yes_no_40.jsonl`, `mitigation_results/followup_nonyesno40/` |
| Latency and memory (Table IX) | `resource_profile/sample_resources.ps1` (during a run) | `summarize_overhead.py` | `resource_profile/` |
| Earlier-code mitigation arms (Sec. V-F) | `run_planted_mitigation.py` | `score_mitigation_ab.py` | `mitigation_results/planted200/` |
| deepset gate test (Table VIII)² | `run_deepset_gate_eval.py` | same script | `external_eval/deepset_gate_rows.jsonl`, `deepset_gate_report.json` |
| deepset gate + graded flag, threshold chosen on the train split (Sec. V-G) | `run_deepset_gate_eval.py --data <deepset train parquet> --tag deepset_gate_train`, `collect_graded_heldout_benign.py` | `fit_gate_graded.py` | `external_eval/gate_graded_report.json`, `deepset_graded_scores.jsonl` |
| deepset end to end (Sec. V-G) | `run_full_pipeline.py --slice external_eval/deepset_e2e_slice.jsonl` | `score_improvements.py` | `external_eval/deepset_e2e.jsonl`, `improvements_report.json` |
| Graded floors on held-out benign questions (Sec. V-B) | `collect_graded_heldout_benign.py` | `fit_graded_floors_heldout.py` | `external_eval/graded_heldout_benign.jsonl`, `graded_floors_heldout_report.json` |
| Ablation at n = 424 (Sec. V-E, "Larger sample") | `build_planted_attacks.py --yesno-only --seed 42`, `run_mitigation_ab.py --slice planted_attacks_extra224.jsonl` | `score_improvements.py` | `planted_attacks_extra224.jsonl`, `mitigation_results/extended424/` |
| Semantic slice, fused vs. isolate-and-aggregate vs. routed generator (Sec. V-H) | `run_full_pipeline.py --generator {fused,isolate,routed}` | `score_improvements.py` | `external_eval/semantic40_*.jsonl`, `benign40_*.jsonl` |
| Encoders on CPU vs. GPU (Sec. V-I) | `run_gpu_latency.sh` (`bench_encoder_device.py`, `run_benign_cost.py` with `RAG_ENCODER_DEVICE=cuda`) | `bench_encoder_device.py` | `external_eval/encoder_latency_*.json`, `mitigation_results/gpu_encoders/` |

The queue that produced the deepset, held-out, semantic, and n = 424 outputs is `run_improvement_queue.py` (sequential, resumable).

¹ Needs the BIPIA file from `multivector_extra_datasets/row4_indirect_injection_rag_poisoning/download_row4_repos.py`;
the derived slices it produces are already committed.<br>
² Needs the deepset test split from `python download_datasets.py` (repository root). The per-row gate
decisions are committed, so the summary can be checked without re-running.

**Removed files.** Exploratory, debugging, and superseded files (earlier deepset probes, one-off
diagnostic scripts, smoke-test runs, old pipeline variants, and bundled third-party data copies)
were moved out of this repository to the private backup repository
`nahinafrin/multivector-thesis-backup`, with their original paths. None of them is needed to
reproduce the paper.

Most runners print their options with `--help`. Generation uses seed 42 unless `RAG_GEN_SEED` is set.

### Rendering the paper without LaTeX

```powershell
cd paper_revision
python build_pdf.py     # writes main.pdf via headless Edge/Chrome (KaTeX math, SVG figures)
```

This preview is approximate. The reference build for submission is `IEEEtran` with real LaTeX:
upload `main.tex` and `refs.bib` to Overleaf (pdfLaTeX), or compile locally with
[Tectonic](https://tectonic-typesetting.github.io/), which fetches the packages it needs:

```powershell
cd paper_revision
tectonic -X compile main.tex --outdir build_tex
```

## Data and licences

Third-party data used in the paper is listed below. Small derived slices and the raw
PoisonedRAG/BIPIA inputs they were built from (`step4 dataset/external_datasets_raw_source/`) are
committed. Larger datasets are not stored here; `python download_datasets.py` re-creates them.

| Dataset / model | Use | Licence |
|---|---|---|
| [rag-datasets/rag-mini-wikipedia](https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia) | Knowledge base and benign questions | CC-BY-3.0 |
| [PoisonedRAG](https://github.com/sleeepeer/PoisonedRAG) (Zou et al., USENIX Security 2025) | External knowledge-corruption set | MIT |
| [BIPIA](https://github.com/microsoft/BIPIA) (Yi et al., KDD 2025) | External indirect-injection set | MIT (some sub-datasets CC-BY-SA-4.0; see upstream) |
| [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) | External gate test | Apache-2.0 |
| [protectai/deberta-v3-base-prompt-injection-v2](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2) | Injection scanner | Apache-2.0 |
| Llama 3.2, Llama Guard 3, Mistral 7B, Qwen2.5 | Generators and safety classifier (via Ollama) | Their respective model licences |

No human participants were involved and no personal data was collected. All attack and benign
rows come from public datasets or were constructed synthetically.

## Licence

The code and the constructed evaluation slices in this repository are released under the
[MIT Licence](LICENSE). Third-party datasets and models keep their own licences (table above).

## Citation

If you use this code or the evaluation slices, please cite the paper (citation details will be
added on publication).
