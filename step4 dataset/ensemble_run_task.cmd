@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set RAG_ENSEMBLE=large
python ensemble_generalization_check.py --slice semantic_slice.jsonl --small-out grounded_semantic_small_ensemble.jsonl --large-out grounded_semantic_large_ensemble.jsonl > ensemble_run_task.log 2> ensemble_run_task.err.log
