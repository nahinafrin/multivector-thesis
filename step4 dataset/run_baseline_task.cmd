@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
python run_baseline_comparison.py --slice adversarial_slice.jsonl > baseline_run_task.log 2> baseline_run_task.err.log
