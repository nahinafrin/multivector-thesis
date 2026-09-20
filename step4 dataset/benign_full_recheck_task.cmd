@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if not exist mitigation_results\planted200_recheck mkdir mitigation_results\planted200_recheck
copy /Y mitigation_results\planted200\benign_off.jsonl mitigation_results\planted200_recheck\benign_off.jsonl > nul
python run_benign_cost.py --qa data\question-answer\test.jsonl --n 200 --skip-off --run-dir mitigation_results\planted200_recheck --label full > benign_full_recheck.log 2> benign_full_recheck.err.log
if errorlevel 1 exit /b %errorlevel%
python score_benign_cost.py --off mitigation_results\planted200_recheck\benign_off.jsonl --on mitigation_results\planted200_recheck\benign_on.jsonl --asr-report mitigation_results\planted200\report_full.json --out mitigation_results\planted200_recheck\benign_cost_full.json >> benign_full_recheck.log 2>> benign_full_recheck.err.log
exit /b %errorlevel%