#!/usr/bin/env bash
# run_pilot.sh -- preregistered 50 + 50 ADO pilot on Protocol B (rows in pilot_selection.json).
# Resumable: rerunning skips rows already written. Uses the GPU environment (bitsandbytes, CUDA torch).
# Requires: Ollama running with llama3.1:8b (controller) and llama3.2:3b (generator); the 4-bit
# checkpoint unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit in the Hugging Face cache.
set -u
cd "$(dirname "$0")"
P="../../step4 dataset/.venv311gpu/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
"../../step4 dataset/.venv311/Scripts/python.exe" check_alignment.py --selection pilot_selection.json || exit 1
for s in attack benign; do
  "$P" run_ado_protocolB.py --selection pilot_selection.json --split $s --out pilot_${s}_results.jsonl \
       --av hf4bit >> pilot_${s}_stdout.txt 2>&1
  echo "$s rc=$? rows=$(wc -l < pilot_${s}_results.jsonl)"
done
"../../step4 dataset/.venv311/Scripts/python.exe" check_alignment.py --selection pilot_selection.json --outputs pilot
