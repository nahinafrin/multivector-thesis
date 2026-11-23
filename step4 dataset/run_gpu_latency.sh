#!/usr/bin/env bash
# run_gpu_latency.sh -- item 5: encoder latency on CPU vs GPU, isolated and end to end.
#   1-2. per-stage encoder benchmark (no LLM call), 50 held-out benign questions, CPU then GPU
#   3-4. full mitigation stack on the first 50 benign questions of Protocol B (seed 42 order),
#        CPU encoders then GPU encoders, with Ollama generation sharing the 8 GB GPU
# Run from "step4 dataset" after nothing else is using Ollama.
set -u
CPU=.venv311/Scripts/python.exe
GPU=.venv311gpu/Scripts/python.exe
D=mitigation_results/gpu_encoders
mkdir -p "$D"
$CPU bench_encoder_device.py --n 50 > external_eval/bench_cpu_log.txt 2>&1; echo "bench cpu rc=$?"
RAG_ENCODER_DEVICE=cuda $GPU bench_encoder_device.py --n 50 > external_eval/bench_cuda_log.txt 2>&1; echo "bench cuda rc=$?"
$CPU run_benign_cost.py --n 50 --skip-off --run-dir "$D" --label full_cpu \
  > "$D/full_cpu_log.txt" 2>&1; echo "benign full cpu rc=$?"
RAG_ENCODER_DEVICE=cuda $GPU run_benign_cost.py --n 50 --skip-off --run-dir "$D" --label full_cuda \
  > "$D/full_cuda_log.txt" 2>&1; echo "benign full cuda rc=$?"
# 5-6. same with fp16 encoder weights on the GPU (RAG_ENCODER_FP16=1)
RAG_ENCODER_DEVICE=cuda RAG_ENCODER_FP16=1 $GPU bench_encoder_device.py --n 50 > external_eval/bench_cuda_fp16_log.txt 2>&1; echo "bench cuda fp16 rc=$?"
RAG_ENCODER_DEVICE=cuda RAG_ENCODER_FP16=1 $GPU run_benign_cost.py --n 50 --skip-off --run-dir "$D" --label full_cuda_fp16 \
  > "$D/full_cuda_fp16_log.txt" 2>&1; echo "benign full cuda fp16 rc=$?"
echo done
