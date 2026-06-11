#!/usr/bin/env bash
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
export PATH="${CUDA_HOME}/bin:${PATH}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/Data/yangpd/envs/airworld-latent/bin/python}"

mkdir -p profiling/ncu

ncu \
  --set full \
  --target-processes all \
  --force-overwrite \
  --export profiling/ncu/cuda_csa_decode \
  "${PYTHON_BIN}" -m hybrid_attention.benchmark \
    --mode cuda-csa \
    --device cuda \
    --dtype bfloat16 \
    --batch 1 \
    --heads 8 \
    --seq-len 32768 \
    --head-dim 128 \
    --chunk-size 64 \
    --top-k 8 \
    --warmup 5 \
    --iters 10 \
    --nvtx
