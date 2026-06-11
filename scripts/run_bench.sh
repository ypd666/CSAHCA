#!/usr/bin/env bash
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
CSAHCA_VENV="${CSAHCA_VENV:-/mnt/Data/yangpd/envs/csahca}"
export PATH="${CSAHCA_VENV}/bin:${CUDA_HOME}/bin:${PATH}"
PYTHON_BIN="${PYTHON_BIN:-${CSAHCA_VENV}/bin/python}"

mkdir -p results

for mode in torch-csa cuda-csa cuda-csa-tiled; do
  for seq_len in 4096 8192 16384 32768; do
    "${PYTHON_BIN}" -m hybrid_attention.benchmark \
      --mode "${mode}" \
      --device cuda \
      --dtype bfloat16 \
      --batch 1 \
      --heads 8 \
      --seq-len "${seq_len}" \
      --head-dim 128 \
      --chunk-size 64 \
      --tile-size 8 \
      --top-k 8 \
      --warmup 10 \
      --iters 50 \
      --out results/results.csv
  done
done
