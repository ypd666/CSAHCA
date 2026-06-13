#!/usr/bin/env bash
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
CSAHCA_VENV="${CSAHCA_VENV:-/mnt/Data/yangpd/envs/csahca}"
export PATH="${CSAHCA_VENV}/bin:${CUDA_HOME}/bin:${PATH}"
PYTHON_BIN="${PYTHON_BIN:-${CSAHCA_VENV}/bin/python}"

mkdir -p results

for selection in precomputed dynamic; do
  for backend in torch-csa cuda-csa cuda-csa-tiled; do
    "${PYTHON_BIN}" -m hybrid_attention.model_inference \
      --backend "${backend}" \
      --selection "${selection}" \
      --device cuda \
      --dtype bfloat16 \
      --batch 1 \
      --heads 8 \
      --seq-len 32768 \
      --head-dim 128 \
      --chunk-size 64 \
      --top-k 8 \
      --tile-size 8 \
      --steps 200 \
      --warmup 20 \
      --mlp-ratio 0.0 \
      --out results/inference_demo.csv
  done
done

