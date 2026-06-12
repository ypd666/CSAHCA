#!/usr/bin/env bash
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
CSAHCA_VENV="${CSAHCA_VENV:-${HOME}/envs/csahca}"
export PATH="${CSAHCA_VENV}/bin:${CUDA_HOME}/bin:${PATH}"
PYTHON_BIN="${PYTHON_BIN:-${CSAHCA_VENV}/bin/python}"

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"

DTYPE="${DTYPE:-bfloat16}"
BATCH="${BATCH:-1}"
HEADS="${HEADS:-8}"
SEQ_LEN="${SEQ_LEN:-32768}"
HEAD_DIM="${HEAD_DIM:-128}"
CHUNK_SIZE="${CHUNK_SIZE:-64}"
TOP_K="${TOP_K:-8}"
TILE_SIZE="${TILE_SIZE:-8}"
STEPS="${STEPS:-200}"
WARMUP="${WARMUP:-20}"
MLP_RATIO="${MLP_RATIO:-2.0}"
SELECTION="${SELECTION:-precomputed}"
OUT="${OUT:-results/model_kernel_ab_gpu${GPU_ID}_${SEQ_LEN}_${DTYPE}.csv}"

mkdir -p "$(dirname "${OUT}")"

for backend in torch-full torch-csa cuda-csa-tiled; do
  "${PYTHON_BIN}" -m hybrid_attention.model_inference \
    --backend "${backend}" \
    --selection "${SELECTION}" \
    --device cuda \
    --dtype "${DTYPE}" \
    --batch "${BATCH}" \
    --heads "${HEADS}" \
    --seq-len "${SEQ_LEN}" \
    --head-dim "${HEAD_DIM}" \
    --chunk-size "${CHUNK_SIZE}" \
    --top-k "${TOP_K}" \
    --tile-size "${TILE_SIZE}" \
    --steps "${STEPS}" \
    --warmup "${WARMUP}" \
    --mlp-ratio "${MLP_RATIO}" \
    --out "${OUT}"
done

echo "Wrote ${OUT}"
