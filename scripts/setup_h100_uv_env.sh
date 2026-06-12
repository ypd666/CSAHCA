#!/usr/bin/env bash
set -euo pipefail

UV_BIN="${UV_BIN:-${HOME}/.local/bin/uv}"
CSAHCA_VENV="${CSAHCA_VENV:-${HOME}/envs/csahca}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"

if [[ "${CSAHCA_RECREATE:-0}" == "1" ]]; then
  UV_VENV_CLEAR=1 "${UV_BIN}" venv "${CSAHCA_VENV}" --python "${PYTHON_BIN}"
elif [[ ! -x "${CSAHCA_VENV}/bin/python" ]]; then
  "${UV_BIN}" venv "${CSAHCA_VENV}" --python "${PYTHON_BIN}"
else
  echo "Using existing uv environment: ${CSAHCA_VENV}"
fi

"${UV_BIN}" pip install \
  --python "${CSAHCA_VENV}/bin/python" \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match \
  torch==2.11.0+cu128 \
  numpy \
  pytest \
  setuptools \
  wheel \
  ninja

export PATH="${CSAHCA_VENV}/bin:${CUDA_HOME}/bin:${PATH}"
export CUDA_HOME

"${CSAHCA_VENV}/bin/python" setup.py build_ext --inplace --force
"${CSAHCA_VENV}/bin/python" -m hybrid_attention.correctness \
  --device cuda \
  --dtype bfloat16 \
  --seq-len 1024 \
  --heads 2 \
  --head-dim 128 \
  --chunk-size 64 \
  --top-k 4 \
  --kernel v1 \
  --require-extension

"${CSAHCA_VENV}/bin/python" -m hybrid_attention.correctness \
  --device cuda \
  --dtype bfloat16 \
  --seq-len 1024 \
  --heads 2 \
  --head-dim 128 \
  --chunk-size 64 \
  --top-k 4 \
  --kernel tiled \
  --require-extension
