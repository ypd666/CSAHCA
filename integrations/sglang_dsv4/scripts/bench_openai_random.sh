#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
SGLANG_SRC="${SGLANG_SRC:-${HOME}/src/sglang-main}"
PY="${PY:-${HOME}/envs/dsv4_flash/bin/python}"
VENV_ROOT="${VENV_ROOT:-$(dirname "$(dirname "${PY}")")}"
SITE_PACKAGES="${SITE_PACKAGES:-${VENV_ROOT}/lib/python3.10/site-packages}"
MODEL_PATH="${MODEL_PATH:-${HOME}/checkpoints/DeepSeek-V4-Flash-HF}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
NUM_PROMPTS="${NUM_PROMPTS:-16}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-1024}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-64}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-}"
OUT="${OUT:-${PROJECT_ROOT}/results/sglang_dsv4_random_${PORT}_${RANDOM_INPUT_LEN}_${RANDOM_OUTPUT_LEN}.jsonl}"

export PYTHONPATH="${SGLANG_SRC}/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${SITE_PACKAGES}/nvidia/cu13/lib:${SITE_PACKAGES}/nvidia/cu13/lib64:${SITE_PACKAGES}/torch/lib:${SITE_PACKAGES}/tvm_ffi/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$(dirname "${OUT}")"

args=(
  -m sglang.bench_serving
  --backend sglang-oai-chat
  --host "${HOST}"
  --port "${PORT}"
  --model "${MODEL_PATH}"
  --tokenizer "${MODEL_PATH}"
  --dataset-name random
  --random-input-len "${RANDOM_INPUT_LEN}"
  --random-output-len "${RANDOM_OUTPUT_LEN}"
  --random-range-ratio 0
  --num-prompts "${NUM_PROMPTS}"
  --request-rate "${REQUEST_RATE}"
  --output-file "${OUT}"
)

if [[ -n "${MAX_CONCURRENCY}" ]]; then
  args+=(--max-concurrency "${MAX_CONCURRENCY}")
fi

"${PY}" "${args[@]}"
