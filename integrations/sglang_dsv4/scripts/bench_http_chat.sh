#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PY="${PY:-${HOME}/envs/dsv4_flash/bin/python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
MODEL_PATH="${MODEL_PATH:-${HOME}/checkpoints/DeepSeek-V4-Flash-HF}"
NUM_PROMPTS="${NUM_PROMPTS:-4}"
CONCURRENCY="${CONCURRENCY:-1}"
INPUT_WORDS="${INPUT_WORDS:-64}"
MAX_TOKENS="${MAX_TOKENS:-16}"
IGNORE_EOS="${IGNORE_EOS:-1}"
OUT="${OUT:-${PROJECT_ROOT}/results/http_chat_${PORT}_${INPUT_WORDS}_${MAX_TOKENS}.json}"

ignore_eos_arg="--ignore-eos"
if [[ "${IGNORE_EOS}" == "0" || "${IGNORE_EOS,,}" == "false" || "${IGNORE_EOS,,}" == "no" ]]; then
  ignore_eos_arg="--no-ignore-eos"
fi

mkdir -p "$(dirname "${OUT}")"

"${PY}" "${SCRIPT_DIR}/bench_http_chat.py" \
  --host "${HOST}" \
  --port "${PORT}" \
  --model "${MODEL_PATH}" \
  --num-prompts "${NUM_PROMPTS}" \
  --concurrency "${CONCURRENCY}" \
  --input-words "${INPUT_WORDS}" \
  --max-tokens "${MAX_TOKENS}" \
  "${ignore_eos_arg}" \
  --out "${OUT}"
