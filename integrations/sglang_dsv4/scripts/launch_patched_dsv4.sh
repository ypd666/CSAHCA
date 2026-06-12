#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATCH_ROOT="${PATCH_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${PATCH_ROOT}/../.." && pwd)}"
SGLANG_SRC="${SGLANG_SRC:-${HOME}/src/sglang-main}"
PY="${PY:-${HOME}/envs/dsv4_flash/bin/python}"
SGLANG_BIN="${SGLANG_BIN:-${HOME}/envs/dsv4_flash/bin/sglang}"
VENV_ROOT="${VENV_ROOT:-$(dirname "$(dirname "${PY}")")}"
SITE_PACKAGES="${SITE_PACKAGES:-${VENV_ROOT}/lib/python3.10/site-packages}"
CUDA_HOME="${CUDA_HOME:-${SITE_PACKAGES}/nvidia/cu13}"
GCC12_HOME="${GCC12_HOME:-${HOME}/conda-envs/gcc12}"
MODEL_PATH="${MODEL_PATH:-${HOME}/checkpoints/DeepSeek-V4-Flash-HF}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30001}"
TP_SIZE="${TP_SIZE:-4}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-1048576}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4}"
LOG_FILE="${LOG_FILE:-${PROJECT_ROOT}/logs/dsv4_flash/sglang_h100x4_csahca_patch.log}"
SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-}"

export CUDA_VISIBLE_DEVICES
export CUDA_HOME
export CC="${CC:-${GCC12_HOME}/bin/x86_64-conda-linux-gnu-gcc}"
export CXX="${CXX:-${GCC12_HOME}/bin/x86_64-conda-linux-gnu-g++}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-${CXX}}"
if [[ -z "${NVCC_PREPEND_FLAGS:-}" ]]; then
  export NVCC_PREPEND_FLAGS="-ccbin=${CXX}"
else
  export NVCC_PREPEND_FLAGS
fi
export PATH="$(dirname "${PY}"):${CUDA_HOME}/bin:${GCC12_HOME}/bin:${PATH:-}"
export PYTHONPATH="${PATCH_ROOT}:${PROJECT_ROOT}:${SGLANG_SRC}/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${GCC12_HOME}/lib:${SITE_PACKAGES}/nvidia/cu13/lib:${SITE_PACKAGES}/nvidia/cu13/lib64:${SITE_PACKAGES}/torch/lib:${SITE_PACKAGES}/tvm_ffi/lib:${LD_LIBRARY_PATH:-}"
export CSAHCA_SGLANG_DSV4_PATCH="${CSAHCA_SGLANG_DSV4_PATCH:-1}"
export CSAHCA_DSV4_MODE="${CSAHCA_DSV4_MODE:-trace}"
export CSAHCA_DSV4_TRACE_FIRST_N="${CSAHCA_DSV4_TRACE_FIRST_N:-16}"
export CSAHCA_DSV4_TRACE_ABI="${CSAHCA_DSV4_TRACE_ABI:-1}"
export CSAHCA_DSV4_TRACE_ABI_FIRST_N="${CSAHCA_DSV4_TRACE_ABI_FIRST_N:-4}"
export CSAHCA_DSV4_NVTX="${CSAHCA_DSV4_NVTX:-1}"
export CSAHCA_DSV4_SUMMARY_INTERVAL_S="${CSAHCA_DSV4_SUMMARY_INTERVAL_S:-30}"
export CSAHCA_DSV4_COMPARE_FIRST_N="${CSAHCA_DSV4_COMPARE_FIRST_N:-2}"
export CSAHCA_DSV4_COMPARE_MAX_CALLS="${CSAHCA_DSV4_COMPARE_MAX_CALLS:-16}"
export CSAHCA_DSV4_COMPARE_LAYER_IDS="${CSAHCA_DSV4_COMPARE_LAYER_IDS:-0}"
export CSAHCA_DSV4_COMPARE_MAX_Q="${CSAHCA_DSV4_COMPARE_MAX_Q:-64}"
export CSAHCA_DSV4_COMPARE_FORWARD_MODES="${CSAHCA_DSV4_COMPARE_FORWARD_MODES:-EXTEND}"
export CSAHCA_DSV4_REPLACE_FORWARD_MODES="${CSAHCA_DSV4_REPLACE_FORWARD_MODES:-DECODE}"
export CSAHCA_DSV4_DELEGATE_LOG_FIRST_N="${CSAHCA_DSV4_DELEGATE_LOG_FIRST_N:-0}"

mkdir -p "$(dirname "${LOG_FILE}")"

echo "Launching patched DeepSeek-V4 SGLang service" | tee "${LOG_FILE}"
echo "  port=${PORT} cuda=${CUDA_VISIBLE_DEVICES} mode=${CSAHCA_DSV4_MODE}" | tee -a "${LOG_FILE}"
echo "  patch_root=${PATCH_ROOT}" | tee -a "${LOG_FILE}"
echo "  extra_args=${SGLANG_EXTRA_ARGS}" | tee -a "${LOG_FILE}"

read -r -a EXTRA_ARGS <<< "${SGLANG_EXTRA_ARGS}"

"${SGLANG_BIN}" serve \
  --model-path "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tp-size "${TP_SIZE}" \
  --context-length "${CONTEXT_LENGTH}" \
  --mem-fraction-static 0.80 \
  --trust-remote-code \
  --moe-runner-backend marlin \
  --fp4-gemm-backend marlin \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "${LOG_FILE}"
