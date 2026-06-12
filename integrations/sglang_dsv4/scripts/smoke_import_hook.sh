#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATCH_ROOT="${PATCH_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SGLANG_SRC="${SGLANG_SRC:-${HOME}/src/sglang-main}"
PY="${PY:-${HOME}/envs/dsv4_flash/bin/python}"
VENV_ROOT="${VENV_ROOT:-$(dirname "$(dirname "${PY}")")}"
SITE_PACKAGES="${SITE_PACKAGES:-${VENV_ROOT}/lib/python3.10/site-packages}"

export PYTHONPATH="${PATCH_ROOT}:${SGLANG_SRC}/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${SITE_PACKAGES}/nvidia/cu13/lib:${SITE_PACKAGES}/nvidia/cu13/lib64:${SITE_PACKAGES}/torch/lib:${SITE_PACKAGES}/tvm_ffi/lib:${LD_LIBRARY_PATH:-}"
export CSAHCA_SGLANG_DSV4_PATCH="${CSAHCA_SGLANG_DSV4_PATCH:-1}"
export CSAHCA_DSV4_MODE="${CSAHCA_DSV4_MODE:-trace}"

"${PY}" - <<'PY'
import inspect
from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend

forward = DeepseekV4AttnBackend.forward
print({
    "patched": bool(getattr(forward, "_csahca_dsv4_patched", False)),
    "forward_module": getattr(forward, "__module__", None),
    "source": inspect.getsourcefile(forward),
})
PY
