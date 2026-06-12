#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-30000}"
CONFIRM_RESTART="${CONFIRM_RESTART:-0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${CONFIRM_RESTART}" != "1" ]]; then
  cat >&2 <<EOF
Refusing to restart service on port ${PORT}.

Set CONFIRM_RESTART=1 to stop the existing SGLang process on this port and
launch the CSAHCA-patched DeepSeek-V4 service.

Example:
  CONFIRM_RESTART=1 PORT=${PORT} CSAHCA_DSV4_MODE=trace \\
    bash ${SCRIPT_DIR}/restart_patched_on_port.sh
EOF
  exit 2
fi

pid="$(ss -ltnp 2>/dev/null | awk -v port=":${PORT}" '$4 ~ port {print $0}' | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | head -n 1)"
if [[ -n "${pid}" ]]; then
  echo "Stopping existing process on port ${PORT}: pid=${pid}"
  kill "${pid}"
  for _ in $(seq 1 60); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "${pid}" 2>/dev/null; then
    echo "Process ${pid} did not exit after 60s; sending SIGKILL"
    kill -9 "${pid}"
  fi
fi

exec bash "${SCRIPT_DIR}/launch_patched_dsv4.sh"
