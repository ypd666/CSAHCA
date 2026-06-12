#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-30000}"
HOST="${HOST:-127.0.0.1}"
CONFIRM_RESTART="${CONFIRM_RESTART:-0}"
WAIT_READY_SECONDS="${WAIT_READY_SECONDS:-900}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
LOG_FILE="${LOG_FILE:-${PROJECT_ROOT}/logs/dsv4_flash/sglang_h100x4_csahca_patch.log}"
PID_FILE="${PID_FILE:-${PROJECT_ROOT}/logs/dsv4_flash/sglang_h100x4_csahca_patch.pid}"
LAUNCH_LOG="${LAUNCH_LOG:-${PROJECT_ROOT}/logs/dsv4_flash/sglang_h100x4_csahca_patch.launcher.log}"

if [[ "${CONFIRM_RESTART}" != "1" ]]; then
  cat >&2 <<EOF
Refusing to restart service on port ${PORT}.

Set CONFIRM_RESTART=1 to stop the existing SGLang process on this port and
launch the CSAHCA-patched DeepSeek-V4 service in the background.

Example:
  CONFIRM_RESTART=1 PORT=${PORT} CSAHCA_DSV4_MODE=trace \\
    bash ${SCRIPT_DIR}/restart_patched_detached.sh
EOF
  exit 2
fi

mkdir -p "$(dirname "${LOG_FILE}")"

pid="$(ss -ltnp 2>/dev/null | awk -v port=":${PORT}" '$4 ~ port {print $0}' | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | head -n 1)"
if [[ -n "${pid}" ]]; then
  echo "Stopping existing process on ${HOST}:${PORT}: pid=${pid}"
  kill "${pid}"
  for _ in $(seq 1 90); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "${pid}" 2>/dev/null; then
    echo "Process ${pid} did not exit after 90s; sending SIGKILL"
    kill -9 "${pid}"
  fi
fi

echo "Launching patched service on ${HOST}:${PORT}; log=${LOG_FILE}"
nohup bash "${SCRIPT_DIR}/launch_patched_dsv4.sh" \
  > "${LAUNCH_LOG}" 2>&1 &
new_pid="$!"
echo "${new_pid}" > "${PID_FILE}"
echo "launcher_pid=${new_pid}"

deadline=$((SECONDS + WAIT_READY_SECONDS))
while (( SECONDS < deadline )); do
  if curl -fsS --max-time 3 "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "ready=true"
    echo "pid_file=${PID_FILE}"
    echo "log_file=${LOG_FILE}"
    exit 0
  fi
  if ! kill -0 "${new_pid}" 2>/dev/null; then
    echo "launcher process exited before service became ready. Last log lines:" >&2
    tail -n 80 "${LOG_FILE}" >&2 || true
    tail -n 80 "${LAUNCH_LOG}" >&2 || true
    exit 1
  fi
  sleep 5
done

echo "Timed out waiting for ${HOST}:${PORT} after ${WAIT_READY_SECONDS}s" >&2
tail -n 80 "${LOG_FILE}" >&2 || true
tail -n 80 "${LAUNCH_LOG}" >&2 || true
exit 1
