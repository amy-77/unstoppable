#!/bin/bash
# Background launcher for generate_teacher.py.
# Resumes incrementally; safe to re-run after a crash.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LANG="${LANG:-zh}"
PARALLEL="${PARALLEL:-8}"
MAX_TOKENS="${MAX_TOKENS:-10000}"
LIMIT_FLAG=""
if [[ -n "${LIMIT:-}" ]]; then
    LIMIT_FLAG="--limit $LIMIT"
fi

PYTHON="${PYTHON:-/home/qwang/software/miniforge3/envs/retroinfer/bin/python}"

unset LD_LIBRARY_PATH || true
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-https://zyapi.xmsxb.com}"
export ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:?must set ANTHROPIC_AUTH_TOKEN}"
export ANTHROPIC_MODEL="${ANTHROPIC_MODEL:-mco-4}"

mkdir -p "${SCRIPT_DIR}/logs"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${SCRIPT_DIR}/logs/generate_teacher_${LANG}_${TS}.log"
PID_FILE="${SCRIPT_DIR}/.generate_teacher_${LANG}.pid"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "[run] generate_teacher (lang=${LANG}) already running with PID $(cat "$PID_FILE")"
    exit 1
fi

echo "[run] starting generate_teacher | lang=${LANG} parallel=${PARALLEL} log=${LOG_FILE}"

cd "$(dirname "$SCRIPT_DIR")"  # matchmaker/

nohup "$PYTHON" -u "${SCRIPT_DIR}/generate_teacher.py" \
    --lang "$LANG" \
    --parallel "$PARALLEL" \
    --max-tokens "$MAX_TOKENS" \
    $LIMIT_FLAG \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"
sleep 2
if kill -0 "$PID" 2>/dev/null; then
    echo "[run] PID=$PID started, tail with: tail -f $LOG_FILE"
else
    echo "[run] failed to start, see $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi
