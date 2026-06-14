#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -z "${ANTHROPIC_AUTH_TOKEN:-${DEEPSEEK_API_KEY:-}}" ]]; then
  echo "Error: set DEEPSEEK_API_KEY or ANTHROPIC_AUTH_TOKEN first." >&2
  echo "Example:" >&2
  echo "  export DEEPSEEK_API_KEY='sk-...'" >&2
  echo "  CUDA_VISIBLE_DEVICES=0 bash run_self_evolve_10_deepseek_pro.sh" >&2
  exit 1
fi

if [[ -z "${ANTHROPIC_AUTH_TOKEN:-}" ]]; then
  export ANTHROPIC_AUTH_TOKEN="$DEEPSEEK_API_KEY"
fi

export ANTHROPIC_BASE_URL="https://api.deepseek.com/anthropic"
export ANTHROPIC_MODEL="deepseek-v4-pro"
unset USE_OFFICIAL_ANTHROPIC

exec bash run_self_evolve_10.sh
