#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -z "${JUDGE_API_KEY:-${OPENAI_API_KEY:-}}" ]]; then
  echo "Error: set JUDGE_API_KEY first." >&2
  echo "Example:" >&2
  echo "  export JUDGE_API_KEY='sk-...'" >&2
  echo "  CUDA_VISIBLE_DEVICES=0 bash run_cost_probe_1_claude_proxy.sh" >&2
  exit 1
fi

export JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://vip.yi-zhan.top/v1}"
export JUDGE_MODEL="${JUDGE_MODEL:-claude-opus-4-7}"
export GENERATOR_BASE_URL="${GENERATOR_BASE_URL:-$JUDGE_BASE_URL}"
export GENERATOR_MODEL="${GENERATOR_MODEL:-$JUDGE_MODEL}"
OUT_DIR="${OUT_DIR:-self_evolve_runs/cost_probe_1_$(date +%Y%m%d_%H%M%S)}"
GENERATOR_MODE="${GENERATOR_MODE:-api}"
GENERATOR_MAX_TOKENS="${GENERATOR_MAX_TOKENS:-2048}"

mkdir -p "$OUT_DIR"

echo "============================================================"
echo "Cost probe"
echo "  cases:     1"
echo "  base_url:  ${JUDGE_BASE_URL}"
echo "  generator: ${GENERATOR_MODE} / ${GENERATOR_MODEL}"
echo "  judge:     ${JUDGE_MODEL}"
echo "  out_dir:   ${OUT_DIR}"
echo "============================================================"

python self_evolve.py evaluate \
  --limit 1 \
  --generator "$GENERATOR_MODE" \
  --generator-model "$GENERATOR_MODEL" \
  --max-new-tokens "$GENERATOR_MAX_TOKENS" \
  --judge-model "$JUDGE_MODEL" \
  --out "${OUT_DIR}/eval_report.json"

REPORT_PATH="${OUT_DIR}/eval_report.json" python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["REPORT_PATH"])
report = json.loads(path.read_text(encoding="utf-8"))
usage = report.get("usage", {})
gen = report.get("generator_usage", {})
judge = report.get("judge_usage", {})
inp = usage.get("input_tokens", 0)
out = usage.get("output_tokens", 0)

# Claude Opus 4.7 official baseline pricing. If your proxy uses different
# pricing, replace these rates with the proxy's actual billing rates.
input_rate = 5.0 / 1_000_000
output_rate = 25.0 / 1_000_000
cost = inp * input_rate + out * output_rate

print()
print("Usage summary")
print(f"  generator_usage: {gen}")
print(f"  judge_usage:     {judge}")
print(f"  input_tokens:  {inp}")
print(f"  output_tokens: {out}")
print(f"  estimated official Opus 4.7 cost: ${cost:.4f}")
print(f"  report: {path}")
PY
