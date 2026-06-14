#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -z "${JUDGE_API_KEY:-${OPENAI_API_KEY:-}}" ]]; then
  echo "Error: set JUDGE_API_KEY first." >&2
  echo "Example:" >&2
  echo "  export JUDGE_API_KEY='sk-...'" >&2
  echo "  CUDA_VISIBLE_DEVICES=0 bash run_self_evolve_10_api_claude_proxy.sh" >&2
  exit 1
fi

export JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://vip.yi-zhan.top/v1}"
export JUDGE_MODEL="${JUDGE_MODEL:-claude-opus-4-7}"
export GENERATOR_BASE_URL="${GENERATOR_BASE_URL:-$JUDGE_BASE_URL}"
export GENERATOR_MODEL="${GENERATOR_MODEL:-$JUDGE_MODEL}"

LIMIT="${LIMIT:-10}"
OFFSET="${OFFSET:-0}"
SKILL_DIR="${SKILL_DIR:-../zh}"
OUT_DIR="${OUT_DIR:-self_evolve_runs/api_claude_$(date +%Y%m%d_%H%M%S)_n${LIMIT}_offset${OFFSET}}"
REFINE_MODE="${REFINE_MODE:-dry-run}"
GENERATOR_MAX_TOKENS="${GENERATOR_MAX_TOKENS:-2048}"

mkdir -p "$OUT_DIR"

echo "============================================================"
echo "API self-evolve smoke run"
echo "  cases:           ${LIMIT}"
echo "  offset:          ${OFFSET}"
echo "  generator_url:   ${GENERATOR_BASE_URL}"
echo "  generator_model: ${GENERATOR_MODEL}"
echo "  generator_max:   ${GENERATOR_MAX_TOKENS}"
echo "  judge_url:       ${JUDGE_BASE_URL}"
echo "  judge_model:     ${JUDGE_MODEL}"
echo "  skill_dir:       ${SKILL_DIR}"
echo "  out_dir:         ${OUT_DIR}"
echo "  refine_mode:     ${REFINE_MODE}"
echo "============================================================"

echo
echo "[1/3] Evaluate -> ${OUT_DIR}/eval_report.json"
python self_evolve.py evaluate \
  --generator api \
  --generator-model "$GENERATOR_MODEL" \
  --judge-model "$JUDGE_MODEL" \
  --limit "$LIMIT" \
  --offset "$OFFSET" \
  --max-new-tokens "$GENERATOR_MAX_TOKENS" \
  --out "${OUT_DIR}/eval_report.json"

echo
echo "[2/3] Diagnose -> ${OUT_DIR}/diagnosis.json"
python self_evolve.py diagnose \
  --eval-report "${OUT_DIR}/eval_report.json" \
  --skill-dir "$SKILL_DIR" \
  --model "$JUDGE_MODEL" \
  --out "${OUT_DIR}/diagnosis.json"

echo
echo "[3/3] Refine -> ${OUT_DIR}/refine_results.json"
if [[ "$REFINE_MODE" == "apply" ]]; then
  python self_evolve.py refine \
    --diagnosis "${OUT_DIR}/diagnosis.json" \
    --skill-dir "$SKILL_DIR" \
    --out "${OUT_DIR}/refine_results.json"
else
  python self_evolve.py refine \
    --dry-run \
    --diagnosis "${OUT_DIR}/diagnosis.json" \
    --skill-dir "$SKILL_DIR" \
    --out "${OUT_DIR}/refine_results.json"
fi

RUN_OUT_DIR="$OUT_DIR" python - <<'PY'
import json
import os
from pathlib import Path

out_dir = Path(os.environ["RUN_OUT_DIR"])
path = out_dir / "eval_report.json"
report = json.loads(path.read_text(encoding="utf-8"))
gen = report.get("generator_usage", {})
judge = report.get("judge_usage", {})
total = report.get("usage", {})

def official_opus_cost(usage):
    return usage.get("input_tokens", 0) * 5 / 1_000_000 + usage.get("output_tokens", 0) * 25 / 1_000_000

print()
print("Usage summary")
print(f"  generator_usage: {gen}")
print(f"  judge_usage:     {judge}")
print(f"  total_usage:     {total}")
print(f"  estimated official Opus 4.7 cost: ${official_opus_cost(total):.4f}")
print()
print("Done.")
print(f"  eval_report:    {out_dir / 'eval_report.json'}")
print(f"  diagnosis:      {out_dir / 'diagnosis.json'}")
print(f"  refine_results: {out_dir / 'refine_results.json'}")
PY
