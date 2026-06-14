#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

LIMIT="${LIMIT:-10}"
OFFSET="${OFFSET:-0}"
SKILL_DIR="${SKILL_DIR:-../zh}"
OUT_DIR="${OUT_DIR:-self_evolve_runs/$(date +%Y%m%d_%H%M%S)_n${LIMIT}_offset${OFFSET}}"
REFINE_MODE="${REFINE_MODE:-dry-run}"
USE_OFFICIAL_ANTHROPIC="${USE_OFFICIAL_ANTHROPIC:-0}"

mkdir -p "$OUT_DIR"

if [[ -z "${ANTHROPIC_AUTH_TOKEN:-${ANTHROPIC_API_KEY:-}}" ]]; then
  echo "Error: set ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY first." >&2
  exit 1
fi

if [[ "$USE_OFFICIAL_ANTHROPIC" == "1" ]]; then
  unset ANTHROPIC_BASE_URL
  unset ANTHROPIC_MODEL
fi

echo "============================================================"
echo "Self-evolve smoke run"
echo "  cases:       ${LIMIT}"
echo "  offset:      ${OFFSET}"
echo "  skill_dir:   ${SKILL_DIR}"
echo "  out_dir:     ${OUT_DIR}"
echo "  refine_mode: ${REFINE_MODE}"
echo "  official_api:${USE_OFFICIAL_ANTHROPIC}"
echo "============================================================"

echo
echo "[1/3] Evaluate -> ${OUT_DIR}/eval_report.json"
python self_evolve.py evaluate \
  --limit "$LIMIT" \
  --offset "$OFFSET" \
  --out "${OUT_DIR}/eval_report.json"

echo
echo "[2/3] Diagnose -> ${OUT_DIR}/diagnosis.json"
python self_evolve.py diagnose \
  --eval-report "${OUT_DIR}/eval_report.json" \
  --skill-dir "$SKILL_DIR" \
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

echo
echo "Done."
echo "  eval_report:    ${OUT_DIR}/eval_report.json"
echo "  diagnosis:      ${OUT_DIR}/diagnosis.json"
echo "  refine_results: ${OUT_DIR}/refine_results.json"
echo
echo "To apply the suggested edits after inspecting dry-run output:"
echo "  python self_evolve.py refine --diagnosis '${OUT_DIR}/diagnosis.json' --skill-dir '${SKILL_DIR}' --out '${OUT_DIR}/refine_results_apply.json'"
