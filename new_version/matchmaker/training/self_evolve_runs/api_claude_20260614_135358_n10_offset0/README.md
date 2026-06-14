# Self-Evolve Run: API Claude, 10 Cases

This run used API generation and API judging:

- generator: `claude-opus-4-7`
- judge: `claude-opus-4-7`
- max generated tokens: `2048`
- refine mode: `dry-run`

## Stage Files

### 1. Evaluate

- `eval_report.json`: aggregate evaluation report.
- `eval_cases.jsonl`: one JSONL row per evaluated case.

Each `eval_cases.jsonl` row contains:

- `generator_input`: system prompt + case input sent to the generator.
- `generator_output`: generated matchmaking analysis.
- `reference_target`: expert target answer.
- `judge_input`: full prompt sent to the judge.
- `judge_output_raw`: raw judge model response.
- `scores`, `avg_score`, `comment`.
- `generator_usage`, `judge_usage`.

### 2. Diagnose

- `diagnosis.json`: structured root-cause diagnosis and patch recommendations.
- `diagnosis_trace.jsonl`: one JSONL row with the diagnose prompt, raw model output, parsed diagnosis, and usage.

### 3. Refine

- `refine_results.json`: aggregate patch application result.
- `refine_results.jsonl`: one JSONL row per proposed patch operation.

This run used `--dry-run`, so no real files were modified.

## Summary

- evaluated cases: `10`
- overall average: `4.14`
- weakest dimension: `strategy_direction`
- dimension averages:
  - `conflict_insight`: `4.4`
  - `strategy_direction`: `3.5`
  - `logic_depth`: `4.4`
  - `persona_read`: `4.4`
  - `actionability`: `4.0`
- total usage: `77725` input tokens, `30319` output tokens
- generator usage: `34664` input tokens, `27116` output tokens
- judge usage: `43061` input tokens, `3203` output tokens
- refine dry-run: `5` operations, `3` would apply, `2` failed anchor/path checks
