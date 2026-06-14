# Self-Evolve Workflow and 10-Case Run Summary

This document summarizes the current Evaluate -> Diagnose -> Refine workflow used for the zh matchmaking skill.

## Initial Skill Source

The initial `zh/SKILL.md` follows a reference-first construction pattern inspired by the Nuwa Skill examples:

1. Collect domain references: raw cases, expert targets, recurring patterns, and market statistics.
2. Organize those references into reusable documents such as `case_digest.md`, `pattern_library.md`, `case_archive.md`, and `statistics_snapshot.md`.
3. Distill the references into a skill containing mental models, archetypes, heuristics, anti-patterns, and output schema rules.
4. Use the Evaluate -> Diagnose -> Refine loop to iteratively improve the skill.

Reference example:
https://github.com/alchaincyf/nuwa-skill/blob/main/examples/elon-musk-perspective/references/Elon-Musk-%E6%80%9D%E6%83%B3%E4%BD%93%E7%B3%BB%E8%B0%83%E7%A0%94-20260404.md

## Stage E1: Evaluate

**Purpose:** Generate analyses for benchmark cases and score them with an LLM-as-Judge.

**Inputs:**

- Test inputs: `/data/qwang/q/thalia/agentic_hackson/agent/datasets/test/inputs.json`
- Expert targets: `/data/qwang/q/thalia/agentic_hackson/agent/datasets/test/outputs.json`
- Generator system prompt: `training/data/sft_train_v2.jsonl` first system message
- API configuration:
  - `JUDGE_BASE_URL`
  - `JUDGE_MODEL`
  - `GENERATOR_BASE_URL`
  - `GENERATOR_MODEL`

**Outputs:**

- `eval_report.json`: aggregate scores, dimension averages, weakest dimension, usage, and per-case results.
- `eval_cases.jsonl`: one row per case with generator input/output, reference target, judge prompt/output, scores, and usage.

## Stage D1: Diagnose

**Purpose:** Use the evaluation report and current skill context to identify weak dimensions, root causes, and prioritized improvement suggestions.

**Inputs:**

- `eval_report.json`
- Current skill: `zh/SKILL.md`
- References: `zh/references/*.md`
- Current prompt context: `training/data/sft_train_v2.jsonl`

**Outputs:**

- `diagnosis.json`: weakest dimension, three root causes, and prioritized improvements.
- `diagnosis_trace.jsonl`: one trace row containing the full diagnose prompt, raw model output, parsed diagnosis, and token usage.

## Stage R1: Refine

**Purpose:** Convert diagnosis improvements into deterministic patch operations over `SKILL.md` and `references/*.md`.

**Inputs:**

- `diagnosis.json`
- Editable files:
  - `zh/SKILL.md`
  - `zh/references/*.md`

**Patch operations:**

- `replace`
- `insert_after`
- `append`

`insert_after` uses an `anchor`, i.e. exact existing text in the target file that determines where new content should be inserted.

**Outputs:**

- `refine_results.json`: aggregate patch status.
- `refine_results.jsonl`: one row per patch operation.

`--dry-run` validates patches without changing files.

## 10-Case API Run

Run directory:

```text
training/self_evolve_runs/api_claude_20260614_135358_n10_offset0
```

Configuration:

- Generator: `claude-opus-4-7`
- Judge: `claude-opus-4-7`
- Generated output limit: `2048`
- Refine mode: `dry-run`

Results:

- Cases: `10`
- Overall average: `4.14`
- Weakest dimension: `strategy_direction`
- Dimension averages:
  - `conflict_insight`: `4.4`
  - `strategy_direction`: `3.5`
  - `logic_depth`: `4.4`
  - `persona_read`: `4.4`
  - `actionability`: `4.0`
- Evaluate usage: `77,725` input tokens, `30,319` output tokens
- Diagnose usage: `86,261` input tokens, `3,226` output tokens
- Refine dry-run: `5` proposed operations, `3` would apply, `2` failed anchor/text checks

## Paper-Ready Description

We implement a self-evolving skill optimization loop consisting of three stages. In the Evaluate stage, the current skill-conditioned prompt is used by an API-based generator to produce structured matchmaking analyses for benchmark cases, and an LLM judge scores each output along five dimensions. In the Diagnose stage, the evaluation report, current `SKILL.md`, reference files, and prompt context are provided to an LLM diagnostician, which identifies the weakest dimension, summarizes root causes, and proposes prioritized document-level edits. In the Refine stage, the proposed edits are converted into deterministic patch operations over `SKILL.md` and reference files. A dry-run mode validates patch anchors and reports which edits would be applied without mutating the source files.
