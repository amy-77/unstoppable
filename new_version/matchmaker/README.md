# Matchmaker Skill

An AI-powered matchmaking analysis skill with automated iterative optimization.

It takes raw dating/matchmaking case data and produces structured diagnostics: personality reads, market positioning, core conflicts, and actionable strategy recommendations.

## Project Structure

```text
matchmaker/
  README.md
  zh/
    SKILL.md
    references/
      case_archive.md
      case_digest.md
      pattern_library.md
      statistics_snapshot.md
  en/
    SKILL.md
    references/
      ...
  scripts/
    iterate_pipeline.py
    evaluate.py
    diagnose.py
    refine.py
    distill_pipeline.py
    i18n.py
    iteration_history/
```

## Skill Layers

### `SKILL.md`

The core skill prompt. Contains mental models, heuristic rules, archetype definitions, and output format specifications. This is what the LLM reads at inference time.

### `references/`

Supporting knowledge base loaded alongside the skill:

- `case_digest.md` — condensed case summaries with strategy tags
- `pattern_library.md` — recurring conflict patterns and resolution models
- `case_archive.md` — full case transcripts for grounding
- `statistics_snapshot.md` — market data and demographic baselines

## Iteration Pipeline

The skill is optimized through an automated evaluate-diagnose-refine loop.

```text
┌─────────┐     ┌──────────┐     ┌────────┐
│ Evaluate │ ──▶ │ Diagnose │ ──▶ │ Refine │ ──┐
└─────────┘     └──────────┘     └────────┘   │
      ▲                                        │
      └────────────────────────────────────────┘
              (repeat until convergence)
```

### Usage

```bash
# Full iterative optimization (default: 3 rounds, target 4.9)
python3 scripts/iterate_pipeline.py --lang zh --target-score 4.9 --max-rounds 3 --parallel 10

# Evaluate only (no modification)
python3 scripts/evaluate.py --skill-path zh/SKILL.md --sample 20

# Diagnose weaknesses from an eval report
python3 scripts/diagnose.py --eval-report scripts/iteration_history/<round>/eval_report.json

# Translate skill between languages
python3 scripts/i18n.py --source zh --target en
```

### Pipeline Scripts

| Script | Purpose |
|--------|---------|
| `iterate_pipeline.py` | Orchestrates the full loop: evaluate → diagnose → refine → re-evaluate |
| `evaluate.py` | Generates analysis for test cases, scores via LLM-as-Judge, extracts structured fields |
| `diagnose.py` | Identifies weak dimensions, root causes, and missing archetypes |
| `refine.py` | Generates and applies patches to SKILL.md, references, and prompts |
| `distill_pipeline.py` | Distills raw case data into structured training/test datasets |
| `i18n.py` | Translates skill and references across language variants |

### Environment Variables

```bash
ANTHROPIC_API_KEY=<key>        # or ANTHROPIC_AUTH_TOKEN
ANTHROPIC_BASE_URL=<url>       # optional, for proxy/gateway
ANTHROPIC_MODEL=<model>        # model for generation and judging
EXTRACT_MODEL=<model>          # model for structured field extraction (falls back to ANTHROPIC_MODEL)
```

### Evaluation Dimensions

Each case is scored 1-5 on:

1. **conflict_insight** — identifies the core contradiction
2. **strategy_direction** — strategy aligns with expert recommendation
3. **logic_depth** — reasoning chain is deep and self-consistent
4. **persona_read** — personality/psychology interpretation accuracy
5. **actionability** — recommendations are concrete and executable

### Iteration History

Each round saves to `scripts/iteration_history/round_<N>_<timestamp>/`:

- `eval_report.json` — per-case scores, generated docs, extracted fields
- `diagnosis.json` — identified weaknesses and improvement suggestions
- `refine_results.json` — patches applied and their outcomes
- `backups/` — pre-modification file snapshots

## Output Schema

After evaluation, each case in `per_case_results` includes `extracted_fields`:

```json
{
  "psychology_and_traits": {
    "personality_tags": ["tag1", "tag2", "..."],
    "behavioral_logic": "one-sentence behavioral pattern summary"
  },
  "matchmaking_intelligence": {
    "core_conflict": "the central contradiction identified",
    "market_value_assessment": "market positioning summary",
    "expert_strategy": "recommended core strategy",
    "target_portrait": "ideal match profile",
    "logic_chain": ["step1", "step2", "..."]
  }
}
```

## Current Performance

Score trajectory across full 100-case test set:

```
4.57 → 4.67 (converged after 2 rounds)
```

## What Is Not Committed

- Raw case transcripts with PII
- API keys or tokens
- `iteration_history/` intermediate files (gitignored)
- `datasets/` raw data
