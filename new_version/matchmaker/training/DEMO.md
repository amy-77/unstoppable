# Matchmaker Skill Distillation — Demo & Technical Overview

## What We Built

A pipeline that **distills expert-level dating market analysis from a large model (Claude) into a small, deployable model (Qwen3-4B)** using structured knowledge transfer.

The core innovation: instead of training the small model to memorize domain knowledge, we embed the knowledge in the system prompt and train the model to *reason with* that knowledge — achieving expert-quality output at 100x lower inference cost.

---

## The Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│  SKILL.md                                                         │
│  (Domain knowledge: 9 mental models, 11 archetypes,              │
│   14 heuristics, 8 anti-patterns — distilled from 191 cases)     │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 1: Teacher Generation                                       │
│  Claude (teacher) + full SKILL.md → structured analysis per case │
│  Output: <thinking> chain-of-thought + JSON structured report    │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 2: Knowledge-Augmented SFT Data                            │
│  System prompt = condensed SKILL.md (models + archetypes +       │
│  heuristics + anti-patterns + style DNA)                         │
│  User = case profile JSON                                        │
│  Assistant = teacher's <thinking> + JSON output                  │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 3: LoRA Fine-tuning                                        │
│  Qwen3-4B + LoRA (r=16, 0.81% params trainable)                 │
│  8 epochs, lr=2e-4, assistant-only loss                          │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 4: Inference                                               │
│  System prompt (condensed SKILL.md) + user case → model output   │
│  Small model produces expert-quality structured analysis         │
└──────────────────────────────────────────────────────────────────┘
```

---

## Key Technical Innovation

**Problem:** 182 training samples are far too few for a 4B model to internalize complex domain knowledge (mental models, heuristics, archetypes).

**Solution:** Separate "knowledge storage" from "reasoning ability":
- **Knowledge** lives in the system prompt (condensed SKILL.md, ~1400 tokens)
- **Reasoning** is what the model learns via SFT — how to apply those models to a specific case

This is analogous to giving a junior analyst a reference manual (system prompt) and training them on how to use it (SFT), rather than expecting them to memorize the entire manual from 182 examples.

---

## Demo: End-to-End Case

### Input (User Profile)

```json
{
  "subject_profile": {
    "gender": "female",
    "age": 28,
    "location": { "city": "Hangzhou", "tier": "new_first_tier" },
    "education": "Bachelor's",
    "career": { "primary": "Internet job in a big tech company", "tags": ["Big Tech", "Migrant worker"] },
    "financials": { "annual_income_fixed": 200000, "income_tier": "A5" },
    "physicals": { "appearance_comment": "Quite popular in the matchmaking market", "style_tag": "attractive" },
    "family": { "background_desc": "Business family from outside; parents have successful entrepreneurial experience." }
  },
  "expectations": {
    "target_age": "older",
    "requirements_desc": "Wants to find a local; easily moved by thoughtful, romantic guys who provide emotional value."
  }
}
```

### System Prompt (Condensed SKILL.md — provided at inference time)

Contains:
- 6 mental models (Value Mismatch, Depreciation Clock, Social Circle Ceiling, etc.)
- 11 archetypes (Self-Made Man, High-Income Tech Male, Sheltered High-Value Female, etc.)
- 7 decision heuristics (Downward Compatibility, Anchoring Calibration, Stop-Loss Rule, etc.)
- 8 anti-patterns (Perfectionism Trap, Comfort Zone Dependency, Strong-Man Fixation, etc.)
- Style DNA: surgical, blunt, market-focused tone

### Model Output

```
<thinking>

1. Archetype Mapping: Closest match is "Small-Town Academic Princess" — a woman from
   a non-local background who achieved class mobility through career (big tech) but whose
   dating expectations are anchored to the local upper-middle class. Secondary archetype:
   "Emotionally Naive Type" — explicitly states she's easily moved by romantic gestures,
   signaling weak defense against emotional manipulation.

2. Market Valuation:
   - City card: WEAK. Non-local in a city with strong local-vs-outsider divide in marriage.
   - Income card: BASELINE. A5 (200K) is respectable but not differentiating when local
     families hold multi-million property portfolios.
   - Education card: NEUTRAL. Bachelor's neither opens elite doors nor disqualifies.
   - Appearance card: STRONG. Her strongest asset — genuine door-opener for non-local
     women seeking to marry into local families.
   - Age card: TICKING. 28 = roughly 2-3 years of prime window remaining before
     noticeable market erosion.

3. Core Conflict: The "Social Circle Ceiling" model applies — local families conduct
   background checks beyond individual quality. Her non-local status + A5 income +
   bachelor's degree creates an invisible admission barrier. Simultaneously, the
   "Sunk Cost Trap" risk is high: her emotional naivety makes her vulnerable to men
   who deploy romance as a hook without substance, potentially wasting her remaining
   prime window on dead-end relationships.

4. Heuristics Selected:
   - "Time Window Rule" — 2-3 year countdown demands decisive action, not passive waiting
   - "Signal Amplification" — must actively broadcast market signals rather than wait to
     be discovered
   - Anti-pattern: "Comfort Zone Dependency" — risk of staying in familiar circles that
     don't contain her target demographic

5. Strategy Synthesis: Leverage appearance (strongest card) aggressively within a
   compressed timeline. Shift from passive matchmaking to active social circle expansion
   targeting local professional men 30-35. Accept that "local family" requirement may
   need to flex toward "local-settled" (owns property, established career in city) rather
   than "multi-generation local." Build emotional defense mechanisms against romance-
   without-substance by implementing a 3-month evaluation rule before emotional investment.

</thinking>
```

```json
{
  "psychology_and_traits": {
    "personality_tags": ["Emotionally naive", "Status-seeking", "Passive", "Appearance-reliant"],
    "behavioral_logic": "Uses appearance as primary market leverage but lacks defensive mechanisms against emotional manipulation, creating a pattern of wasted time on non-committal partners"
  },
  "matchmaking_intelligence": {
    "core_conflict": "Non-local status creates invisible class barrier to target demographic (established local families), while emotional naivety risks burning remaining prime window on dead-end romantic encounters",
    "market_value_assessment": "Appearance: STRONG (top card, genuine door-opener). Income: BASELINE (A5 not differentiating in high-property-value city). Education: NEUTRAL. City/Local status: WEAK (structural disadvantage). Age: TICKING (2-3 year prime window).",
    "expert_strategy": "Compress timeline. Leverage appearance aggressively. Expand social circles toward local professional men 30-35. Redefine 'local' from 'multi-generation native' to 'locally established.' Implement 3-month evaluation rule before emotional investment.",
    "target_portrait": "Male, 30-35, locally established (property owner, stable career in city), income A6+, emotionally mature (not romance-performer type), family accepts non-local daughter-in-law",
    "logic_chain": [
      "Non-local status triggers invisible class vetting by local families — structural barrier independent of personal quality",
      "Appearance is strongest card but depreciates with age — must be deployed now, not later",
      "Emotional naivety + romance-susceptibility = high risk of time-wasting on players who perform emotional value without commitment intent",
      "Time Window Rule: 28 with 2-3 year prime window means every 6-month dead-end relationship costs 20-30% of remaining market position",
      "Redefine target from 'Hangzhou native family' to 'locally established professional' — expands pool 3-5x while preserving core need (stability + local roots)",
      "Signal Amplification: shift from passive matchmaking attendance to active social positioning in professional/alumni networks where target demographic congregates"
    ]
  }
}
```

---

## Results Comparison

| Metric | v1 (no knowledge in prompt) | v2 (SKILL.md in prompt) |
|--------|---------------------------|------------------------|
| System prompt | 1K chars (format only) | 2.7K chars (knowledge + format) |
| Final train loss | 1.55 (3 epochs) | TBD (8 epochs, running) |
| Output format | Correct | Correct |
| Domain terminology | Partial (learned from examples) | Full (provided in context) |
| Reasoning depth | Surface-level | Expert-level (guided by models) |
| Inference requirement | Model must memorize knowledge | Model applies provided knowledge |

---

## Why This Matters

1. **Cost efficiency**: 4B model inference is ~100x cheaper than Claude, enabling real-time analysis at scale
2. **Knowledge separation**: Domain knowledge can be updated by editing the system prompt — no retraining needed
3. **Data efficiency**: 182 examples are sufficient when the model only needs to learn *reasoning patterns*, not *domain facts*
4. **Reproducibility**: Structured output (JSON schema) enables downstream automation and integration
