# Matchmaker SFT Training Pipeline

Distilling Claude's matchmaking analysis capabilities into a Qwen3-4B small model.

## Background: How SKILL.md Was Created

The core knowledge source of this project is `SKILL.md` — a domain skill prompt generated and iteratively optimized by Claude through automated refinement. It contains:

- **Mental Models**: Foundational analytical frameworks for the dating/matchmaking market
- **Heuristic Rules**: Decision trees for quickly identifying core contradictions in cases
- **Archetype Definitions**: Common user persona classifications (e.g., "Status-Chasing Deadlock Woman", "Small-Town Academic Princess")
- **Output Format Specifications**: Structured JSON output schema

### How SKILL Is Generated

SKILL.md is optimized through an automated evaluate-diagnose-refine loop:

```
┌─────────┐     ┌──────────┐     ┌────────┐
│ Evaluate │ ──▶ │ Diagnose │ ──▶ │ Refine │ ──┐
└─────────┘     └──────────┘     └────────┘   │
      ▲                                        │
      └────────────────────────────────────────┘
              (repeat until convergence)
```

1. **Evaluate**: Use SKILL.md to guide Claude in generating analyses for 100 test cases, scored by LLM-as-Judge (5 dimensions, 1-5 each)
2. **Diagnose**: Analyze low-scoring cases, identify weak dimensions and missing archetypes/patterns
3. **Refine**: Automatically patch SKILL.md and references, adding new rules or correcting biases

After 2 rounds of iteration, Claude + SKILL.md achieved **4.57 → 4.67** on the 100-case test set (out of 5.0).

### Supporting Knowledge Base (references/)

- `case_digest.md` — Case summaries with strategy tags
- `pattern_library.md` — Recurring conflict patterns and resolution models
- `case_archive.md` — Full case transcripts for grounding
- `statistics_snapshot.md` — Market data and demographic baselines

### How SKILL Is Used

At inference time, a condensed version of SKILL.md is injected as the system prompt. The model reads this knowledge and generates structured analysis for user-provided case JSON.

See: `matchmaker/zh/SKILL.md`

---

## Distillation Training Pipeline

With high-quality SKILL + Claude generation capability established, we distill it into a Qwen3-4B small model for independent inference.

## Distillation Flow

```
SKILL.md (domain knowledge)
     │
     ▼
┌─────────────────────────┐
│ 1. Teacher Generation    │  generate_teacher.py
│    Claude + SKILL.md     │  → data/teacher_zh.jsonl
└─────────────────────────┘
     │
     ▼
┌─────────────────────────┐
│ 2. Data Preparation      │  prepare_data.py
│    teacher → SFT format  │  → data/sft_train.jsonl / sft_val.jsonl
└─────────────────────────┘
     │
     ▼
┌─────────────────────────┐
│ 3. SFT Training          │  train_sft.py
│    LoRA + SFTTrainer     │  → outputs/<run_name>/adapter/
└─────────────────────────┘
     │
     ▼
┌─────────────────────────┐
│ 4. Inference Validation  │  Load base + adapter, run on test cases
└─────────────────────────┘
```

## Step-by-Step Details

### Step 1: Teacher Signal Generation

Use Claude as the teacher, equipped with full SKILL.md knowledge, to generate structured analyses for each user case.

```bash
export ANTHROPIC_BASE_URL=...
export ANTHROPIC_AUTH_TOKEN=...
python generate_teacher.py --lang zh --parallel 8
```

Output: `data/teacher_zh.jsonl`, each entry contains a `<thinking>` reasoning chain + JSON structured analysis.

### Step 2: Data Format Conversion

Convert teacher outputs to SFT chat-message format (system / user / assistant).

```bash
python prepare_data.py --inputs data/teacher_zh.jsonl --val-frac 0.05
```

Output: `data/sft_train.jsonl` + `data/sft_val.jsonl`

**Key Design Decision: System Prompt Strategy**

- **v1** (`sft_train.jsonl`): System prompt contains only brief format instructions, no SKILL.md knowledge. Expects the model to memorize knowledge into weights.
- **v2** (`sft_train_v2.jsonl`): System prompt includes a condensed SKILL.md (mental models, archetypes, heuristics, anti-patterns, expression DNA). Also provided at inference time — model only needs to learn "how to apply knowledge."

**Conclusion: v2 is more appropriate.** 182 training samples are insufficient for a 4B model to internalize all domain knowledge, but sufficient to teach it how to apply knowledge from the system prompt.

### Step 3: LoRA SFT Training

```bash
CUDA_VISIBLE_DEVICES=0 python train_sft.py \
    --base /path/to/Qwen3-4B-Thinking-2507 \
    --train data/sft_train_v2.jsonl --val data/sft_val_v2.jsonl \
    --run-name qwen3_4b_zh_skill_ep8_lr2e4 \
    --epochs 8 --batch-size 2 --grad-accum 4 --lr 2e-4 \
    --warmup-ratio 0.03 --max-seq 4096
```

Key parameters:
- LoRA: r=16, alpha=32, target=all-linear, trainable params ~33M (0.81%)
- `assistant_only_loss=True`: Loss computed only on assistant responses
- Gradient checkpointing enabled, runs on a single A100 80GB

### Step 4: Inference Validation

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained(base_path, dtype=torch.bfloat16, device_map={"": 0})
model = PeftModel.from_pretrained(model, "outputs/<run_name>/adapter")

# Construct messages (system prompt uses v2 version with condensed SKILL.md knowledge)
# apply_chat_template → generate
```

## Experiment Results

| Run | Data | Epochs | LR | Final Loss | Notes |
|-----|------|--------|-----|-----------|------|
| `qwen3_4b_zh_skill_ep8_lr2e4` | v2 (with knowledge) | 8 | 2e-4 | 0.55 | Final model, LLM-as-Judge avg 4.0/5 |

Training details:
- Base model: Qwen3-4B-Thinking-2507
- LoRA: r=16, alpha=32, target=all-linear, trainable params ~33M (0.81%)
- Training time: ~2h (single A100 80GB)
- Average train loss: 1.02 → final step loss: 0.55
- Token accuracy: 84%

## Demo Case

Below is the model's actual inference result on test case "鳌烨101" (full output in `test_inference_result.txt`).

### Input

32-year-old female, New York, Master's degree (Journalism/Media), Green Card holder, annual income ¥450K, family assets at A8.5 level, 165cm/56kg, has tattoos, edgy/cool style. Migrated from Malaysia to the US, obtained Green Card through personal effort.

**Partner Requirements:** Seeking a culturally aligned ABC (American-Born Chinese) or elite Chinese professional who shares similar spending values and desires a marriage with traditional roots.

### Model Output (Excerpts)

**Persona Classification:**
> Closest to the "Small-Town Academic Princess" archetype but with key differences — she didn't achieve class mobility through education only to face social circle scrutiny. Instead, through personal effort (migration → Green Card → A8.5 assets), she has already integrated into the high-net-worth circle. Also carries traces of the "Status-Chasing Deadlock Woman" since her explicit requirement for "culturally aligned ABC or elite Chinese" is essentially an extremely high demand for male intellectual/cultural supply capacity.

**Core Conflict:**
> Structural deadlock between her extremely high demand for male intellectual supply capacity (cultural alignment + wit) and the extreme scarcity of men who simultaneously satisfy "traditional family that accepts tattoos and edgy style."

**Target Portrait:**
> Age 28-35, family assets A7-A8, overseas Chinese background, finance/tech/law professionals, strong cultural consumption ability but relatively relaxed family screening (e.g., second-generation or new immigrant elites), clean-cut appearance, intellectual personality but not too introverted — essentially seeking "wealthy with social circles but whose intellectual world she can fill," rather than a "traditional + deep" unicorn.

### Ground Truth

> Prefer long-term US residents or second-generation Chinese Americans (ABC) with extremely high cultural alignment, who can accept an independent lifestyle and tattoo history, and are financially matched to prevent exploitation.

### LLM-as-Judge Scores

| Dimension | Score |
|-----------|-------|
| conflict_insight (core conflict identification) | 4/5 |
| strategy_direction (strategy alignment) | 3/5 |
| logic_depth (reasoning depth) | 5/5 |
| persona_read (personality interpretation) | 3/5 |
| actionability (actionable recommendations) | 5/5 |
| **Overall Average** | **4.0/5** |

**Assessment:** Successfully identified the tension between independent identity and traditional partner-seeking. Logic chain is complete and highly actionable. Strategy direction diverges from ground truth — the reference answer suggests not compromising but switching pools, while the model recommends proactively lowering standards.

## Inference API

The trained model is deployed as a FastAPI service. See [README_API.md](README_API.md) for details.

```bash
# Start service
CUDA_VISIBLE_DEVICES=0 python3 serve.py --port 8000

# Call
curl -X POST http://172.27.96.38:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"subject_profile": {...}, "expectations": {...}}'
```

## File Structure

```
training/
├── generate_teacher.py        # Step 1: Claude teacher generation
├── run_generate_teacher.sh    # Teacher generation launch script
├── prepare_data.py            # Step 2: teacher → SFT format
├── build_new_system_prompt.py # Build v2 data (with SKILL.md knowledge)
├── train_sft.py               # Step 3: LoRA SFT training
├── serve.py                   # FastAPI inference service
├── test_inference.py          # Single-case inference + LLM-as-Judge evaluation
├── test_inference_result.txt  # Demo case inference result
├── README_API.md              # API usage guide
├── data/
│   ├── teacher_zh.jsonl       # Teacher raw output
│   ├── sft_train_v2.jsonl     # v2 training set (with SKILL.md knowledge)
│   └── sft_val_v2.jsonl       # v2 validation set
├── outputs/
│   └── qwen3_4b_zh_skill_ep8_lr2e4/adapter/  # Final LoRA adapter
└── logs/
```

