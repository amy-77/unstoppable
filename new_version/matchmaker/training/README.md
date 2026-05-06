# Matchmaker SFT Training Pipeline

将大模型（Claude）的婚恋市场分析能力蒸馏到 Qwen3-4B 小模型中。

## Pipeline 概览

```
SKILL.md (领域知识)
     │
     ▼
┌─────────────────────┐
│ 1. Teacher 生成      │  generate_teacher.py
│    Claude + SKILL.md │  → data/teacher_zh.jsonl
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│ 2. 数据准备          │  prepare_data.py
│    teacher → SFT格式 │  → data/sft_train.jsonl / sft_val.jsonl
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│ 3. SFT 训练         │  train_sft.py
│    LoRA + SFTTrainer │  → outputs/<run_name>/adapter/
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│ 4. 推理验证          │  加载 base + adapter，输入用户案例，检查输出质量
└─────────────────────┘
```

## 各步骤详解

### Step 1: Teacher 信号生成

用 Claude 作为 teacher，带着完整 SKILL.md 知识，对每个用户案例生成结构化分析。

```bash
export ANTHROPIC_BASE_URL=...
export ANTHROPIC_AUTH_TOKEN=...
python generate_teacher.py --lang zh --parallel 8
```

输出：`data/teacher_zh.jsonl`，每条包含 `<thinking>` 推理链 + JSON 结构化分析。

### Step 2: 数据格式转换

将 teacher 输出转为 SFT chat-message 格式（system / user / assistant）。

```bash
python prepare_data.py --inputs data/teacher_zh.jsonl --val-frac 0.05
```

输出：`data/sft_train.jsonl` + `data/sft_val.jsonl`

**关键设计决策：system prompt 的选择**

- **v1**（`sft_train.jsonl`）：system prompt 只有简短的格式指令，不含 SKILL.md 知识。期望模型把知识"背"进权重。
- **v2**（`sft_train_v2.jsonl`）：system prompt 包含 SKILL.md 精简版（心智模型、原型、启发式、反模式、表达DNA）。推理时也带上，模型只需学会"如何运用知识"。

**结论：v2 更合理。** 182 条数据不足以让 4B 模型内化全部领域知识，但足以教会它如何运用 system prompt 中的知识做分析。

### Step 3: LoRA SFT 训练

```bash
CUDA_VISIBLE_DEVICES=0 python train_sft.py \
    --base /path/to/Qwen3-4B-Thinking-2507 \
    --train data/sft_train_v2.jsonl --val data/sft_val_v2.jsonl \
    --run-name qwen3_4b_zh_skill_ep8_lr2e4 \
    --epochs 8 --batch-size 2 --grad-accum 4 --lr 2e-4 \
    --warmup-ratio 0.03 --max-seq 4096
```

关键参数：
- LoRA: r=16, alpha=32, target=all-linear, trainable params ~33M (0.81%)
- `assistant_only_loss=True`：只在 assistant 回复上计算 loss
- gradient checkpointing 开启，单卡 A100 80GB 可跑

### Step 4: 推理验证

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained(base_path, dtype=torch.bfloat16, device_map={"": 0})
model = PeftModel.from_pretrained(model, "outputs/<run_name>/adapter")

# 构造 messages（system prompt 用 v2 版本，包含 SKILL.md 精简知识）
# apply_chat_template → generate
```

## 已完成的实验

| Run | 数据 | Epochs | LR | 最终 Loss | 备注 |
|-----|------|--------|-----|-----------|------|
| `qwen3_4b_zh_full_ep3_lr1e4` | v1 (无知识) | 3 | 1e-4 | 1.55 | 收敛不充分，loss偏高 |
| `qwen3_4b_zh_skill_ep8_lr2e4` | v2 (含知识) | 8 | 2e-4 | TBD | 待完成 |

## 文件结构

```
training/
├── generate_teacher.py      # Step 1: Claude teacher 生成
├── run_generate_teacher.sh  # teacher 生成的启动脚本
├── prepare_data.py          # Step 2: teacher → SFT 格式
├── build_new_system_prompt.py  # 构建 v2 数据（含 SKILL.md 知识）
├── train_sft.py             # Step 3: LoRA SFT 训练
├── data/
│   ├── teacher_zh.jsonl     # teacher 原始输出
│   ├── sft_train.jsonl      # v1 训练集（简短 system prompt）
│   ├── sft_val.jsonl        # v1 验证集
│   ├── sft_train_v2.jsonl   # v2 训练集（含 SKILL.md 精简知识）
│   └── sft_val_v2.jsonl     # v2 验证集
├── outputs/
│   ├── qwen3_4b_zh_full_ep3_lr1e4/adapter/   # v1 实验
│   └── qwen3_4b_zh_skill_ep8_lr2e4/adapter/  # v2 实验（进行中）
└── logs/
```

## 待办

- [ ] 完成 v2 数据的 ep8 训练，对比 loss 和生成质量
- [ ] 如果 loss 降到 ~1.0 以下，做 `--save-merged` 导出完整模型
- [ ] 考虑增加训练数据量（当前仅 182 条）
