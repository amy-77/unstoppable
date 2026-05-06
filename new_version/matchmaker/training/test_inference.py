#!/usr/bin/env python3
"""
Test inference: load trained Qwen3-4B LoRA model, run on a test case,
then use LLM-as-Judge (Anthropic API) to score against target_portrait.
"""
import json
import os
import sys
import re
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Paths
BASE_MODEL = "/data/qwang/q/thalia/kvcache/ANNCache/models/Qwen3-4B-Thinking-2507"
ADAPTER_PATH = "outputs/qwen3_4b_zh_skill_ep8_lr2e4/adapter"
TEST_INPUTS = "/data/qwang/q/thalia/agentic_hackson/agent/datasets/test/inputs.json"
TEST_OUTPUTS = "/data/qwang/q/thalia/agentic_hackson/agent/datasets/test/outputs.json"
TRAIN_DATA = "data/sft_train_v2.jsonl"
LOG_FILE = "test_inference_result.txt"

CASE_IDX = 0  # test case index


def get_system_prompt(train_path):
    with open(train_path) as f:
        first_line = json.loads(f.readline())
    for msg in first_line['messages']:
        if msg['role'] == 'system':
            return msg['content']
    return ""


def load_model():
    print("[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    print("[2/4] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    print("[3/4] Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH, device_map="cuda:0")
    print("[3.5/4] Merging adapter...")
    model = model.merge_and_unload()
    model.eval()
    print("[4/4] Model ready.")
    return model, tokenizer


def run_inference(model, tokenizer, system_prompt, user_input):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_input, ensure_ascii=False)},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    generated = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    return response


def judge_with_anthropic(generated_doc: str, target: dict, case_input: dict) -> str:
    """Use Anthropic API as judge, same logic as scripts/evaluate.py"""
    try:
        import anthropic
    except ImportError:
        return "Error: pip install anthropic"

    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "Error: need ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN"

    kwargs = {"api_key": api_key}
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    client = anthropic.Anthropic(**kwargs)

    model = os.environ.get("JUDGE_MODEL") or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    prompt = f"""你是一个婚恋分析质量评估专家。请对比"生成的分析"和"参考答案"，从语义层面评分。
不要求措辞一致，只要核心判断方向一致即可得高分。

## 评分维度（每项 1-5 分）

1. conflict_insight: 是否识别出与参考答案相同/等价的核心冲突
2. strategy_direction: 策略方向是否与专家建议一致（措辞不同没关系）
3. logic_depth: 分析逻辑链是否有深度、自洽
4. persona_read: 对人物心理/性格的解读是否到位
5. actionability: 建议是否具体可执行（而非泛泛而谈）

## 案例原始信息
{json.dumps(case_input, ensure_ascii=False, indent=2)[:1500]}

## 参考答案
- 核心冲突: {target.get('matchmaking_intelligence', {}).get('core_conflict', 'N/A')}
- 市场评估: {target.get('matchmaking_intelligence', {}).get('market_value_assessment', 'N/A')}
- 专家策略: {target.get('matchmaking_intelligence', {}).get('expert_strategy', 'N/A')}
- 目标画像(target_portrait): {target.get('matchmaking_intelligence', {}).get('target_portrait', 'N/A')}
- 逻辑链: {json.dumps(target.get('matchmaking_intelligence', {}).get('logic_chain', []), ensure_ascii=False)}
- 性格标签: {json.dumps(target.get('psychology_and_traits', {}).get('personality_tags', []), ensure_ascii=False)}

## 生成的分析
{generated_doc}

## 输出格式（严格 JSON）
```json
{{
  "conflict_insight": <1-5>,
  "strategy_direction": <1-5>,
  "logic_depth": <1-5>,
  "persona_read": <1-5>,
  "actionability": <1-5>,
  "comment": "<一句话总结主要差距>"
}}
```"""

    msg = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = msg.content[0].text.strip()

    # Parse scores
    try:
        json_match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
        if json_match:
            scores = json.loads(json_match.group())
            dims = ["conflict_insight", "strategy_direction", "logic_depth", "persona_read", "actionability"]
            total = sum(scores.get(d, 0) for d in dims)
            avg = total / len(dims)
            result = f"各维度评分:\n"
            for d in dims:
                result += f"  {d}: {scores.get(d, '?')}/5\n"
            result += f"\n综合均分: {avg:.1f}/5\n"
            result += f"评价: {scores.get('comment', 'N/A')}"
            return result
    except Exception:
        pass
    return raw


def main():
    with open(TEST_INPUTS) as f:
        inputs = json.load(f)
    with open(TEST_OUTPUTS) as f:
        outputs = json.load(f)

    case_input = inputs[CASE_IDX]
    case_target = outputs[CASE_IDX]
    print(f"Case: {case_input['case_id']}")

    system_prompt = get_system_prompt(TRAIN_DATA)
    model, tokenizer = load_model()

    print("Running inference...")
    model_output = run_inference(model, tokenizer, system_prompt, case_input['input'])

    print("Running LLM-as-Judge (Anthropic API)...")
    judge_result = judge_with_anthropic(model_output, case_target['target'], case_input['input'])

    # Build log
    target_portrait = case_target['target']['matchmaking_intelligence']['target_portrait']
    log_content = f"""{'='*80}
案例ID: {case_input['case_id']}
{'='*80}

{'─'*40}
【INPUT】
{'─'*40}
{json.dumps(case_input['input'], ensure_ascii=False, indent=2)}

{'─'*40}
【MODEL OUTPUT】
{'─'*40}
{model_output}

{'─'*40}
【标准答案 - target_portrait】
{'─'*40}
{target_portrait}

{'─'*40}
【LLM-as-Judge 评分】
{'─'*40}
{judge_result}

{'='*80}
"""

    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(log_content)

    print(f"\n{'─'*40}")
    print(f"Results saved to: {LOG_FILE}")
    print(f"{'─'*40}")
    print(f"\n【Model Output (preview)】")
    print(model_output[:600])
    print(f"\n【Judge Result】")
    print(judge_result)


if __name__ == "__main__":
    main()
