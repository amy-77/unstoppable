#!/usr/bin/env python3
"""
Evaluate: 用当前 SKILL.md 对 test 集生成分析文档，再用 LLM-as-Judge 语义评分。

Usage:
    python3 scripts/evaluate.py
    python3 scripts/evaluate.py --skill-path zh/SKILL.md --sample 10
"""

import argparse
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR.parent / "datasets"


def get_client():
    try:
        import anthropic
    except ImportError:
        print("Error: pip install anthropic")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: need ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN")
        sys.exit(1)

    kwargs = {"api_key": api_key}
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return anthropic.Anthropic(**kwargs)


def load_skill(skill_path: Path) -> str:
    return skill_path.read_text(encoding="utf-8")


def load_references(skill_dir: Path) -> str:
    refs = []
    ref_dir = skill_dir / "references"
    if ref_dir.exists():
        for f in sorted(ref_dir.glob("*.md")):
            refs.append(f"## {f.stem}\n\n{f.read_text(encoding='utf-8')[:3000]}")
    return "\n\n---\n\n".join(refs)


def load_test_data():
    inputs_path = DATASETS_DIR / "test" / "inputs.json"
    outputs_path = DATASETS_DIR / "test" / "outputs.json"
    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    outputs = json.loads(outputs_path.read_text(encoding="utf-8"))
    output_map = {item["case_id"]: item["target"] for item in outputs}
    return inputs, output_map


def generate_analysis(client, model: str, skill_text: str, refs_text: str, case_input: dict) -> str:
    system_prompt = f"""{skill_text}

---

## 参考资料

{refs_text}"""

    user_msg = f"""请对以下相亲案例进行完整分析。输出一份可读性强的分析文档，包含：
- 核心牌面（优势和劣势）
- 核心矛盾/冲突
- 市场定位
- 匹配策略建议
- 逻辑推导链

案例信息：
{json.dumps(case_input, ensure_ascii=False, indent=2)}"""

    msg = client.messages.create(
        model=model,
        max_tokens=2000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_msg}]
    )
    return msg.content[0].text


def extract_structured_fields(client, generated_doc: str) -> dict:
    """Extract structured fields from generated analysis for frontend rendering."""
    extract_model = os.environ.get("EXTRACT_MODEL") or os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    prompt = f"""From the following matchmaking analysis document, extract these structured fields. Output ONLY valid JSON, no markdown fences.

{{
  "psychology_and_traits": {{
    "personality_tags": ["list of 3-6 short personality trait tags"],
    "behavioral_logic": "one sentence summarizing their core behavioral pattern"
  }},
  "matchmaking_intelligence": {{
    "core_conflict": "the central contradiction/conflict identified",
    "market_value_assessment": "brief market positioning summary",
    "expert_strategy": "the recommended core strategy in one sentence",
    "target_portrait": "ideal match profile description",
    "logic_chain": ["step1", "step2", "step3", "...key reasoning steps"]
  }}
}}

Document:
{generated_doc}"""

    try:
        msg = client.messages.create(
            model=extract_model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        text = msg.content[0].text.strip()
        if text.startswith("```"):
            import re
            text = re.sub(r'```(?:json)?\s*\n?', '', text)
            text = re.sub(r'\n?```', '', text)
        return json.loads(text.strip())
    except Exception:
        return {}


def judge_analysis(client, model: str, generated_doc: str, target: dict, case_input: dict) -> dict:
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
    text = msg.content[0].text

    import re
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {"conflict_insight": 3, "strategy_direction": 3, "logic_depth": 3,
            "persona_read": 3, "actionability": 3, "comment": "parse failed"}


def evaluate_case(client, model, skill_text, refs_text, case, target):
    case_id = case["case_id"]
    case_input = case["input"]
    generated = generate_analysis(client, model, skill_text, refs_text, case_input)
    scores = judge_analysis(client, model, generated, target, case_input)
    extracted = extract_structured_fields(client, generated)
    return {
        "case_id": case_id,
        "generated_doc": generated,
        "scores": scores,
        "judge_comment": scores.get("comment", ""),
        "extracted_fields": extracted
    }


def run_evaluation(skill_path: Path, sample_size: int = None, parallel: int = 3) -> dict:
    client = get_client()
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    print(f"[Evaluate] Model: {model}")

    skill_text = load_skill(skill_path)
    refs_text = load_references(skill_path.parent)
    inputs, output_map = load_test_data()

    if sample_size:
        inputs = inputs[:sample_size]

    print(f"[Evaluate] {len(inputs)} cases to evaluate...")

    results = []
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {}
        for case in inputs:
            case_id = case["case_id"]
            target = output_map.get(case_id, {})
            if not target:
                continue
            f = executor.submit(evaluate_case, client, model, skill_text, refs_text, case, target)
            futures[f] = case_id

        for i, f in enumerate(as_completed(futures), 1):
            case_id = futures[f]
            try:
                result = f.result()
                results.append(result)
                avg = sum(result["scores"].get(d, 3) for d in
                          ["conflict_insight", "strategy_direction", "logic_depth",
                           "persona_read", "actionability"]) / 5
                print(f"  [{i}/{len(futures)}] {case_id}: {avg:.1f}")
            except Exception as e:
                print(f"  [{i}/{len(futures)}] {case_id}: ERROR - {e}")

    dimensions = ["conflict_insight", "strategy_direction", "logic_depth",
                  "persona_read", "actionability"]
    dim_scores = {}
    for d in dimensions:
        scores_list = [r["scores"].get(d, 3) for r in results]
        dim_scores[d] = round(sum(scores_list) / max(len(scores_list), 1), 2)

    overall = round(sum(dim_scores.values()) / len(dim_scores), 2)

    worst_cases = sorted(results, key=lambda r: sum(
        r["scores"].get(d, 3) for d in dimensions))[:5]

    report = {
        "overall_score": overall,
        "dimension_scores": dim_scores,
        "total_cases": len(results),
        "per_case_results": results,
        "worst_cases": [{"case_id": c["case_id"], "scores": c["scores"],
                         "comment": c["judge_comment"]} for c in worst_cases],
    }

    print(f"\n[Evaluate] Overall: {overall}")
    for d, s in dim_scores.items():
        print(f"  {d}: {s}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate skill quality against test set")
    parser.add_argument("--skill-path", default="zh/SKILL.md", help="Path to SKILL.md relative to matchmaker/")
    parser.add_argument("--sample", type=int, default=None, help="Only evaluate N cases")
    parser.add_argument("--parallel", type=int, default=3, help="Parallel workers")
    parser.add_argument("--output", default=None, help="Output report path")
    args = parser.parse_args()

    skill_path = BASE_DIR / args.skill_path
    if not skill_path.exists():
        print(f"Error: {skill_path} not found")
        sys.exit(1)

    report = run_evaluation(skill_path, args.sample, args.parallel)

    out_path = Path(args.output) if args.output else BASE_DIR / "scripts" / "iteration_history" / "eval_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[Evaluate] Report saved: {out_path}")


if __name__ == "__main__":
    main()
