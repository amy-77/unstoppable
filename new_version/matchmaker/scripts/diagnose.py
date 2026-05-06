#!/usr/bin/env python3
"""
Diagnose: 分析评估结果，识别系统性错误模式，输出改进建议。

Usage:
    python3 scripts/diagnose.py --eval-report scripts/iteration_history/eval_report.json
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def get_client():
    try:
        import anthropic
    except ImportError:
        print("Error: pip install anthropic")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: need ANTHROPIC_API_KEY")
        sys.exit(1)

    kwargs = {"api_key": api_key}
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return anthropic.Anthropic(**kwargs)


def load_skill(skill_path: Path) -> str:
    return skill_path.read_text(encoding="utf-8")


def load_references(skill_path: Path) -> str:
    """Load references directory content summary"""
    ref_dir = skill_path.parent / "references"
    if not ref_dir.exists():
        return "(references 目录不存在)"

    parts = []
    for f in sorted(ref_dir.glob("*.md")):
        content = f.read_text(encoding="utf-8")
        parts.append(f"### {f.name} ({len(content)}字)\n{content[:800]}")
    return "\n\n".join(parts) if parts else "(无 reference 文件)"


def load_prompts_context(skill_path: Path) -> str:
    """Load distill pipeline prompts for diagnosis context"""
    # Load from i18n.py
    lang = skill_path.parent.name  # "zh" or "en"
    i18n_path = BASE_DIR / "scripts" / "i18n.py"
    if not i18n_path.exists():
        return "(i18n.py 不存在)"

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("i18n", i18n_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cfg = mod.I18N.get(lang, {})
        parts = []
        if "prompt_1" in cfg:
            parts.append(f"**prompt_1** (提炼心智模型/启发式):\n{cfg['prompt_1'][:600]}")
        if "prompt_2" in cfg:
            parts.append(f"**prompt_2** (提炼原型/反模式):\n{cfg['prompt_2'][:600]}")
        if "digest_prompt" in cfg:
            parts.append(f"**digest_prompt** (案例脱敏摘要):\n{cfg['digest_prompt'][:400]}")
        return "\n\n".join(parts) if parts else "(未找到 prompt 配置)"
    except Exception as e:
        return f"(加载 prompts 失败: {e})"


def run_diagnosis(eval_report: dict, skill_path: Path) -> dict:
    client = get_client()
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    print(f"[Diagnose] Model: {model}")

    skill_text = load_skill(skill_path)
    references_text = load_references(skill_path)
    prompts_text = load_prompts_context(skill_path)

    worst_cases = eval_report.get("worst_cases", [])
    dim_scores = eval_report.get("dimension_scores", {})
    overall = eval_report.get("overall_score", 0)

    all_comments = [r.get("judge_comment", "") for r in eval_report.get("per_case_results", []) if r.get("judge_comment")]

    weakest_dim = min(dim_scores, key=dim_scores.get) if dim_scores else "unknown"

    worst_details = []
    for case in eval_report.get("per_case_results", []):
        case_avg = sum(case["scores"].get(d, 3) for d in dim_scores.keys()) / max(len(dim_scores), 1)
        if case_avg < overall - 0.5:
            worst_details.append({
                "case_id": case["case_id"],
                "scores": case["scores"],
                "comment": case["judge_comment"],
                "generated_excerpt": case.get("generated_doc", "")[:500]
            })

    prompt = f"""你是一个 Skill 质量诊断专家。根据以下评估数据，分析当前系统的问题并给出改进建议。

改进建议必须覆盖三个目标：
- **skill_md**: SKILL.md 本身（心智模型、启发式、原型、反模式、表达DNA）
- **references**: references/ 目录下的参考文件（pattern_library.md, case_digest.md, statistics_snapshot.md, case_archive.md）
- **prompts**: distill_pipeline 中的 prompt 模板（影响分析生成质量的源头）

## 当前评估结果
- 总分: {overall}/5.0
- 各维度: {json.dumps(dim_scores, ensure_ascii=False)}
- 最弱维度: {weakest_dim}
- 评估案例数: {eval_report.get('total_cases', 0)}

## 评委反馈汇总（所有 case 的评语）
{chr(10).join(f'- {c}' for c in all_comments[:30])}

## 表现最差的案例
{json.dumps(worst_details[:8], ensure_ascii=False, indent=2)[:4000]}

## 当前 SKILL.md 结构（前3000字）
{skill_text[:3000]}

## 当前 References 内容摘要
{references_text[:2000]}

## 当前 Distill Prompts
{prompts_text[:1500]}

## 请输出诊断结果（严格 JSON）

分析以下维度的问题：
1. mental_models: 哪些心智模型解释力不足或缺失
2. heuristics: 哪些启发式规则需要补充或修正
3. archetypes: 哪些人物原型覆盖不足
4. anti_patterns: 哪些反模式需要新增
5. expression_dna: 表达风格是否导致分析偏差
6. references: pattern_library/case_digest 是否缺少关键模式或案例覆盖
7. prompts: distill prompt 是否引导不足导致生成质量瓶颈

**重要**：improvements 必须分布到所有三个 target。具体分配原则：
- `skill_md`: 心智模型/启发式/原型/反模式/表达DNA 的增删改
- `references`: pattern_library 缺少的冲突模式、case_digest 缺少的案例摘要、statistics_snapshot 需要更新的统计数据
- `prompts`: 当生成分析的质量瓶颈来自 prompt 引导不足时（如缺少某类分析视角的指令、输出格式约束不够等）

每个 target 至少输出 1-2 条改进建议。

```json
{{
  "weakest_dimension": "<最需要改进的评分维度>",
  "root_causes": ["<根因1>", "<根因2>", ...],
  "improvements": [
    {{
      "target": "<skill_md|references|prompts>",
      "section": "<具体要改的部分，如 mental_models / pattern_library / prompt_1 / case_digest 等>",
      "action": "<add|modify|remove>",
      "priority": <1-3>,
      "description": "<具体改什么>",
      "rationale": "<为什么这样改，引用具体案例>"
    }}
  ],
  "missing_archetypes": ["<test中出现但skill未覆盖的类型>"],
  "question_priorities": ["<对准确性影响最大的信息维度，按重要性排序>"]
}}
```"""

    msg = client.messages.create(
        model=model,
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )
    text = msg.content[0].text

    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        try:
            diagnosis = json.loads(m.group(0))
            imps = diagnosis.get("improvements", [])
            from collections import Counter
            target_dist = Counter(i.get("target", "?") for i in imps)
            print(f"[Diagnose] Found {len(imps)} improvements: {dict(target_dist)}")
            print(f"[Diagnose] Root causes: {diagnosis.get('root_causes', [])}")
            return diagnosis
        except json.JSONDecodeError:
            pass

    print("[Diagnose] Warning: failed to parse LLM response, returning raw")
    return {"raw_response": text, "improvements": [], "root_causes": ["parse_failed"]}


def main():
    parser = argparse.ArgumentParser(description="Diagnose skill weaknesses from eval report")
    parser.add_argument("--eval-report", required=True, help="Path to eval_report.json")
    parser.add_argument("--skill-path", default="zh/SKILL.md", help="Path to SKILL.md")
    parser.add_argument("--output", default=None, help="Output diagnosis path")
    args = parser.parse_args()

    report_path = Path(args.eval_report)
    if not report_path.exists():
        print(f"Error: {report_path} not found")
        sys.exit(1)

    eval_report = json.loads(report_path.read_text(encoding="utf-8"))
    skill_path = BASE_DIR / args.skill_path

    diagnosis = run_diagnosis(eval_report, skill_path)

    out_path = Path(args.output) if args.output else report_path.parent / "diagnosis.json"
    out_path.write_text(json.dumps(diagnosis, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Diagnose] Saved: {out_path}")


if __name__ == "__main__":
    main()
