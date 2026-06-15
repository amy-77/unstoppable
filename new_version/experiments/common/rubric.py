"""9-dimension matchmaking-analysis judging rubric (shared by exp1/2/3).

Two groups:
  A  Accuracy  — scored against the expert reference (recall of expert's points;
                 do NOT penalize correct extra depth, since expert answers are terse).
  B  Quality   — intrinsic quality of the analysis itself (may exceed the expert);
                 reward substance, NOT length (anti-verbosity).
"""

from __future__ import annotations

import json
import re

# Dimension groups -----------------------------------------------------------
DIMS_A = [
    "conflict_insight",
    "market_read_accuracy",
    "strategy_direction",
    "target_portrait_match",
    "persona_read",
]
DIMS_B = [
    "logic_depth",
    "insight_nonobviousness",
    "risk_anti_pattern",
    "actionability",
]
DIMS = DIMS_A + DIMS_B

# Judge prompt ---------------------------------------------------------------
RUBRIC_PROMPT = """你是资深婚恋市场分析质量评审。下面给你一个相亲案例的【输入资料】、一份【专家参考答案】、以及一份【待评分析】。请按 9 个维度，每维 1-5 分，严格打分。

## 评分总则（务必遵守）
- 5 分极稀缺：只给几乎无可挑剔的表现。泛泛而谈、放之四海皆准的话最多 3 分。
- A 类（准确性）= 看【待评分析】是否命中【专家参考】的关键判断点。命中得分；若待评分析补充了**正确的额外深度**，不要因此扣分（专家答案本身很简短）。
- B 类（分析质量）= 看分析**本身**的质量，不要求对标专家，允许且鼓励超过专家。奖励**实质洞察与深度**，**不奖励篇幅长短**；啰嗦冗长要扣分。

## 维度（每维 1-5）
A 准确性：
1. conflict_insight：是否识别出与专家等价的核心矛盾及其结构性根源。5=精确命中且点出同样结构性根源；3=只抓到次要/泛泛矛盾；1=没识别。
2. market_read_accuracy：市场估值是否(a)扎实落在输入事实上（收入/学历/外形/年龄/城市），(b)方向与专家总结一致。5=两者都满足且精准；3=部分对或脱离事实；1=明显错判。
3. strategy_direction：策略方向是否与专家建议一致（措辞不同没关系）。5=核心方向一致；3=沾边但偏；1=相反/无关。
4. target_portrait_match：理想对象画像是否与专家一致（圈层/年龄/性格/家庭）。5=高度一致且具体；3=笼统/部分；1=偏离。
5. persona_read：性格与行为逻辑解读是否到位。5=准确且抓到深层动机；3=只有表面标签；1=误读。

B 质量：
6. logic_depth：推理链是否有深度且自洽（诊断→策略推得出来）。5=层层递进、因果扎实；3=有逻辑但浅；1=跳跃/自相矛盾。
7. insight_nonobviousness：是否给出反直觉的非显然洞察（参考专家 logic_chain 那种密度）。5=有≥1条真正非显然且成立的洞察；3=基本是常识；1=空洞。
8. risk_anti_pattern：是否识别关键风险/陷阱（被收割、面子驱动、期望错配等）。5=精准点出主要风险；3=泛泛提风险；1=没提。
9. actionability：建议是否具体可落地（非空话），且不啰嗦。5=具体到可执行步骤且简洁；3=方向有但不具体；1=空泛口号。

## 输入资料
{case_input}

## 专家参考答案
{expert_reference}

## 待评分析
{candidate}

## 输出（严格 JSON，一行，不要 markdown 代码块）
{{"conflict_insight":<1-5>,"market_read_accuracy":<1-5>,"strategy_direction":<1-5>,"target_portrait_match":<1-5>,"persona_read":<1-5>,"logic_depth":<1-5>,"insight_nonobviousness":<1-5>,"risk_anti_pattern":<1-5>,"actionability":<1-5>,"comment":"<一句话指出主要差距>"}}"""


def render_target_as_analysis(target: dict) -> str:
    """Render an expert/structured target into a readable analysis document."""
    pt = target.get("psychology_and_traits", {})
    mi = target.get("matchmaking_intelligence", {})
    lines = ["## 性格与心理"]
    lines.append(f"- 性格标签：{('、'.join(pt.get('personality_tags', [])) or 'N/A')}")
    lines.append(f"- 行为逻辑：{pt.get('behavioral_logic', 'N/A')}")
    lines.append("\n## 婚恋情报")
    lines.append(f"- 核心矛盾：{mi.get('core_conflict', 'N/A')}")
    lines.append(f"- 市场估值：{mi.get('market_value_assessment', 'N/A')}")
    lines.append(f"- 专家策略：{mi.get('expert_strategy', 'N/A')}")
    lines.append(f"- 理想对象：{mi.get('target_portrait', 'N/A')}")
    chain = mi.get("logic_chain", [])
    if chain:
        lines.append("- 逻辑链：")
        for i, step in enumerate(chain, 1):
            lines.append(f"  {i}. {step}")
    return "\n".join(lines)


def build_prompt(case_input: dict, expert_target: dict, candidate_doc: str) -> str:
    return RUBRIC_PROMPT.format(
        case_input=json.dumps(case_input, ensure_ascii=False, indent=2),
        expert_reference=render_target_as_analysis(expert_target),
        candidate=candidate_doc,
    )


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_scores(text: str) -> dict | None:
    """Pull the JSON score object out of a judge's raw output."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    m = _JSON_RE.search(t)
    if not m:
        return None
    for candidate in (m.group(0), re.sub(r",(\s*[}\]])", r"\1", m.group(0))):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        # keep only valid integer-ish dim scores
        out = {}
        for d in DIMS:
            v = obj.get(d)
            if isinstance(v, (int, float)) and 1 <= v <= 5:
                out[d] = float(v)
        if len(out) == len(DIMS):
            out["comment"] = str(obj.get("comment", ""))[:300]
            return out
    return None
