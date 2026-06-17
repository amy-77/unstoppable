"""Opus generator for exp1, two conditions (controlled: only the system prompt differs).

  - noskill : generic analyst system prompt, NO domain skill/heuristics/archetypes
  - skillv1 : current zh/SKILL.md + references injected as system prompt

Both produce the SAME output schema (psychology_and_traits + matchmaking_intelligence)
so the rendered candidate is apples-to-apples with the expert answer.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]          # new_version/
SKILL_DIR = _ROOT / "matchmaker" / "zh"
GEN_MODEL = os.environ.get("GEN_MODEL", "claude-opus-4-8")
MAX_TOKENS = 4000

SCHEMA_BLOCK = """```json
{
  "psychology_and_traits": {
    "personality_tags": ["3-6 个简短性格标签"],
    "behavioral_logic": "1-2 句核心行为逻辑"
  },
  "matchmaking_intelligence": {
    "core_conflict": "1-2 句结构性核心矛盾",
    "market_value_assessment": "按城市/收入/学历/外形/年龄五维展开的市场定位，2-4 句",
    "expert_strategy": "具体可执行的核心策略，2-4 句",
    "target_portrait": "理想匹配对象画像，具体到圈层/年龄/性格/家庭，2-4 句",
    "logic_chain": ["关键推导步骤1", "步骤2", "步骤3", "..."]
  }
}
```"""

USER_TEMPLATE = """请对以下相亲案例进行完整分析。

先输出一个 `<thinking>` 块（你的推理过程），然后输出一个严格遵守以下 schema 的 JSON（包在 ```json 代码块里）：

{schema}

## 案例信息
```json
{case_json}
```

只输出 `<thinking>` 块和 JSON 代码块，不要其它内容。"""

NOSKILL_SYSTEM = (
    "你是一位婚恋市场分析师。根据当事人的资料和择偶期望，"
    "对其婚恋市场定位做出结构化分析。"
)


def load_references_text() -> str:
    refs = []
    ref_dir = SKILL_DIR / "references"
    if ref_dir.exists():
        for f in sorted(ref_dir.glob("*.md")):
            refs.append(f"## {f.stem}\n\n{f.read_text(encoding='utf-8')}")
    return "\n\n---\n\n".join(refs)


def build_skill_system(skill_text: str) -> str:
    """System prompt = given SKILL.md text + the (fixed) references."""
    return skill_text + "\n\n---\n\n## 参考资料\n\n" + load_references_text()


def _load_skill_system() -> str:
    skill = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    return build_skill_system(skill)


def system_prompt_for(condition: str) -> str:
    if condition == "noskill":
        return NOSKILL_SYSTEM
    if condition == "skillv1":
        return _load_skill_system()
    raise ValueError(f"unknown condition: {condition}")


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_json(text: str) -> dict | None:
    blocks = _JSON_BLOCK_RE.findall(text)
    candidate = blocks[-1] if blocks else None
    if candidate is None:
        last = text.rfind("}")
        if last != -1:
            depth, start = 0, -1
            for i, ch in enumerate(text[: last + 1]):
                if ch == "{":
                    if depth == 0:
                        start = i
                    depth += 1
                elif ch == "}":
                    depth -= 1
            candidate = text[start: last + 1] if start != -1 else None
    if not candidate:
        return None
    for c in (candidate, re.sub(r",(\s*[}\]])", r"\1", candidate)):
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    return None


def _anthropic_client():
    import anthropic
    return anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"),
        timeout=600.0, max_retries=2,   # 600s: large skill-author outputs (~14K tok) legitimately take minutes; still bounded so a stalled connection can't hang forever
        **({"base_url": os.environ["ANTHROPIC_BASE_URL"]} if os.environ.get("ANTHROPIC_BASE_URL") else {}),
    )


def generate_with_system(case_input: dict, system: str, retries: int = 3) -> tuple[dict | None, str]:
    """Generate using an explicit system prompt. Returns (parsed_json, raw_text).

    The large system prompt is sent with cache_control so repeated calls within a
    round hit the prompt cache (~10x cheaper on the cached prefix)."""
    client = _anthropic_client()
    user = USER_TEMPLATE.format(schema=SCHEMA_BLOCK,
                                case_json=json.dumps(case_input, ensure_ascii=False, indent=2))
    system_blocks = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
    text = ""
    for attempt in range(1, retries + 1):
        try:
            msg = client.messages.create(
                model=GEN_MODEL, max_tokens=MAX_TOKENS, system=system_blocks,
                messages=[{"role": "user", "content": user}],
            )
            text = msg.content[0].text if msg.content else ""
            parsed = _parse_json(text)
            if parsed:
                return parsed, text
        except Exception as e:  # noqa: BLE001
            text = f"[ERROR] {type(e).__name__}: {str(e)[:120]}"
        time.sleep(3 * attempt)
    return None, text


def generate_with_skill(case_input: dict, skill_text: str, retries: int = 3) -> tuple[dict | None, str]:
    return generate_with_system(case_input, build_skill_system(skill_text), retries)


def generate(case_input: dict, condition: str, retries: int = 3) -> tuple[dict | None, str]:
    """Returns (parsed_json, raw_text). raw_text always preserved (incl <thinking>)."""
    client = _anthropic_client()
    system = system_prompt_for(condition)
    user = USER_TEMPLATE.format(schema=SCHEMA_BLOCK,
                                case_json=json.dumps(case_input, ensure_ascii=False, indent=2))
    text = ""
    for attempt in range(1, retries + 1):
        try:
            msg = client.messages.create(
                model=GEN_MODEL, max_tokens=MAX_TOKENS, system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = msg.content[0].text if msg.content else ""
            parsed = _parse_json(text)
            if parsed:
                return parsed, text
        except Exception as e:  # noqa: BLE001
            text = f"[ERROR] {type(e).__name__}: {str(e)[:120]}"
        time.sleep(3 * attempt)
    print(f"    [generate failed: {text[:80]}]")
    return None, text
