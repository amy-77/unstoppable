#!/usr/bin/env python3
"""Generate teacher signal for distillation.

Given a SKILL.md + references and the train cases, run Claude (teacher) to
produce, for each case, a structured response containing:

  1) <thinking>...</thinking>      reasoning trace, the "skill" being applied
  2) {json}                         final analysis (psychology_and_traits +
                                    matchmaking_intelligence)

Both pieces are kept together; the student model will be trained to imitate
this combined output (CoT-style sequence-level distillation).

Resumable: writes one record per case to a JSONL file, skipping cases that
already have a parsed JSON output.

Usage:
    export ANTHROPIC_BASE_URL=...
    export ANTHROPIC_AUTH_TOKEN=...
    export ANTHROPIC_MODEL=...
    python generate_teacher.py --lang zh --parallel 8 --limit 5    # smoke test
    python generate_teacher.py --lang zh --parallel 8              # full run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # agent/matchmaker
DATASETS_DIR = BASE_DIR.parent / "datasets"        # agent/datasets

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

TEACHER_USER_TEMPLATE_ZH = """请按照本 Skill 的分析框架，对以下相亲案例进行完整分析。

## 重要约束（必须遵守）

⚠️ **禁止使用编号引用**：不要写「触发模型 11」「应用启发式 6」「触发反模式 3」等编号。
   直接说出该模型/启发式/反模式的**核心规则名称和内容**。
   例如不要写「应用启发式 6: 短板即护城河法则」，而要直接论述「在跨阶层匹配中，
   对方明显的可接受短板（矮、丑、话少）反而是关系稳定的安全保障，因为短板降低了对方的外部选择权」。
   这样做的原因：student 模型推理时不会带 SKILL.md 在上下文里，编号引用对 student 是无意义的。

✅ **鼓励引用具体历史案例**：在推理中可以引用 case_digest / case_archive 中的具体 case_id（如 CN_007、TS_005 等），
   作为"案例锚点"。例如：「类似 CN_007 红娘所提出的反直觉定律——跨级相亲中男方的明显短板反而是女方安全的保障」。

## 输出要求（极其严格）

第一段必须是 `<thinking>` 块，包含 5 步推理（每步 4-8 句话，详细展开）：
1. **画像定位**：这个 case 最像 SKILL.md 中的哪一个或几个典型原型？为什么？这个原型的典型困境是什么？
2. **市场估值**：从城市/收入/学历/外形/年龄五维度逐一展开，每张牌具体硬在哪里、软在哪里。
3. **核心冲突识别**：用具体的心智模型（按规则名称而非编号）解释结构性矛盾。冲突是怎么产生的？
4. **启发式 / 反模式 / 历史案例锚定**：列出 2-4 条最相关的决策启发式或反模式（按规则内容而非编号），
   并尽量引用 references 里的具体 case_id 作为锚点（比如「与 CN_007 相似」）。
5. **策略综合**：把上述结论合成成具体可执行的最终策略，包括圈层定位、优势放大、期望校准、反模式规避等。

第二段必须是一个 JSON 对象（包在 ```json 代码块里），严格遵守以下 schema：

```json
{{
  "psychology_and_traits": {{
    "personality_tags": ["3-6 个简短性格标签"],
    "behavioral_logic": "用 1-2 句总结其核心行为逻辑"
  }},
  "matchmaking_intelligence": {{
    "core_conflict": "用 1-2 句说明结构性核心矛盾（不要使用编号引用）",
    "market_value_assessment": "市场定位的具体说明（按五维度展开 2-4 句）",
    "expert_strategy": "推荐的核心策略，要具体可执行（2-4 句）",
    "target_portrait": "理想匹配对象的画像描述（具体到圈层、年龄段、性格倾向、家庭背景，2-4 句）",
    "logic_chain": ["关键推导步骤 1（一句话）", "步骤 2", "步骤 3", "..."]
  }}
}}
```

logic_chain 步数建议 5-8 步，每步要内容实质，不要凑数。

## 案例信息

```json
{case_json}
```

请严格按"先 `<thinking>` 块，再 JSON 代码块"的顺序输出，不要有任何其他前后文。"""


TEACHER_USER_TEMPLATE_EN = """Analyze the following matchmaking case using this Skill's framework.

## IMPORTANT CONSTRAINTS

⚠️ **No numbered references**: Do NOT write "Triggers Model 11", "Applies Heuristic 6",
   "Anti-pattern 3", etc. Instead, state the rule's **name and content** directly.
   E.g., do not write "Apply Heuristic 6: Weakness as Moat", but instead
   "In cross-class matching, the other party's obvious acceptable weakness
   (short, plain, quiet) is actually a stability safeguard for the relationship,
   because the weakness reduces their external choice power."
   Reason: the student model will not have SKILL.md in context at inference,
   so numbered references are meaningless to it.

✅ **Encouraged: reference specific historical cases**: In reasoning you may cite
   specific case_ids from references (e.g., CN_007, TS_005) as case anchors.
   E.g., "Similar to the counterintuitive principle from CN_007 — in cross-class
   matchmaking the man's obvious weakness is actually the woman's safety guarantee."

## Output Format (strict)

First, a `<thinking>` block with a 5-step reasoning trace (each step 4-8 sentences, fully expanded):
1. **Archetype mapping**: which archetype(s) from SKILL.md best fits this case? Why? What's their typical dilemma?
2. **Market valuation**: walk through city / income / education / appearance / age one by one;
   spell out exactly where each card is strong or weak.
3. **Core conflict identification**: explain the structural contradiction using the specific
   mental model (by rule name, not by number). How does the conflict arise?
4. **Heuristics / anti-patterns / historical anchors**: list 2-4 most relevant decision heuristics
   or anti-patterns (by content, not number), and try to cite specific case_ids from references
   as anchors when relevant.
5. **Strategy synthesis**: compose a concrete actionable final strategy from the above,
   including circle positioning, strength amplification, expectation calibration, anti-pattern avoidance.

Second, a JSON object (in a ```json code block) strictly conforming to:

```json
{{
  "psychology_and_traits": {{
    "personality_tags": ["3-6 short personality tags"],
    "behavioral_logic": "1-2 sentences summarizing core behavioral pattern"
  }},
  "matchmaking_intelligence": {{
    "core_conflict": "1-2 sentences on the structural core conflict (no numbered references)",
    "market_value_assessment": "concrete market positioning across the five dimensions, 2-4 sentences",
    "expert_strategy": "concrete actionable strategy (2-4 sentences)",
    "target_portrait": "ideal match profile, specific to circle, age range, personality, family background (2-4 sentences)",
    "logic_chain": ["one-sentence reasoning step", "step2", "step3", "..."]
  }}
}}
```

logic_chain length suggested 5-8 steps, each substantive (no padding).

## Case

```json
{case_json}
```

Output ONLY the `<thinking>` block followed by the JSON code block, nothing else."""


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def get_client():
    import anthropic

    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: need ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY in env", file=sys.stderr)
        sys.exit(1)
    kwargs = {"api_key": api_key}
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return anthropic.Anthropic(**kwargs)


def load_skill(skill_path: Path) -> str:
    return skill_path.read_text(encoding="utf-8")


def load_references(skill_dir: Path, max_per_file: int = 200000) -> str:
    """Load all reference files. Default no real truncation (200K chars cap is
    only a safety guard against runaway files).
    """
    refs = []
    ref_dir = skill_dir / "references"
    if ref_dir.exists():
        for f in sorted(ref_dir.glob("*.md")):
            text = f.read_text(encoding="utf-8")
            if len(text) > max_per_file:
                text = text[:max_per_file] + "\n... (truncated)"
            refs.append(f"## {f.stem}\n\n{text}")
    return "\n\n---\n\n".join(refs)


def load_train_cases(lang: str) -> list[dict]:
    if lang == "zh":
        path = DATASETS_DIR / "train" / "inputs_outputs.json"
    elif lang == "en":
        path = DATASETS_DIR / "train" / "inputs_outputs_en.json"
    else:
        raise ValueError(f"unsupported lang: {lang}")
    return json.loads(path.read_text(encoding="utf-8"))


def case_to_input_only(case: dict) -> dict:
    """Strip the case down to fields the model is allowed to see."""
    return {
        "subject_profile": case.get("subject_profile", {}),
        "expectations": case.get("expectations", {}),
    }


def case_to_target(case: dict) -> dict:
    return {
        "psychology_and_traits": case.get("psychology_and_traits", {}),
        "matchmaking_intelligence": case.get("matchmaking_intelligence", {}),
    }


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

THINKING_RE = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL)
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def parse_teacher_output(text: str) -> tuple[str | None, dict | None]:
    """Return (thinking, parsed_json). Both can be None on failure."""
    thinking = None
    m = THINKING_RE.search(text)
    if m:
        thinking = m.group(1).strip()

    parsed = None
    blocks = JSON_BLOCK_RE.findall(text)
    if blocks:
        candidate = blocks[-1].strip()
    else:
        last_brace = text.rfind("}")
        if last_brace == -1:
            candidate = None
        else:
            depth = 0
            start = -1
            for i, ch in enumerate(text[: last_brace + 1]):
                if ch == "{":
                    if depth == 0:
                        start = i
                    depth += 1
                elif ch == "}":
                    depth -= 1
            candidate = text[start : last_brace + 1] if start != -1 else None

    if candidate:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                cleaned = re.sub(r",(\s*[}\]])", r"\1", candidate)
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                parsed = None

    return thinking, parsed


# ---------------------------------------------------------------------------
# Teacher call
# ---------------------------------------------------------------------------


def call_teacher(client, model: str, system_prompt: str, user_msg: str,
                 max_tokens: int = 10000, retries: int = 3,
                 backoff: float = 4.0) -> tuple[str, dict]:
    """Returns (raw_text, usage_dict)."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = msg.content[0].text if msg.content else ""
            usage = {
                "input_tokens": getattr(msg.usage, "input_tokens", 0) if msg.usage else 0,
                "output_tokens": getattr(msg.usage, "output_tokens", 0) if msg.usage else 0,
            }
            return text, usage
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < retries:
                wait = backoff * (2 ** (attempt - 1))
                time.sleep(wait)
    raise RuntimeError(f"call_teacher failed after {retries} retries: {last_err}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def get_case_id(case: dict, idx: int) -> str:
    return case.get("metadata", {}).get("case_id") or f"idx_{idx:04d}"


def already_done_ids(out_path: Path) -> set[str]:
    done = set()
    if not out_path.exists():
        return done
    for line in out_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("teacher_json"):
            done.add(r["case_id"])
    return done


def process_one(client, model: str, system_prompt: str, user_template: str,
                case: dict, idx: int, max_tokens: int) -> dict:
    case_id = get_case_id(case, idx)
    case_input = case_to_input_only(case)
    target = case_to_target(case)

    user_msg = user_template.format(
        case_json=json.dumps(case_input, ensure_ascii=False, indent=2)
    )
    t0 = time.time()
    raw_text, usage = call_teacher(
        client, model, system_prompt, user_msg, max_tokens=max_tokens
    )
    elapsed = time.time() - t0

    thinking, teacher_json = parse_teacher_output(raw_text)

    return {
        "case_id": case_id,
        "case_input": case_input,
        "expert_target": target,
        "teacher_raw": raw_text,
        "teacher_thinking": thinking,
        "teacher_json": teacher_json,
        "parse_ok": teacher_json is not None,
        "usage": usage,
        "elapsed_s": round(elapsed, 2),
        "model": model,
    }


def append_jsonl(path: Path, record: dict, lock: threading.Lock):
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lang", default="zh", choices=["zh", "en"], help="data language")
    p.add_argument("--parallel", type=int, default=8, help="concurrent API workers")
    p.add_argument("--limit", type=int, default=None,
                   help="limit number of cases (for smoke testing)")
    p.add_argument("--max-tokens", type=int, default=10000,
                   help="max output tokens per teacher call")
    p.add_argument("--out", type=str, default=None,
                   help="output JSONL path (default: training/data/teacher_<lang>.jsonl)")
    p.add_argument("--no-resume", action="store_true",
                   help="overwrite output file even if cases are already done")
    args = p.parse_args()

    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    client = get_client()

    skill_path = BASE_DIR / args.lang / "SKILL.md"
    if not skill_path.exists():
        print(f"Error: {skill_path} not found", file=sys.stderr)
        sys.exit(1)

    skill_text = load_skill(skill_path)
    refs_text = load_references(skill_path.parent)
    system_prompt = f"{skill_text}\n\n---\n\n## 参考资料\n\n{refs_text}"

    user_template = TEACHER_USER_TEMPLATE_ZH if args.lang == "zh" else TEACHER_USER_TEMPLATE_EN

    cases = load_train_cases(args.lang)
    if args.limit:
        cases = cases[: args.limit]

    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / f"teacher_{args.lang}.jsonl"

    if args.no_resume and out_path.exists():
        out_path.unlink()
    done_ids = already_done_ids(out_path)

    todo = []
    for idx, case in enumerate(cases):
        cid = get_case_id(case, idx)
        if cid in done_ids:
            continue
        todo.append((idx, case))

    print(f"=" * 70)
    print(f"Generate teacher signal | model={model} lang={args.lang}")
    print(f"  cases total:     {len(cases)}")
    print(f"  already done:    {len(done_ids)}")
    print(f"  to process:      {len(todo)}")
    print(f"  parallel:        {args.parallel}")
    print(f"  out file:        {out_path}")
    print(f"  system prompt:   {len(system_prompt)} chars (~{len(system_prompt)//3} tokens)")
    print(f"=" * 70)
    if not todo:
        print("Nothing to do; everything already cached. Exiting.")
        return

    lock = threading.Lock()
    parse_ok = 0
    parse_fail = 0
    total_in = 0
    total_out = 0
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = {
            ex.submit(process_one, client, model, system_prompt, user_template,
                      case, idx, args.max_tokens): (idx, case)
            for idx, case in todo
        }

        for i, fut in enumerate(as_completed(futures), 1):
            idx, case = futures[fut]
            cid = get_case_id(case, idx)
            try:
                rec = fut.result()
                append_jsonl(out_path, rec, lock)
                if rec["parse_ok"]:
                    parse_ok += 1
                else:
                    parse_fail += 1
                total_in += rec["usage"].get("input_tokens", 0)
                total_out += rec["usage"].get("output_tokens", 0)
                elapsed_total = time.time() - t_start
                rate = i / max(elapsed_total, 0.001)
                eta = (len(todo) - i) / max(rate, 0.001)
                status = "OK" if rec["parse_ok"] else "PARSE_FAIL"
                print(f"  [{i}/{len(todo)}] {cid} {status} "
                      f"in={rec['usage']['input_tokens']} out={rec['usage']['output_tokens']} "
                      f"t={rec['elapsed_s']}s | "
                      f"rate={rate:.2f}/s eta={eta:.0f}s")
            except Exception as e:  # noqa: BLE001
                print(f"  [{i}/{len(todo)}] {cid} FAILED: {e}", file=sys.stderr)
                fail_rec = {
                    "case_id": cid,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
                append_jsonl(out_path, fail_rec, lock)
                parse_fail += 1

    total_elapsed = time.time() - t_start
    print()
    print("=" * 70)
    print(f"Done in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  parse OK:   {parse_ok}")
    print(f"  parse FAIL: {parse_fail}")
    print(f"  tokens:     in={total_in:,}  out={total_out:,}")
    print(f"  output:     {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
