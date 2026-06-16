#!/usr/bin/env python3
"""Exp2 — teacher-data generation for the skill-vs-no-skill distillation study.

Generates two SFT teacher datasets over the SAME 191 train cases with the SAME
Opus teacher; the ONLY difference is whether the teacher sees the v2 skill:

  --arm skill     system prompt = v2 best SKILL.md (skill-only, NO references,
                  per exp1 inference discipline)
  --arm noskill   system prompt = output-format-only persona (no skill, no refs)

Everything else is identical (model, cases, output schema, parsing). The two
user templates share the same output format / JSON schema and differ ONLY in
the skill-specific framing (step 1 archetype mapping + step 4 heuristics), so
the controlled variable is strictly "did the teacher use the skill".

Output is the combined teacher signal per case:
  1) <thinking>...</thinking>   reasoning trace
  2) {json}                     final analysis (psychology_and_traits +
                                matchmaking_intelligence)

Resumable (one JSONL record per case; cases with a parsed json are skipped),
concurrent, every raw LLM output recorded verbatim, long API timeout + retries.

Usage (run from new_version/, .env at repo root is auto-loaded):
    python experiments/exp2_skill_vs_noskill_distill/gen_teacher.py --arm skill   --limit 2   # smoke
    python experiments/exp2_skill_vs_noskill_distill/gen_teacher.py --arm noskill --limit 2   # smoke
    python experiments/exp2_skill_vs_noskill_distill/gen_teacher.py --arm skill                # full 191
    python experiments/exp2_skill_vs_noskill_distill/gen_teacher.py --arm noskill              # full 191
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

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
SCRIPT_DIR = Path(__file__).resolve().parent
NEW_VERSION = SCRIPT_DIR.parents[1]                 # new_version/
REPO_ROOT = NEW_VERSION.parent                      # Unstoppable/unstoppable/
DATASETS_DIR = NEW_VERSION / "datasets"
DATA_OUT_DIR = SCRIPT_DIR / "data"

# v2 best skill from exp1 (all 3 per-dimension files are byte-identical; this is
# the canonical pick recorded in the experiment state).
V2_SKILL_PATH = (
    NEW_VERSION / "experiments" / "exp1_self_evolve_effectiveness"
    / "results" / "refine_v2" / "run" / "round_08" / "SKILL_market_read_accuracy.md"
)

DEFAULT_MODEL = "claude-opus-4-8"


# --------------------------------------------------------------------------- #
# env loading (repo-root .env, no hard dependency)
# --------------------------------------------------------------------------- #
def load_env():
    env_path = REPO_ROOT / ".env"
    try:
        from dotenv import load_dotenv
        if env_path.exists():
            load_dotenv(env_path)
            return
    except Exception:
        pass
    # fallback: minimal KEY=VALUE parser
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# --------------------------------------------------------------------------- #
# Prompts — shared output contract; skill-specific bits differ by arm
# --------------------------------------------------------------------------- #

# Shared schema + tail used by BOTH arms (verbatim identical → controlled).
_SHARED_SCHEMA_TAIL = """
第二段必须是一个 JSON 对象（包在 ```json 代码块里），严格遵守以下 schema：

```json
{{
  "psychology_and_traits": {{
    "personality_tags": ["3-6 个简短性格标签"],
    "behavioral_logic": "用 1-2 句总结其核心行为逻辑"
  }},
  "matchmaking_intelligence": {{
    "core_conflict": "用 1-2 句说明结构性核心矛盾",
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


# ---- SKILL arm (teacher sees the v2 skill in the system prompt) ----
TEACHER_USER_TEMPLATE_SKILL = """请按照本 Skill 的分析框架，对以下相亲案例进行完整分析。

## 重要约束（必须遵守）

⚠️ **禁止使用编号引用**：不要写「触发模型 11」「应用启发式 6」「触发反模式 3」等编号。
   直接说出该模型/启发式/反模式的**核心规则名称和内容**。
   例如不要写「应用启发式 6: 短板即护城河法则」，而要直接论述「在跨阶层匹配中，
   对方明显的可接受短板（矮、丑、话少）反而是关系稳定的安全保障，因为短板降低了对方的外部选择权」。
   这样做的原因：student 模型推理时上下文里不一定能用上编号，编号引用对它是无意义的。

## 输出要求（极其严格）

第一段必须是 `<thinking>` 块，包含 5 步推理（每步 4-8 句话，详细展开）：
1. **画像定位**：这个 case 最像本 Skill 中的哪一个或几个典型原型？为什么？这个原型的典型困境是什么？
2. **市场估值**：从城市/收入/学历/外形/年龄五维度逐一展开，每张牌具体硬在哪里、软在哪里。
3. **核心冲突识别**：用具体的心智模型（按规则名称而非编号）解释结构性矛盾。冲突是怎么产生的？
4. **启发式 / 反模式锚定**：列出 2-4 条最相关的决策启发式或反模式（按规则内容而非编号），说明为何适用本 case。
5. **策略综合**：把上述结论合成成具体可执行的最终策略，包括圈层定位、优势放大、期望校准、反模式规避等。
""" + _SHARED_SCHEMA_TAIL


# ---- NO-SKILL arm (teacher has no skill; pure expert reasoning) ----
TEACHER_USER_TEMPLATE_NOSKILL = """请作为资深婚恋市场分析师，凭你的专业判断，对以下相亲案例进行完整分析。

## 输出要求（极其严格）

第一段必须是 `<thinking>` 块，包含 5 步推理（每步 4-8 句话，详细展开）：
1. **类型定位**：这个求偶者属于什么类型？其典型困境是什么？为什么？
2. **市场估值**：从城市/收入/学历/外形/年龄五维度逐一展开，每张牌具体硬在哪里、软在哪里。
3. **核心冲突识别**：解释这个 case 的结构性矛盾。冲突是怎么产生的？
4. **关键决策原则 / 常见误区**：列出 2-4 条最相关的决策原则或要规避的常见误区，说明为何适用本 case。
5. **策略综合**：把上述结论合成成具体可执行的最终策略，包括圈层定位、优势放大、期望校准、风险规避等。
""" + _SHARED_SCHEMA_TAIL


NOSKILL_SYSTEM_PROMPT = (
    "你是一位资深的婚恋市场分析师，擅长把一个求偶者的画像和择偶预期，"
    "蒸馏成残酷但可执行的市场诊断。请严格遵守用户消息中给出的输出格式要求。"
)


# --------------------------------------------------------------------------- #
# Client / IO helpers
# --------------------------------------------------------------------------- #
def get_client(timeout: float = 600.0):
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if not api_key:
        print("Error: need ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN in env / .env",
              file=sys.stderr)
        sys.exit(1)
    kwargs = {"api_key": api_key, "timeout": timeout}
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return anthropic.Anthropic(**kwargs)


def load_train_cases(lang: str) -> list[dict]:
    if lang == "zh":
        path = DATASETS_DIR / "train" / "inputs_outputs.json"
    elif lang == "en":
        path = DATASETS_DIR / "train" / "inputs_outputs_en.json"
    else:
        raise ValueError(f"unsupported lang: {lang}")
    return json.loads(path.read_text(encoding="utf-8"))


def case_to_input_only(case: dict) -> dict:
    return {
        "subject_profile": case.get("subject_profile", {}),
        "expectations": case.get("expectations", {}),
    }


def case_to_target(case: dict) -> dict:
    return {
        "psychology_and_traits": case.get("psychology_and_traits", {}),
        "matchmaking_intelligence": case.get("matchmaking_intelligence", {}),
    }


def get_case_id(case: dict, idx: int) -> str:
    return case.get("metadata", {}).get("case_id") or f"idx_{idx:04d}"


# --------------------------------------------------------------------------- #
# Parsing (same logic as exp1 teacher gen)
# --------------------------------------------------------------------------- #
THINKING_RE = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL)
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def parse_teacher_output(text: str) -> tuple[str | None, dict | None]:
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


# --------------------------------------------------------------------------- #
# Teacher call
# --------------------------------------------------------------------------- #
def call_teacher(client, model: str, system_prompt: str, user_msg: str,
                 max_tokens: int = 10000, retries: int = 4,
                 backoff: float = 4.0) -> tuple[str, dict]:
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
                time.sleep(backoff * (2 ** (attempt - 1)))
    raise RuntimeError(f"call_teacher failed after {retries} retries: {last_err}")


def process_one(client, model: str, arm: str, system_prompt: str, user_template: str,
                case: dict, idx: int, max_tokens: int) -> dict:
    case_id = get_case_id(case, idx)
    case_input = case_to_input_only(case)
    target = case_to_target(case)
    user_msg = user_template.format(
        case_json=json.dumps(case_input, ensure_ascii=False, indent=2)
    )
    t0 = time.time()
    raw_text, usage = call_teacher(client, model, system_prompt, user_msg, max_tokens=max_tokens)
    elapsed = time.time() - t0
    thinking, teacher_json = parse_teacher_output(raw_text)
    return {
        "case_id": case_id,
        "arm": arm,
        "case_input": case_input,
        "expert_target": target,
        "teacher_raw": raw_text,
        "teacher_thinking": thinking,
        "teacher_json": teacher_json,
        "parse_ok": teacher_json is not None,
        "usage": usage,
        "elapsed_s": round(elapsed, 2),
        "model": model,
        "timestamp": datetime.now().isoformat(),
    }


def append_jsonl(path: Path, record: dict, lock: threading.Lock):
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


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


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    load_env()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=["skill", "noskill"],
                   help="skill = teacher sees v2 SKILL.md; noskill = no skill")
    p.add_argument("--lang", default="zh", choices=["zh", "en"], help="data language")
    p.add_argument("--parallel", type=int, default=16, help="concurrent API workers")
    p.add_argument("--limit", type=int, default=None, help="limit #cases (smoke test)")
    p.add_argument("--max-tokens", type=int, default=10000, help="max output tokens/call")
    p.add_argument("--timeout", type=float, default=600.0, help="per-call API timeout (s)")
    p.add_argument("--model", default=None,
                   help=f"override teacher model (default ${{GEN_MODEL}} or {DEFAULT_MODEL})")
    p.add_argument("--out", type=str, default=None, help="override output JSONL path")
    p.add_argument("--no-resume", action="store_true", help="overwrite output file")
    args = p.parse_args()

    model = args.model or os.environ.get("GEN_MODEL") or os.environ.get("ANTHROPIC_MODEL") or DEFAULT_MODEL
    client = get_client(timeout=args.timeout)

    # Build system prompt + user template per arm.
    if args.arm == "skill":
        if not V2_SKILL_PATH.exists():
            print(f"Error: v2 skill not found at {V2_SKILL_PATH}", file=sys.stderr)
            sys.exit(1)
        system_prompt = V2_SKILL_PATH.read_text(encoding="utf-8")  # skill-only, NO references
        user_template = TEACHER_USER_TEMPLATE_SKILL
    else:
        system_prompt = NOSKILL_SYSTEM_PROMPT
        user_template = TEACHER_USER_TEMPLATE_NOSKILL

    cases = load_train_cases(args.lang)
    if args.limit:
        cases = cases[: args.limit]

    DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else DATA_OUT_DIR / f"teacher_{args.arm}_{args.lang}.jsonl"

    if args.no_resume and out_path.exists():
        out_path.unlink()
    done_ids = already_done_ids(out_path)

    todo = [(idx, c) for idx, c in enumerate(cases) if get_case_id(c, idx) not in done_ids]

    print("=" * 72)
    print(f"Exp2 teacher gen | arm={args.arm} model={model} lang={args.lang}")
    print(f"  cases total:   {len(cases)}")
    print(f"  already done:  {len(done_ids)}")
    print(f"  to process:    {len(todo)}")
    print(f"  parallel:      {args.parallel}   timeout={args.timeout}s")
    print(f"  out file:      {out_path}")
    print(f"  system prompt: {len(system_prompt)} chars (~{len(system_prompt)//3} tok)")
    print("=" * 72)
    if not todo:
        print("Nothing to do; all cases cached. Exiting.")
        return

    lock = threading.Lock()
    parse_ok = parse_fail = total_in = total_out = 0
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = {
            ex.submit(process_one, client, model, args.arm, system_prompt,
                      user_template, case, idx, args.max_tokens): (idx, case)
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
                      f"t={rec['elapsed_s']}s | rate={rate:.2f}/s eta={eta:.0f}s")
            except Exception as e:  # noqa: BLE001
                print(f"  [{i}/{len(todo)}] {cid} FAILED: {e}", file=sys.stderr)
                append_jsonl(out_path, {"case_id": cid, "arm": args.arm,
                                        "error": str(e),
                                        "timestamp": datetime.now().isoformat()}, lock)
                parse_fail += 1

    total_elapsed = time.time() - t_start
    print()
    print("=" * 72)
    print(f"Done in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min) | arm={args.arm}")
    print(f"  parse OK:   {parse_ok}")
    print(f"  parse FAIL: {parse_fail}")
    print(f"  tokens:     in={total_in:,}  out={total_out:,}")
    print(f"  output:     {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
