#!/usr/bin/env python3
"""Evaluate -> Diagnose -> Refine loop for the matchmaking skill.

Outputs:
  1. eval_report.json
  2. diagnosis.json
  3. refine_results.json, plus edits to SKILL.md / references/*.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR.parent / "datasets"
DEFAULT_BASE_MODEL = "/data/qwang/q/thalia/kvcache/ANNCache/models/Qwen3-4B-Thinking-2507"
DEFAULT_ADAPTER_PATH = Path(__file__).resolve().parent / "outputs/qwen3_4b_zh_skill_ep8_lr2e4/adapter"
DEFAULT_TRAIN_DATA = Path(__file__).resolve().parent / "data/sft_train_v2.jsonl"

SCORE_DIMS = [
    "conflict_insight",
    "strategy_direction",
    "logic_depth",
    "persona_read",
    "actionability",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first valid top-level JSON object from a model response."""
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates = fenced + [text]
    for candidate in candidates:
        for start in [m.start() for m in re.finditer(r"\{", candidate)]:
            depth = 0
            in_string = False
            escape = False
            for i, ch in enumerate(candidate[start:], start=start):
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        raw = candidate[start : i + 1]
                        try:
                            return json.loads(raw)
                        except json.JSONDecodeError:
                            break
    raise ValueError("no valid JSON object found in model response")


def post_json_with_retries(url: str, body: dict[str, Any], headers: dict[str, str], timeout: int = 300, retries: int = 3) -> dict[str, Any]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"POST {url} failed after {retries} attempts: {last_err}")


def anthropic_client():
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError("Install dependency first: pip install anthropic") from exc

    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("need ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY in env")

    kwargs: dict[str, Any] = {"api_key": api_key}
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return anthropic.Anthropic(**kwargs)


def call_anthropic_json(prompt: str, *, model: str | None = None, max_tokens: int = 2000) -> tuple[dict[str, Any], str, dict[str, int]]:
    judge_base_url = os.environ.get("JUDGE_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    if judge_base_url:
        return call_openai_compatible_json(prompt, model=model, max_tokens=max_tokens)

    client = anthropic_client()
    model_name = model or os.environ.get("ANTHROPIC_MODEL") or os.environ.get("JUDGE_MODEL") or "claude-sonnet-4-6"
    msg = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text.strip() if msg.content else ""
    usage = {
        "input_tokens": getattr(msg.usage, "input_tokens", 0) if msg.usage else 0,
        "output_tokens": getattr(msg.usage, "output_tokens", 0) if msg.usage else 0,
    }
    return extract_json_object(raw), raw, usage


def call_openai_compatible_json(prompt: str, *, model: str | None = None, max_tokens: int = 2000) -> tuple[dict[str, Any], str, dict[str, int]]:
    base_url = (os.environ.get("JUDGE_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "").rstrip("/")
    api_key = os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
    model_name = model or os.environ.get("JUDGE_MODEL") or os.environ.get("OPENAI_MODEL") or os.environ.get("ANTHROPIC_MODEL")
    if not base_url:
        raise RuntimeError("need JUDGE_BASE_URL or OPENAI_BASE_URL for OpenAI-compatible judge")
    if not api_key:
        raise RuntimeError("need JUDGE_API_KEY, OPENAI_API_KEY, or ANTHROPIC_AUTH_TOKEN")
    if not model_name:
        raise RuntimeError("need JUDGE_MODEL or OPENAI_MODEL")

    body = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": 0,
    }
    data = post_json_with_retries(
        base_url + "/chat/completions",
        body,
        {
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
        timeout=180,
    )
    raw = data["choices"][0]["message"].get("content") or ""
    usage_data = data.get("usage") or {}
    usage = {
        "input_tokens": usage_data.get("prompt_tokens", usage_data.get("input_tokens", 0)),
        "output_tokens": usage_data.get("completion_tokens", usage_data.get("output_tokens", 0)),
    }
    return extract_json_object(raw), raw, usage


def call_openai_compatible_text(
    messages: list[dict[str, str]],
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> tuple[str, dict[str, int]]:
    base = (
        base_url
        or os.environ.get("GENERATOR_BASE_URL")
        or os.environ.get("JUDGE_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or ""
    ).rstrip("/")
    key = (
        api_key
        or os.environ.get("GENERATOR_API_KEY")
        or os.environ.get("JUDGE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    model_name = (
        model
        or os.environ.get("GENERATOR_MODEL")
        or os.environ.get("JUDGE_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or os.environ.get("ANTHROPIC_MODEL")
    )
    if not base:
        raise RuntimeError("need GENERATOR_BASE_URL, JUDGE_BASE_URL, or OPENAI_BASE_URL")
    if not key:
        raise RuntimeError("need GENERATOR_API_KEY, JUDGE_API_KEY, OPENAI_API_KEY, or ANTHROPIC_AUTH_TOKEN")
    if not model_name:
        raise RuntimeError("need GENERATOR_MODEL, JUDGE_MODEL, or OPENAI_MODEL")

    body = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": temperature,
    }
    data = post_json_with_retries(
        base + "/chat/completions",
        body,
        {
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
        },
        timeout=300,
    )
    raw = data["choices"][0]["message"].get("content") or ""
    usage_data = data.get("usage") or {}
    usage = {
        "input_tokens": usage_data.get("prompt_tokens", usage_data.get("input_tokens", 0)),
        "output_tokens": usage_data.get("completion_tokens", usage_data.get("output_tokens", 0)),
    }
    return raw, usage


def preflight_anthropic(model: str | None = None) -> None:
    call_anthropic_json(
        'Return exactly this JSON object and nothing else: {"ok": true}',
        model=model,
        max_tokens=20,
    )


def load_system_prompt(train_data: Path) -> str:
    first_line = json.loads(train_data.read_text(encoding="utf-8").splitlines()[0])
    for msg in first_line["messages"]:
        if msg["role"] == "system":
            return msg["content"]
    return ""


def load_cases(inputs_path: Path, outputs_path: Path, limit: int | None, offset: int) -> list[dict[str, Any]]:
    inputs = read_json(inputs_path)
    outputs = read_json(outputs_path)
    if len(inputs) != len(outputs):
        raise ValueError(f"input/output case count mismatch: {len(inputs)} != {len(outputs)}")
    rows = []
    for idx, (case_input, case_output) in enumerate(zip(inputs, outputs)):
        if idx < offset:
            continue
        rows.append({
            "idx": idx,
            "case_id": case_input.get("case_id") or case_input.get("input", {}).get("metadata", {}).get("case_id") or f"idx_{idx:04d}",
            "input": case_input.get("input", case_input),
            "target": case_output.get("target", case_output),
        })
        if limit is not None and len(rows) >= limit:
            break
    return rows


def load_predictions(path: Path) -> dict[str, str]:
    records = read_json(path) if path.suffix == ".json" else None
    if isinstance(records, dict):
        if "cases" in records and isinstance(records["cases"], list):
            rows = records["cases"]
        else:
            return {str(k): str(v) for k, v in records.items() if isinstance(v, str)}
    elif isinstance(records, list):
        rows = records
    else:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    out: dict[str, str] = {}
    for row in rows:
        case_id = str(row.get("case_id") or row.get("id"))
        text = row.get("generated_doc") or row.get("model_output") or row.get("output") or row.get("text")
        if case_id and text:
            out[case_id] = str(text)
    return out


def load_local_model(base_model: str, adapter_path: Path):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_path), device_map="cuda:0")
    model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def generate_local(model, tokenizer, system_prompt: str, case_input: dict[str, Any], max_new_tokens: int) -> str:
    import torch

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(case_input, ensure_ascii=False)},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def generate_api(system_prompt: str, case_input: dict[str, Any], model: str | None, max_tokens: int) -> tuple[str, dict[str, int]]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(case_input, ensure_ascii=False)},
    ]
    return call_openai_compatible_text(
        messages,
        model=model,
        max_tokens=max_tokens,
        temperature=0.7,
    )


def judge_prompt(case_input: dict[str, Any], target: dict[str, Any], generated_doc: str) -> str:
    mi = target.get("matchmaking_intelligence", {})
    pt = target.get("psychology_and_traits", {})
    return f"""你是一个婚恋分析质量评估专家。请对比"生成的分析"和"参考答案"，从语义层面评分。
不要求措辞一致，只要核心判断方向一致即可得高分。

## 评分维度（每项 1-5 分）

1. conflict_insight: 是否识别出与参考答案相同/等价的核心冲突
2. strategy_direction: 策略方向是否与专家建议一致（措辞不同没关系）
3. logic_depth: 分析逻辑链是否有深度、自洽
4. persona_read: 对人物心理/性格的解读是否到位
5. actionability: 建议是否具体可执行（而非泛泛而谈）

## 案例原始信息
{json.dumps(case_input, ensure_ascii=False, indent=2)[:2000]}

## 参考答案
- 核心冲突: {mi.get("core_conflict", "N/A")}
- 市场评估: {mi.get("market_value_assessment", "N/A")}
- 专家策略: {mi.get("expert_strategy", "N/A")}
- 目标画像(target_portrait): {mi.get("target_portrait", "N/A")}
- 逻辑链: {json.dumps(mi.get("logic_chain", []), ensure_ascii=False)}
- 性格标签: {json.dumps(pt.get("personality_tags", []), ensure_ascii=False)}

## 生成的分析
{generated_doc}

## 输出格式（严格 JSON，不要 Markdown）
{{
  "conflict_insight": <1-5>,
  "strategy_direction": <1-5>,
  "logic_depth": <1-5>,
  "persona_read": <1-5>,
  "actionability": <1-5>,
  "comment": "<一句话总结主要差距>"
}}"""


def cmd_evaluate(args: argparse.Namespace) -> None:
    cases = load_cases(Path(args.inputs), Path(args.outputs), args.limit, args.offset)
    predictions = load_predictions(Path(args.predictions)) if args.predictions else {}
    model = tokenizer = None
    system_prompt = ""
    print("Checking judge credentials...", flush=True)
    preflight_anthropic(args.judge_model)
    system_prompt = load_system_prompt(Path(args.train_data))
    if args.generator == "api" and not predictions:
        print("Checking API generator credentials...", flush=True)
        generate_api("Return exactly: ok", {"ping": "ok"}, args.generator_model, 20)
    if not predictions:
        if args.generator == "local":
            print("Loading local model...", flush=True)
            model, tokenizer = load_local_model(args.base_model, Path(args.adapter_path))
        else:
            print("Using API generator.", flush=True)

    out_path = Path(args.out)

    def build_report(elapsed_s: float) -> dict[str, Any]:
        n = max(len(results), 1)
        dim_avgs = {dim: round(total / n, 3) for dim, total in totals.items()}
        return {
            "created_at": datetime.now().isoformat(),
            "num_cases": len(results),
            "requested_cases": len(cases),
            "complete": len(results) == len(cases),
            "overall_avg": round(sum(dim_avgs.values()) / len(SCORE_DIMS), 3),
            "dimension_avgs": dim_avgs,
            "weakest_dimension": min(dim_avgs, key=dim_avgs.get) if results else None,
            "usage": {
                "input_tokens": generator_usage_total["input_tokens"] + judge_usage_total["input_tokens"],
                "output_tokens": generator_usage_total["output_tokens"] + judge_usage_total["output_tokens"],
            },
            "generator": args.generator,
            "generator_model": args.generator_model if args.generator == "api" else "local",
            "judge_model": args.judge_model or os.environ.get("JUDGE_MODEL") or os.environ.get("ANTHROPIC_MODEL"),
            "generator_usage": generator_usage_total,
            "judge_usage": judge_usage_total,
            "elapsed_s": round(elapsed_s, 2),
            "cases": results,
        }

    results = []
    trace_records = []
    totals = {dim: 0.0 for dim in SCORE_DIMS}
    judge_usage_total = {"input_tokens": 0, "output_tokens": 0}
    generator_usage_total = {"input_tokens": 0, "output_tokens": 0}
    t0 = time.time()
    for i, case in enumerate(cases, 1):
        case_id = case["case_id"]
        print(f"[{i}/{len(cases)}] evaluate {case_id}", flush=True)
        generated_doc = predictions.get(case_id)
        generator_usage = {"input_tokens": 0, "output_tokens": 0}
        if generated_doc is None:
            if args.generator == "local":
                generated_doc = generate_local(model, tokenizer, system_prompt, case["input"], args.max_new_tokens)
            else:
                generated_doc, generator_usage = generate_api(
                    system_prompt,
                    case["input"],
                    args.generator_model,
                    args.max_new_tokens,
                )

        prompt = judge_prompt(case["input"], case["target"], generated_doc)
        scores, raw_judge, usage = call_anthropic_json(
            prompt,
            model=args.judge_model,
            max_tokens=args.judge_max_tokens,
        )
        dim_scores = {dim: int(scores.get(dim, 0)) for dim in SCORE_DIMS}
        avg = sum(dim_scores.values()) / len(SCORE_DIMS)
        for dim, val in dim_scores.items():
            totals[dim] += val
        judge_usage_total["input_tokens"] += usage["input_tokens"]
        judge_usage_total["output_tokens"] += usage["output_tokens"]
        generator_usage_total["input_tokens"] += generator_usage["input_tokens"]
        generator_usage_total["output_tokens"] += generator_usage["output_tokens"]
        results.append({
            "case_id": case_id,
            "idx": case["idx"],
            "case_input": case["input"],
            "scores": dim_scores,
            "avg_score": round(avg, 2),
            "comment": scores.get("comment", ""),
            "generated_doc": generated_doc,
            "target": case["target"],
            "raw_judge": raw_judge,
            "generator_usage": generator_usage,
            "judge_usage": usage,
        })
        trace_records.append({
            "stage": "evaluate",
            "case_id": case_id,
            "idx": case["idx"],
            "generator": args.generator,
            "generator_model": args.generator_model if args.generator == "api" else "local",
            "judge_model": args.judge_model or os.environ.get("JUDGE_MODEL") or os.environ.get("ANTHROPIC_MODEL"),
            "generator_input": {
                "system_prompt": system_prompt,
                "case_input": case["input"],
            },
            "generator_output": generated_doc,
            "reference_target": case["target"],
            "judge_input": prompt,
            "judge_output_raw": raw_judge,
            "scores": dim_scores,
            "avg_score": round(avg, 2),
            "comment": scores.get("comment", ""),
            "generator_usage": generator_usage,
            "judge_usage": usage,
        })
        write_json(out_path, build_report(time.time() - t0))
        write_jsonl(out_path.with_name("eval_cases.jsonl"), trace_records)

    write_json(out_path, build_report(time.time() - t0))
    write_jsonl(out_path.with_name("eval_cases.jsonl"), trace_records)
    print(f"wrote {args.out}")


def load_skill_context(skill_dir: Path, train_data: Path | None = None) -> dict[str, str]:
    files: dict[str, str] = {}
    skill_path = skill_dir / "SKILL.md"
    if skill_path.exists():
        files["SKILL.md"] = skill_path.read_text(encoding="utf-8")
    ref_dir = skill_dir / "references"
    if ref_dir.exists():
        for path in sorted(ref_dir.glob("*.md")):
            files[f"references/{path.name}"] = path.read_text(encoding="utf-8")
    if train_data and train_data.exists():
        files["prompts/system_prompt_from_train_data"] = load_system_prompt(train_data)
    return files


def diagnose_prompt(report: dict[str, Any], skill_files: dict[str, str]) -> str:
    compact_cases = []
    for row in sorted(report.get("cases", []), key=lambda r: r.get("avg_score", 0))[:12]:
        compact_cases.append({
            "case_id": row.get("case_id"),
            "scores": row.get("scores"),
            "avg_score": row.get("avg_score"),
            "comment": row.get("comment"),
            "target": row.get("target"),
            "generated_doc_excerpt": str(row.get("generated_doc", ""))[:2000],
        })
    skill_excerpt = {
        name: text[:12000] + ("\n...TRUNCATED..." if len(text) > 12000 else "")
        for name, text in skill_files.items()
    }
    return f"""你是 Skill 自进化诊断器。请基于评估报告和当前 SKILL/references，找出低分根因并给出可直接执行的修复建议。

要求：
1. 找出最弱维度。
2. 给出 exactly 3 条根因，每条必须绑定证据 case_id 和当前文档缺口。
3. 给出按优先级排序的改进建议。每条建议必须指明文件、section/anchor、操作和新增/替换文本。
4. 只能修改 SKILL.md 或 references/*.md。
5. 输出严格 JSON，不要 Markdown。

允许的 patch 操作：
- replace: 需要 file, old_text, new_text
- insert_after: 需要 file, anchor, content
- append: 需要 file, content

## 评估摘要
{json.dumps({k: report.get(k) for k in ["num_cases", "overall_avg", "dimension_avgs", "weakest_dimension"]}, ensure_ascii=False, indent=2)}

## 低分案例
{json.dumps(compact_cases, ensure_ascii=False, indent=2)}

## 当前文档
{json.dumps(skill_excerpt, ensure_ascii=False, indent=2)}

## 输出格式
{{
  "weakest_dimension": "<dimension>",
  "root_causes": [
    {{"id": "R1", "evidence_case_ids": ["..."], "finding": "...", "document_gap": "..."}},
    {{"id": "R2", "evidence_case_ids": ["..."], "finding": "...", "document_gap": "..."}},
    {{"id": "R3", "evidence_case_ids": ["..."], "finding": "...", "document_gap": "..."}}
  ],
  "improvements": [
    {{
      "priority": 1,
      "root_cause_id": "R1",
      "file": "SKILL.md",
      "section": "## ...",
      "operation": "insert_after",
      "anchor": "exact existing text anchor",
      "content": "text to insert"
    }}
  ]
}}"""


def cmd_diagnose(args: argparse.Namespace) -> None:
    report = read_json(Path(args.eval_report))
    skill_files = load_skill_context(Path(args.skill_dir), Path(args.train_data))
    prompt = diagnose_prompt(report, skill_files)
    diagnosis, raw, usage = call_anthropic_json(
        prompt,
        model=args.model,
        max_tokens=args.max_tokens,
    )
    diagnosis["_meta"] = {
        "created_at": datetime.now().isoformat(),
        "eval_report": args.eval_report,
        "skill_dir": args.skill_dir,
        "usage": usage,
        "raw_response": raw,
    }
    out_path = Path(args.out)
    write_json(out_path, diagnosis)
    write_jsonl(out_path.with_name("diagnosis_trace.jsonl"), [{
        "stage": "diagnose",
        "eval_report": args.eval_report,
        "skill_dir": args.skill_dir,
        "model": args.model or os.environ.get("JUDGE_MODEL") or os.environ.get("ANTHROPIC_MODEL"),
        "diagnose_input": {
            "prompt": prompt,
            "eval_summary": {k: report.get(k) for k in ["num_cases", "overall_avg", "dimension_avgs", "weakest_dimension"]},
            "skill_files": list(skill_files.keys()),
        },
        "diagnose_output_raw": raw,
        "diagnosis": {k: v for k, v in diagnosis.items() if k != "_meta"},
        "usage": usage,
    }])
    print(f"wrote {args.out}")


def resolve_edit_path(skill_dir: Path, rel_file: str) -> Path:
    if rel_file == "SKILL.md":
        path = skill_dir / "SKILL.md"
    elif rel_file.startswith("references/") and rel_file.endswith(".md"):
        path = skill_dir / rel_file
    else:
        raise ValueError(f"refusing to edit unsupported file: {rel_file}")
    root = skill_dir.resolve()
    resolved = path.resolve()
    if root not in resolved.parents and resolved != root:
        raise ValueError(f"refusing path outside skill_dir: {rel_file}")
    return resolved


def apply_one_change(skill_dir: Path, change: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    rel_file = change.get("file", "")
    operation = change.get("operation", "")
    path = resolve_edit_path(skill_dir, rel_file)
    before = path.read_text(encoding="utf-8") if path.exists() else ""
    after = before

    if operation == "replace":
        old_text = change.get("old_text") or change.get("anchor") or ""
        new_text = change.get("new_text") or change.get("content") or ""
        if not old_text:
            raise ValueError("replace requires old_text")
        if old_text not in before:
            raise ValueError("replace old_text not found")
        after = before.replace(old_text, new_text, 1)
    elif operation == "insert_after":
        anchor = change.get("anchor") or ""
        content = change.get("content") or ""
        if not anchor or not content:
            raise ValueError("insert_after requires anchor and content")
        idx = before.find(anchor)
        if idx == -1:
            raise ValueError("insert_after anchor not found")
        insert_at = idx + len(anchor)
        spacer = "\n\n" if not content.startswith("\n") else ""
        after = before[:insert_at] + spacer + content.rstrip() + "\n" + before[insert_at:]
    elif operation == "append":
        content = change.get("content") or ""
        if not content:
            raise ValueError("append requires content")
        after = before.rstrip() + "\n\n" + content.rstrip() + "\n"
    else:
        raise ValueError(f"unsupported operation: {operation}")

    changed = after != before
    if changed and not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(after, encoding="utf-8")
    return {
        "file": rel_file,
        "operation": operation,
        "changed": changed,
        "dry_run": dry_run,
        "before_chars": len(before),
        "after_chars": len(after),
    }


def cmd_refine(args: argparse.Namespace) -> None:
    diagnosis = read_json(Path(args.diagnosis))
    changes = diagnosis.get("improvements", [])
    if not isinstance(changes, list):
        raise ValueError("diagnosis.improvements must be a list")

    results = []
    ok = failed = changed = 0
    for item in sorted(changes, key=lambda c: c.get("priority", 999)):
        try:
            result = apply_one_change(Path(args.skill_dir), item, args.dry_run)
            ok += 1
            changed += 1 if result["changed"] else 0
            result["priority"] = item.get("priority")
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            results.append({
                "file": item.get("file"),
                "operation": item.get("operation"),
                "priority": item.get("priority"),
                "changed": False,
                "error": str(exc),
            })

    summary = {
        "created_at": datetime.now().isoformat(),
        "diagnosis": args.diagnosis,
        "skill_dir": args.skill_dir,
        "dry_run": args.dry_run,
        "total": len(changes),
        "applied": ok,
        "failed": failed,
        "changed": changed,
        "results": results,
    }
    write_json(Path(args.out), summary)
    write_jsonl(Path(args.out).with_suffix(".jsonl"), [
        {
            "stage": "refine",
            "diagnosis": args.diagnosis,
            "skill_dir": args.skill_dir,
            "dry_run": args.dry_run,
            **result,
        }
        for result in results
    ])
    print(f"wrote {args.out} | applied={ok} failed={failed} changed={changed}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("evaluate", help="run cases and write eval_report.json")
    p.add_argument("--inputs", default=str(DATASETS_DIR / "test" / "inputs.json"))
    p.add_argument("--outputs", default=str(DATASETS_DIR / "test" / "outputs.json"))
    p.add_argument("--train-data", default=str(DEFAULT_TRAIN_DATA))
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--adapter-path", default=str(DEFAULT_ADAPTER_PATH))
    p.add_argument("--predictions", default=None, help="optional JSON/JSONL with case_id + output/generated_doc")
    p.add_argument("--generator", choices=["local", "api"], default="local",
                   help="generate analyses with the local LoRA model or an OpenAI-compatible API")
    p.add_argument("--generator-model", default=None,
                   help="API model for --generator api; defaults to GENERATOR_MODEL/JUDGE_MODEL")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--judge-model", default=None)
    p.add_argument("--judge-max-tokens", type=int, default=800)
    p.add_argument("--out", default="eval_report.json")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("diagnose", help="diagnose eval_report.json and write diagnosis.json")
    p.add_argument("--eval-report", default="eval_report.json")
    p.add_argument("--skill-dir", default=str(BASE_DIR / "zh"))
    p.add_argument("--train-data", default=str(DEFAULT_TRAIN_DATA))
    p.add_argument("--model", default=None)
    p.add_argument("--max-tokens", type=int, default=5000)
    p.add_argument("--out", default="diagnosis.json")
    p.set_defaults(func=cmd_diagnose)

    p = sub.add_parser("refine", help="apply diagnosis improvements to SKILL.md/references")
    p.add_argument("--diagnosis", default="diagnosis.json")
    p.add_argument("--skill-dir", default=str(BASE_DIR / "zh"))
    p.add_argument("--out", default="refine_results.json")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_refine)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
