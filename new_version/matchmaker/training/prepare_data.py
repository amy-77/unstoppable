#!/usr/bin/env python3
"""Convert teacher_<lang>.jsonl into SFT chat-message format.

Each output record looks like:

  {
    "case_id": "CN_007",
    "lang": "zh",
    "messages": [
      {"role": "system",    "content": <SYSTEM_PROMPT>},
      {"role": "user",      "content": <case input JSON pretty-printed>},
      {"role": "assistant", "content": <teacher_raw, i.e. <thinking>+JSON>}
    ]
  }

The student model learns: case input → reasoning trace + structured analysis.
The SKILL.md is intentionally NOT in the system prompt — knowledge must be
internalized into the student weights (CoT distillation).

A defensive instruction is added in system prompt:
  "Do not reference rule IDs — explain rules by their content"
to suppress occasional teacher-style "Triggers Model 11" leaks.

Usage:
    python prepare_data.py --inputs data/teacher_zh.jsonl
    python prepare_data.py --inputs data/teacher_zh.jsonl data/teacher_en.jsonl
    python prepare_data.py --inputs data/teacher_zh.jsonl --val-frac 0.05
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a senior matchmaking-market analyst. Given a person's profile and "
    "their expectations, produce a structured analysis of their dating-market "
    "position.\n\n"
    "## Output format (strict)\n\n"
    "1. First, a `<thinking>...</thinking>` block with a 5-step reasoning trace:\n"
    "   (1) archetype mapping, (2) market valuation across city / income / "
    "education / appearance / age, (3) core conflict identification, "
    "(4) heuristics & anti-patterns selected, (5) strategy synthesis.\n"
    "2. Then a JSON object inside a ```json code block, with this schema:\n"
    "   {\n"
    "     \"psychology_and_traits\": {\"personality_tags\": [...], "
    "\"behavioral_logic\": \"...\"},\n"
    "     \"matchmaking_intelligence\": {\"core_conflict\": \"...\", "
    "\"market_value_assessment\": \"...\", \"expert_strategy\": \"...\", "
    "\"target_portrait\": \"...\", \"logic_chain\": [\"...\", \"...\"]}\n"
    "   }\n\n"
    "## Rules\n\n"
    "- Do NOT reference rules by numeric ID (no \"Model 11\", \"Heuristic 6\", "
    "\"Anti-pattern 3\"). Explain rules by their content inline.\n"
    "- Tone: surgical, blunt, market-focused. No empty comfort.\n"
    "- Output ONLY the `<thinking>` block followed by the json code block — "
    "nothing before or after."
)


# Strip numeric rule references from teacher output, preserving the rule's
# CONTENT NAME (which conveys meaning) while dropping the ID number.
#
# Examples after stripping:
#   "「模型1：价值错配模型」"   →  "「价值错配模型」"
#   "触发「模型1: 价值错配」"    →  "触发「价值错配」"
#   "反模式5「面子驱动型婚姻」"  →  "「面子驱动型婚姻」"
#   "启发式13「暂停入场法则」"   →  "「暂停入场法则」"
#   "applies Heuristic 6: weakness as moat" → "applies weakness as moat"
#   "Triggers Model 11"          →  "Triggers" (bare ID, name-less)
#
# Strategy:
#   PASS 1 (prefix-with-name): "<TAG><N>[:：]<NAME>" → "<NAME>"
#   PASS 2 (bare):             "<TAG><N>"           → ""
ZH_TAG = r"(?:心智模型|模型|启发式|反模式|原型)"
EN_TAG = r"(?:mental\s*model|model|heuristic|anti[- ]pattern|archetype)"

PREFIX_NAMED_ZH = re.compile(rf"{ZH_TAG}\s*\d+\s*[:：]\s*", re.IGNORECASE)
PREFIX_NAMED_EN = re.compile(rf"\b{EN_TAG}\s*#?\s*\d+\s*[:：]\s*", re.IGNORECASE)
BARE_ZH = re.compile(rf"{ZH_TAG}\s*\d+", re.IGNORECASE)
BARE_EN = re.compile(rf"\b{EN_TAG}\s*#?\s*\d+\b", re.IGNORECASE)


def strip_rule_ids(text: str) -> str:
    out = text
    out = PREFIX_NAMED_ZH.sub("", out)
    out = PREFIX_NAMED_EN.sub("", out)
    out = BARE_ZH.sub("", out)
    out = BARE_EN.sub("", out)
    out = re.sub(r"「\s*」", "", out)
    out = re.sub(r"[（(]\s*[)）]", "", out)
    out = re.sub(r"  +", " ", out)
    out = re.sub(r"\s+([,.；。，、])", r"\1", out)
    return out


def case_input_text(case_input: dict) -> str:
    """Render the case input as the user message body."""
    payload = {
        "subject_profile": case_input.get("subject_profile", {}),
        "expectations":    case_input.get("expectations", {}),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def lang_of_path(p: Path) -> str:
    n = p.stem.lower()
    if n.endswith("_en"): return "en"
    if n.endswith("_zh"): return "zh"
    return "unk"


def load_records(in_path: Path) -> list[dict]:
    recs = []
    for line in in_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not r.get("parse_ok"):
            continue
        if not r.get("teacher_raw"):
            continue
        recs.append(r)
    return recs


def to_message_record(r: dict, lang: str, scrub_ids: bool) -> dict:
    user_body = case_input_text(r["case_input"])
    asst_body = r["teacher_raw"]
    if scrub_ids:
        asst_body = strip_rule_ids(asst_body)

    return {
        "case_id": r["case_id"],
        "lang":    lang,
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_body},
            {"role": "assistant", "content": asst_body},
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="one or more teacher_*.jsonl files")
    ap.add_argument("--out-dir", default="data",
                    help="directory to write sft_train.jsonl + sft_val.jsonl")
    ap.add_argument("--val-frac", type=float, default=0.05,
                    help="fraction held out for validation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-scrub-ids", action="store_true",
                    help="disable numeric rule-id scrubbing in assistant text")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_recs: list[dict] = []
    for f in args.inputs:
        p = Path(f)
        lang = lang_of_path(p)
        rs = load_records(p)
        print(f"  {p}: {len(rs)} records  lang={lang}")
        for r in rs:
            all_recs.append(to_message_record(r, lang, scrub_ids=not args.no_scrub_ids))

    rng = random.Random(args.seed)
    rng.shuffle(all_recs)
    n_val = max(1, int(len(all_recs) * args.val_frac))
    val_recs   = all_recs[:n_val]
    train_recs = all_recs[n_val:]

    train_path = out_dir / "sft_train.jsonl"
    val_path   = out_dir / "sft_val.jsonl"
    with train_path.open("w", encoding="utf-8") as f:
        for r in train_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with val_path.open("w", encoding="utf-8") as f:
        for r in val_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nwrote {train_path} ({len(train_recs)} records)")
    print(f"wrote {val_path}   ({len(val_recs)} records)")

    if all_recs:
        s = all_recs[0]
        print(f"\nsample case_id={s['case_id']} lang={s['lang']}")
        print(f"  user_chars={len(s['messages'][1]['content'])} "
              f"asst_chars={len(s['messages'][2]['content'])}")
        from collections import Counter
        cnt = Counter(r["lang"] for r in all_recs)
        print(f"  lang dist: {dict(cnt)}")


if __name__ == "__main__":
    main()
