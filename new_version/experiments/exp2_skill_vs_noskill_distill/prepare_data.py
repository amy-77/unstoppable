#!/usr/bin/env python3
"""Exp2 — turn teacher_{arm}_zh.jsonl into SFT chat-message data.

Per the locked exp2 design: BOTH arms (skill / noskill) put the **v2 best skill**
in the student's system prompt at train time (= deployment/eval condition), so the
only controlled variable is whether the *teacher* used the skill when producing
the assistant target.

Output record:
  {
    "case_id": "...", "arm": "skill|noskill", "lang": "zh",
    "messages": [
      {"role": "system",    "content": <v2 skill + output-format spec>},
      {"role": "user",      "content": <case input JSON>},
      {"role": "assistant", "content": <teacher_raw: <thinking>+json>}
    ]
  }

The system prompt built here is the SINGLE SOURCE OF TRUTH — the eval step MUST
build the student system prompt with `build_student_system()` so train/inference
match exactly.

Both arms cover the same 190 case_ids (dedup already done upstream). Records are
sorted by case_id before the seeded shuffle so the train/val split is IDENTICAL
across arms.

Usage (run from new_version/):
    python experiments/exp2_skill_vs_noskill_distill/prepare_data.py --arm both
    python experiments/exp2_skill_vs_noskill_distill/prepare_data.py --arm skill --val-frac 0.05
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
NEW_VERSION = SCRIPT_DIR.parents[1]
DATA_DIR = SCRIPT_DIR / "data"

V2_SKILL_PATH = (
    NEW_VERSION / "experiments" / "exp1_self_evolve_effectiveness"
    / "results" / "refine_v2" / "run" / "round_08" / "SKILL_market_read_accuracy.md"
)

# Concise output-format spec appended after the skill so the student reliably
# emits <thinking> + the JSON schema at inference (the skill itself describes a
# different narrative format). Mirrors the teacher user-template contract.
OUTPUT_FORMAT_SPEC = """## 输出格式（严格）

1. 先输出一个 `<thinking>...</thinking>` 块，包含 5 步推理：
   (1) 画像/类型定位 (2) 市场估值（城市/收入/学历/外形/年龄）
   (3) 核心冲突识别 (4) 启发式与反模式 (5) 策略综合。
2. 然后输出一个 ```json 代码块，严格遵守 schema：

```json
{
  "psychology_and_traits": {
    "personality_tags": ["..."],
    "behavioral_logic": "..."
  },
  "matchmaking_intelligence": {
    "core_conflict": "...",
    "market_value_assessment": "...",
    "expert_strategy": "...",
    "target_portrait": "...",
    "logic_chain": ["...", "..."]
  }
}
```

只输出 `<thinking>` 块 + json 代码块，前后不要有任何其他内容。"""


def build_student_system() -> str:
    """The student's system prompt = v2 skill + output-format spec.
    Used by BOTH arms at train time and MUST be reused verbatim at eval time."""
    skill = V2_SKILL_PATH.read_text(encoding="utf-8")
    return f"{skill}\n\n---\n\n{OUTPUT_FORMAT_SPEC}"


def case_input_text(case_input: dict) -> str:
    payload = {
        "subject_profile": case_input.get("subject_profile", {}),
        "expectations": case_input.get("expectations", {}),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def load_teacher(arm: str, lang: str) -> list[dict]:
    path = DATA_DIR / f"teacher_{arm}_{lang}.jsonl"
    recs = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not r.get("parse_ok") or not r.get("teacher_raw"):
            continue
        cid = r["case_id"]
        if cid in seen:                      # safety: one record per case_id
            continue
        seen.add(cid)
        recs.append(r)
    return recs


def to_sft_record(r: dict, system: str, arm: str, lang: str) -> dict:
    return {
        "case_id": r["case_id"],
        "arm": arm,
        "lang": lang,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": case_input_text(r["case_input"])},
            {"role": "assistant", "content": r["teacher_raw"]},
        ],
    }


def prepare_arm(arm: str, lang: str, system: str, val_frac: float, seed: int):
    recs = load_teacher(arm, lang)
    recs.sort(key=lambda r: r["case_id"])                # identical order across arms
    sft = [to_sft_record(r, system, arm, lang) for r in recs]

    rng = random.Random(seed)
    rng.shuffle(sft)                                     # deterministic; same per arm
    n_val = max(1, int(len(sft) * val_frac))
    val, train = sft[:n_val], sft[n_val:]

    train_path = DATA_DIR / f"sft_{arm}_train.jsonl"
    val_path = DATA_DIR / f"sft_{arm}_val.jsonl"
    for path, rows in [(train_path, train), (val_path, val)]:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    sys_chars = len(system)
    asst = [len(s["messages"][2]["content"]) for s in sft]
    print(f"[{arm}] total={len(sft)} train={len(train)} val={len(val)} | "
          f"system={sys_chars} chars (~{sys_chars//2}-{sys_chars} tok) | "
          f"assistant chars mean={sum(asst)//len(asst)} max={max(asst)}")
    print(f"        -> {train_path.name} , {val_path.name}")
    return {s["case_id"] for s in val}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="both", choices=["skill", "noskill", "both"])
    ap.add_argument("--lang", default="zh", choices=["zh", "en"])
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not V2_SKILL_PATH.exists():
        raise SystemExit(f"v2 skill not found: {V2_SKILL_PATH}")
    system = build_student_system()

    arms = ["skill", "noskill"] if args.arm == "both" else [args.arm]
    val_sets = {}
    for arm in arms:
        val_sets[arm] = prepare_arm(arm, args.lang, system, args.val_frac, args.seed)

    if len(arms) == 2:
        same = val_sets["skill"] == val_sets["noskill"]
        print(f"\nval split identical across arms: {same} "
              f"({'OK — controlled' if same else 'WARNING — splits differ!'})")


if __name__ == "__main__":
    main()
