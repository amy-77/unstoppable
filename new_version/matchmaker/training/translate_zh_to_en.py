#!/usr/bin/env python3
"""Translate the iterated zh/SKILL.md + zh/references/*.md into English.

Why: zh/ was iterated through 2 rounds of evaluate-diagnose-refine, scoring
4.67/5. The current en/ was auto-generated and never iterated. Since the
training cases are 1-to-1 translations (CN_001 zh ↔ CN_001 en), we can
inherit the iterated quality of zh/ by translating the whole skill folder.

Output goes to en_v2/ (not en/) so we can compare; orchestrator script
swaps en/ to en_v2/ atomically when ready.

Strategy:
  - SKILL.md, statistics_snapshot.md, pattern_library.md, case_archive.md:
    each < 30K chars → single-shot translation.
  - case_digest.md: ~106K chars in 191 ### CN_xxx blocks → split by case
    header, translate batches of 25 cases in parallel, then concat.

Constraints (passed to Claude in system prompt):
  - Preserve every case_id exactly (CN_001, TS_005, etc.).
  - Preserve frontmatter, markdown headers, code blocks, and emoji.
  - Translate Chinese cultural terms idiomatically, not literally
    (e.g., 慕强 → status-seeking; 内耗 → self-corrosive emotional pattern;
    体制内 → state-sector / civil-service; 卷 → grind; 顶级 → top-tier).
  - Output English only — no Chinese characters in body, except case_ids
    that contain only ASCII (CN_*, TS_* are fine).

Usage:
    export ANTHROPIC_BASE_URL=...
    export ANTHROPIC_AUTH_TOKEN=...
    python translate_zh_to_en.py --parallel 6
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
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # agent/matchmaker

SYSTEM_PROMPT = """You are a professional translator. Translate the given Chinese
markdown content into natural, fluent English suitable for a Skill / knowledge
base used by an LLM matchmaking analyst.

## HARD RULES — must preserve EXACTLY

1. **Markdown structure**: every header (#, ##, ###), bullet, code fence,
   table, and HTML/special character must remain.
2. **Case IDs**: every identifier matching `[A-Z]{2,3}_\\d{3,}` (CN_001,
   TS_005, etc.) must be copied verbatim — NEVER translated, renumbered,
   or paraphrased.
3. **Frontmatter** (`---` … `---` blocks at the very top) must remain as
   YAML/markdown frontmatter. Translate the values, but keep the keys
   (`name`, `description`, etc.) unchanged. Suffix the `name` value with
   `-en` if it ends with `-zh`.
4. **Numbered headings** like `### 心智模型 1: ...` → translate the heading
   (e.g., `### Mental Model 1: ...`) but KEEP the SAME number.
5. **Quotes / examples** that are themselves direct case-language should
   stay as English equivalents (translate, don't keep Chinese).

## TRANSLATION STYLE

- Idiomatic, not literal. Match the analytical / blunt tone of the source.
- Cultural terms — use these conventions:
  - 慕强 → "status-seeking" / "drawn to higher-status partners"
  - 内耗 → "self-corrosive" / "emotionally self-eroding"
  - 体制内 → "state sector" / "civil service" / "stable government job"
  - 体制外 → "private sector"
  - 顶级 → "top-tier"
  - 一线城市 → "tier-1 city"  (二线 → tier-2, 三线 → tier-3, etc.)
  - 海王 / 渣男 → "player"
  - 老实 / 踏实 → "steady" / "low-key" / "down-to-earth"
  - 圈子 / 圈层 → "social circle" / "class stratum"
  - 红娘 → "matchmaker"
  - 长相 / 外形 → "appearance"
  - 短板 → "weakness" (or "Achilles' heel" when emphatic)
  - 硬伤 → "hard disqualifier" / "structural weakness"
  - 反直觉 → "counter-intuitive"
  - 跨级 / 跨阶层 → "cross-class"
  - A-tier income labels (A4, A5, A6, A7, A8+) → keep as is
- Preserve emphatic emoji (⚠️, ✅) in place.

## OUTPUT FORMAT

Output ONLY the translated English markdown. No preamble, no explanation,
no surrounding code fences. Begin with the first character of the
translation (which may be `---`, `#`, `>`, etc.).
"""


# ---------------------------------------------------------------------------
# Anthropic client
# ---------------------------------------------------------------------------


def get_client():
    try:
        import anthropic
    except ImportError:
        sys.exit("pip install anthropic")
    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("set ANTHROPIC_AUTH_TOKEN")
    kwargs = {"api_key": api_key}
    base = os.environ.get("ANTHROPIC_BASE_URL")
    if base:
        kwargs["base_url"] = base
    return anthropic.Anthropic(**kwargs)


def call_translate(client, model: str, text: str, max_tokens: int,
                   retries: int = 3, backoff: float = 4.0) -> tuple[str, dict]:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": text}],
            )
            out = msg.content[0].text if msg.content else ""
            usage = {
                "in":  getattr(msg.usage, "input_tokens",  0) if msg.usage else 0,
                "out": getattr(msg.usage, "output_tokens", 0) if msg.usage else 0,
            }
            return out, usage
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < retries:
                time.sleep(backoff * (2 ** (attempt - 1)))
    raise RuntimeError(f"call_translate failed: {last_err}")


# ---------------------------------------------------------------------------
# Case-digest splitter
# ---------------------------------------------------------------------------

CASE_HDR_RE = re.compile(r"^### [A-Z]{2,3}_\d+", re.MULTILINE)


def split_case_digest(text: str) -> tuple[str, list[str]]:
    """Return (preamble, [case_block, ...]). Each case_block starts with `### `."""
    matches = list(CASE_HDR_RE.finditer(text))
    if not matches:
        return text, []
    preamble = text[: matches[0].start()].rstrip() + "\n"
    blocks = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        blocks.append(text[m.start():end].rstrip() + "\n")
    return preamble, blocks


def make_batches(blocks: list[str], batch_size: int) -> list[str]:
    """Group consecutive case blocks into batches joined with blank line."""
    batches = []
    for i in range(0, len(blocks), batch_size):
        batches.append("\n".join(blocks[i:i + batch_size]))
    return batches


# ---------------------------------------------------------------------------
# Quality check + fixups
# ---------------------------------------------------------------------------

CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def cjk_count(text: str) -> int:
    return len(CJK_RE.findall(text))


# mco-4 has a tokenizer glitch: it outputs "鳌烨" where "案例" (case) should
# be translated. Always strip these in post-processing.
GLITCH_FIXUPS = [
    (re.compile(r"鳌烨\s*(\d+)"), r"Case \1"),
    (re.compile(r"鳌烨"),         r"Case"),
]


def postprocess(text: str) -> str:
    for rx, repl in GLITCH_FIXUPS:
        text = rx.sub(repl, text)
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="zh", help="source skill folder (under matchmaker/)")
    p.add_argument("--dst", default="en_v2", help="destination skill folder")
    p.add_argument("--parallel", type=int, default=8,
                   help="global parallel workers across ALL sub-tasks")
    p.add_argument("--batch-size", type=int, default=20,
                   help="cases per case_digest batch")
    p.add_argument("--max-tokens", type=int, default=12000)
    args = p.parse_args()

    src_dir = BASE_DIR / args.src
    dst_dir = BASE_DIR / args.dst
    if not src_dir.exists():
        sys.exit(f"missing {src_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    (dst_dir / "references").mkdir(exist_ok=True)

    client = get_client()
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
    print(f"[translate] src={src_dir}  dst={dst_dir}  model={model}")
    print(f"[translate] parallel={args.parallel}  batch_size={args.batch_size}  max_tokens={args.max_tokens}")

    # ------------------------------------------------------------------
    # Build a flat list of translation tasks across ALL files.
    # Each task = (task_key, src_chars, payload, on_done(text))
    # ------------------------------------------------------------------
    tasks: list[tuple[str, str]] = []          # (task_key, payload)

    # Track each file: list of task_keys that compose it (in order),
    # plus the destination path. Results merged after all tasks done.
    composition: dict[Path, list[str]] = {}

    def add_simple(src: Path, dst: Path, label: str):
        tasks.append((label, src.read_text("utf-8")))
        composition[dst] = [label]

    add_simple(src_dir / "SKILL.md",
               dst_dir / "SKILL.md", "SKILL")
    add_simple(src_dir / "references" / "statistics_snapshot.md",
               dst_dir / "references" / "statistics_snapshot.md", "stats")
    add_simple(src_dir / "references" / "pattern_library.md",
               dst_dir / "references" / "pattern_library.md", "patterns")
    add_simple(src_dir / "references" / "case_archive.md",
               dst_dir / "references" / "case_archive.md", "archive")

    # case_digest: split into preamble + N batches
    digest_src = src_dir / "references" / "case_digest.md"
    digest_dst = dst_dir / "references" / "case_digest.md"
    digest_text = digest_src.read_text("utf-8")
    preamble, blocks = split_case_digest(digest_text)
    batches = make_batches(blocks, args.batch_size)
    print(f"  case_digest: {len(blocks)} cases → {len(batches)} batches")
    digest_keys = ["digest_pre"]
    tasks.append(("digest_pre", preamble))
    for i, b in enumerate(batches, start=1):
        k = f"digest_b{i:02d}"
        tasks.append((k, b))
        digest_keys.append(k)
    composition[digest_dst] = digest_keys

    print(f"[translate] total {len(tasks)} sub-tasks across {len(composition)} files")

    # ------------------------------------------------------------------
    # Run in parallel
    # ------------------------------------------------------------------
    lock = threading.Lock()
    stats = {"in": 0, "out": 0, "done": 0}
    results: dict[str, str] = {}
    t0 = time.time()

    def worker(key: str, payload: str):
        out, usage = call_translate(client, model, payload, args.max_tokens)
        out = postprocess(out)
        cjk = cjk_count(out)
        with lock:
            stats["in"]  += usage["in"]
            stats["out"] += usage["out"]
            stats["done"] += 1
            d = stats["done"]
        print(f"  [{d:2d}/{len(tasks)}] ✓ {key:14s}  "
              f"in_chars={len(payload):6d} out_chars={len(out):6d} cjk={cjk:4d}  "
              f"toks in={usage['in']:6d} out={usage['out']:6d}",
              flush=True)
        return key, out

    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = [ex.submit(worker, k, p) for k, p in tasks]
        for f in as_completed(futs):
            k, o = f.result()
            results[k] = o

    # ------------------------------------------------------------------
    # Merge per file
    # ------------------------------------------------------------------
    for dst, keys in composition.items():
        dst.parent.mkdir(parents=True, exist_ok=True)
        merged = "\n\n".join(results[k].rstrip() for k in keys) + "\n"
        dst.write_text(merged, encoding="utf-8")
        cjk = cjk_count(merged)
        print(f"  -> wrote {dst.relative_to(BASE_DIR)}  chars={len(merged)} cjk_left={cjk}")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Translation complete in {elapsed:.1f}s")
    print(f"  files written: {len(composition)}  sub-tasks: {len(tasks)}")
    print(f"  total tokens  in={stats['in']:,}  out={stats['out']:,}")
    cost = stats["in"] / 1e6 * 3.0 + stats["out"] / 1e6 * 15.0
    print(f"  est cost (sonnet pricing): ${cost:.2f}")
    print(f"  output dir: {dst_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
