"""Jury of 3 diverse LLM judges (PoLL-style) for matchmaking-analysis scoring.

Generator is Anthropic Opus, so NONE of the judges is Anthropic (no self-pref);
NONE is Qwen (the distillation student) either — disjoint families:
  - gpt-5.5            (OpenAI)
  - gemini-3.1-pro     (Google)
  - deepseek-v4-pro    (DeepSeek)

Default samples=1: the 3-judge jury already provides continuity + variance
reduction, so per-model multi-sampling is redundant.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rubric import DIMS, DIMS_A, DIMS_B, build_prompt, extract_scores

_ROOT = Path(__file__).resolve().parents[2]  # new_version/
MAX_TOKENS = 4000


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader (no external dependency)."""
    p = path or (_ROOT / ".env")
    if not p.exists():
        # also try repo root (one level above new_version)
        p = _ROOT.parent / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


# --- provider calls ---------------------------------------------------------

def _call_openai_compatible(prompt: str, *, model: str, api_key: str,
                            base_url: str | None, max_param: str) -> str:
    from openai import OpenAI
    _kw = {"api_key": api_key, "timeout": 120.0, "max_retries": 2}
    if base_url:
        _kw["base_url"] = base_url
    client = OpenAI(**_kw)
    kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}], max_param: MAX_TOKENS}
    r = client.chat.completions.create(**kwargs)
    return r.choices[0].message.content or ""


def call_openai(prompt: str, model: str = "gpt-5.5") -> str:
    return _call_openai_compatible(
        prompt, model=model, api_key=os.environ["OPENAI_API_KEY"],
        base_url=None, max_param="max_completion_tokens")


def call_deepseek(prompt: str, model: str = "deepseek-v4-pro") -> str:
    return _call_openai_compatible(
        prompt, model=model, api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com/v1", max_param="max_tokens")


def call_gemini(prompt: str, model: str = "gemini-3.1-pro-preview") -> str:
    key = os.environ["GOOGLE_GENERATIVE_AI_API_KEY"]
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 1.0, "maxOutputTokens": MAX_TOKENS},
    }).encode()
    for m in (model, "gemini-2.5-pro"):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={key}"
        try:
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=120).read())
            return r["candidates"][0]["content"]["parts"][0]["text"]
        except (urllib.error.HTTPError, KeyError, IndexError):
            continue
    return ""


JUDGES = [
    ("gpt-5.5", call_openai),
    ("gemini-3.1-pro", call_gemini),
    ("deepseek-v4-pro", call_deepseek),
]


def _call_with_retry(fn, prompt: str, retries: int = 2) -> str:
    last = ""
    for attempt in range(retries):
        try:
            txt = fn(prompt)
            if txt:
                return txt
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {str(e)[:120]}"
        time.sleep(2 * (attempt + 1))
    if last:
        print(f"      [retry exhausted] {last}")
    return ""


def _judge_one(case_id, candidate_label, judge_name, fn, s, prompt) -> dict:
    raw = _call_with_retry(fn, prompt)
    scores = extract_scores(raw)
    if scores is None:
        print(f"      [{case_id}|{judge_name}|s{s}] PARSE_FAIL")
        return {"case_id": case_id, "candidate": candidate_label,
                "judge": judge_name, "sample": s, "parse_ok": False, "raw": raw}
    row = {"case_id": case_id, "candidate": candidate_label,
           "judge": judge_name, "sample": s, "parse_ok": True}
    for d in DIMS:
        row[d] = scores[d]
    row["A_avg"] = round(sum(scores[d] for d in DIMS_A) / len(DIMS_A), 3)
    row["B_avg"] = round(sum(scores[d] for d in DIMS_B) / len(DIMS_B), 3)
    row["overall"] = round(sum(scores[d] for d in DIMS) / len(DIMS), 3)
    row["comment"] = scores.get("comment", "")
    row["raw"] = raw  # full raw judge response preserved (jsonl only; CSV ignores it)
    return row


def score_candidate(case_id: str, case_input: dict, expert_target: dict,
                    candidate_doc: str, candidate_label: str,
                    samples: int = 1, judges: list | None = None) -> list[dict]:
    """Run the jury (or a subset) — all (judge, sample) calls IN PARALLEL.
    Returns one raw row per (judge, sample)."""
    prompt = build_prompt(case_input, expert_target, candidate_doc)
    tasks = [(jn, fn, s) for jn, fn in (judges or JUDGES) for s in range(samples)]
    with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
        rows = list(ex.map(
            lambda t: _judge_one(case_id, candidate_label, t[0], t[1], t[2], prompt), tasks))
    return rows


def aggregate(rows: list[dict]) -> dict:
    """Aggregate valid rows into per-dimension means + group means."""
    ok = [r for r in rows if r.get("parse_ok")]
    if not ok:
        return {}
    agg = {}
    for d in DIMS:
        vals = [r[d] for r in ok]
        agg[d] = round(sum(vals) / len(vals), 3)
    agg["A_avg"] = round(sum(agg[d] for d in DIMS_A) / len(DIMS_A), 3)
    agg["B_avg"] = round(sum(agg[d] for d in DIMS_B) / len(DIMS_B), 3)
    agg["overall"] = round(sum(agg[d] for d in DIMS) / len(DIMS), 3)
    agg["n_scores"] = len(ok)
    return agg
