#!/usr/bin/env python3
"""Experiment 1 (nuwa-style) — bootstrap a SKILL.md FROM ZERO via self-evolve.

Idea (matches nuwa-skill: references = source material -> distill a compact SKILL.md):
  - references/ (the 191-case distilled knowledge) are SOURCE material the optimizer
    reads while WRITING the skill; they are NOT injected at inference.
  - inference uses the compact SKILL.md ONLY (cheap), or nothing (no-skill baseline).

Loop:
  round 0 : NO skill (pure Opus) -> judge -> collect ALL critiques        (floor)
  round 1 : optimizer reads [references + round-0 outputs + ALL critiques]
            -> AUTHORS SKILL.md v1 from scratch -> validation gate
  round N : optimizer reads [references + current-skill outputs + ALL critiques]
            -> revises SKILL.md -> validation gate
  accept a candidate only if it strictly beats current best on selection.

Discipline: 3-way split, validation gate, per-round snapshots, FULLY RESUMABLE,
min-rounds floor + auto-stop. Never touches matchmaker/zh/SKILL.md.

Usage (smoke): python3 run_bootstrap.py --selection 3 --rounds 2 --min-rounds 1
Usage (full):  python3 run_bootstrap.py --selection 40 --rounds 5 --min-rounds 3 --final-test
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "common"))

from rubric import DIMS, DIMS_A, DIMS_B, render_target_as_analysis  # noqa: E402
from judges import load_dotenv, score_candidate, JUDGES  # noqa: E402
from generator import (generate, generate_with_system, _anthropic_client,  # noqa: E402
                       GEN_MODEL, load_references_text)

RUN = _HERE / "results" / "bootstrap"
GATE_JUDGES = JUDGES   # full 3-judge jury inside the loop too (lower gate noise, consistent with final)
OPT_MODEL = GEN_MODEL                        # optimizer (skill author) = Opus


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def append_jsonl(p: Path, rec: dict) -> None:
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def opus_text(system: str, user: str, max_tokens: int = 14000, retries: int = 3) -> str:
    """Author/refine call. Resilient: retries on timeout/error instead of crashing
    the whole run (a single failed author call must not kill the loop)."""
    import time as _t
    last = None
    for attempt in range(1, retries + 1):
        try:
            client = _anthropic_client()
            msg = client.messages.create(model=OPT_MODEL, max_tokens=max_tokens,
                                         system=system, messages=[{"role": "user", "content": user}])
            return msg.content[0].text if msg.content else ""
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"      [opus_text retry {attempt}/{retries}] {type(e).__name__}", flush=True)
            _t.sleep(5 * attempt)
    print(f"      [opus_text FAILED after {retries}] {type(last).__name__}", flush=True)
    return ""  # caller falls back (keeps current skill); never crashes the loop


# --------------------------------------------------------------------------- #
# evaluate (resumable). skill_text=None -> no-skill baseline; else skill-only.
# --------------------------------------------------------------------------- #
def _eval_one(c: dict, skill_text: str | None, expert_map: dict, judges, label: str) -> dict:
    cid = c["case_id"]
    if skill_text is None:
        gen, gen_raw = generate(c["input"], "noskill")
    else:
        gen, gen_raw = generate_with_system(c["input"], skill_text)  # skill-only, no refs
    if not gen:
        return {"case_id": cid, "gen": None, "gen_raw": gen_raw, "judge_rows": [], "case_mean": None}
    doc = render_target_as_analysis(gen)
    rows = score_candidate(cid, c["input"], expert_map[cid], doc, label, judges=judges)
    ok = [r for r in rows if r.get("parse_ok")]
    cmean = round(sum(r["overall"] for r in ok) / len(ok), 3) if ok else None
    return {"case_id": cid, "gen": gen, "gen_raw": gen_raw, "judge_rows": rows, "case_mean": cmean}


def eval_cases(skill_text: str | None, cases: list[dict], expert_map: dict,
               eval_path: Path, judges, label: str, concurrency: int = 8) -> tuple[float, list[dict]]:
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {r["case_id"]: r for r in read_jsonl(eval_path)}
    per_case = list(existing.values())
    todo = [c for c in cases if c["case_id"] not in existing]
    print(f"    [eval:{label}] {len(existing)} cached, {len(todo)} to do (concurrency={concurrency})")
    # cases run concurrently; results appended in the main thread as they finish
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(_eval_one, c, skill_text, expert_map, judges, label): c for c in todo}
        for fut in as_completed(futs):
            rec = fut.result()
            append_jsonl(eval_path, rec)   # single-consumer (main thread) -> safe
            per_case.append(rec)
            print(f"      {rec['case_id']}: {rec['case_mean']}")
    means = [r["case_mean"] for r in per_case if r["case_mean"] is not None]
    return (round(sum(means) / len(means), 3) if means else 0.0), per_case


def dim_means(per_case: list[dict]) -> dict:
    agg = {d: [] for d in DIMS}
    for r in per_case:
        for jr in r.get("judge_rows", []):
            if jr.get("parse_ok"):
                for d in DIMS:
                    agg[d].append(jr[d])
    return {d: (round(sum(v) / len(v), 3) if v else None) for d, v in agg.items()}


def all_critiques(per_case: list[dict]) -> str:
    """ALL cases' generated gist + every judge comment + dim scores (the user wants
    the optimizer to see every comment, not just the worst cases)."""
    blocks = []
    for r in per_case:
        gen = r.get("gen") or {}
        mi = gen.get("matchmaking_intelligence", {})
        gist = f"核心矛盾={mi.get('core_conflict','?')[:60]} | 策略={mi.get('expert_strategy','?')[:60]}"
        comments = " ; ".join(jr.get("comment", "") for jr in r["judge_rows"] if jr.get("parse_ok"))
        blocks.append(f"- {r['case_id']} (score={r['case_mean']}): {gist}\n  评委意见: {comments}")
    return "\n".join(blocks)


# --------------------------------------------------------------------------- #
# author / evolve the skill (reads references as source material)
# --------------------------------------------------------------------------- #
AUTHOR_SYS = ("你是一位 skill 作者与优化专家。你擅长把领域调研资料(references)蒸馏成"
              "一份紧凑、可复用的婚恋市场分析 SKILL.md(认知操作系统:心智模型、启发式、"
              "人物原型、反模式、输出格式)。produce 的 skill 会被单独注入(推理时没有 references),"
              "所以必须把关键知识内化进 skill 本身,且保持紧凑。")

_REFS = None
def refs_source() -> str:
    global _REFS
    if _REFS is None:
        _REFS = load_references_text()
    return _REFS


def author_skill(current_skill: str | None, per_case: list[dict], dims: dict, raw_path: Path) -> str:
    refs = refs_source()
    crit = all_critiques(per_case)
    if current_skill is None:
        task = f"""现在还【没有 skill】。下面是模型在无 skill 时对 selection 案例的输出与**全部评委意见**。
请基于 references(源材料)+ 这些失败模式,从零写出第一版 SKILL.md。"""
        cur_block = ""
    else:
        task = f"""下面是【当前 SKILL.md】、用它生成的输出、以及**全部评委意见**。
请针对评委指出的薄弱点改进 SKILL.md(可新增/强化心智模型、启发式、输出要求),保持结构与篇幅克制。"""
        cur_block = f"\n## 当前 SKILL.md\n<<<CUR_START>>>\n{current_skill}\n<<<CUR_END>>>\n"
    user = f"""{task}

## references(源材料,可蒸馏但不要整段照搬)
{refs}
{cur_block}
## 各维度平均分(1-5)
{json.dumps(dims, ensure_ascii=False)}

## selection 全部案例的输出与评委意见
{crit}

请输出**完整的 SKILL.md**,包在 <<<SKILL_START>>> 和 <<<SKILL_END>>> 之间。要求:含 YAML frontmatter、心智模型/启发式/人物原型/反模式/输出格式;紧凑(目标 < 16K 字符);针对上面评委意见对症下药;不要解释。"""
    raw = opus_text(AUTHOR_SYS, user, max_tokens=14000)
    raw_path.write_text(raw, encoding="utf-8")
    m = re.search(r"<<<SKILL_START>>>(.*?)<<<SKILL_END>>>", raw, re.DOTALL)
    return m.group(1).strip() if m else (current_skill or "")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--min-rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-gain", type=float, default=0.03)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=8, help="cases evaluated in parallel")
    ap.add_argument("--final-test", action="store_true")
    args = ap.parse_args()
    C = args.concurrency

    load_dotenv()
    from data_split import load_split
    selection, test, expert_map = load_split(selection_n=args.selection, seed=args.seed)
    RUN.mkdir(parents=True, exist_ok=True)
    ledger_path = RUN / "ledger.json"

    # ---- round 0: no-skill baseline (resumable) ----
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        done = max([h["round"] for h in ledger["history"]], default=0)
        if ledger.get("stopped") and done < args.min_rounds:
            ledger["stopped"] = False
            print(f"[resume] clearing early-stop ({done} < min {args.min_rounds})")
        print(f"[resume] best_round={ledger['best_round']} best_score={ledger['best_score']}")
    else:
        print("[round 0] no-skill baseline on selection...")
        ns_score, ns_pc = eval_cases(None, selection, expert_map,
                                     RUN / "round_00_noskill" / "eval.jsonl", GATE_JUDGES, "noskill", concurrency=C)
        ledger = {"best_round": 0, "best_score": ns_score, "best_skill_path": None,
                  "noskill_score": ns_score, "reject_streak": 0, "stopped": False,
                  "history": [{"round": 0, "score": ns_score, "decision": "noskill_baseline"}]}
        ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[round 0] no-skill = {ns_score}")

    # ---- evolve rounds ----
    for rnd in range(1, args.rounds + 1):
        if any(h["round"] == rnd for h in ledger["history"]):
            continue
        if ledger.get("stopped"):
            break
        rdir = RUN / f"round_{rnd:02d}"
        rdir.mkdir(parents=True, exist_ok=True)
        if (rdir / "round_status.json").exists():
            continue

        # current best's per-case eval (source of critiques for authoring)
        if ledger["best_skill_path"] is None:           # still on no-skill floor
            base_eval = RUN / "round_00_noskill" / "eval.jsonl"
            _, base_pc = eval_cases(None, selection, expert_map, base_eval, GATE_JUDGES, "noskill", concurrency=C)
            cur_skill = None
        else:
            best_dir = Path(ledger["best_skill_path"]).parent
            _, base_pc = eval_cases(Path(ledger["best_skill_path"]).read_text(encoding="utf-8"),
                                    selection, expert_map, best_dir / "eval.jsonl", GATE_JUDGES, f"best@r{rnd}", concurrency=C)
            cur_skill = Path(ledger["best_skill_path"]).read_text(encoding="utf-8")

        print(f"\n[round {rnd}] authoring/refining skill from {len(base_pc)} cases' critiques...")
        cand_skill = author_skill(cur_skill, base_pc, dim_means(base_pc), rdir / "author_raw.txt")
        (rdir / "SKILL.md").write_text(cand_skill, encoding="utf-8")
        print(f"[round {rnd}] candidate skill written ({len(cand_skill)} chars). gate eval...")

        cand_score, _ = eval_cases(cand_skill, selection, expert_map, rdir / "eval.jsonl", GATE_JUDGES, f"cand@r{rnd}", concurrency=C)
        gain = round(cand_score - ledger["best_score"], 3)
        accepted = cand_score > ledger["best_score"]
        if accepted:
            ledger.update(best_round=rnd, best_score=cand_score, best_skill_path=str(rdir / "SKILL.md"))
            ledger["reject_streak"] = 0; decision = "accept"
        else:
            ledger["reject_streak"] += 1; decision = "reject"
        ledger["history"].append({"round": rnd, "score": cand_score, "gain": gain, "decision": decision})

        stop_reason = None
        if rnd >= args.min_rounds:
            if accepted and gain < args.min_gain:
                stop_reason = f"gain {gain} < {args.min_gain}"
            elif ledger["reject_streak"] >= args.patience:
                stop_reason = f"{ledger['reject_streak']} consecutive rejects"
        if stop_reason:
            ledger["stopped"] = True; ledger["stop_reason"] = stop_reason
        (rdir / "round_status.json").write_text(json.dumps(
            {"round": rnd, "cand_score": cand_score, "best_score": ledger["best_score"],
             "gain": gain, "decision": decision, "stop_reason": stop_reason}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[round {rnd}] cand={cand_score} gain={gain} -> {decision}" + (f" | STOP: {stop_reason}" if stop_reason else ""))
        if stop_reason:
            break

    print(f"\n[loop done] best_round={ledger['best_round']} best_score={ledger['best_score']}")
    print("  trajectory:", [(h['round'], h['score'], h.get('decision')) for h in ledger["history"]])

    # ---- final held-out test with full jury: no-skill vs best-auto-skill ----
    if args.final_test:
        from judges import JUDGES
        fdir = RUN / "final"; fdir.mkdir(parents=True, exist_ok=True)
        print(f"\n[final] held-out test ({len(test)} cases), full jury {[j[0] for j in JUDGES]}")
        m_ns, _ = eval_cases(None, test, expert_map, fdir / "eval_noskill.jsonl", JUDGES, "test:noskill", concurrency=C)
        print(f"  [final] no-skill: {m_ns}")
        if ledger["best_skill_path"]:
            best = Path(ledger["best_skill_path"]).read_text(encoding="utf-8")
            m_best, _ = eval_cases(best, test, expert_map, fdir / "eval_best.jsonl", JUDGES, "test:best", concurrency=C)
            print(f"  [final] best-auto-skill (round {ledger['best_round']}): {m_best}")


if __name__ == "__main__":
    main()
