#!/usr/bin/env python3
"""Experiment 1 — v2 refiner: per-dimension BOUNDED edits + no-regression gate.

Differences from v1 (run_bootstrap.py):
  - round 1 = genesis (author skill from scratch from no-skill critiques) — SAME as v1
  - rounds 2+ = TARGETED coordinate-ascent:
      * pick the K weakest dimensions on selection
      * for each, make a SMALL bounded edit aimed ONLY at that dimension
        (the rest of SKILL.md must stay verbatim) — K candidates IN PARALLEL
      * evaluate all K candidates (flattened into ONE thread pool -> max parallelism)
      * NO-REGRESSION GATE: accept the best candidate iff overall strictly up AND
        no dimension drops by more than --tol
      * rejected-edit buffer per dimension; a dim that fails twice is "exhausted"
  - stop when all dims exhausted / patience consecutive rejected rounds / cap

Maximizes parallelism: K candidate edits authored in parallel; all
(candidate x case) evals in one pool; the 3 judges per case also run in parallel.

FULLY RESUMABLE (per-candidate eval.jsonl skips done cases; round_status written last).

Usage: python3 run_bootstrap_v2.py --selection 40 --rounds 8 --concurrency 16 --final-test
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
from generator import generate, generate_with_system, _anthropic_client, GEN_MODEL, load_references_text  # noqa: E402
# reuse v1 helpers
from run_bootstrap import (opus_text, author_skill, dim_means, all_critiques,  # noqa: E402
                           read_jsonl, append_jsonl, _eval_one, refs_source)

RUN = _HERE / "results" / "refine_v2" / "run"
GATE_JUDGES = JUDGES
K_CANDIDATES = 3
TOL = 0.15  # a dimension may not drop by more than this for a targeted edit to be accepted


# --------------------------------------------------------------------------- #
# parallel multi-candidate eval (flatten candidate x case into one pool)
# --------------------------------------------------------------------------- #
def eval_one_skill(skill_text, cases, expert_map, eval_path, label, concurrency):
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {r["case_id"]: r for r in read_jsonl(eval_path)}
    per_case = list(existing.values())
    todo = [c for c in cases if c["case_id"] not in existing]
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(_eval_one, c, skill_text, expert_map, GATE_JUDGES, label): c for c in todo}
        for fut in as_completed(futs):
            rec = fut.result()
            append_jsonl(eval_path, rec); per_case.append(rec)
    means = [r["case_mean"] for r in per_case if r["case_mean"] is not None]
    overall = round(sum(means) / len(means), 3) if means else 0.0
    return overall, per_case


# --------------------------------------------------------------------------- #
# targeted bounded edit (one dimension)
# --------------------------------------------------------------------------- #
TARGETED_SYS = ("你是 skill 优化专家。你只做**有界的、针对单一评分维度**的小幅编辑:"
                "新增或强化与该维度相关的一小段(规则/原型/输出要求),其余部分必须**逐字保留**。")

DIM_DESC = {
    "conflict_insight": "识别与专家等价的核心矛盾及结构性根源",
    "market_read_accuracy": "市场估值扎实落在输入事实(收入/学历/外形/年龄/城市)且方向正确",
    "strategy_direction": "策略方向与专家一致",
    "target_portrait_match": "理想对象画像具体且与专家一致(圈层/年龄/性格/家庭)",
    "persona_read": "性格与行为逻辑解读到位、抓深层动机",
    "logic_depth": "推理链有深度且自洽",
    "insight_nonobviousness": "给出反直觉的非显然洞察",
    "risk_anti_pattern": "识别关键风险/陷阱(被收割/面子驱动/期望错配)",
    "actionability": "建议具体可落地、不啰嗦",
}


def case_dim_score(rec, dim):
    js = [jr[dim] for jr in rec.get("judge_rows", []) if jr.get("parse_ok")]
    return sum(js) / len(js) if js else None


def targeted_refine(skill_text, dim, best_pc, rejected_notes, raw_path):
    worst = sorted([r for r in best_pc if case_dim_score(r, dim) is not None],
                   key=lambda r: case_dim_score(r, dim))[:5]
    blocks = []
    for r in worst:
        comments = " ; ".join(jr.get("comment", "") for jr in r["judge_rows"] if jr.get("parse_ok"))
        blocks.append(f"- {r['case_id']} ({dim}={round(case_dim_score(r, dim),2)}): {comments}")
    rej = ("\n## 这些改法之前试过且失败(别重复):\n" + "\n".join(f"- {n}" for n in rejected_notes)) if rejected_notes else ""
    user = f"""当前 SKILL.md:
<<<CUR_START>>>
{skill_text}
<<<CUR_END>>>

现在**只提升这一个维度**:`{dim}`（{DIM_DESC.get(dim, dim)}）——它是当前最弱项之一。

该维度得分最低的 case + 评委针对它的意见:
{chr(10).join(blocks)}
{rej}
## 可参考的源材料片段
{refs_source()[:20000]}

请做**一处有界编辑**专门提升 `{dim}`:新增或强化与该维度直接相关的一小段,
其余内容**逐字不变**。输出**完整 SKILL.md**(包在 <<<SKILL_START>>> 和 <<<SKILL_END>>> 之间),
除目标改动外必须与原文一致;不要解释。"""
    raw = opus_text(TARGETED_SYS, user, max_tokens=14000)
    raw_path.write_text(raw, encoding="utf-8")
    m = re.search(r"<<<SKILL_START>>>(.*?)<<<SKILL_END>>>", raw, re.DOTALL)
    return m.group(1).strip() if m else skill_text


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--k", type=int, default=K_CANDIDATES)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--final-test", action="store_true")
    args = ap.parse_args()
    C = args.concurrency

    load_dotenv()
    from data_split import load_split
    selection, test, expert_map = load_split(selection_n=args.selection, seed=args.seed)
    RUN.mkdir(parents=True, exist_ok=True)
    ledger_path = RUN / "ledger.json"

    # ---- round 0: no-skill baseline ----
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        print(f"[resume] best_round={ledger['best_round']} best_score={ledger['best_score']}")
    else:
        print("[round 0] no-skill baseline...")
        ns, ns_pc = eval_one_skill(None, selection, expert_map, RUN / "round_00_noskill" / "eval.jsonl", "noskill", C)
        ledger = {"best_round": 0, "best_score": ns, "best_skill_path": None, "noskill_score": ns,
                  "rejected": {d: [] for d in DIMS}, "stopped": False,
                  "history": [{"round": 0, "score": ns, "decision": "noskill_baseline", "dims": dim_means(ns_pc)}]}
        ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[round 0] no-skill = {ns}")

    fail_streak = 0
    for rnd in range(1, args.rounds + 1):
        if any(h["round"] == rnd for h in ledger["history"]):
            continue
        if ledger.get("stopped"):
            break
        rdir = RUN / f"round_{rnd:02d}"; rdir.mkdir(parents=True, exist_ok=True)
        if (rdir / "round_status.json").exists():
            continue

        # current best per-case eval
        if ledger["best_skill_path"] is None:
            best_eval = RUN / "round_00_noskill" / "eval.jsonl"
            _, best_pc = eval_one_skill(None, selection, expert_map, best_eval, "noskill", C)
            best_skill = None
        else:
            bdir = Path(ledger["best_skill_path"]).parent
            best_skill = Path(ledger["best_skill_path"]).read_text(encoding="utf-8")
            _, best_pc = eval_one_skill(best_skill, selection, expert_map, bdir / "eval.jsonl", "best", C)
        best_dims = dim_means(best_pc)
        is_genesis = best_skill is None

        # ---- round 1: genesis (author from scratch) ----
        if best_skill is None:
            print(f"\n[round {rnd}] GENESIS: author first skill from no-skill critiques...")
            candidates = {"genesis": author_skill(None, best_pc, best_dims, rdir / "author_genesis.txt")}
            targeted_dims = {"genesis": None}
        else:
            # ---- rounds 2+: pick K weakest non-exhausted dims, bounded-edit each (parallel) ----
            exhausted = {d for d in DIMS if len(ledger["rejected"][d]) >= 2}
            ranked = sorted([d for d in DIMS if d not in exhausted], key=lambda d: best_dims[d] or 5)
            target_dims = ranked[:args.k]
            if not target_dims:
                ledger["stopped"] = True; ledger["stop_reason"] = "all dims exhausted"
                ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
                print("[stop] all dims exhausted"); break
            print(f"\n[round {rnd}] targeting weakest dims {[(d, best_dims[d]) for d in target_dims]} (parallel bounded edits)")
            candidates, targeted_dims = {}, {}
            with ThreadPoolExecutor(max_workers=len(target_dims)) as ex:
                futs = {ex.submit(targeted_refine, best_skill, d, best_pc, ledger["rejected"][d],
                                  rdir / f"edit_{d}.txt"): d for d in target_dims}
                for fut in as_completed(futs):
                    d = futs[fut]
                    candidates[d] = fut.result(); targeted_dims[d] = d

        # ---- evaluate all candidates (each in its own resumable file) ----
        for lbl, sk in candidates.items():
            (rdir / f"SKILL_{lbl}.md").write_text(sk, encoding="utf-8")
        cand_scores = {}
        for lbl, sk in candidates.items():
            ov, pc = eval_one_skill(sk, selection, expert_map, rdir / f"eval_{lbl}.jsonl", f"cand:{lbl}", C)
            cand_scores[lbl] = (ov, dim_means(pc))
            print(f"    candidate[{lbl}] overall={ov}")

        # ---- no-regression gate: pick best candidate that beats best & regresses no dim > tol ----
        def passes(lbl):
            ov, dd = cand_scores[lbl]
            if ov <= ledger["best_score"]:
                return False
            if is_genesis:
                return True  # first skill: any overall improvement over no-skill is accepted
            return all((dd[d] - best_dims[d]) >= -args.tol for d in DIMS if dd[d] is not None and best_dims[d] is not None)
        winners = [lbl for lbl in candidates if passes(lbl)]
        winner = max(winners, key=lambda lbl: cand_scores[lbl][0]) if winners else None

        if winner is not None:
            ov = cand_scores[winner][0]
            ledger.update(best_round=rnd, best_score=ov, best_skill_path=str(rdir / f"SKILL_{winner}.md"))
            decision = f"accept[{winner}]"; fail_streak = 0
        else:
            decision = "reject"; fail_streak += 1
            for lbl, d in targeted_dims.items():
                if d is not None:
                    ledger["rejected"][d].append(f"r{rnd}: overall={cand_scores[lbl][0]} 未通过无回归门")

        gain = round((cand_scores[winner][0] if winner else max(c[0] for c in cand_scores.values())) - ledger["history"][-1]["score"], 3) if cand_scores else 0
        ledger["history"].append({"round": rnd, "score": ledger["best_score"], "decision": decision,
                                  "candidates": {l: cand_scores[l][0] for l in candidates}})
        stop_reason = None
        if rnd >= 3 and fail_streak >= args.patience:
            stop_reason = f"{fail_streak} consecutive rejected rounds"; ledger["stopped"] = True; ledger["stop_reason"] = stop_reason
        (rdir / "round_status.json").write_text(json.dumps(
            {"round": rnd, "decision": decision, "best_score": ledger["best_score"],
             "candidates": {l: cand_scores[l][0] for l in candidates}, "stop_reason": stop_reason},
            ensure_ascii=False, indent=2), encoding="utf-8")
        ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[round {rnd}] -> {decision} | best={ledger['best_score']}" + (f" | STOP {stop_reason}" if stop_reason else ""))
        if stop_reason:
            break

    print(f"\n[loop done] best_round={ledger['best_round']} best_score={ledger['best_score']}")
    print("  trajectory:", [(h['round'], h['score'], h['decision']) for h in ledger["history"]])

    if args.final_test:
        fdir = RUN / "final"; fdir.mkdir(parents=True, exist_ok=True)
        print(f"\n[final] held-out test ({len(test)}), full jury")
        m_ns, _ = eval_one_skill(None, test, expert_map, fdir / "eval_noskill.jsonl", "test:noskill", C)
        print(f"  no-skill: {m_ns}")
        if ledger["best_skill_path"]:
            best = Path(ledger["best_skill_path"]).read_text(encoding="utf-8")
            m_b, _ = eval_one_skill(best, test, expert_map, fdir / "eval_best.jsonl", "test:best", C)
            print(f"  best-auto-skill (round {ledger['best_round']}): {m_b}")


if __name__ == "__main__":
    main()
