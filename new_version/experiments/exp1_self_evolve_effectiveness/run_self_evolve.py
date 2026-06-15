#!/usr/bin/env python3
"""Experiment 1 — self-evolve loop (Evaluate -> Diagnose -> Refine -> Validation Gate).

Discipline (SkillOpt-style):
  - 3-way split: selection (gate/diagnose) + held-out test (final report only)
  - validation gate: a refined skill is accepted ONLY if it strictly beats the
    current best on the selection set
  - refine edits ONLY a per-round COPY of SKILL.md; the canonical
    matchmaker/zh/SKILL.md is never mutated. References are held fixed.
  - everything recorded; FULLY RESUMABLE (re-run the same command to continue):
      run/round_NN/{SKILL.md, eval.jsonl, diagnosis.json, refine_raw.txt, round_status.json}
      run/ledger.json
      run/final/eval_<variant>.jsonl
    A case is skipped if already in eval.jsonl; a round is skipped if its
    round_status.json (written last) exists.

Auto-stop (so it can run unattended):
  hard cap --rounds; stop early if accepted gain < --min-gain, or
  --patience consecutive rejected rounds.

Usage (smoke):  python3 run_self_evolve.py --selection 3 --rounds 1
Usage (full):   python3 run_self_evolve.py --selection 40 --rounds 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "common"))

from rubric import DIMS, DIMS_A, DIMS_B, render_target_as_analysis  # noqa: E402
from judges import load_dotenv, score_candidate, call_openai  # noqa: E402
from generator import generate_with_skill, _anthropic_client, GEN_MODEL, SKILL_DIR  # noqa: E402

RUN = _HERE / "results" / "self_evolve"
GATE_JUDGES = [("gpt-5.5", call_openai)]      # cheap single judge inside the loop
OPT_MODEL = GEN_MODEL                          # optimizer (diagnose + refine) = Opus


# --------------------------------------------------------------------------- #
# small IO helpers
# --------------------------------------------------------------------------- #
def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def append_jsonl(p: Path, rec: dict) -> None:
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def opus_text(system: str, user: str, max_tokens: int = 8000) -> str:
    client = _anthropic_client()
    msg = client.messages.create(model=OPT_MODEL, max_tokens=max_tokens,
                                 system=system, messages=[{"role": "user", "content": user}])
    return msg.content[0].text if msg.content else ""


# --------------------------------------------------------------------------- #
# evaluate (resumable, per-case)
# --------------------------------------------------------------------------- #
def eval_skill(skill_text: str, cases: list[dict], expert_map: dict,
               eval_path: Path, judges, label: str) -> tuple[float, list[dict]]:
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {r["case_id"]: r for r in read_jsonl(eval_path)}
    per_case = list(existing.values())
    todo = [c for c in cases if c["case_id"] not in existing]
    print(f"    [eval:{label}] {len(existing)} cached, {len(todo)} to do")
    for c in todo:
        cid = c["case_id"]
        target = expert_map[cid]
        gen, gen_raw = generate_with_skill(c["input"], skill_text)
        if not gen:
            rec = {"case_id": cid, "gen": None, "gen_raw": gen_raw,
                   "judge_rows": [], "case_mean": None}
            append_jsonl(eval_path, rec); per_case.append(rec)
            print(f"      {cid}: GEN_FAIL"); continue
        doc = render_target_as_analysis(gen)
        rows = score_candidate(cid, c["input"], target, doc, label, judges=judges)
        ok = [r for r in rows if r.get("parse_ok")]
        cmean = round(sum(r["overall"] for r in ok) / len(ok), 3) if ok else None
        rec = {"case_id": cid, "gen": gen, "gen_raw": gen_raw,
               "judge_rows": rows, "case_mean": cmean}
        append_jsonl(eval_path, rec); per_case.append(rec)
        print(f"      {cid}: {cmean}")
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


# --------------------------------------------------------------------------- #
# diagnose + refine
# --------------------------------------------------------------------------- #
DIAGNOSE_SYS = "你是一位 prompt/skill 优化专家，擅长诊断婚恋分析 skill 的薄弱环节。"

def diagnose(skill_text: str, per_case: list[dict], dims: dict, out_path: Path) -> dict:
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))
    worst = sorted([r for r in per_case if r["case_mean"] is not None],
                   key=lambda r: r["case_mean"])[:6]
    blocks = []
    for r in worst:
        comments = " | ".join(jr.get("comment", "") for jr in r["judge_rows"] if jr.get("parse_ok"))
        blocks.append(f"### case {r['case_id']} (score={r['case_mean']})\n裁判意见：{comments}")
    user = f"""下面是当前 skill 在 selection 集上的评测结果。各维度平均分：
{json.dumps(dims, ensure_ascii=False)}

最低分的几个 case 及裁判意见：
{chr(10).join(blocks)}

请诊断 skill 最该改进的地方。严格输出 JSON：
{{"weak_dims": ["最弱的1-3个维度"], "root_causes": ["根因1","根因2","..."], "suggestions": ["对 SKILL.md 的具体改进建议1","建议2","..."]}}"""
    raw = opus_text(DIAGNOSE_SYS, user, max_tokens=3000)
    import re
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    diag = json.loads(m.group(0)) if m else {"weak_dims": [], "root_causes": [], "suggestions": [raw[:500]]}
    out_path.write_text(json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")
    return diag


REFINE_SYS = "你是一位 skill 优化专家。你会在保持结构和篇幅克制的前提下，改进给定的 SKILL.md。"

def refine(skill_text: str, diag: dict, raw_path: Path) -> str:
    user = f"""这是当前的 SKILL.md：

<<<SKILL_START>>>
{skill_text}
<<<SKILL_END>>>

诊断结论：
- 薄弱维度：{json.dumps(diag.get('weak_dims', []), ensure_ascii=False)}
- 根因：{json.dumps(diag.get('root_causes', []), ensure_ascii=False)}
- 改进建议：{json.dumps(diag.get('suggestions', []), ensure_ascii=False)}

请据此产出**改进后的完整 SKILL.md**。要求：
- 只针对诊断的薄弱点做有界改进（新增/强化相关心智模型、启发式、输出要求），不要大改无关部分
- 保留 YAML frontmatter 和整体结构，篇幅不要明显膨胀
- 不要解释，直接输出完整 SKILL.md，包在 <<<SKILL_START>>> 和 <<<SKILL_END>>> 之间"""
    raw = opus_text(REFINE_SYS, user, max_tokens=12000)
    raw_path.write_text(raw, encoding="utf-8")
    import re
    m = re.search(r"<<<SKILL_START>>>(.*?)<<<SKILL_END>>>", raw, re.DOTALL)
    return m.group(1).strip() if m else skill_text  # fallback: keep old skill


# --------------------------------------------------------------------------- #
# main loop
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=5, help="hard cap on rounds")
    ap.add_argument("--min-rounds", type=int, default=3,
                    help="auto-stop is suppressed until at least this many rounds have run")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-gain", type=float, default=0.03)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--final-test", action="store_true",
                    help="after the loop, score no-skill/v1/best on held-out test with full jury")
    args = ap.parse_args()

    load_dotenv()
    from data_split import load_split
    selection, test, expert_map = load_split(selection_n=args.selection, seed=args.seed)
    RUN.mkdir(parents=True, exist_ok=True)
    ledger_path = RUN / "ledger.json"

    v1_skill = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")

    # ---- resume: restore ledger or init with v1 baseline ----
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        done_rounds = max([h["round"] for h in ledger["history"]], default=0)
        if ledger.get("stopped") and done_rounds < args.min_rounds:
            ledger["stopped"] = False  # honor the min-rounds floor on resume
            print(f"[resume] clearing early-stop: only {done_rounds} rounds done, min is {args.min_rounds}")
        print(f"[resume] ledger found: best_round={ledger['best_round']} best_score={ledger['best_score']}")
    else:
        print("[round 0] baseline eval of skill v1 on selection...")
        v1_score, v1_pc = eval_skill(v1_skill, selection, expert_map,
                                     RUN / "round_00_v1" / "eval.jsonl", GATE_JUDGES, "v1")
        (RUN / "round_00_v1" / "SKILL.md").write_text(v1_skill, encoding="utf-8")
        ledger = {"best_round": 0, "best_score": v1_score, "best_skill_path": str(RUN / "round_00_v1" / "SKILL.md"),
                  "reject_streak": 0, "stopped": False, "history": [{"round": 0, "score": v1_score, "decision": "baseline"}]}
        ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[round 0] v1 baseline = {v1_score}")

    # ---- self-evolve rounds ----
    for rnd in range(1, args.rounds + 1):
        if any(h["round"] == rnd for h in ledger["history"]):
            continue  # already done (resume)
        if ledger.get("stopped"):
            break
        rdir = RUN / f"round_{rnd:02d}"
        rdir.mkdir(parents=True, exist_ok=True)
        status_path = rdir / "round_status.json"
        if status_path.exists():
            continue

        best_skill = Path(ledger["best_skill_path"]).read_text(encoding="utf-8")
        print(f"\n[round {rnd}] diagnosing (best so far = {ledger['best_score']})...")
        # need best skill's per-case eval for diagnosis
        best_eval_path = Path(ledger["best_skill_path"]).parent / "eval.jsonl"
        _, best_pc = eval_skill(best_skill, selection, expert_map, best_eval_path, GATE_JUDGES, f"best@r{rnd}")
        diag = diagnose(best_skill, best_pc, dim_means(best_pc), rdir / "diagnosis.json")
        print(f"[round {rnd}] weak_dims={diag.get('weak_dims')}")

        cand_skill = refine(best_skill, diag, rdir / "refine_raw.txt")
        (rdir / "SKILL.md").write_text(cand_skill, encoding="utf-8")

        print(f"[round {rnd}] validation-gate eval of candidate...")
        cand_score, _ = eval_skill(cand_skill, selection, expert_map, rdir / "eval.jsonl", GATE_JUDGES, f"cand@r{rnd}")
        gain = round(cand_score - ledger["best_score"], 3)
        accepted = cand_score > ledger["best_score"]

        if accepted:
            ledger["best_round"] = rnd
            ledger["best_score"] = cand_score
            ledger["best_skill_path"] = str(rdir / "SKILL.md")
            ledger["reject_streak"] = 0
            decision = "accept"
        else:
            ledger["reject_streak"] += 1
            decision = "reject"
        ledger["history"].append({"round": rnd, "score": cand_score, "gain": gain, "decision": decision})

        # stopping rule — suppressed until the min-rounds floor is reached
        stop_reason = None
        if rnd >= args.min_rounds:
            if accepted and gain < args.min_gain:
                stop_reason = f"gain {gain} < min_gain {args.min_gain}"
            elif ledger["reject_streak"] >= args.patience:
                stop_reason = f"{ledger['reject_streak']} consecutive rejects"
        if stop_reason:
            ledger["stopped"] = True
            ledger["stop_reason"] = stop_reason

        status = {"round": rnd, "cand_score": cand_score, "best_score": ledger["best_score"],
                  "gain": gain, "decision": decision, "stop_reason": stop_reason}
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[round {rnd}] cand={cand_score} gain={gain} -> {decision}" + (f" | STOP: {stop_reason}" if stop_reason else ""))
        if stop_reason:
            break

    print(f"\n[loop done] best_round={ledger['best_round']} best_score={ledger['best_score']}")
    print("  trajectory:", [(h['round'], h['score'], h.get('decision')) for h in ledger["history"]])

    # ---- final held-out test with full jury ----
    if args.final_test:
        from judges import JUDGES
        fdir = RUN / "final"; fdir.mkdir(parents=True, exist_ok=True)
        best_skill = Path(ledger["best_skill_path"]).read_text(encoding="utf-8")
        print(f"\n[final] held-out test ({len(test)} cases), full jury {[j[0] for j in JUDGES]}")
        # v1 and best use generate_with_skill; noskill handled below
        variants = {"v1": v1_skill, "best": best_skill}
        for name, sk in variants.items():
            m, _ = eval_skill(sk, test, expert_map, fdir / f"eval_{name}.jsonl", JUDGES, f"test:{name}")
            print(f"  [final] {name}: {m}")
        # noskill on test
        from generator import generate as gen_cond
        ns_path = fdir / "eval_noskill.jsonl"
        existing = {r["case_id"] for r in read_jsonl(ns_path)}
        ns_means = [r["case_mean"] for r in read_jsonl(ns_path) if r.get("case_mean") is not None]
        for c in test:
            if c["case_id"] in existing:
                continue
            gen, gen_raw = gen_cond(c["input"], "noskill")
            if not gen:
                append_jsonl(ns_path, {"case_id": c["case_id"], "gen": None, "gen_raw": gen_raw,
                                       "judge_rows": [], "case_mean": None}); continue
            doc = render_target_as_analysis(gen)
            rows = score_candidate(c["case_id"], c["input"], expert_map[c["case_id"]], doc, "noskill", judges=JUDGES)
            ok = [r for r in rows if r.get("parse_ok")]
            cm = round(sum(r["overall"] for r in ok) / len(ok), 3) if ok else None
            append_jsonl(ns_path, {"case_id": c["case_id"], "gen": gen, "gen_raw": gen_raw,
                                   "judge_rows": rows, "case_mean": cm})
            if cm is not None:
                ns_means.append(cm)
        if ns_means:
            print(f"  [final] noskill: {round(sum(ns_means)/len(ns_means),3)}")


if __name__ == "__main__":
    main()
