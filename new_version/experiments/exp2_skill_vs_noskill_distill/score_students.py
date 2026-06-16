#!/usr/bin/env python3
"""Exp2 scoring — judge the two students' held-out generations with the fixed
3-judge jury + 9-dim rubric (same as exp1, so scores are directly comparable).

Reads results/gen_student_{skill,noskill}_outputs.jsonl (from eval_modal.py),
scores each against the expert target, and prints/saves the skill-vs-noskill
comparison (overall / A / B / per-dimension + delta).

Resumable (append per-case rows; skip already-scored case_ids), concurrent
(cases parallel; each case's 3 judges already run in parallel inside the jury).

    python experiments/exp2_skill_vs_noskill_distill/score_students.py
    python experiments/exp2_skill_vs_noskill_distill/score_students.py --arm skill --parallel 16
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEW_VERSION = HERE.parents[1]
sys.path.insert(0, str(NEW_VERSION / "experiments" / "common"))

from judges import load_dotenv, score_candidate, aggregate, JUDGES  # noqa: E402
from rubric import render_target_as_analysis, DIMS, DIMS_A, DIMS_B  # noqa: E402
from data_split import load_split  # noqa: E402

RESULTS = HERE / "results"


def load_gen(arm: str) -> dict[str, dict]:
    path = RESULTS / f"gen_student_{arm}_outputs.jsonl"
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out[r["case_id"]] = r
    return out


def already_scored(detail_path: Path) -> set[str]:
    done = set()
    if not detail_path.exists():
        return done
    for line in detail_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("parse_ok"):
            done.add(r["case_id"])
    return done


def build_todo(arm: str, expert_map: dict):
    """Return (todo, detail_path). todo = [(arm, cid, output)] still to score."""
    gen = load_gen(arm)
    detail_path = RESULTS / f"gen_student_{arm}_judge_detail.jsonl"
    done = already_scored(detail_path)
    todo, n_parsefail = [], 0
    for cid, r in gen.items():
        if r.get("output") is None:           # student JSON didn't parse → can't score
            n_parsefail += 1
            continue
        if cid in done or cid not in expert_map:
            continue
        todo.append((arm, cid, r["output"]))
    print(f"[{arm}] gen={len(gen)} student_parsefail={n_parsefail} "
          f"already_scored={len(done)} to_score={len(todo)}")
    return todo, detail_path


def aggregate_arm(arm: str) -> dict:
    detail_path = RESULTS / f"gen_student_{arm}_judge_detail.jsonl"
    all_rows = [json.loads(l) for l in detail_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    ok = [r for r in all_rows if r.get("parse_ok")]
    agg = aggregate(ok)
    agg["n_cases"] = len({r["case_id"] for r in ok})
    csv_path = RESULTS / f"gen_student_{arm}_raw.csv"
    cols = ["case_id", "candidate", "judge", "sample", "parse_ok",
            *DIMS, "A_avg", "B_avg", "overall", "comment"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="both", choices=["skill", "noskill", "both"])
    ap.add_argument("--parallel", type=int, default=16)
    args = ap.parse_args()

    load_dotenv()
    _, test, expert_map = load_split(selection_n=40, seed=42)
    input_map = {c["case_id"]: c["input"] for c in test}

    arms = ["skill", "noskill"] if args.arm == "both" else [args.arm]

    # Pool BOTH arms' cases into ONE concurrent queue (no idle gap between arms);
    # each case's 3 judges also run in parallel inside score_candidate.
    pooled, detail_paths = [], {}
    for arm in arms:
        todo, dp = build_todo(arm, expert_map)
        pooled += todo
        detail_paths[arm] = dp
    locks = {arm: threading.Lock() for arm in arms}

    def work(item):
        arm, cid, output = item
        candidate_doc = render_target_as_analysis(output)
        rows = score_candidate(cid, input_map[cid], expert_map[cid],
                               candidate_doc, candidate_label=f"student_{arm}")
        with locks[arm], detail_paths[arm].open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        agg = aggregate(rows)
        return arm, cid, (agg.get("overall", "NA") if agg else "NA")

    print(f"[pool] scoring {len(pooled)} cases across {len(arms)} arm(s) @ parallel={args.parallel}")
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = {ex.submit(work, it): it for it in pooled}
        for i, fut in enumerate(as_completed(futs), 1):
            arm, cid, _ = futs[fut][0], futs[fut][1], None
            try:
                arm, cid, o = fut.result()
                print(f"  [{i}/{len(pooled)}] {arm} {cid} overall={o}")
            except Exception as e:  # noqa: BLE001
                print(f"  [{i}/{len(pooled)}] FAILED: {e}", file=sys.stderr)

    aggs = {arm: aggregate_arm(arm) for arm in arms}

    # comparison
    print("\n" + "=" * 64)
    print(f"EXP2 — student comparison (held-out {len(test)}, jury={[j[0] for j in JUDGES]})")
    print("=" * 64)
    hdr = f"{'metric':<24}" + "".join(f"{a:>12}" for a in arms)
    if len(arms) == 2:
        hdr += f"{'Δ(skill-noskill)':>18}"
    print(hdr)
    metrics = ["overall", "A_avg", "B_avg", *DIMS]
    for m in metrics:
        line = f"{m:<24}" + "".join(f"{aggs[a].get(m, 0):>12.3f}" for a in arms)
        if len(arms) == 2 and "skill" in aggs and "noskill" in aggs:
            d = aggs["skill"].get(m, 0) - aggs["noskill"].get(m, 0)
            line += f"{d:>+18.3f}"
        print(line)
    for a in arms:
        print(f"  [{a}] n_cases={aggs[a].get('n_cases')} n_scores={aggs[a].get('n_scores')}")

    (RESULTS / "compare_summary.json").write_text(
        json.dumps(aggs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved: {RESULTS/'compare_summary.json'}")


if __name__ == "__main__":
    main()
