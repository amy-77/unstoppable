#!/usr/bin/env python3
"""Exp3 — score every (size, mode) generation with the fixed 3-judge jury, then
build the scaling table: per size  student vs baseline  → distillation lift.

Local only (judge APIs, no GPU). Resumable (skip already-scored case_ids per
size/mode), and ALL (size, mode, case) are pooled into one concurrent queue
(parallel 16; each case's 3 judges run in parallel inside the jury).

    python experiments/exp3_scaling_law/score.py                 # all sizes found in results/
    python experiments/exp3_scaling_law/score.py --sizes 0.6B,4B --parallel 16
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEW_VERSION = HERE.parents[1]
sys.path.insert(0, str(NEW_VERSION / "experiments" / "common"))

from judges import load_dotenv, score_candidate, aggregate, JUDGES  # noqa: E402
from rubric import render_target_as_analysis, DIMS                  # noqa: E402
from data_split import load_split                                   # noqa: E402

from config import PHASE1, MODES                                    # noqa: E402

RESULTS = HERE / "results"


def _done_ids(detail: Path) -> set[str]:
    done = set()
    if detail.exists():
        for line in detail.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("parse_ok"):
                    done.add(r["case_id"])
    return done


def _build_todo(size: str, mode: str, expert_map: dict):
    gen_path = RESULTS / f"gen_{size}_{mode}_outputs.jsonl"
    if not gen_path.exists():
        return [], None, 0
    detail = RESULTS / f"judge_{size}_{mode}.jsonl"
    done = _done_ids(detail)
    todo, pf = [], 0
    for line in gen_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("output") is None:
            pf += 1
            continue
        cid = r["case_id"]
        if cid in done or cid not in expert_map:
            continue
        todo.append((size, mode, cid, r["output"]))
    print(f"[{size}:{mode}] parsefail={pf} done={len(done)} todo={len(todo)}")
    return todo, detail, pf


def _agg(size: str, mode: str) -> dict:
    detail = RESULTS / f"judge_{size}_{mode}.jsonl"
    if not detail.exists():
        return {}
    rows = [json.loads(l) for l in detail.read_text(encoding="utf-8").splitlines() if l.strip()]
    ok = [r for r in rows if r.get("parse_ok")]
    a = aggregate(ok)
    if a:
        a["n_cases"] = len({r["case_id"] for r in ok})
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="")
    ap.add_argument("--parallel", type=int, default=16)
    args = ap.parse_args()

    load_dotenv()
    _, test, expert_map = load_split(selection_n=40, seed=42)
    input_map = {c["case_id"]: c["input"] for c in test}
    sizes = [x.strip() for x in args.sizes.split(",") if x.strip()] or PHASE1

    pooled, details = [], {}
    for size in sizes:
        for mode in MODES:
            todo, detail, _ = _build_todo(size, mode, expert_map)
            if detail is None:
                continue
            pooled += todo
            details[(size, mode)] = (detail, threading.Lock())

    def work(item):
        size, mode, cid, output = item
        doc = render_target_as_analysis(output)
        rows = score_candidate(cid, input_map[cid], expert_map[cid], doc,
                               candidate_label=f"{size}_{mode}")
        detail, lock = details[(size, mode)]
        with lock, detail.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return size, mode, cid

    print(f"[pool] scoring {len(pooled)} (size,mode,case) @ parallel={args.parallel}")
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = [ex.submit(work, it) for it in pooled]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                size, mode, cid = fut.result()
                print(f"  [{i}/{len(pooled)}] {size}:{mode} {cid}")
            except Exception as e:  # noqa: BLE001
                print(f"  [{i}/{len(pooled)}] FAILED: {e}", file=sys.stderr)

    # ----- scaling table: student vs baseline → lift, per size
    table = {}
    for size in sizes:
        st, bl = _agg(size, "student"), _agg(size, "baseline")
        table[size] = {
            "student_overall": st.get("overall"), "student_A": st.get("A_avg"), "student_B": st.get("B_avg"),
            "baseline_overall": bl.get("overall"), "baseline_A": bl.get("A_avg"), "baseline_B": bl.get("B_avg"),
            "lift_overall": (round(st["overall"] - bl["overall"], 3)
                             if st.get("overall") is not None and bl.get("overall") is not None else None),
            "n_student": st.get("n_cases"), "n_baseline": bl.get("n_cases"),
        }

    print("\n" + "=" * 70)
    print(f"EXP3 — scaling (held-out 60, jury={[j[0] for j in JUDGES]})")
    print("=" * 70)
    print(f"{'size':<8}{'student':>10}{'baseline':>10}{'lift':>10}")
    for size in sizes:
        t = table[size]
        so = f"{t['student_overall']:.3f}" if t['student_overall'] is not None else "—"
        bo = f"{t['baseline_overall']:.3f}" if t['baseline_overall'] is not None else "—"
        lf = f"{t['lift_overall']:+.3f}" if t['lift_overall'] is not None else "—"
        print(f"{size:<8}{so:>10}{bo:>10}{lf:>10}")

    # MERGE into the existing summary (never drop sizes scored in other runs).
    summary_path = RESULTS / "scaling_summary.json"
    merged = {}
    if summary_path.exists():
        try:
            merged = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            merged = {}
    merged.update(table)                       # only the sizes scored this run
    # keep a stable size order
    order = ["0.6B", "1.7B", "4B", "8B", "14B", "32B"]
    merged = {k: merged[k] for k in order if k in merged} | {
        k: v for k, v in merged.items() if k not in order}
    summary_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved (merged, {len(merged)} sizes): {summary_path}")


if __name__ == "__main__":
    main()
