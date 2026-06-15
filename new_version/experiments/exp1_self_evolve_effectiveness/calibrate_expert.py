#!/usr/bin/env python3
"""Calibration: score the EXPERT answers themselves under our 9-dim jury rubric.

Purpose:
  1. smoke-test the jury scorer end-to-end
  2. answer "what score does the expert ground-truth get under our rubric?"
  3. diagnose verbosity bias (a terse-but-dense expert should still score well on B)
  4. produce the human reference line for exp1's trajectory chart

Outputs (under results/):
  - expert_calibration_raw.csv   one row per (case, judge, sample), all 9 metrics + A/B/overall
  - expert_calibration_raw.jsonl raw judge outputs (full fidelity)
  - prints per-case + global aggregate

Usage:
  set -a && source ../../../.env && set +a   # or rely on auto .env load
  python3 calibrate_expert.py --cases 6 --samples 1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# make common/ importable
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "common"))

from rubric import DIMS, DIMS_A, DIMS_B, render_target_as_analysis  # noqa: E402
from judges import load_dotenv, score_candidate, aggregate, JUDGES  # noqa: E402

_NEW_VERSION = _HERE.parents[1]  # new_version/
DATASETS = _NEW_VERSION / "datasets"
RESULTS = _HERE / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=6, help="number of cases to score")
    ap.add_argument("--samples", type=int, default=1, help="samples per judge (jury makes >1 redundant)")
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()

    load_dotenv()
    RESULTS.mkdir(parents=True, exist_ok=True)

    inputs = json.loads((DATASETS / "test" / "inputs.json").read_text(encoding="utf-8"))
    out_map = {o["case_id"]: o["target"] for o in
               json.loads((DATASETS / "test" / "outputs.json").read_text(encoding="utf-8"))}

    subset = inputs[args.offset: args.offset + args.cases]
    print(f"Jury: {[j[0] for j in JUDGES]}  | samples/judge={args.samples}")
    print(f"Scoring {len(subset)} expert answers...\n")

    all_rows: list[dict] = []
    raw_path = RESULTS / "expert_calibration_raw.jsonl"
    raw_f = raw_path.open("w", encoding="utf-8")

    per_case_overall = []
    for case in subset:
        cid = case["case_id"]
        target = out_map.get(cid)
        if not target:
            continue
        candidate_doc = render_target_as_analysis(target)  # expert IS the candidate
        rows = score_candidate(cid, case["input"], target, candidate_doc,
                               candidate_label="expert", samples=args.samples)
        for r in rows:
            raw_f.write(json.dumps(r, ensure_ascii=False) + "\n")
        all_rows.extend(rows)
        agg = aggregate(rows)
        if agg:
            per_case_overall.append(agg["overall"])
            print(f"  {cid}: overall={agg['overall']}  A={agg['A_avg']}  B={agg['B_avg']}  (n={agg['n_scores']})")
        else:
            print(f"  {cid}: ALL JUDGES FAILED")
    raw_f.close()

    # --- write CSV (one row per case/judge/sample) ---
    csv_path = RESULTS / "expert_calibration_raw.csv"
    cols = ["case_id", "candidate", "judge", "sample", "parse_ok",
            *DIMS, "A_avg", "B_avg", "overall", "comment"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    # --- global aggregate ---
    print("\n" + "=" * 60)
    glob = aggregate(all_rows)
    if glob:
        print("EXPERT ANSWER — global score under our rubric")
        print("=" * 60)
        print("  A 准确性维度:")
        for d in DIMS_A:
            print(f"    {d:24s} {glob[d]}")
        print("  B 质量维度:")
        for d in DIMS_B:
            print(f"    {d:24s} {glob[d]}")
        print(f"  ----  A_avg={glob['A_avg']}  B_avg={glob['B_avg']}  OVERALL={glob['overall']}")
        # per-judge breakdown (agreement check)
        print("\n  per-judge overall (一致性):")
        for jname, _ in JUDGES:
            jrows = [r for r in all_rows if r.get("parse_ok") and r["judge"] == jname]
            if jrows:
                jo = round(sum(r["overall"] for r in jrows) / len(jrows), 3)
                print(f"    {jname:18s} {jo}  (n={len(jrows)})")
    n_fail = sum(1 for r in all_rows if not r.get("parse_ok"))
    print(f"\n  parse failures: {n_fail}/{len(all_rows)}")
    print(f"\n  CSV : {csv_path}")
    print(f"  JSONL: {raw_path}")


if __name__ == "__main__":
    main()
