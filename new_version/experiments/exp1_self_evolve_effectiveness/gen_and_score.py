#!/usr/bin/env python3
"""Generate Opus analyses (no-skill or skill-v1) and score them with the jury.

Usage:
  python3 gen_and_score.py --condition noskill --cases 3
  python3 gen_and_score.py --condition skillv1 --cases 3

Outputs (results/):
  - gen_<condition>_raw.csv     one row per (case, judge), 9 metrics + A/B/overall
  - gen_<condition>_outputs.jsonl  the generated analyses (full fidelity)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "common"))

from rubric import DIMS, DIMS_A, DIMS_B, render_target_as_analysis  # noqa: E402
from judges import load_dotenv, score_candidate, aggregate, JUDGES  # noqa: E402
from generator import generate, GEN_MODEL  # noqa: E402

_NEW_VERSION = _HERE.parents[1]
DATASETS = _NEW_VERSION / "datasets"
RESULTS = _HERE / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["noskill", "skillv1"])
    ap.add_argument("--cases", type=int, default=3)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--samples", type=int, default=1)
    args = ap.parse_args()

    load_dotenv()
    RESULTS.mkdir(parents=True, exist_ok=True)

    inputs = json.loads((DATASETS / "test" / "inputs.json").read_text(encoding="utf-8"))
    out_map = {o["case_id"]: o["target"] for o in
               json.loads((DATASETS / "test" / "outputs.json").read_text(encoding="utf-8"))}
    subset = inputs[args.offset: args.offset + args.cases]

    print(f"Generator: {GEN_MODEL} | condition={args.condition} | jury={[j[0] for j in JUDGES]}")
    print(f"Generating + scoring {len(subset)} cases...\n")

    all_rows: list[dict] = []
    gen_path = RESULTS / f"gen_{args.condition}_outputs.jsonl"
    gen_f = gen_path.open("w", encoding="utf-8")

    for case in subset:
        cid = case["case_id"]
        target = out_map.get(cid)
        if not target:
            continue
        gen, gen_raw = generate(case["input"], args.condition)
        if not gen:
            print(f"  {cid}: GENERATE FAILED")
            gen_f.write(json.dumps({"case_id": cid, "condition": args.condition,
                                    "output": None, "raw": gen_raw}, ensure_ascii=False) + "\n")
            continue
        # save parsed JSON + full raw Opus text (includes <thinking>)
        gen_f.write(json.dumps({"case_id": cid, "condition": args.condition,
                                "output": gen, "raw": gen_raw}, ensure_ascii=False) + "\n")
        candidate_doc = render_target_as_analysis(gen)
        rows = score_candidate(cid, case["input"], target, candidate_doc,
                               candidate_label=args.condition, samples=args.samples)
        all_rows.extend(rows)
        agg = aggregate(rows)
        if agg:
            print(f"  {cid}: overall={agg['overall']}  A={agg['A_avg']}  B={agg['B_avg']}  (n={agg['n_scores']})")
    gen_f.close()

    # full judge detail (9 dims + A/B/overall + comment + raw judge text) -> jsonl
    judge_jsonl = RESULTS / f"gen_{args.condition}_judge_detail.jsonl"
    with judge_jsonl.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # clean numeric view -> csv (drops the long raw text)
    csv_path = RESULTS / f"gen_{args.condition}_raw.csv"
    cols = ["case_id", "candidate", "judge", "sample", "parse_ok",
            *DIMS, "A_avg", "B_avg", "overall", "comment"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    glob = aggregate(all_rows)
    print("\n" + "=" * 60)
    print(f"OPUS [{args.condition}] — global score")
    print("=" * 60)
    if glob:
        for d in DIMS_A:
            print(f"  A {d:24s} {glob[d]}")
        for d in DIMS_B:
            print(f"  B {d:24s} {glob[d]}")
        print(f"  ---- A_avg={glob['A_avg']}  B_avg={glob['B_avg']}  OVERALL={glob['overall']}")
        print("\n  per-judge overall:")
        for jname, _ in JUDGES:
            jr = [r for r in all_rows if r.get("parse_ok") and r["judge"] == jname]
            if jr:
                print(f"    {jname:18s} {round(sum(r['overall'] for r in jr)/len(jr),3)}")
    print(f"\n  CSV: {csv_path}\n  judge detail: {judge_jsonl}\n  gen outputs: {gen_path}")


if __name__ == "__main__":
    main()
