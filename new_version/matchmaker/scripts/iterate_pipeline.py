#!/usr/bin/env python3
"""
Skill 自迭代主控脚本：evaluate → diagnose → refine → re-evaluate，直到收敛。

Usage:
    python3 scripts/iterate_pipeline.py
    python3 scripts/iterate_pipeline.py --max-rounds 3 --target-score 4.0 --sample 10
    python3 scripts/iterate_pipeline.py --lang zh --parallel 5
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import run_evaluation
from diagnose import run_diagnosis
from refine import run_refinement


def iterate(
    lang: str = "zh",
    max_rounds: int = 5,
    target_score: float = 4.2,
    min_improvement: float = 0.1,
    sample_size: int = None,
    parallel: int = 3,
):
    skill_path = BASE_DIR / lang / "SKILL.md"
    if not skill_path.exists():
        print(f"Error: {skill_path} not found")
        sys.exit(1)

    history_dir = BASE_DIR / "scripts" / "iteration_history"
    history_dir.mkdir(parents=True, exist_ok=True)

    history = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"{'='*60}")
    print(f"  Skill Self-Iteration Pipeline")
    print(f"  Language: {lang}")
    print(f"  Target score: {target_score}")
    print(f"  Max rounds: {max_rounds}")
    print(f"  Sample size: {sample_size or 'all'}")
    print(f"{'='*60}\n")

    for round_num in range(1, max_rounds + 1):
        round_dir = history_dir / f"round_{round_num}_{timestamp}"
        round_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'─'*40}")
        print(f"  Round {round_num}/{max_rounds}")
        print(f"{'─'*40}\n")

        # Step 1: Evaluate
        print(f"[Round {round_num}] Step 1: Evaluate...")
        eval_report = run_evaluation(skill_path, sample_size, parallel)
        eval_path = round_dir / "eval_report.json"
        eval_report["round"] = round_num
        eval_report["timestamp"] = datetime.now().isoformat()
        eval_path.write_text(json.dumps(eval_report, ensure_ascii=False, indent=2), encoding="utf-8")
        history.append(eval_report)

        overall = eval_report["overall_score"]
        print(f"\n[Round {round_num}] Score: {overall:.2f}/5.0")

        # Step 2: Check convergence
        if overall >= target_score:
            print(f"\n✅ Target score reached ({overall:.2f} >= {target_score}). Stopping.")
            break

        if round_num > 1:
            prev_score = history[-2]["overall_score"]
            improvement = overall - prev_score
            print(f"[Round {round_num}] Improvement: {improvement:+.2f}")
            if improvement < min_improvement:
                print(f"\n⚠️  Converged (improvement {improvement:.2f} < {min_improvement}). Stopping.")
                break

        # Step 3: Diagnose
        print(f"\n[Round {round_num}] Step 2: Diagnose...")
        diagnosis = run_diagnosis(eval_report, skill_path)
        diag_path = round_dir / "diagnosis.json"
        diag_path.write_text(json.dumps(diagnosis, ensure_ascii=False, indent=2), encoding="utf-8")

        # Step 4: Refine
        print(f"\n[Round {round_num}] Step 3: Refine...")
        refine_results = run_refinement(diag_path, skill_path, round_dir)
        refine_path = round_dir / "refine_results.json"
        refine_path.write_text(json.dumps(refine_results, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"\n[Round {round_num}] Complete. Files in: {round_dir}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"  Iteration Complete")
    print(f"{'='*60}")
    print(f"  Rounds: {len(history)}")
    scores_str = " → ".join(f"{h['overall_score']:.2f}" for h in history)
    print(f"  Score trajectory: {scores_str}")
    if len(history) > 1:
        total_improvement = history[-1]["overall_score"] - history[0]["overall_score"]
        print(f"  Total improvement: {total_improvement:+.2f}")
    print(f"  Final score: {history[-1]['overall_score']:.2f}/5.0")

    # Save summary
    summary = {
        "timestamp": timestamp,
        "lang": lang,
        "rounds": len(history),
        "score_trajectory": [h["overall_score"] for h in history],
        "dimension_trajectory": [h["dimension_scores"] for h in history],
        "final_score": history[-1]["overall_score"],
        "converged": history[-1]["overall_score"] >= target_score or
                     (len(history) > 1 and history[-1]["overall_score"] - history[-2]["overall_score"] < min_improvement),
    }
    summary_path = history_dir / f"summary_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Skill self-iteration pipeline")
    parser.add_argument("--lang", default="zh", choices=["zh", "en"], help="Language")
    parser.add_argument("--max-rounds", type=int, default=5, help="Max iteration rounds")
    parser.add_argument("--target-score", type=float, default=4.2, help="Target score to stop")
    parser.add_argument("--min-improvement", type=float, default=0.1, help="Min improvement per round")
    parser.add_argument("--sample", type=int, default=None, help="Only evaluate N cases per round")
    parser.add_argument("--parallel", type=int, default=3, help="Parallel workers for evaluation")
    args = parser.parse_args()

    iterate(
        lang=args.lang,
        max_rounds=args.max_rounds,
        target_score=args.target_score,
        min_improvement=args.min_improvement,
        sample_size=args.sample,
        parallel=args.parallel,
    )


if __name__ == "__main__":
    main()
