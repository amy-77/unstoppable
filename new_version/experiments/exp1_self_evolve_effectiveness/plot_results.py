#!/usr/bin/env python3
"""Generate v1-vs-v2 comparison figures for the README."""
import json
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
DIMS = ["conflict_insight", "market_read_accuracy", "strategy_direction",
        "target_portrait_match", "persona_read", "logic_depth",
        "insight_nonobviousness", "risk_anti_pattern", "actionability"]
EXPERT_REF = 4.91
C1, C2 = "#d1495b", "#2e86ab"   # v1 red, v2 blue


def best_so_far(ledger_path):
    l = json.load(open(ledger_path))
    xs, ys, best = [], [], -1
    for h in l["history"]:
        best = max(best, h["score"])
        xs.append(h["round"]); ys.append(round(best, 3))
    return xs, ys


def dm(path):
    jr = [x for line in open(path) for x in json.loads(line).get("judge_rows", []) if x.get("parse_ok")]
    return {d: sum(x[d] for x in jr) / len(jr) for d in DIMS}, sum(x["overall"] for x in jr) / len(jr)


v1l, v2l = HERE / "results/refine_v1/run/ledger.json", HERE / "results/refine_v2/run/ledger.json"
v1f, v2f = HERE / "results/refine_v1/run/final", HERE / "results/refine_v2/run/final"

# ---- Fig 1: best-so-far selection trajectory ----
x1, y1 = best_so_far(v1l); x2, y2 = best_so_far(v2l)
plt.figure(figsize=(7, 4.3))
plt.axhline(EXPERT_REF, ls="--", c="gray", lw=1, label=f"expert reference ({EXPERT_REF})")
plt.axhline(y1[0], ls=":", c="black", lw=1, label=f"no-skill baseline ({y1[0]})")
plt.plot(x1, y1, "-o", c=C1, label="v1 (full rewrite)")
plt.plot(x2, y2, "-o", c=C2, label="v2 (per-dim bounded edit)")
plt.xlabel("self-evolve round"); plt.ylabel("best-so-far score (selection, 1-5)")
plt.title("Self-evolve trajectory: skill built from zero")
plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(FIG / "fig_trajectory.png", dpi=130); plt.close()

# ---- Fig 2: held-out test bars ----
v1d, v1o = dm(v1f / "eval_noskill.jsonl"); v1bd, v1bo = dm(v1f / "eval_best.jsonl")
v2d, v2o = dm(v2f / "eval_noskill.jsonl"); v2bd, v2bo = dm(v2f / "eval_best.jsonl")
labels = ["no-skill", "v1 auto-skill", "v2 auto-skill"]
vals = [round((v1o + v2o) / 2, 3), round(v1bo, 3), round(v2bo, 3)]
plt.figure(figsize=(6, 4.3))
bars = plt.bar(labels, vals, color=["#999999", C1, C2])
plt.axhline(EXPERT_REF, ls="--", c="gray", lw=1, label=f"expert reference ({EXPERT_REF})")
for b, v in zip(bars, vals):
    plt.text(b.get_x() + b.get_width() / 2, v + .01, f"{v}", ha="center", fontsize=10)
plt.ylim(3.8, 5.0); plt.ylabel("overall score (held-out test, 60 cases)")
plt.title("Held-out generalization: no-skill vs auto-built skill")
plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(FIG / "fig_heldout.png", dpi=130); plt.close()

# ---- Fig 3: per-dimension lift (v1 vs v2) ----
v1lift = [v1bd[d] - v1d[d] for d in DIMS]
v2lift = [v2bd[d] - v2d[d] for d in DIMS]
import numpy as np
y = np.arange(len(DIMS)); h = 0.38
plt.figure(figsize=(7.5, 5))
plt.barh(y + h/2, v1lift, h, color=C1, label="v1")
plt.barh(y - h/2, v2lift, h, color=C2, label="v2")
plt.yticks(y, DIMS, fontsize=8); plt.gca().invert_yaxis()
plt.xlabel("score lift on held-out (no-skill -> auto-skill)")
plt.title("Per-dimension lift: v1 vs v2")
plt.axvline(0, c="black", lw=.6); plt.legend(); plt.grid(axis="x", alpha=.3); plt.tight_layout()
plt.savefig(FIG / "fig_perdim.png", dpi=130); plt.close()

print("wrote:", *[p.name for p in sorted(FIG.glob('*.png'))])
