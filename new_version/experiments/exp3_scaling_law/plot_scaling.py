#!/usr/bin/env python3
"""Plot the exp3 scaling law: distillation lift (student - baseline) vs model size.

Reads results/scaling_summary.json (merged by score.py) and writes
figures/fig_scaling_lift.png — a two-panel figure:
  L: student vs baseline absolute (both with-skill); the gap IS the lift.
  R: the lift curve vs size (log-param x) — shows it is NOT an inverted-U
     (32B passes the 4B peak).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
SUMMARY = HERE / "results" / "scaling_summary.json"

C_STU, C_BASE, C_LIFT = "#2e86ab", "#999999", "#d1495b"   # student blue, baseline gray, lift red


def parse_b(k: str) -> float:        # "1.7B" -> 1.7, "0.6B" -> 0.6
    return float(k.rstrip("Bb"))


data = json.loads(SUMMARY.read_text(encoding="utf-8"))
pts = sorted(((parse_b(k), k, v) for k, v in data.items()), key=lambda t: t[0])
xs   = [p for p, _, _ in pts]
labs = [k for _, k, _ in pts]
stu  = [v["student_overall"] for _, _, v in pts]
base = [v["baseline_overall"] for _, _, v in pts]
lift = [v["lift_overall"] for _, _, v in pts]

# the pre-32B local peak (4B) — the line 32B has to clear to break the inverted-U
peak4b = data["4B"]["lift_overall"] if "4B" in data else max(lift[:-1])

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.8))

# ---- Left: absolute student vs baseline (both with-skill); shaded gap = lift ----
axL.fill_between(xs, base, stu, color=C_LIFT, alpha=0.12, label="distillation lift (gap)")
axL.plot(xs, stu,  "-o", c=C_STU,  lw=2, label="student (SFT'd, with skill)")
axL.plot(xs, base, "-o", c=C_BASE, lw=2, label="baseline (no SFT, skill in prompt)")
axL.set_xscale("log"); axL.set_xticks(xs); axL.set_xticklabels(labs); axL.minorticks_off()
axL.set_xlabel("model size (params, log scale)")
axL.set_ylabel("overall score (held-out 60, 3-judge, 1–5)")
axL.set_title("Student vs baseline across size")
axL.legend(fontsize=8, loc="lower right"); axL.grid(alpha=.3)

# ---- Right: the lift curve ----
axR.plot(xs, lift, "-o", c=C_LIFT, lw=2)
axR.axhline(peak4b, ls=":", c="gray", lw=1)
axR.text(xs[0], peak4b + 0.006, f"4B local peak ({peak4b:.3f}) — surpassed by 32B",
         fontsize=8, color="gray", va="bottom")
for x, l in zip(xs, lift):
    axR.annotate(f"+{l:.3f}", (x, l), textcoords="offset points", xytext=(0, 8),
                 ha="center", fontsize=8)
axR.set_xscale("log"); axR.set_xticks(xs); axR.set_xticklabels(labs); axR.minorticks_off()
axR.set_ylim(0, max(lift) * 1.18)
axR.set_xlabel("model size (params, log scale)")
axR.set_ylabel("distillation lift  (student − baseline)")
axR.set_title("Distillation lift vs size — NOT an inverted-U")
axR.grid(alpha=.3)

fig.suptitle("Exp3 — does the distillation lift hold across model size?", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = FIG / "fig_scaling_lift.png"
fig.savefig(out, dpi=130); plt.close(fig)
print("wrote:", out)
