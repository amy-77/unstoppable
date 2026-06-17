# Experiment 3 — Scaling law: does the distillation lift hold across model size?

**Question.** Take the skill-teacher SFT data fixed (from exp2) and train students of
different sizes. **How does the distillation lift scale with model size?** I.e. as the
student grows, does distilling the skill-taught data into its weights help *more* or
*less* (does a bigger model need it less, or exploit it better)?

**Controlled design.** The ONLY variable is the base model's **parameter count**, within
**Qwen3 dense, same generation** — no architecture confound (MoE excluded: a 30B-A3B is
~3B active, doesn't sit on a dense N-axis; cross-gen/cross-family excluded too). For a
big point, use `Qwen3-32B` dense, not 30B-MoE / 27B-other-gen.

Per size we run **two evals**, both WITH the v2 skill in the system prompt:
- **student** — the SFT'd model (skill-teacher data distilled into weights)
- **baseline** — the same base model, **no SFT** (skill only in context)
- **lift = student − baseline** = value of distillation at that size.

## Models (Qwen3 dense, base lineage — see `config.py`)

| size | HF id | gpu | phase |
|---|---|---|---|
| 0.6B | `Qwen/Qwen3-0.6B` | A100-40GB | 1 |
| 1.7B | `Qwen/Qwen3-1.7B` | A100-40GB | 1 |
| 4B | `Qwen/Qwen3-4B` | A100-40GB | 1 |
| 8B | `Qwen/Qwen3-8B` | A100-80GB | 1 |
| 14B | `Qwen/Qwen3-14B` | A100-80GB | 2 (after top-up) |
| 32B | `Qwen/Qwen3-32B` | H100:2 (FSDP) | 2 (done) |

> exp2's 4B used `Qwen3-4B-Thinking-2507`; exp3 uses **base** `Qwen3-4B` for a clean
> same-lineage curve, so the 4B point is re-trained here (cheap).

## Setup (shared, fixed)

- **SFT data**: exp2's `sft_skill_{train,val}.jsonl` verbatim (skill-teacher targets,
  v2 skill in the student system prompt). Same data on every base.
- **Engine**: `matchmaker/training/train_sft.py` (LoRA all-linear r16, `assistant_only_loss`,
  `max_seq=12288`), wrapped by `train_modal.py`. ≤14B trains single-GPU (bf16, `device_map`);
  **32B uses FSDP full-shard (ZeRO-3) bf16 LoRA on 2×H100** (`--fsdp`, `accelerate launch
  --num_processes 2`) — same recipe, effective batch still 8 (per-rank `grad_accum=4` × 2),
  so it stays on the same curve. 32B ships **adapter-only** (no 65GB merge in the GPU window);
  eval serves base + LoRA via vLLM (numerically identical to a merged model).
- **Eval set / judges / rubric**: held-out 60 (seed 42), `gpt-5.5 + gemini-3.1-pro +
  deepseek-v4-pro`, 9-dim rubric — identical to exp1/exp2, so all numbers are comparable.
- **Inference**: vLLM, FlashAttention-2 backend (`VLLM_ATTENTION_BACKEND=FLASH_ATTN`,
  flashinfer disabled → no nvcc), greedy, with-skill.

## Pipeline (all on Modal; model size is the only knob)

```
config.py        SIZES (single source of truth) — size → hf_id, gpu
train_modal.py   ::sweep   LoRA SFT skill data on each base (parallel sizes; size+step resume)
eval_modal.py    ::sweep   vLLM gen on held-out 60, modes={student,baseline} (parallel; chunk resume)
score.py                   3-judge jury per (size,mode) → scaling table: student/baseline/lift
```

Abstraction: every script is parameterized over **size** (and **mode** for eval). Modal's
GPU is static per function, so there are two thin wrappers per stage (`*_40` / `*_80`) and
the sweep dispatches each size to the right one by `config.gpu`.

Resumability: size-level (skip if merged exists), step-level (`train_sft.py --resume`),
chunk-level (eval commits each chunk to the volume), case-level (scoring skips done).
Concurrency: sizes/modes via `spawn`; vLLM batches within; scoring pools all cases @ 16.
`timeout` caps + `retries=2` bound cost; `modal app stop` stale apps before relaunch.

## Reproduce

```bash
cd new_version
# 0) exp2's skill SFT data must exist (experiments/exp2_.../data/sft_skill_*.jsonl)
#    + vendor the FA2 wheels into experiments/exp3_scaling_law/wheels/ (both ABIs;
#      gitignored, 244M each — see exp3-modal-flashattn-liger recipe).
# 0b) OPTIONAL stack smoke (validate FA2 wheel ABI + Liger on the new image, ~$0.2):
modal run experiments/exp3_scaling_law/train_modal.py::smoke --size 0.6B
# 1) train — Phase-1 sizes in parallel, or one big size at a time
modal run experiments/exp3_scaling_law/train_modal.py::sweep --sizes 0.6B,1.7B,4B,8B
modal run --detach experiments/exp3_scaling_law/train_modal.py::one --size 14B   # big: detach + --resume
# 2) eval both modes (student + baseline), parallel
modal run experiments/exp3_scaling_law/eval_modal.py::sweep --sizes 0.6B,1.7B,4B,8B,14B
# 3) score + scaling table (local, 3-judge)
python experiments/exp3_scaling_law/score.py --sizes 0.6B,1.7B,4B,8B,14B

# --- 32B (FSDP on 2×H100) ---
modal run experiments/exp3_scaling_law/train_modal.py::stage  --size 32B   # CPU prestage weights → volume (no GPU)
modal run experiments/exp3_scaling_law/train_modal.py::smoke  --size 32B   # 2-step FSDP smoke on 2×H100 (validate before full)
modal run experiments/exp3_scaling_law/train_modal.py::one    --size 32B   # full FSDP run, adapter-only
modal run experiments/exp3_scaling_law/eval_modal.py::sweep   --sizes 32B  # vLLM TP=2; student = base + LoRA adapter
python experiments/exp3_scaling_law/score.py --sizes 32B                   # merges into scaling_summary.json
```

## Layout
```
config.py        size→hf_id→gpu (the only thing that varies)
train_modal.py   parameterized LoRA SFT (wraps train_sft.py); entrypoints: smoke / one / sweep (/ stage prestage)
eval_modal.py    parameterized vLLM gen (student/baseline modes), sweep entrypoint
score.py         jury scoring + scaling table (student/baseline/lift vs size)
wheels/          vendored flash-attn wheels (both C++ ABIs, 244M each) — gitignored, copied into the train image
results/         gen_{size}_{mode}_outputs.jsonl, judge_{size}_{mode}.jsonl, scaling_summary.json
```

## Status
Pipeline validated end-to-end; **0.6B–32B complete — all 6 points**.
Results (held-out 60, **base Qwen3 dense, same lineage**, with-skill):

| size | baseline | student | **lift** | scored n/60 (stu / base) |
|---|---|---|---|---|
| 0.6B | 1.481 | 1.724 | +0.243 | 50 / 59 |
| 1.7B | 1.819 | 2.559 | +0.740 | 59 / 56 |
| **4B** | 2.271 | 3.307 | +1.036 | 59 / 56 |
| 8B | 2.684 | 3.596 | +0.912 | 60 / 60 |
| 14B | 2.877 | 3.762 | +0.885 | 60 / 60 |
| **32B** | 2.905 | 4.127 | **+1.222** | 60 / 60 |

**Headline: the distillation lift is NON-MONOTONIC in size — the earlier "inverted-U" is
overturned by 32B.** Through 14B the lift looked like a clean inverted-U (peak 4B +1.04, then
8B +0.91, 14B +0.89). **32B breaks it: lift = +1.22, above the 4B peak** — the lift does not
keep tapering with size.

- Small (0.6B): model too weak to exploit the skill-distilled data → small lift.
- Mid (4B): strong enough to absorb it while its baseline is still weak → local peak.
- 8B–14B: baseline catches up (2.27→2.68→2.88), lift dips/plateaus (+1.04→+0.91→+0.89) —
  the stretch that *looked* like a post-peak decline.
- 32B: the **baseline plateaus** (14B→32B: 2.877→2.905, +0.03) while the **student jumps**
  (3.762→4.127, +0.365) → lift widens to +1.22. The big model exploits the distilled data
  *more*, not less, because its in-context baseline has stopped improving.

**Shape is underdetermined — don't overclaim.** Two readings fit the 6 points: (a) *peak–dip–
peak* (genuine non-monotonicity), or (b) *lift broadly rises with log-params*, with 8B–14B a
closely-spaced near-plateau that only looks like a dip. The x-axis is uneven: **14B→32B is a
~2.3× parameter jump** (second-biggest step on the curve) vs only ~1.75× for 8B→14B, so part
of the 32B rise is simply a bigger step along size — the reversal may be less abrupt than the
table suggests. A point near ~20–24B and/or repeat seeds would separate (a) from (b). With 6
unevenly-spaced points and one seed we report only: **the lift is not an inverted-U; it
exceeds the 4B peak at 32B; the exact curve shape is open.**

(An earlier 3-point read [0.6/1.7/8B] looked monotonic; the clean **base** 4B point revealed a
local peak; 14B looked like a confirmed decline; 32B overturned it. exp2's 4B was
`Qwen3-4B-Thinking-2507` — off-lineage, side-reference only, not on this curve.)

**Reading the "scored n/60" column.** Each model's score is averaged over *only* the cases
whose output parsed to valid JSON; unparseable outputs are dropped before scoring. Two
consequences:
1. **Parse rate is itself a quality signal** — it rises with size (0.6B 50/60 → every point
   ≥8B is 60/60): small models fail the strict output format more often.
2. **Survivorship bias at the small end** — when student and baseline parse *different* subsets
   (e.g. 0.6B: 50 vs 59), their averages aren't over the same cases, so the 0.6B–4B lifts are
   the least clean. Every point **8B–32B is 60/60** (full, identical subsets) → the upper half
   of the curve, including the 32B reversal, is bias-free. TODO: intersection-only scoring to
   clean the small-size points.

Three bugs the 0.6B smoke caught & fixed: sibling `config.py` → `add_local_python_source`;
12K×151K-vocab logits OOM → `batch_size=1` (later solved properly by the **Liger** fused
lm_head+CE) ; small-model greedy degeneration → `repetition_penalty=1.1`. Training now runs
**FlashAttention-2 + Liger kernel** (FA2 wheels vendored locally — see `exp3-modal-flashattn-liger`
recipe; never pip-from-github on the Modal builder, it hangs). FA2+Liger cut 4B from ~100min(est, old
sdpa path) to ~30min; **14B ran at batch_size=2 on one A100-80GB without OOM** (Liger frees the
~7GB fp32 logits tensor), ~49–55 s/optimizer-step, 69 steps.

**Preemption.** The 14B run was preempted by a Modal worker at step 46/69; `retries=2` did
NOT auto-resume a detached run (left `Terminated`, Tasks:0). Recovery that works: Modal Volume
background-commit had persisted `checkpoint-40`, so `modal app stop` + relaunch
`::one --size 14B` (always passes `--resume`) picked up from step 40 and finished — ~6 steps
redone. Lesson: expect preemption on big GPUs; rely on `--resume` + `save_steps=10`, and when
monitoring, watch for `Runner interrupted`/`preempt` **and** log-staleness (silence ≠ healthy).
