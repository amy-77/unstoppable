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
| 32B | `Qwen/Qwen3-32B` | H100 | optional big point |

> exp2's 4B used `Qwen3-4B-Thinking-2507`; exp3 uses **base** `Qwen3-4B` for a clean
> same-lineage curve, so the 4B point is re-trained here (cheap).

## Setup (shared, fixed)

- **SFT data**: exp2's `sft_skill_{train,val}.jsonl` verbatim (skill-teacher targets,
  v2 skill in the student system prompt). Same data on every base.
- **Engine**: `matchmaker/training/train_sft.py` (LoRA all-linear r16, `assistant_only_loss`,
  `max_seq=12288`), wrapped by `train_modal.py`.
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
# 1) train (Phase 1 in parallel)
modal run experiments/exp3_scaling_law/train_modal.py::sweep --sizes 0.6B,1.7B,4B,8B
# 2) eval both modes (student + baseline), all sizes, parallel
modal run experiments/exp3_scaling_law/eval_modal.py::sweep --sizes 0.6B,1.7B,4B,8B
# 3) score + scaling table (local)
python experiments/exp3_scaling_law/score.py --sizes 0.6B,1.7B,4B,8B
```

## Layout
```
config.py        size→hf_id→gpu (the only thing that varies)
train_modal.py   parameterized LoRA SFT (wraps train_sft.py), sweep entrypoint
eval_modal.py    parameterized vLLM gen (student/baseline modes), sweep entrypoint
score.py         jury scoring + scaling table (student/baseline/lift vs size)
results/         gen_{size}_{mode}_outputs.jsonl, judge_{size}_{mode}.jsonl, scaling_summary.json
```

## Status (in progress)
Pipeline validated end-to-end. Results (held-out 60, **base Qwen3 dense, same lineage**, with-skill):

| size | baseline | student | **lift** | parse (s/b) |
|---|---|---|---|---|
| 0.6B | 1.481 | 1.724 | +0.243 | 50/59 |
| 1.7B | 1.819 | 2.559 | +0.740 | 59/56 |
| **4B** | 2.271 | 3.307 | **+1.036** | 59/56 |
| 8B | 2.684 | 3.596 | +0.912 | 60/60 |

**Headline: the distillation lift is an inverted-U in model size — it peaks at 4B (+1.04)
and starts to decline at 8B (+0.91).**
- Small (0.6B): model too weak to exploit the skill-distilled data → small lift.
- Mid (4B): strong enough to absorb it while its baseline is still weak → max lift.
- Large (8B+): the baseline catches up faster than the student (4B→8B baseline +0.41 vs
  student +0.29) → the model needs the distillation less → lift declines.

(An earlier 3-point read [0.6/1.7/8B] looked monotonic; the clean **base** 4B point revealed
the peak. exp2's 4B was `Qwen3-4B-Thinking-2507` — off-lineage, used only as a side-reference,
not on this curve.) 14B (running) will show whether the post-peak decline continues or plateaus;
32B (needs QLoRA) pending.

Parse reliability also rises with size (0.6B 50/60 → 8B 60/60). Caveat: unequal parse counts
(student vs baseline) → mild survivorship bias; TODO intersection-only scoring.

Three bugs the 0.6B smoke caught & fixed: sibling `config.py` → `add_local_python_source`;
12K×151K-vocab logits OOM → `batch_size=1` (later solved properly by the **Liger** fused
lm_head+CE) ; small-model greedy degeneration → `repetition_penalty=1.1`. Training now runs
**FlashAttention-2 + Liger kernel** (FA2 wheels vendored locally — see `exp3-modal-flashattn-liger`
recipe; never pip-from-github on the Modal builder, it hangs). FA2+Liger cut 4B from ~100min(est, old
sdpa path) to ~30min.
