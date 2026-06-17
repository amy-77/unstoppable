# Experiment 2 — Does the skill improve distillation data?

**Question.** Both students are deployed/evaluated *with* the v2 skill in their
system prompt. Given that, **does it still matter whether the teacher used the
skill when it generated the SFT targets?**

**Answer (held-out 60, 3-judge jury): yes — the skill-taught student beats the
no-skill-taught student by `+0.584` overall, and on all 9 rubric dimensions.**
Because both students carry the same skill at inference, this is the *conservative
marginal* value of skill-in-teacher-data on top of skill-in-context.

---

## Setup

| | |
|---|---|
| **Teacher** | `claude-opus-4-8` |
| **Student base** | `Qwen/Qwen3-4B-Thinking-2507`, LoRA (all-linear, r16/α32), `assistant_only_loss` |
| **Train cases** | 190 (`datasets/train`, `CN030` near-dup removed) |
| **Controlled variable** | only **whether the teacher saw the v2 skill** when writing the target |
| **Both arms identical otherwise** | same cases, base, hyperparams, eval, judges; **v2 skill in the student system prompt at train AND inference** |
| **v2 skill** | `exp1/.../refine_v2/run/round_08/SKILL_market_read_accuracy.md` (skill-only, no references) |
| **Eval set** | held-out **60** (`鳌烨xxx`, disjoint from the train `CN/TS/AY` cases; seed 42) |
| **Jury (PoLL)** | `gpt-5.5` + `gemini-3.1-pro` + `deepseek-v4-pro` (no Anthropic / no Qwen → no self-pref) |
| **Rubric** | 9 dims, 1–5; A = accuracy vs expert, B = intrinsic quality |

Two SFT datasets over the same 190 cases:
- **A `skill`** — teacher prompted **with** the v2 skill.
- **B `noskill`** — teacher prompted with **no skill** (output-format persona only).

Measured: longest example = 11,199 Qwen tokens → `max_seq=12288` (no truncation);
the v2 skill alone ≈ 8,526 tokens in the system prompt.

---

## Pipeline

```
gen_teacher.py   --arm skill|noskill   →  data/teacher_{arm}_zh.jsonl   (Opus teacher signal)
prepare_data.py  --arm both            →  data/sft_{arm}_{train,val}.jsonl  (skill in student system prompt; val split identical across arms)
modal_train.py   ::both                →  LoRA SFT both arms on Modal (A100-80GB), merged models in volume
eval_modal.py    --arm skill|noskill   →  results/gen_student_{arm}_outputs.jsonl  (vLLM, with-skill, held-out 60)
score_students.py --arm both           →  results/compare_summary.json  (3-judge jury, skill vs noskill)
```

Both students use `prepare_data.build_student_system()` (v2 skill + output-format
spec) **verbatim** at train and inference, so the only difference is the teacher data.

---

## Results — held-out 60, jury mean

| dimension | skill | noskill | Δ (skill−noskill) |
|---|---|---|---|
| **overall** | **3.518** | **2.934** | **+0.584** |
| A_avg (accuracy vs expert) | 3.406 | 2.909 | +0.497 |
| B_avg (intrinsic quality) | 3.659 | 2.965 | +0.694 |
| conflict_insight | 3.444 | 3.022 | +0.422 |
| market_read_accuracy | 3.607 | 3.033 | +0.574 |
| strategy_direction | 3.242 | 2.550 | +0.692 |
| target_portrait_match | 3.365 | 2.878 | +0.487 |
| persona_read | 3.371 | 3.061 | +0.310 |
| logic_depth | 3.500 | 3.106 | +0.394 |
| insight_nonobviousness | 3.528 | 2.744 | **+0.784** |
| risk_anti_pattern | 3.854 | 3.000 | **+0.854** |
| actionability | 3.753 | 3.011 | **+0.742** |

`n_scores`: skill 178/180, noskill 180/180 (2 judge parse-fails on skill, excluded).

---

## Findings

1. **Skill-taught distillation data is strictly better** — every one of the 9
   dimensions favours the skill arm; overall `+0.584`.
2. **Biggest lifts are exactly the skill's signature moves**: `risk_anti_pattern`
   (+0.85), `insight_nonobviousness` (+0.78), `actionability` (+0.74),
   `strategy_direction` (+0.69) — the anti-pattern avoidance, counter-intuitive
   reframes, concrete next-actions, and play/hold direction the skill encodes
   transferred into the student's weights.
3. **Conservative read.** Both students have the full skill in context at
   inference, so `+0.584` is the *marginal* gain of skill-in-teacher-data **on top
   of** skill-in-context. The total value of the skill is larger (would need a
   "student infers without skill" 2×2 arm to measure — out of scope here).
4. **Distillation gap.** Student absolute scores (~3.5 skill) sit well below the
   Opus teacher (~4.7) and expert (~4.9) — a 4B LoRA loses a lot vs the teacher,
   but the skill arm clearly wins the controlled comparison.

---

## Reproduce

```bash
cd new_version
# 1) teacher data (Opus; ~$10, resumable, concurrency 16)
python experiments/exp2_skill_vs_noskill_distill/gen_teacher.py --arm skill
python experiments/exp2_skill_vs_noskill_distill/gen_teacher.py --arm noskill
# 2) SFT format (local, free)
python experiments/exp2_skill_vs_noskill_distill/prepare_data.py --arm both
# 3) SFT both arms on Modal (GPU); ALWAYS smoke first
modal run experiments/exp2_skill_vs_noskill_distill/modal_train.py::main --arm skill --smoke
modal run experiments/exp2_skill_vs_noskill_distill/modal_train.py::both
# 4) generate on held-out 60 (vLLM, with-skill) — resumable, 40-min cap
modal run experiments/exp2_skill_vs_noskill_distill/eval_modal.py --arm skill
modal run experiments/exp2_skill_vs_noskill_distill/eval_modal.py --arm noskill
# 5) score (local, 3-judge jury, resumable, parallel 16)
python experiments/exp2_skill_vs_noskill_distill/score_students.py --arm both
```

### Layout
```
gen_teacher.py        Opus teacher generation (--arm skill|noskill, skill-only)
prepare_data.py       teacher jsonl → SFT chat format (skill in student system prompt)
modal_train.py        LoRA SFT on Modal, wraps matchmaker/training/train_sft.py
eval_modal.py         vLLM generation on held-out 60 (with-skill), resumable
score_students.py     3-judge jury scoring + skill-vs-noskill comparison
data/                 teacher_{arm}_zh.jsonl  (sft_*.jsonl are derived → gitignored)
results/              gen_student_*_outputs.jsonl, *_judge_detail.jsonl, *_raw.csv, compare_summary.json
```

### Notes / gotchas baked into the scripts
- transformers ≥ 4.51 required for the Qwen3 architecture.
- vLLM: disable flashinfer (`VLLM_ATTENTION_BACKEND=FLASH_ATTN` +
  `VLLM_USE_FLASHINFER_SAMPLER=0`) to avoid its runtime nvcc JIT on slim images.
- `train_sft.py --resume` + `eval_modal.py` chunk-commit give step/chunk-level
  resume; Modal `timeout` + `retries` bound cost; `modal app stop` stale apps
  before relaunch to avoid double-billing.
