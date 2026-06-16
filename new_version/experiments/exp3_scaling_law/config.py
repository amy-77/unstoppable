"""Exp3 scaling-law config — the ONLY thing that varies across runs is the base model.

Single source of truth for which Qwen3 *dense* sizes the sweep covers (same family,
same generation, so the x-axis is purely parameter count — no architecture confound).
Exp3 reuses exp2's skill-teacher SFT data verbatim; each size trains the same data on a
different base, then we eval (a) the SFT student and (b) the no-SFT base, both WITH the
v2 skill in the system prompt → the gap is the distillation lift at that size.
"""

# gpu is sized for LoRA training at max_seq=12288 (the heavier of train/eval).
SIZES = [
    {"size": "0.6B", "hf_id": "Qwen/Qwen3-0.6B", "gpu": "A100-40GB"},
    {"size": "1.7B", "hf_id": "Qwen/Qwen3-1.7B", "gpu": "A100-40GB"},
    {"size": "4B",   "hf_id": "Qwen/Qwen3-4B",   "gpu": "A100-40GB"},
    {"size": "8B",   "hf_id": "Qwen/Qwen3-8B",   "gpu": "A100-80GB"},
    {"size": "14B",  "hf_id": "Qwen/Qwen3-14B",  "gpu": "A100-80GB"},
    # 32B bf16 LoRA (~64GB weights) won't fit one 80GB card. Keep bf16 (NOT QLoRA —
    # 4-bit would be a scaling-law confound vs the bf16 points) and split layers
    # across 2×H100 via naive model-parallel (train --device-map auto).
    {"size": "32B",  "hf_id": "Qwen/Qwen3-32B",  "gpu": "H100:2"},
]

# Order: smoke 0.6B → Phase 1 new points → 4B last → Phase 2 big (after top-up).
PHASE1 = ["0.6B", "1.7B", "8B"]          # 4B deferred to last (re-trained on base)
PHASE2 = ["14B", "32B"]                  # after Modal top-up; 32B = bf16 on 2×H100

MODES = ["student", "baseline"]   # student = SFT'd; baseline = base model, no SFT (both with-skill)
VOLUME = "exp3-scaling-models"    # Modal volume: base/ + outputs/ + eval_out/


def by_size(name: str) -> dict:
    for s in SIZES:
        if s["size"] == name:
            return s
    raise KeyError(f"unknown size {name!r}; known: {[s['size'] for s in SIZES]}")
