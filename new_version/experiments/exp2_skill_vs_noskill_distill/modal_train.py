#!/usr/bin/env python3
"""Exp2 — LoRA SFT on Modal, REUSING the repo's matchmaker/training/train_sft.py.

This script does NOT re-implement training. It builds a GPU image, downloads the
Qwen3-4B base into a persistent Volume, and invokes the existing `train_sft.py`
(the one already used for the legacy distillation) per arm. The only deltas vs a
local run are: base model is pulled from HF into the Volume, and `--max-seq` is
set to 12288 (measured: longest exp2 example = 11199 Qwen tokens → no truncation).

Speed: the two arms (skill / noskill) are INDEPENDENT runs, so `both` spawns them
on two GPUs concurrently → same GPU-hours, ~half wall-clock. The base model is
downloaded ONCE first (a cheap CPU step) to avoid two arms racing on the volume.

Prereqs (run locally first, free, no GPU):
    python experiments/exp2_skill_vs_noskill_distill/prepare_data.py --arm both
    # -> data/sft_{skill,noskill}_{train,val}.jsonl

Setup once:
    pip install modal && modal token new       # active profile = the one with credit

Run. ALWAYS smoke first to validate the stack + assistant-loss mask:
    modal run experiments/exp2_skill_vs_noskill_distill/modal_train.py::main --arm skill --smoke
    # then both arms in parallel:
    modal run experiments/exp2_skill_vs_noskill_distill/modal_train.py::both

Fetch the trained adapters/merged models back:
    modal volume get exp2-distill-models outputs/exp2_skill   ./out_skill
    modal volume get exp2-distill-models outputs/exp2_noskill ./out_noskill
"""

from __future__ import annotations

import pathlib

import modal

DEFAULT_BASE = "Qwen/Qwen3-4B-Thinking-2507"

# --------------------------------------------------------------------------- #
# Image — modern training stack (torch>=2.4); train_sft.py uses sdpa attention,
# so NO flash-attn build is needed. Pins are a known-coherent set; bump if the
# smoke run errors on an API change.
# --------------------------------------------------------------------------- #
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Frozen to the set validated by the green smoke run (2026-06-15).
        # transformers>=4.51 is required for the Qwen3 architecture; this stack
        # also has SFTConfig(max_length=, assistant_only_loss=) + the dtype=
        # kwarg that train_sft.py uses.
        "torch==2.7.1",
        "transformers==4.57.6",
        "trl==0.23.1",
        "peft==0.19.1",
        "accelerate==1.14.0",
        "datasets==5.0.0",
        "sentencepiece==0.2.1",
    )
    .env({"HF_HOME": "/models/hf", "TOKENIZERS_PARALLELISM": "false"})
)

# Local-path code (mounts + asserts) must run ONLY on the client — inside the
# container __file__ is /root/modal_train.py and these paths don't exist.
if modal.is_local():
    HERE = pathlib.Path(__file__).resolve().parent
    NEW_VERSION = HERE.parents[1]
    TRAIN_SCRIPT = NEW_VERSION / "matchmaker" / "training" / "train_sft.py"
    SFT_DATA_DIR = HERE / "data"
    assert TRAIN_SCRIPT.exists(), f"train_sft.py not found at {TRAIN_SCRIPT}"
    assert (SFT_DATA_DIR / "sft_skill_train.jsonl").exists(), \
        "run prepare_data.py --arm both first (sft_*_train.jsonl missing)"
    image = (
        base_image
        .add_local_file(str(TRAIN_SCRIPT), "/root/train_sft.py")
        .add_local_dir(str(SFT_DATA_DIR), "/root/data")
    )
else:
    image = base_image

# Persistent volume: base-model cache (/models/base) + run outputs (/models/outputs)
models_vol = modal.Volume.from_name("exp2-distill-models", create_if_missing=True)
# Existing HF token secret on the account (avoids anonymous-download rate limits;
# Qwen3 is ungated so this is robustness, not strictly required).
hf_secret = modal.Secret.from_name("huggingface-secret")

app = modal.App("exp2-distill-sft", image=image)


def _local_base_path(base_model: str) -> str:
    return f"/models/base/{base_model.split('/')[-1]}"


@app.function(volumes={"/models": models_vol}, secrets=[hf_secret], timeout=3600)
def download_base(base_model: str = DEFAULT_BASE) -> str:
    """CPU-only: pull the base model into the volume ONCE so parallel train()
    calls don't race on the same local_dir."""
    import os

    from huggingface_hub import snapshot_download

    local_base = _local_base_path(base_model)
    if os.path.isdir(local_base) and any(
        f.endswith(".safetensors") for f in os.listdir(local_base)
    ):
        print(f"[download] already cached: {local_base}")
        return local_base
    print(f"[download] {base_model} -> {local_base}")
    snapshot_download(
        repo_id=base_model,
        local_dir=local_base,
        ignore_patterns=["*.pth", "*.gguf", "original/*"],
    )
    models_vol.commit()
    return local_base


@app.function(gpu="A100-80GB", volumes={"/models": models_vol},
              secrets=[hf_secret], timeout=4 * 3600, retries=2)
def train(
    arm: str,
    base_model: str = DEFAULT_BASE,
    epochs: float = 3.0,
    max_seq: int = 12288,
    batch_size: int = 2,
    grad_accum: int = 4,
    lr: float = 1e-4,
    save_steps: int = 10,
    smoke: bool = False,
    download_if_missing: bool = True,
):
    import os
    import subprocess
    import sys

    assert arm in ("skill", "noskill"), arm
    models_vol.reload()  # see whatever download_base / a prior run committed
    local_base = _local_base_path(base_model)

    if not os.path.isdir(local_base):
        if not download_if_missing:
            raise RuntimeError(f"base model missing: {local_base} (run download_base first)")
        from huggingface_hub import snapshot_download
        print(f"[train:{arm}] base not found, downloading {base_model}")
        snapshot_download(repo_id=base_model, local_dir=local_base,
                          ignore_patterns=["*.pth", "*.gguf", "original/*"])
        models_vol.commit()

    run_name = f"exp2_{arm}" + ("_smoke" if smoke else "")
    cmd = [
        sys.executable, "/root/train_sft.py",
        "--base", local_base,
        "--train", f"/root/data/sft_{arm}_train.jsonl",
        "--val", f"/root/data/sft_{arm}_val.jsonl",
        "--out-dir", "/models/outputs",
        "--run-name", run_name,
        "--max-seq", str(max_seq),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--grad-accum", str(grad_accum),
        "--lr", str(lr),
        "--save-steps", str(save_steps),
        "--resume",               # step-level resume from last checkpoint in run_dir
        "--save-merged",          # also write a merged model for vLLM eval
    ]
    if smoke:
        cmd.append("--smoke")
    print(f"[train:{arm}] running:", " ".join(cmd))
    # commit in finally so checkpoints persist to the Volume even if the run
    # crashes mid-training → a re-run with --resume continues from the last step.
    try:
        subprocess.run(cmd, check=True)
    finally:
        models_vol.commit()
    print(f"[train:{arm}] done -> volume:exp2-distill-models /outputs/{run_name}")
    return run_name


# --------------------------------------------------------------------------- #
# Entrypoints
# --------------------------------------------------------------------------- #
@app.local_entrypoint()
def main(
    arm: str = "skill",
    base_model: str = DEFAULT_BASE,
    epochs: float = 3.0,
    max_seq: int = 12288,
    batch_size: int = 2,
    grad_accum: int = 4,
    lr: float = 1e-4,
    save_steps: int = 10,
    smoke: bool = False,
):
    """Single arm (use for the --smoke validation)."""
    train.remote(arm=arm, base_model=base_model, epochs=epochs, max_seq=max_seq,
                 batch_size=batch_size, grad_accum=grad_accum, lr=lr,
                 save_steps=save_steps, smoke=smoke)


@app.local_entrypoint()
def both(
    base_model: str = DEFAULT_BASE,
    epochs: float = 3.0,
    max_seq: int = 12288,
    batch_size: int = 2,
    grad_accum: int = 4,
    lr: float = 1e-4,
    save_steps: int = 10,
):
    """Train BOTH arms concurrently on two GPUs (download base once first)."""
    print("[both] downloading base model once...")
    download_base.remote(base_model)
    print("[both] spawning skill + noskill in parallel...")
    calls = {
        arm: train.spawn(arm=arm, base_model=base_model, epochs=epochs,
                         max_seq=max_seq, batch_size=batch_size,
                         grad_accum=grad_accum, lr=lr, save_steps=save_steps,
                         download_if_missing=False)
        for arm in ("skill", "noskill")
    }
    for arm, c in calls.items():
        print(f"[both] {arm} -> {c.get()}")
    print("[both] all done.")
