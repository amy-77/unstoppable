#!/usr/bin/env python3
"""Exp3 — LoRA SFT the (fixed) skill-teacher data on each Qwen3 dense size.

Reuses exp2's `matchmaker/training/train_sft.py` (the engine) and exp2's
`sft_skill_{train,val}.jsonl` (the data) verbatim — the ONLY thing that varies
per run is `--base` (the model size from config.py).

Resumable at two levels: size-level (skip a size whose merged model already
exists) + step-level (`train_sft.py --resume` from the last checkpoint). Each
size auto-retries on flaky GPUs (retries=2). Sizes run concurrently via spawn.

Run (ALWAYS confirm before launching — Modal billing):
    modal run experiments/exp3_scaling_law/train_modal.py::one   --size 0.6B
    modal run experiments/exp3_scaling_law/train_modal.py::sweep              # PHASE1 in parallel
    modal run experiments/exp3_scaling_law/train_modal.py::sweep --sizes 0.6B,1.7B,4B,8B
"""

from __future__ import annotations

import pathlib

import modal

from config import SIZES, PHASE1, VOLUME, by_size  # noqa: E402

DATASETS_STACK = [
    "torch==2.7.1", "transformers==4.57.6", "trl==0.23.1",
    "peft==0.19.1", "accelerate==1.14.0", "datasets==5.0.0", "sentencepiece==0.2.1",
]

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*DATASETS_STACK)
    .env({"HF_HOME": "/models/hf", "TOKENIZERS_PARALLELISM": "false"})
)

# Local-only: mount the shared engine + exp2's skill SFT data (guarded for container).
if modal.is_local():
    HERE = pathlib.Path(__file__).resolve().parent
    NEW_VERSION = HERE.parents[1]
    TRAIN_SCRIPT = NEW_VERSION / "matchmaker" / "training" / "train_sft.py"
    EXP2_DATA = NEW_VERSION / "experiments" / "exp2_skill_vs_noskill_distill" / "data"
    assert TRAIN_SCRIPT.exists(), TRAIN_SCRIPT
    assert (EXP2_DATA / "sft_skill_train.jsonl").exists(), \
        "run exp2 prepare_data.py --arm skill first (sft_skill_train.jsonl missing)"
    image = (
        base_image
        .add_local_python_source("config")          # sibling module → importable in container
        .add_local_file(str(TRAIN_SCRIPT), "/root/train_sft.py")
        .add_local_file(str(EXP2_DATA / "sft_skill_train.jsonl"), "/root/data/sft_skill_train.jsonl")
        .add_local_file(str(EXP2_DATA / "sft_skill_val.jsonl"), "/root/data/sft_skill_val.jsonl")
    )
else:
    image = base_image

vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")
app = modal.App("exp3-scaling-train", image=image)


def _train_impl(size: str, hf_id: str, epochs: float, max_seq: int,
                batch_size: int, grad_accum: int, lr: float, save_steps: int):
    import os
    import subprocess
    import sys

    vol.reload()
    local_base = f"/models/base/{hf_id.split('/')[-1]}"
    run_name = f"scale_{size}"
    merged = f"/models/outputs/{run_name}/merged"

    if os.path.isdir(merged) and os.listdir(merged):
        print(f"[train:{size}] merged already exists → skip"); return run_name

    if not (os.path.isdir(local_base) and any(f.endswith('.safetensors') for f in os.listdir(local_base))):
        from huggingface_hub import snapshot_download
        print(f"[train:{size}] downloading {hf_id}")
        snapshot_download(repo_id=hf_id, local_dir=local_base,
                          ignore_patterns=["*.pth", "*.gguf", "original/*"])
        vol.commit()

    cmd = [
        sys.executable, "/root/train_sft.py",
        "--base", local_base,
        "--train", "/root/data/sft_skill_train.jsonl",
        "--val", "/root/data/sft_skill_val.jsonl",
        "--out-dir", "/models/outputs", "--run-name", run_name,
        "--max-seq", str(max_seq), "--epochs", str(epochs),
        "--batch-size", str(batch_size), "--grad-accum", str(grad_accum),
        "--lr", str(lr), "--save-steps", str(save_steps),
        "--resume", "--save-merged",
    ]
    print(f"[train:{size}] {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    finally:
        vol.commit()          # persist checkpoints even on failure → step-level resume
    print(f"[train:{size}] done → volume:{VOLUME} /outputs/{run_name}")
    return run_name


# Modal GPU is static per function, so one thin wrapper per GPU tier; dispatch by config.
def _common(**kw):
    # batch_size=1: the 12K-seq × 151K-vocab logits (+ fp32 grad) are the memory
    # bomb, not the weights — batch 2 OOMs even 0.6B on 40GB. grad_accum=8 keeps
    # the effective batch at 8.
    return dict(epochs=kw.get("epochs", 3.0), max_seq=kw.get("max_seq", 12288),
                batch_size=kw.get("batch_size", 1), grad_accum=kw.get("grad_accum", 8),
                lr=kw.get("lr", 1e-4), save_steps=kw.get("save_steps", 10))


@app.function(gpu="A100-40GB", volumes={"/models": vol}, secrets=[hf_secret], timeout=4 * 3600, retries=2)
def train_40(size: str, hf_id: str, **kw):
    return _train_impl(size, hf_id, **_common(**kw))


@app.function(gpu="A100-80GB", volumes={"/models": vol}, secrets=[hf_secret], timeout=6 * 3600, retries=2)
def train_80(size: str, hf_id: str, **kw):
    # 80GB fits batch_size=2 even with the 12K logits → ~2x faster than batch 1.
    # grad_accum=4 keeps the effective batch at 8. timeout 6h for 14B.
    kw.setdefault("batch_size", 2)
    kw.setdefault("grad_accum", 4)
    return _train_impl(size, hf_id, **_common(**kw))


def _fn_for(gpu: str):
    return train_80 if "80GB" in gpu else train_40


@app.local_entrypoint()
def one(size: str = "0.6B"):
    s = by_size(size)
    print(f"[exp3] train {size} ({s['hf_id']}) on {s['gpu']}")
    _fn_for(s["gpu"]).remote(size=s["size"], hf_id=s["hf_id"])


@app.local_entrypoint()
def sweep(sizes: str = ""):
    names = [x.strip() for x in sizes.split(",") if x.strip()] or PHASE1
    targets = [by_size(n) for n in names]
    print(f"[exp3] training {[t['size'] for t in targets]} concurrently...")
    calls = [(t["size"], _fn_for(t["gpu"]).spawn(size=t["size"], hf_id=t["hf_id"])) for t in targets]
    for name, c in calls:
        print(f"[exp3] {name} → {c.get()}")
    print("[exp3] sweep done.")
