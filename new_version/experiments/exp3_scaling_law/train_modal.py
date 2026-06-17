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
    "liger-kernel==0.8.0",   # fused lm_head+CE → kills the 151K-vocab logits memory bomb
    "einops",                # flash_attn import dep (we install the wheel with --no-deps)
]

# flash-attn 2.8.3.post1 — official cu12torch2.7 cp311 wheels, BOTH C++ ABIs.
# Hard lesson: letting the Modal builder `pip install` straight from the GitHub
# release URL hangs (EU builder → github releases gets throttled to a near-stall,
# no progress, no timeout). So we vendor BOTH wheels locally (see wheels/, fetched
# by hand) and copy them into the image; the build step then `pip install`s the one
# matching torch's compiled ABI. No builder→github, no ABI guess.
FA_VERSION = "2.8.3.post1"
def _fa_whl(abi: str) -> str:
    return f"flash_attn-{FA_VERSION}+cu12torch2.7cxx11abi{abi}-cp311-cp311-linux_x86_64.whl"

# Detect torch's ABI at build time, install the matching vendored wheel (--no-deps:
# don't let pip re-resolve against the index), then prove the import works.
FA_INSTALL = (
    "ABI=$(python -c \"import torch;print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')\") && "
    "echo \"[fa] torch cxx11abi=$ABI\" && "
    f"pip install --no-deps /root/wheels/flash_attn-{FA_VERSION}+cu12torch2.7cxx11abi${{ABI}}-cp311-cp311-linux_x86_64.whl && "
    "python -c \"import flash_attn, importlib.metadata as m; print('[fa] flash_attn', m.version('flash_attn'), 'import OK')\""
)

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*DATASETS_STACK)
    .env({"HF_HOME": "/models/hf", "TOKENIZERS_PARALLELISM": "false"})
)

# Local-only: vendor flash-attn wheels + mount the shared engine + exp2's skill SFT
# data (guarded for container, where these local paths don't exist).
if modal.is_local():
    HERE = pathlib.Path(__file__).resolve().parent
    NEW_VERSION = HERE.parents[1]
    TRAIN_SCRIPT = NEW_VERSION / "matchmaker" / "training" / "train_sft.py"
    EXP2_DATA = NEW_VERSION / "experiments" / "exp2_skill_vs_noskill_distill" / "data"
    WHEELS = HERE / "wheels"
    assert TRAIN_SCRIPT.exists(), TRAIN_SCRIPT
    assert (EXP2_DATA / "sft_skill_train.jsonl").exists(), \
        "run exp2 prepare_data.py --arm skill first (sft_skill_train.jsonl missing)"
    for abi in ("TRUE", "FALSE"):
        assert (WHEELS / _fa_whl(abi)).exists(), \
            f"missing vendored wheel {WHEELS / _fa_whl(abi)} — download both ABI wheels into wheels/"
    image = (
        base_image
        # copy=True → wheels live in the image *at build time* so FA_INSTALL can use them
        .add_local_file(str(WHEELS / _fa_whl("TRUE")), f"/root/wheels/{_fa_whl('TRUE')}", copy=True)
        .add_local_file(str(WHEELS / _fa_whl("FALSE")), f"/root/wheels/{_fa_whl('FALSE')}", copy=True)
        .run_commands(FA_INSTALL)
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
                batch_size: int, grad_accum: int, lr: float, save_steps: int,
                smoke: bool = False, device_map: str = "single",
                save_merged: bool = True, fsdp: bool = False, nproc: int = 1):
    import os
    import subprocess
    import sys

    vol.reload()
    local_base = f"/models/base/{hf_id.split('/')[-1]}"
    run_name = f"smoke_{size}" if smoke else f"scale_{size}"
    # The "this run already finished" marker: merged dir when we merge, else the
    # adapter dir (32B saves adapter only — no 65GB merge in the pricey GPU window).
    done_marker = (f"/models/outputs/{run_name}/merged" if save_merged
                   else f"/models/outputs/{run_name}/adapter")

    # smoke = stack validation (FA2 wheel ABI + Liger Qwen3 patch); never skip on
    # an existing output, and run only 2 steps on 4 samples.
    if not smoke and os.path.isdir(done_marker) and os.listdir(done_marker):
        print(f"[train:{size}] {done_marker} already exists → skip"); return run_name

    if not (os.path.isdir(local_base) and any(f.endswith('.safetensors') for f in os.listdir(local_base))):
        # Fallback download (prefer pre-staging via `stage` so the GPU clock never
        # pays for this 65GB pull — see prestage()).
        from huggingface_hub import snapshot_download
        print(f"[train:{size}] base not on volume → downloading {hf_id} (consider `stage` first)")
        snapshot_download(repo_id=hf_id, local_dir=local_base,
                          ignore_patterns=["*.pth", "*.gguf", "original/*"])
        vol.commit()

    # FSDP needs one process per GPU → accelerate launch. Naive-MP / single → plain python.
    if fsdp:
        launch = [sys.executable, "-m", "accelerate.commands.launch",
                  "--num_processes", str(nproc), "--num_machines", "1",
                  "--mixed_precision", "bf16", "/root/train_sft.py"]
    else:
        launch = [sys.executable, "/root/train_sft.py"]

    cmd = launch + [
        "--base", local_base,
        "--train", "/root/data/sft_skill_train.jsonl",
        "--val", "/root/data/sft_skill_val.jsonl",
        "--out-dir", "/models/outputs", "--run-name", run_name,
        "--max-seq", str(max_seq), "--epochs", str(epochs),
        "--batch-size", str(batch_size), "--grad-accum", str(grad_accum),
        "--lr", str(lr), "--save-steps", str(save_steps),
        "--resume",
    ]
    if fsdp:
        cmd.append("--fsdp")
    elif device_map != "single":
        cmd += ["--device-map", device_map]   # auto = naive MP across the visible GPUs
    if save_merged:
        cmd.append("--save-merged")
    if smoke:
        cmd.append("--smoke")          # 2 steps / 4 samples; --save-merged self-guards off

    env = os.environ.copy()
    if fsdp:
        # accelerate launch (the parent) must see these BEFORE it execs train_sft.py,
        # so it picks the FSDP distributed type and transformers loads ram-efficiently.
        env["ACCELERATE_USE_FSDP"] = "true"
        env["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "true"

    print(f"[train:{size}] {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, env=env)
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
def train_40(size: str, hf_id: str, smoke: bool = False, **kw):
    return _train_impl(size, hf_id, smoke=smoke, **_common(**kw))


@app.function(gpu="A100-80GB", volumes={"/models": vol}, secrets=[hf_secret], timeout=6 * 3600, retries=2)
def train_80(size: str, hf_id: str, smoke: bool = False, **kw):
    # 80GB fits batch_size=2 even with the 12K logits → ~2x faster than batch 1.
    # grad_accum=4 keeps the effective batch at 8. timeout 6h for 14B.
    kw.setdefault("batch_size", 2)
    kw.setdefault("grad_accum", 4)
    return _train_impl(size, hf_id, smoke=smoke, **_common(**kw))


@app.function(gpu="H100:2", volumes={"/models": vol}, secrets=[hf_secret], timeout=6 * 3600, retries=2)
def train_h100x2(size: str, hf_id: str, smoke: bool = False, **kw):
    # 32B bf16 LoRA via FSDP full-shard (ZeRO-3) across 2×H100: shards the frozen
    # base AND data-parallels (both cards compute) → ~2x over naive MP. Effective
    # batch = batch_size(1) × grad_accum(4) × nproc(2) = 8 → comparable to the other
    # points (grad_accum is PER-RANK under FSDP, hence 4 not 8). Save adapter only —
    # no 65GB merge in the 2-GPU window; eval mounts the LoRA.
    # Fallback if FSDP misbehaves: device_map="auto", save_merged=False (naive MP).
    kw.setdefault("batch_size", 1)
    kw.setdefault("grad_accum", 4)
    return _train_impl(size, hf_id, smoke=smoke,
                       fsdp=True, nproc=2, save_merged=False, **_common(**kw))


def _fn_for(gpu: str):
    if gpu.startswith("H100"):
        return train_h100x2
    return train_80 if "80GB" in gpu else train_40


@app.function(volumes={"/models": vol}, secrets=[hf_secret], timeout=2 * 3600)
def prestage(hf_id: str):
    """CPU-only: pull a base model onto the volume so the GPU clock never pays for
    the download. Idempotent — skips if the weights are already there."""
    import os
    from huggingface_hub import snapshot_download

    vol.reload()
    local_base = f"/models/base/{hf_id.split('/')[-1]}"
    if os.path.isdir(local_base) and any(f.endswith('.safetensors') for f in os.listdir(local_base)):
        print(f"[prestage] {local_base} already present → skip"); return
    print(f"[prestage] downloading {hf_id} → {local_base} (no GPU billed)")
    snapshot_download(repo_id=hf_id, local_dir=local_base,
                      ignore_patterns=["*.pth", "*.gguf", "original/*"])
    vol.commit()
    print(f"[prestage] done → volume:{VOLUME} /base/{hf_id.split('/')[-1]}")


@app.local_entrypoint()
def stage(size: str = "32B"):
    # Run this BEFORE training the big sizes: the 65GB pull happens on a cheap CPU
    # container, then train_h100x2 finds the weights already on the volume.
    s = by_size(size)
    print(f"[exp3] prestage {size} ({s['hf_id']}) — CPU download to volume")
    prestage.remote(s["hf_id"])


@app.local_entrypoint()
def smoke(size: str = "0.6B"):
    # Cheap stack check on the new image: FA2 wheel ABI + Liger Qwen3 patch.
    # 2 steps / 4 samples; never skips on existing merged. ~minutes, ~$0.2.
    s = by_size(size)
    print(f"[exp3] SMOKE {size} ({s['hf_id']}) on {s['gpu']} — validate FA2 + Liger")
    _fn_for(s["gpu"]).remote(size=s["size"], hf_id=s["hf_id"], smoke=True)


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
