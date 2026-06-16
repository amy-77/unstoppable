#!/usr/bin/env python3
"""Exp3 — generate held-out 60 analyses per (size, mode) via vLLM, WITH the v2 skill.

mode=student  → the SFT'd merged model for that size  (outputs/scale_{size}/merged)
mode=baseline → the base Qwen3-{size} model, NO SFT    (base/{hf_id})
Both use the SAME system prompt (v2 skill + format) → student−baseline = distillation
lift at that size. Generation only; scoring is local in score.py.

RESUMABLE: chunk-level commit to the volume (eval_out/gen_{size}_{mode}.jsonl); a
re-run skips already-generated case_ids. vLLM continuous batching; flashinfer disabled
(no nvcc needed). Sizes × modes run concurrently via spawn.

Run (confirm before launching):
    modal run experiments/exp3_scaling_law/eval_modal.py::one --size 4B --mode student
    modal run experiments/exp3_scaling_law/eval_modal.py::sweep                 # PHASE1 × both modes
    modal run experiments/exp3_scaling_law/eval_modal.py::sweep --sizes 0.6B,1.7B,4B,8B
"""

from __future__ import annotations

import json
import pathlib

import modal

from config import PHASE1, MODES, VOLUME, by_size  # noqa: E402

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm")
    .env({
        "HF_HOME": "/models/hf",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",      # avoid flashinfer's runtime nvcc JIT
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_DISABLE_COMPILE_CACHE": "1",
    })
)
# config.py is a sibling module imported at top level → must be in the container.
if modal.is_local():
    image = base_image.add_local_python_source("config")   # importable in container
else:
    image = base_image

vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")
app = modal.App("exp3-scaling-eval", image=image)


def _chatml(system: str, user: str) -> str:
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")


def _gen_impl(size: str, hf_id: str, mode: str, system_prompt: str,
              items: list[dict], max_tokens: int = 2560, chunk_size: int = 30):
    import json as _json
    import os

    from vllm import LLM, SamplingParams

    vol.reload()
    if mode == "student":
        model_path = f"/models/outputs/scale_{size}/merged"
        if not os.path.isdir(model_path):
            raise RuntimeError(f"student model missing: {model_path} (train {size} first)")
    else:  # baseline: base model, download if missing
        model_path = f"/models/base/{hf_id.split('/')[-1]}"
        if not (os.path.isdir(model_path) and any(f.endswith('.safetensors') for f in os.listdir(model_path))):
            from huggingface_hub import snapshot_download
            print(f"[gen:{size}:{mode}] downloading {hf_id}")
            snapshot_download(repo_id=hf_id, local_dir=model_path,
                              ignore_patterns=["*.pth", "*.gguf", "original/*"])
            vol.commit()

    vol_out = f"/models/eval_out/gen_{size}_{mode}.jsonl"
    os.makedirs(os.path.dirname(vol_out), exist_ok=True)
    done = set()
    if os.path.exists(vol_out):
        for line in open(vol_out, encoding="utf-8"):
            line = line.strip()
            if line:
                done.add(_json.loads(line)["case_id"])
    todo = [it for it in items if it["case_id"] not in done]
    print(f"[gen:{size}:{mode}] total={len(items)} done={len(done)} todo={len(todo)}")

    if todo:
        llm = LLM(model=model_path, dtype="bfloat16", max_model_len=14000,
                  gpu_memory_utilization=0.90, enforce_eager=True)
        # repetition_penalty breaks the degeneration loops small models fall into
        # under pure greedy (e.g. 0.6B repeating a token until max_tokens → unparseable).
        # Stays near-deterministic; applied uniformly across all sizes for comparability.
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens,
                            stop=["<|im_end|>"], repetition_penalty=1.1)
        for i in range(0, len(todo), chunk_size):
            chunk = todo[i:i + chunk_size]
            outs = llm.generate([_chatml(system_prompt, it["user"]) for it in chunk], sp)
            with open(vol_out, "a", encoding="utf-8") as f:
                for it, o in zip(chunk, outs):
                    f.write(_json.dumps({"case_id": it["case_id"], "raw": o.outputs[0].text},
                                        ensure_ascii=False) + "\n")
            vol.commit()
            print(f"[gen:{size}:{mode}] {min(i+chunk_size, len(todo))}/{len(todo)} (committed)")

    return [_json.loads(l) for l in open(vol_out, encoding="utf-8") if l.strip()]


@app.function(gpu="A100-40GB", volumes={"/models": vol}, secrets=[hf_secret], timeout=2400, retries=2)
def gen_40(size, hf_id, mode, system_prompt, items):
    return _gen_impl(size, hf_id, mode, system_prompt, items)


@app.function(gpu="A100-80GB", volumes={"/models": vol}, secrets=[hf_secret], timeout=2400, retries=2)
def gen_80(size, hf_id, mode, system_prompt, items):
    return _gen_impl(size, hf_id, mode, system_prompt, items)


def _fn_for(gpu: str):
    return gen_80 if "80GB" in gpu else gen_40


def _setup():
    """Build the (system_prompt, items) shared by all (size, mode) — done once locally."""
    import sys
    here = pathlib.Path(__file__).resolve().parent
    nv = here.parents[1]
    sys.path.insert(0, str(nv / "experiments" / "common"))
    sys.path.insert(0, str(nv / "experiments" / "exp2_skill_vs_noskill_distill"))
    from data_split import load_split            # noqa: E402
    from prepare_data import build_student_system, case_input_text  # noqa: E402
    system_prompt = build_student_system()
    _, test, _ = load_split(selection_n=40, seed=42)
    items = [{"case_id": c["case_id"], "user": case_input_text(c["input"])} for c in test]
    return system_prompt, items, here


def _save_local(here, size, mode, results):
    import re
    JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
    out_dir = here / "results"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"gen_{size}_{mode}_outputs.jsonl"
    ok = 0
    with path.open("w", encoding="utf-8") as f:
        for r in results:
            m = JSON_BLOCK.findall(r["raw"])
            parsed = None
            if m:
                try:
                    parsed = json.loads(m[-1])
                except json.JSONDecodeError:
                    try:
                        parsed = json.loads(re.sub(r",(\s*[}\]])", r"\1", m[-1]))
                    except json.JSONDecodeError:
                        parsed = None
            ok += parsed is not None
            f.write(json.dumps({"case_id": r["case_id"], "size": size, "mode": mode,
                                "output": parsed, "raw": r["raw"]}, ensure_ascii=False) + "\n")
    print(f"[eval] {size}:{mode} parsed {ok}/{len(results)} → {path.name}")


@app.local_entrypoint()
def one(size: str = "4B", mode: str = "student"):
    system_prompt, items, here = _setup()
    s = by_size(size)
    results = _fn_for(s["gpu"]).remote(size, s["hf_id"], mode, system_prompt, items)
    _save_local(here, size, mode, results)


@app.local_entrypoint()
def sweep(sizes: str = "", modes: str = ""):
    system_prompt, items, here = _setup()
    names = [x.strip() for x in sizes.split(",") if x.strip()] or PHASE1
    want_modes = [x.strip() for x in modes.split(",") if x.strip()] or MODES
    jobs = [(by_size(n), m) for n in names for m in want_modes]
    print(f"[exp3] generating {[(s['size'], m) for s, m in jobs]} concurrently...")
    calls = [(s["size"], m, _fn_for(s["gpu"]).spawn(s["size"], s["hf_id"], m, system_prompt, items))
             for s, m in jobs]
    for size, mode, c in calls:
        _save_local(here, size, mode, c.get())
    print("[exp3] eval-gen sweep done.")
