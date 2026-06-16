#!/usr/bin/env python3
"""Exp2 eval — generate the two students' analyses on the held-out 60 via vLLM.

Both students run WITH the v2 skill in their system prompt (== training/deploy
condition); the system prompt is built by prepare_data.build_student_system() so
train and inference match byte-for-byte. The user message is rendered exactly as
in training (case_input_text over {subject_profile, expectations}).

This only does GENERATION on GPU. Scoring (the 3-judge jury) runs locally in
score_students.py (it needs the OpenAI/Gemini/DeepSeek keys, not a GPU).

Run (per arm; merged models already in the exp2-distill-models volume):
    modal run experiments/exp2_skill_vs_noskill_distill/eval_modal.py --arm skill   --limit 2   # smoke
    modal run experiments/exp2_skill_vs_noskill_distill/eval_modal.py --arm skill
    modal run experiments/exp2_skill_vs_noskill_distill/eval_modal.py --arm noskill

Writes locally: results/gen_student_<arm>_outputs.jsonl  ({case_id, output, raw}).
"""

from __future__ import annotations

import json
import pathlib

import modal

DEFAULT_BASE = "Qwen/Qwen3-4B-Thinking-2507"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm")
    # Disable flashinfer (it JIT-compiles CUDA kernels at runtime → needs nvcc,
    # which debian_slim lacks). Use vllm's prebuilt flash-attn backend + the
    # non-flashinfer sampler instead → no nvcc needed.
    .env({
        "HF_HOME": "/models/hf",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_DISABLE_COMPILE_CACHE": "1",
    })
)

models_vol = modal.Volume.from_name("exp2-distill-models")
app = modal.App("exp2-eval-gen", image=image)


# Exact ChatML prompt the student was trained on (SIMPLE_CHAT_TEMPLATE rendered).
# Built manually so we bypass the saved chat template's {% generation %} markers.
def _chatml(system: str, user: str) -> str:
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n")


# A100-40GB: same bandwidth/speed as the 80GB but ~43% cheaper; plenty for a 4B.
# retries=2 for flaky-GPU faults.
@app.function(gpu="A100-40GB", volumes={"/models": models_vol}, timeout=2400, retries=2)
def generate(arm: str, system_prompt: str, items: list[dict],
             max_tokens: int = 2560, chunk_size: int = 30) -> list[dict]:
    """items = [{"case_id":..., "user":<rendered user text>}]. vLLM continuous
    batching (5-8x faster than HF generate).

    RESUMABLE: commits each chunk to a volume file, so a kill/crash loses at most
    one chunk — a re-run skips already-generated case_ids. Returns the full
    [{"case_id":..., "raw":...}] from the volume file."""
    import json as _json
    import os

    from vllm import LLM, SamplingParams

    models_vol.reload()
    vol_out = f"/models/eval_out/gen_student_{arm}.jsonl"
    os.makedirs(os.path.dirname(vol_out), exist_ok=True)

    done = set()
    if os.path.exists(vol_out):
        for line in open(vol_out, encoding="utf-8"):
            line = line.strip()
            if line:
                done.add(_json.loads(line)["case_id"])
    todo = [it for it in items if it["case_id"] not in done]
    print(f"[gen:{arm}] total={len(items)} done={len(done)} todo={len(todo)}")

    if todo:
        merged = f"/models/outputs/exp2_{arm}/merged"
        llm = LLM(model=merged, dtype="bfloat16", max_model_len=14000,
                  gpu_memory_utilization=0.90, enforce_eager=True)
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens,
                            stop=["<|im_end|>"])
        for i in range(0, len(todo), chunk_size):
            chunk = todo[i:i + chunk_size]
            outs = llm.generate([_chatml(system_prompt, it["user"]) for it in chunk], sp)
            with open(vol_out, "a", encoding="utf-8") as f:        # incremental
                for it, o in zip(chunk, outs):
                    f.write(_json.dumps({"case_id": it["case_id"],
                                         "raw": o.outputs[0].text},
                                        ensure_ascii=False) + "\n")
            models_vol.commit()                                    # persist chunk
            print(f"[gen:{arm}] {min(i+chunk_size, len(todo))}/{len(todo)} (committed)")

    return [_json.loads(l) for l in open(vol_out, encoding="utf-8") if l.strip()]


@app.local_entrypoint()
def main(arm: str = "skill", limit: int = 0):
    import re
    import sys

    here = pathlib.Path(__file__).resolve().parent
    new_version = here.parents[1]
    sys.path.insert(0, str(new_version / "experiments" / "common"))
    sys.path.insert(0, str(here))
    from data_split import load_split            # noqa: E402
    from prepare_data import build_student_system, case_input_text  # noqa: E402

    system_prompt = build_student_system()
    _, test, _ = load_split(selection_n=40, seed=42)   # held-out 60
    if limit:
        test = test[:limit]
    items = [{"case_id": c["case_id"],
              "user": case_input_text(c["input"])} for c in test]
    print(f"[eval] arm={arm} cases={len(items)} system={len(system_prompt)} chars")

    results = generate.remote(arm, system_prompt, items)

    # parse <thinking> + json locally; save outputs jsonl (parsed + raw)
    JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
    out_dir = here / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"gen_student_{arm}_outputs.jsonl"
    ok = 0
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            raw = r["raw"]
            m = JSON_BLOCK.findall(raw)
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
            f.write(json.dumps({"case_id": r["case_id"], "arm": arm,
                                "output": parsed, "raw": raw}, ensure_ascii=False) + "\n")
    print(f"[eval] arm={arm}: parsed {ok}/{len(results)} -> {out_path}")
