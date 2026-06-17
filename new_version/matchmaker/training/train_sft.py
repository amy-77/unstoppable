#!/usr/bin/env python3
"""LoRA SFT training for the matchmaker distillation.

Inputs:
  data/sft_train.jsonl + data/sft_val.jsonl
  Each record has {"messages": [system, user, assistant], "case_id", "lang"}.

Pipeline:
  1. Load tokenizer + base model (4-bit optional, default bf16).
  2. Wrap in LoRA via peft.
  3. Use trl.SFTTrainer with completions-only loss (assistant tokens only).
  4. Save adapter to outputs/<run_name>/adapter and full merged model
     to outputs/<run_name>/merged (optional).

Usage:
    CUDA_VISIBLE_DEVICES=0 python train_sft.py \
        --base /data/qwang/q/thalia/kvcache/ANNCache/models/Qwen3-4B-Thinking-2507 \
        --train data/sft_train.jsonl --val data/sft_val.jsonl \
        --run-name qwen3_4b_zh_smoke \
        --epochs 3 --batch-size 2 --grad-accum 4 --lr 1e-4 --max-seq 3072

For a smoke test (1 sample, 1 step):
    python train_sft.py --base ... --train data/sft_train.jsonl --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True,
                   help="path to base model (HF dir)")
    p.add_argument("--train", default="data/sft_train.jsonl")
    p.add_argument("--val",   default="data/sft_val.jsonl")
    p.add_argument("--out-dir", default="outputs",
                   help="parent dir for run outputs")
    p.add_argument("--run-name", default=None,
                   help="name of this run (subdir of out-dir)")

    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=2,
                   help="per-device train batch size")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--max-seq", type=int, default=3072,
                   help="max sequence length (tokens) — truncate longer")
    p.add_argument("--seed", type=int, default=42)

    # LoRA
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target", default="all-linear",
                   help="all-linear | comma-separated module names")

    # Misc
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.add_argument("--device-map", default="single",
                   help="single = whole model on cuda:0 (default, fits ≤14B bf16 LoRA "
                        "on one 80GB card) | auto = naive model-parallel, split layers "
                        "across ALL visible GPUs (for models too big for one card, e.g. "
                        "32B bf16 needs 2×H100). HF Trainer sees hf_device_map and runs "
                        "model-parallel — does NOT wrap in DataParallel.")
    p.add_argument("--fsdp", action="store_true",
                   help="train under FSDP full-shard (ZeRO-3): shard the frozen base "
                        "across GPUs AND data-parallel (both cards compute) → ~2x over "
                        "naive MP. MUST be launched via `accelerate launch "
                        "--num_processes N`. Overrides --device-map (model is loaded "
                        "unplaced; FSDP shards it). grad_accum is per-rank, so the "
                        "effective batch = batch × grad_accum × N.")
    p.add_argument("--attn-impl", default="flash_attention_2",
                   help="attention kernel: flash_attention_2 (fast, needs flash-attn "
                        "+ GPU) | sdpa (portable fallback) | eager")
    p.add_argument("--use-liger", action="store_true", default=True,
                   help="fuse lm_head+cross-entropy via Liger kernel — avoids "
                        "materializing the [seq, 151K-vocab] logits (the memory bomb)")
    p.add_argument("--no-liger", dest="use_liger", action="store_false")
    p.add_argument("--save-merged", action="store_true",
                   help="also save a fully merged (LoRA-merged) model")
    p.add_argument("--smoke", action="store_true",
                   help="train for 2 steps on 4 samples; for pipeline check")
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--no-grad-ckpt", dest="gradient_checkpointing",
                   action="store_false")
    p.add_argument("--logging-steps", type=int, default=1,
                   help="print metrics every N optimizer steps (loss appears at these intervals)")
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--eval-steps", type=int, default=20)
    p.add_argument("--resume", action="store_true",
                   help="resume from the last checkpoint in run_dir if one exists "
                        "(step-level resume; safe no-op when none found)")
    return p.parse_args()


def main():
    args = parse_args()

    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainerCallback,
        set_seed,
    )
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig

    set_seed(args.seed)

    # FSDP: this MUST be in the env BEFORE from_pretrained so transformers'
    # is_fsdp_enabled() loads weights on rank0 + meta elsewhere (ram-efficient)
    # instead of a full 64GB copy per process. accelerate launch normally sets it
    # from the env we export; setdefault here covers a direct invocation too.
    if args.fsdp:
        os.environ.setdefault("ACCELERATE_USE_FSDP", "true")
        os.environ.setdefault("FSDP_CPU_RAM_EFFICIENT_LOADING", "true")

    base = Path(args.base)
    if not base.exists():
        sys.exit(f"base model not found: {base}")

    # ----- run dir
    if not args.run_name:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"{base.name}__{ts}"
    run_dir = Path(args.out_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2,
                                                  ensure_ascii=False))
    print(f"[run] {run_dir}")

    # ----- tokenizer
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.padding_side != "right":
        # SFTTrainer wants right-padding for causal LM training
        tok.padding_side = "right"

    # Override the chat template with a simple, training-compatible one that
    # marks the assistant span with `{% generation %}...{% endgeneration %}`,
    # which trl's `assistant_only_loss=True` requires. Qwen3-Thinking's
    # original template also force-injects a `<think>\n` opener for assistant
    # turns, which conflicts with our SFT data that uses `<thinking>...</thinking>`.
    SIMPLE_CHAT_TEMPLATE = (
        "{%- for message in messages -%}"
        "{%- if message.role == 'system' -%}"
        "<|im_start|>system\n{{ message.content }}<|im_end|>\n"
        "{%- elif message.role == 'user' -%}"
        "<|im_start|>user\n{{ message.content }}<|im_end|>\n"
        "{%- elif message.role == 'assistant' -%}"
        "<|im_start|>assistant\n"
        "{% generation %}{{ message.content }}<|im_end|>{% endgeneration %}\n"
        "{%- endif -%}"
        "{%- endfor -%}"
        "{%- if add_generation_prompt -%}"
        "<|im_start|>assistant\n"
        "{%- endif -%}"
    )
    tok.chat_template = SIMPLE_CHAT_TEMPLATE
    print(f"[tok] vocab={len(tok)} pad={tok.pad_token!r} eos={tok.eos_token!r}")
    print(f"[tok] chat template overridden (training-compatible, no forced <think>)")

    # ----- model (bf16, full-precision base; LoRA adds trainable adapters)
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model_kwargs = dict(dtype=dtype, trust_remote_code=True,
                        attn_implementation=args.attn_impl)
    if args.fsdp:
        # FSDP shards the model itself — it must be loaded UNplaced (no device_map).
        # low_cpu_mem_usage + the FSDP env → rank0 loads real weights, others meta.
        model_kwargs["low_cpu_mem_usage"] = True
        print(f"[model] loading {base.name}  dtype={dtype}  mode=FSDP (unplaced)")
    else:
        # single = pin everything to cuda:0; auto = naive model-parallel split across
        # all visible GPUs (the only way 32B bf16 fits on 2×H100 without FSDP).
        model_kwargs["device_map"] = {"": 0} if args.device_map == "single" else args.device_map
        print(f"[model] loading {base.name}  dtype={dtype}  device_map={model_kwargs['device_map']}")
    model = AutoModelForCausalLM.from_pretrained(base, **model_kwargs)
    if getattr(model, "hf_device_map", None) is not None:
        print(f"[model] hf_device_map spans devices: "
              f"{sorted(set(str(d) for d in model.hf_device_map.values()))}")
    print(f"[model] attn_implementation={args.attn_impl}  use_liger={args.use_liger}")
    model.config.use_cache = False  # required for grad ckpt
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # ----- LoRA
    target = args.lora_target if args.lora_target == "all-linear" \
        else [s.strip() for s in args.lora_target.split(",") if s.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target,
    )
    model = get_peft_model(model, lora_cfg)
    if args.fsdp and args.gradient_checkpointing:
        # PEFT + grad-ckpt under FSDP: the frozen base produces activations with no
        # requires_grad, so checkpointing has nothing to recompute grads through —
        # force the input embeddings to require grad to bridge it.
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    # ----- dataset
    print(f"[data] train={args.train}  val={args.val}")
    data_files = {"train": args.train}
    if args.val and Path(args.val).exists():
        data_files["validation"] = args.val
    ds = load_dataset("json", data_files=data_files)
    print(f"[data] sizes: { {k: len(v) for k, v in ds.items()} }")

    # smoke: subsample
    if args.smoke:
        ds["train"] = ds["train"].select(range(min(4, len(ds["train"]))))
        if "validation" in ds:
            ds["validation"] = ds["validation"].select(
                range(min(2, len(ds["validation"])))
            )
        print(f"[smoke] ds: { {k: len(v) for k, v in ds.items()} }")

    # ----- SFT config
    cfg = SFTConfig(
        output_dir=str(run_dir),
        num_train_epochs=args.epochs if not args.smoke else 1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=args.bf16,
        max_grad_norm=1.0,
        logging_steps=max(1, args.logging_steps),
        logging_first_step=True,
        log_level="info",
        disable_tqdm=False,
        save_strategy="steps" if not args.smoke else "no",
        save_steps=args.save_steps,
        eval_strategy="steps" if "validation" in ds and not args.smoke else "no",
        eval_steps=args.eval_steps,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        # FSDP full-shard (ZeRO-3). Empty string → no FSDP (single / naive-MP path).
        # The plugin details below are ignored unless fsdp is set. use_orig_params is
        # REQUIRED for LoRA (mixed frozen/trainable params); FULL_STATE_DICT keeps the
        # tiny adapter save simple (gather to rank0, no sharded checkpoint files).
        fsdp=("full_shard auto_wrap" if args.fsdp else ""),
        fsdp_config=({
            "transformer_layer_cls_to_wrap": "Qwen3DecoderLayer",
            "use_orig_params": True,
            "sync_module_states": True,
            "cpu_ram_efficient_loading": True,
            "backward_prefetch": "backward_pre",
            "forward_prefetch": False,
            "limit_all_gathers": True,
            "state_dict_type": "FULL_STATE_DICT",
            "activation_checkpointing": False,   # we use HF gradient_checkpointing
        } if args.fsdp else None),
        # SFT specifics
        max_length=args.max_seq,
        packing=False,                      # short-form data, no packing
        assistant_only_loss=True,           # mask user+system tokens from loss
        use_liger_kernel=args.use_liger,    # fuse lm_head+CE → no full logits tensor
        max_steps=2 if args.smoke else -1,
    )

    class LossPrinterCallback(TrainerCallback):
        """Explicit console lines so loss is visible above tqdm / nohup logs."""

        def on_log(self, args, state, control, logs=None, **_kw):  # noqa: ARG002
            if not logs:
                return
            if "loss" in logs:
                lr = logs.get("learning_rate", float("nan"))
                epoch = logs.get("epoch", float("nan"))
                print(
                    f"[train] step={state.global_step}/{state.max_steps} "
                    f"epoch={epoch:.4f} loss={logs['loss']:.6f} lr={lr:.3e}",
                    flush=True,
                )
            if "eval_loss" in logs:
                print(f"[eval] step={state.global_step} eval_loss={logs['eval_loss']:.6f}", flush=True)

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        processing_class=tok,
        callbacks=[LossPrinterCallback()],
    )

    resume_ckpt = None
    if args.resume:
        from transformers.trainer_utils import get_last_checkpoint
        resume_ckpt = get_last_checkpoint(str(run_dir)) if run_dir.exists() else None
        print(f"[resume] last checkpoint: {resume_ckpt or 'none — starting fresh'}")

    print("[train] starting...")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # ----- save adapter
    adapter_dir = run_dir / "adapter"
    if args.fsdp:
        # Params are sharded across ranks — trainer.save_model gathers the
        # FULL_STATE_DICT to rank0 and writes the PEFT adapter there.
        trainer.save_model(str(adapter_dir))
        if trainer.accelerator.is_main_process:
            tok.save_pretrained(adapter_dir)
            print(f"[save] adapter (FSDP-gathered) → {adapter_dir}")
        trainer.accelerator.wait_for_everyone()
    else:
        trainer.model.save_pretrained(adapter_dir)
        tok.save_pretrained(adapter_dir)
        print(f"[save] adapter → {adapter_dir}")

    # ----- optional merged model (single-device path only; merging a sharded /
    # naive-MP model is fragile, and 32B intentionally ships adapter-only)
    if args.save_merged and not args.smoke and not args.fsdp:
        print("[merge] merging LoRA adapter into base model...")
        merged = trainer.model.merge_and_unload()
        merged_dir = run_dir / "merged"
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tok.save_pretrained(merged_dir)
        print(f"[save] merged → {merged_dir}")

    print("[done]")


if __name__ == "__main__":
    main()
