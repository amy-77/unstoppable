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
    print(f"[model] loading {base.name}  dtype={dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        base,
        dtype=dtype,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
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
        # SFT specifics
        max_length=args.max_seq,
        packing=False,                      # short-form data, no packing
        assistant_only_loss=True,           # mask user+system tokens from loss
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

    print("[train] starting...")
    trainer.train()

    # ----- save adapter
    adapter_dir = run_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    print(f"[save] adapter → {adapter_dir}")

    # ----- optional merged model
    if args.save_merged and not args.smoke:
        print("[merge] merging LoRA adapter into base model...")
        merged = trainer.model.merge_and_unload()
        merged_dir = run_dir / "merged"
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tok.save_pretrained(merged_dir)
        print(f"[save] merged → {merged_dir}")

    print("[done]")


if __name__ == "__main__":
    main()
