#!/usr/bin/env python3
"""
FastAPI 推理服务 - Qwen3-4B 婚恋分析模型

启动:
    CUDA_VISIBLE_DEVICES=0 python3 serve.py --port 8000

调用:
    curl -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"subject_profile": {...}, "expectations": {...}}'
"""
import argparse
import json
import re
import torch
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "/data/qwang/q/thalia/kvcache/ANNCache/models/Qwen3-4B-Thinking-2507"
ADAPTER_PATH = "outputs/qwen3_4b_zh_skill_ep8_lr2e4/adapter"
TRAIN_DATA = "data/sft_train_v2.jsonl"

model = None
tokenizer = None
system_prompt = None


def load_system_prompt():
    with open(TRAIN_DATA) as f:
        first_line = json.loads(f.readline())
    for msg in first_line['messages']:
        if msg['role'] == 'system':
            return msg['content']
    return ""


def load_model_and_tokenizer():
    print("[1/4] Loading tokenizer...", flush=True)
    tok = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    print("[2/4] Loading base model...", flush=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    print("[3/4] Loading LoRA adapter...", flush=True)
    mdl = PeftModel.from_pretrained(mdl, ADAPTER_PATH, device_map="cuda:0")
    print("[3.5/4] Merging adapter...", flush=True)
    mdl = mdl.merge_and_unload()
    mdl.eval()
    print("[4/4] Model ready.", flush=True)
    return mdl, tok


def generate(user_input: dict) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_input, ensure_ascii=False)},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    generated = outputs[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def try_parse_structured(text: str) -> dict:
    match = re.search(r'```json\s*\n(.*?)\n```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r'\{[^{}]*"matchmaking_intelligence"[^}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, system_prompt
    system_prompt = load_system_prompt()
    model, tokenizer = load_model_and_tokenizer()
    yield


app = FastAPI(title="婚恋分析模型 API", lifespan=lifespan)


class PredictRequest(BaseModel):
    subject_profile: dict
    expectations: dict = {}
    metadata: dict = {}


@app.get("/health")
def health():
    return {"status": "ok", "model": "qwen3_4b_zh_skill_ep8"}


@app.post("/predict")
def predict(req: PredictRequest):
    user_input = req.model_dump(exclude_none=True)
    raw_output = generate(user_input)
    structured = try_parse_structured(raw_output)

    return {
        "case_id": req.metadata.get("case_id", "unknown"),
        "output": raw_output,
        "structured": structured,
    }


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
