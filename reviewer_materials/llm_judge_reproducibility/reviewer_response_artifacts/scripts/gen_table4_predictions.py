#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import BitsAndBytesConfig, GenerationConfig, LlamaTokenizer


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.causal_lm_llama import LlamaContinualForCausalLM
from mpeft import LoraConfig, TaskType, get_peft_model
from tasks.mtl5.dataloader_mtl_causal_llama import PROMPT_TEMPLATE
from utils.inference_utils import generate_memr_response, load_memr_model


TASKS = ["neike", "waike", "erke", "fuchanke", "nanke", "zhongliuke"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate compare-format predictions for TABLE 4.")
    parser.add_argument("--mode", required=True, choices=["memr", "plain"], help="'memr' for MeMR-style checkpoints, 'plain' for single-adapter LoRA checkpoints")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_model_path", default="chinese-alpaca-plus-7b-hf")
    parser.add_argument("--dataset_root", default="datasets/medical_consult")
    parser.add_argument("--meta_embeddings_path", default="metadata_embeddings/keshi_meta_embeddings.pt")
    parser.add_argument("--inference_log_dir", default="/tmp/memr_inference_logs")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    return parser.parse_args()


def build_tokenizer(base_model_path: str):
    tokenizer = LlamaTokenizer.from_pretrained(
        base_model_path,
        add_prefix_space=True,
        padding_side="left",
        trust_remote_code=True,
    )
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 1
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_quantization_config():
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_plain_lora_model(base_model_path: str, checkpoint_dir: Path):
    tokenizer = build_tokenizer(base_model_path)
    quantization_config = build_quantization_config()
    model = LlamaContinualForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        output_dir=str(checkpoint_dir / "lora"),
        target_modules=["q_proj", "v_proj"],
        mpeft_enabled=False,
    )
    model = get_peft_model(model, peft_config, adapter_name="default")
    model.set_adapter("default")
    state_dict = torch.load(checkpoint_dir / "state_dict.pt", map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, tokenizer


def generate_plain_response(model, tokenizer, question: str, max_new_tokens: int) -> str:
    prompt = PROMPT_TEMPLATE.format(instruction=question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=False,
    )
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            generation_config=generation_config,
            active_adapter="default",
        )
    input_length = inputs["input_ids"].shape[1]
    return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()


def load_questions(dataset_root: Path, task_name: str) -> list[dict]:
    with open(dataset_root / task_name / "test.json", "r", encoding="utf-8") as f:
        return json.load(f)["questions"]


def save_compare_json(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"results": rows}, f, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)

    if args.mode == "memr":
        model, tokenizer, query_encoder, task_list = load_memr_model(
            base_model_path=args.base_model_path,
            checkpoint_dir=str(checkpoint_dir),
            meta_embeddings_path=args.meta_embeddings_path,
            inference_log_dir=args.inference_log_dir,
        )
    else:
        model, tokenizer = load_plain_lora_model(args.base_model_path, checkpoint_dir)
        query_encoder = None
        task_list = None

    for task_name in TASKS:
        questions = load_questions(dataset_root, task_name)
        rows = []
        for item in tqdm(questions, desc=f"generating {task_name}", unit="sample"):
            question = item["question"]
            if args.mode == "memr":
                answer = generate_memr_response(
                    model,
                    tokenizer,
                    query_encoder,
                    task_list,
                    question,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.0,
                )
            else:
                answer = generate_plain_response(model, tokenizer, question, args.max_new_tokens)
            rows.append(
                {
                    "id": item["id"],
                    "question": question,
                    "answer": answer,
                }
            )
        save_compare_json(rows, output_dir / f"{task_name}.json")
        print(f"Saved {task_name} -> {output_dir / f'{task_name}.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
