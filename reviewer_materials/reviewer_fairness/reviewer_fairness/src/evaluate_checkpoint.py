"""
Evaluate saved fairness checkpoints on the six CMedCL task dev/test sets.

This utility supports:
  - plain LoRA baselines saved by train_lora_reference.py
  - MeMR checkpoints saved by the original project
  - metadata-only routing evaluation via runtime wrapper
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import BitsAndBytesConfig, GenerationConfig, LlamaModel, LlamaTokenizer

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.causal_lm_llama import LlamaContinualForCausalLM
from mpeft import KeyEncoderConfig, LoraConfig, TaskType, get_peft_model
from reviewer_fairness.src.common import (
    TASK_CODE_TO_NAME,
    config_to_plain,
    load_train_records,
    load_yaml,
    split_train_dev_records,
    stable_task_split_seed,
    write_json,
)
from reviewer_fairness.src.metadata_only_routing import enable_metadata_only_routing
from tasks.mtl5.dataloader_mtl_causal_llama import custom_data_collator, preprocess_function
from utils.compute_metrics import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate reviewer fairness checkpoints.")
    parser.add_argument("--method", required=True, choices=["sequential_ft", "joint_training", "single_task_oracle", "er_lora", "memr", "metadata_only_routing"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--order", required=True)
    parser.add_argument("--use_test_split", action="store_true")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    return config_to_plain(load_yaml(path))


def build_tokenizer(base_model: str):
    tokenizer = LlamaTokenizer.from_pretrained(
        base_model,
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


def load_plain_lora_model(config: Dict[str, Any], checkpoint_dir: Path):
    quantization_config = build_quantization_config()
    model = LlamaContinualForCausalLM.from_pretrained(
        config["base_model"],
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        bias="none",
        output_dir=str(checkpoint_dir / "lora"),
        target_modules=config["lora"]["target_modules"],
        mpeft_enabled=False,
    )
    model = get_peft_model(model, peft_config, adapter_name="default")
    model.set_adapter("default")
    state_dict_path = checkpoint_dir / "state_dict.pt"
    state_dict = torch.load(state_dict_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, None, None


def load_checkpoint_task_list(checkpoint_dir: Path) -> List[str]:
    info_path = checkpoint_dir / "checkpoint_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"checkpoint_info.json not found in {checkpoint_dir}")
    with open(info_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    task_list = payload.get("lora_adapters")
    if not task_list:
        raise ValueError(f"'lora_adapters' missing in {info_path}")
    return list(task_list)


def load_memr_like_model(config: Dict[str, Any], checkpoint_dir: Path, metadata_only: bool):
    quantization_config = build_quantization_config()
    tokenizer = build_tokenizer(config["tokenizer"])
    model = LlamaContinualForCausalLM.from_pretrained(
        config["base_model"],
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    query_encoder = LlamaModel.from_pretrained(
        config["base_model"],
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    for param in query_encoder.parameters():
        param.requires_grad = False
    query_encoder.eval()

    task_names = load_checkpoint_task_list(checkpoint_dir)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        bias="none",
        target_modules=config["lora"]["target_modules"],
        mpeft_enabled=True,
    )
    peft_config.mpeft_config = KeyEncoderConfig(
        task_list=task_names,
        meta_embeddings_path=config["meta_embeddings_path"],
        key_dim=query_encoder.config.hidden_size,
        output_dir=str(Path(config["output_root"]) / "tmp_eval_logs"),
        matching_loss_v2=True,
    )
    for task_name in task_names:
        model = get_peft_model(model, peft_config, adapter_name=task_name)
    state_dict = torch.load(checkpoint_dir / "state_dict.pt", map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    if metadata_only:
        enable_metadata_only_routing(model)
    return model, tokenizer, query_encoder, task_names


def read_eval_records(dataset_root: Path, task_name: str, config: Dict[str, Any], use_test_split: bool) -> List[Dict[str, Any]]:
    if use_test_split:
        with open(dataset_root / task_name / "test.json", "r", encoding="utf-8") as f:
            payload = json.load(f)
        return [{"instruction": item["question"], "input": "", "output": ""} for item in payload["questions"]]
    task_code = next(code for code, name in TASK_CODE_TO_NAME.items() if name == task_name)
    records = load_train_records(dataset_root, task_name)
    _, dev_records = split_train_dev_records(
        records,
        config["training"]["validation_split_percentage"],
        stable_task_split_seed(config["training"]["seed"], task_code),
    )
    return dev_records


def build_loader(records: List[Dict[str, Any]], tokenizer, max_seq_length: int) -> DataLoader:
    dataset = Dataset.from_list(records)
    dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_seq_length, is_eval=True),
        batched=True,
        remove_columns=dataset.column_names,
        desc="tokenizing eval dataset",
    )
    collator = lambda features: custom_data_collator(features, tokenizer)
    return DataLoader(
        dataset,
        batch_size=1,
        sampler=SequentialSampler(dataset),
        collate_fn=collator,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )


def build_query_embed(query_encoder, model_inputs, method: str):
    if query_encoder is None:
        return None
    input_ids = model_inputs["query_input_ids"]
    attention_mask = model_inputs["query_attention_mask"]
    with torch.no_grad():
        hidden_states = query_encoder(input_ids, attention_mask=attention_mask)[0]
    masked_sum = torch.sum(hidden_states * attention_mask.unsqueeze(-1), dim=1)
    num_tokens = torch.sum(attention_mask, dim=1).unsqueeze(-1)
    return masked_sum / (num_tokens + 1e-9)


def postprocess_generated_tokens(tokenizer, generated_tokens, input_length: int) -> List[str]:
    if isinstance(generated_tokens, torch.Tensor):
        generated_tokens = generated_tokens.cpu().numpy()
    generated_part = generated_tokens[:, input_length:]
    predictions = tokenizer.batch_decode(generated_part, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return [pred.strip() for pred in predictions]


def evaluate_model(
    model,
    tokenizer,
    query_encoder,
    config: Dict[str, Any],
    output_dir: Path,
    method: str,
    use_test_split: bool,
    active_adapter: Optional[str] = None,
):
    dataset_root = Path(config["dataset_root"])
    generation_config = GenerationConfig(
        max_new_tokens=config["training"]["max_target_length"],
        do_sample=False,
        repetition_penalty=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=False,
    )
    task_scores = {}
    predictions_root = output_dir / "predictions"
    predictions_root.mkdir(parents=True, exist_ok=True)
    for code, task_name in TASK_CODE_TO_NAME.items():
        records = read_eval_records(dataset_root, task_name, config, use_test_split)
        if not records:
            continue
        if "output" not in records[0] or records[0].get("output", "") == "":
            task_scores[code] = None
            continue
        loader = build_loader(records, tokenizer, config["training"]["max_seq_length"])
        metric_rows = []
        saved_predictions = []
        for batch in tqdm(loader, desc=f"eval {method}:{code}"):
            batch = {
                key: value.to(model.device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            query_embed = build_query_embed(query_encoder, batch, method)
            if method in {"memr", "metadata_only_routing"}:
                generated_tokens = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    generation_config=generation_config,
                    active_adapter=active_adapter,
                    query_embed=query_embed,
                    train=False,
                    final=False,
                )
            else:
                generated_tokens = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    generation_config=generation_config,
                    active_adapter="default",
                )
            predictions = postprocess_generated_tokens(tokenizer, generated_tokens, batch["input_ids"].shape[1])
            references = tokenizer.batch_decode(batch["targets"], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            metric_rows.append(compute_metrics(predictions, references))
            saved_predictions.extend([{"prediction": pred, "reference": ref} for pred, ref in zip(predictions, references)])
        final_metrics = {}
        for key in metric_rows[0].keys():
            final_metrics[key] = round(sum(row[key] for row in metric_rows) / len(metric_rows), 4)
        task_scores[code] = final_metrics["rougeL"]
        write_json(saved_predictions, predictions_root / f"{task_name}.json")
    return task_scores


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = build_tokenizer(config["tokenizer"])

    checkpoint_dir = Path(args.checkpoint_dir)
    if args.method in {"memr", "metadata_only_routing"}:
        model, tokenizer, query_encoder, task_names = load_memr_like_model(
            config,
            checkpoint_dir,
            metadata_only=args.method == "metadata_only_routing",
        )
        active_adapter = task_names[-1]
    else:
        model, _, _ = load_plain_lora_model(config, checkpoint_dir)
        query_encoder = None
        active_adapter = "default"

    task_scores = evaluate_model(
        model,
        tokenizer,
        query_encoder,
        config,
        output_dir,
        args.method,
        args.use_test_split,
        active_adapter=active_adapter,
    )
    write_json(task_scores, output_dir / "task_scores.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
