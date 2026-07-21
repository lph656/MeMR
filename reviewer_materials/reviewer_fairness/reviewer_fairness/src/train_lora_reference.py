"""
Actual LoRA baseline training entrypoint for reviewer fairness experiments.

This file is additive and does not change the original MeMR training scripts.
It provides matched-setup training for:
  - sequential_ft
  - joint_training
  - single_task_oracle
  - er_lora
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from datasets import Dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import BitsAndBytesConfig, GenerationConfig, LlamaTokenizer, set_seed

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.causal_lm_llama import LlamaContinualForCausalLM
from mpeft import LoraConfig, TaskType, get_peft_model
from reviewer_fairness.src.build_replay_buffer import sample_replay_records
from reviewer_fairness.src.common import (
    TASK_CODE_TO_NAME,
    append_text,
    config_to_plain,
    detect_todo_values,
    dump_yaml,
    load_yaml,
    make_metrics_template,
    normalize_score,
    resolve_order_codes,
    split_train_dev_records,
    stable_task_split_seed,
    write_json,
    write_metrics_csv,
)
from tasks.mtl5.dataloader_mtl_causal_llama import custom_data_collator, preprocess_function
from utils.compute_metrics import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fairness LoRA baselines.")
    parser.add_argument("--method", required=True, choices=["sequential_ft", "joint_training", "single_task_oracle", "er_lora"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--order", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def load_config(path: str, method: str, order: str) -> Dict[str, Any]:
    config = config_to_plain(load_yaml(path))
    unresolved = [item for item in detect_todo_values(config) if item not in {"decoding.temperature", "decoding.top_p", "decoding.num_beams"}]
    if unresolved:
        raise RuntimeError(f"Unresolved required TODO values in config: {', '.join(unresolved)}")
    config["method"] = method
    config["order_name"] = order
    return config


def read_train_records(dataset_root: str | Path, task_name: str) -> List[Dict[str, Any]]:
    with open(Path(dataset_root) / task_name / "train.json", "r", encoding="utf-8") as f:
        return json.load(f)


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


def load_lora_model(config: Dict[str, Any], output_dir: Path):
    quantization_config = build_quantization_config()
    model = LlamaContinualForCausalLM.from_pretrained(
        config["base_model"],
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.config.pad_token_id = 1
    model.config.mpeft_enabled = False
    model.config.multi_peft_modules = False
    model.config.disentangle_modules = False

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        bias="none",
        output_dir=str(output_dir / "lora"),
        target_modules=config["lora"]["target_modules"],
        mpeft_enabled=False,
    )
    model = get_peft_model(model, peft_config, adapter_name="default")
    model.set_adapter("default")
    return model


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def build_dataloader_from_records(
    records: List[Dict[str, Any]],
    tokenizer,
    max_seq_length: int,
    batch_size: int,
    is_eval: bool,
    num_workers: int = 0,
) -> DataLoader:
    dataset = Dataset.from_list(records)
    dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_seq_length, is_eval=is_eval),
        batched=True,
        remove_columns=dataset.column_names,
        desc="tokenizing fairness dataset",
    )
    collator = lambda features: custom_data_collator(features, tokenizer)
    sampler = SequentialSampler(dataset) if is_eval else RandomSampler(dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collator,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def compute_parameter_counts(model) -> Tuple[int, int]:
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    return int(trainable), int(total)


def train_one_stage(
    model,
    tokenizer,
    train_records: List[Dict[str, Any]],
    config: Dict[str, Any],
    output_dir: Path,
    stage_name: str,
) -> None:
    train_loader = build_dataloader_from_records(
        train_records,
        tokenizer,
        config["training"]["max_seq_length"],
        config["training"]["batch_size"],
        is_eval=False,
    )
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(params, lr=config["training"]["learning_rate"])
    grad_accum = config["training"]["gradient_accumulation_steps"]
    model.train()
    model.zero_grad()
    device = model.device
    num_epochs = config["training"]["epochs"]
    train_log = output_dir / "train.log"

    for epoch in range(num_epochs):
        progress = tqdm(train_loader, desc=f"{stage_name} epoch {epoch + 1}/{num_epochs}")
        for step, batch in enumerate(progress, start=1):
            batch = to_device(batch, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                loss_mask=batch["loss_mask"],
                active_adapter="default",
            )
            loss = outputs.loss / grad_accum
            loss.backward()
            if step % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
            progress.set_postfix(loss=float(outputs.loss.detach().cpu()))
        append_text(train_log, f"{stage_name} epoch={epoch + 1} completed\n")


def postprocess_generated_tokens(tokenizer, generated_tokens, input_length: int) -> List[str]:
    if isinstance(generated_tokens, torch.Tensor):
        generated_tokens = generated_tokens.cpu().numpy()
    generated_part = generated_tokens[:, input_length:]
    predictions = tokenizer.batch_decode(generated_part, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return [pred.strip() for pred in predictions]


def evaluate_records(
    model,
    tokenizer,
    eval_records: List[Dict[str, Any]],
    config: Dict[str, Any],
    output_dir: Path,
    task_code: str,
    prediction_prefix: str,
) -> Dict[str, float]:
    loader = build_dataloader_from_records(
        eval_records,
        tokenizer,
        config["training"]["max_seq_length"],
        config["training"]["batch_size"],
        is_eval=True,
    )
    device = model.device
    generation_config = GenerationConfig(
        max_new_tokens=config["training"]["max_target_length"],
        do_sample=False,
        repetition_penalty=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=False,
    )
    model.eval()
    all_predictions: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, float]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"eval {task_code}"):
            batch = to_device(batch, device)
            generated_tokens = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                generation_config=generation_config,
                active_adapter="default",
            )
            predictions = postprocess_generated_tokens(tokenizer, generated_tokens, batch["input_ids"].shape[1])
            references = tokenizer.batch_decode(batch["targets"], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            metrics = compute_metrics(predictions, references)
            metric_rows.append(metrics)
            for pred, ref in zip(predictions, references):
                all_predictions.append({"prediction": pred, "reference": ref})

    final_metrics: Dict[str, float] = {}
    for key in metric_rows[0].keys():
        final_metrics[key] = round(sum(row[key] for row in metric_rows) / len(metric_rows), 4)
    write_json(all_predictions, output_dir / "predictions" / f"{prediction_prefix}_{task_code}.json")
    append_text(output_dir / "eval.log", f"{prediction_prefix} {task_code} {final_metrics}\n")
    return final_metrics


def save_checkpoint(model, output_dir: Path, name: str) -> Path:
    ckpt_dir = output_dir / "checkpoints" / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state_dict = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if "lora" in key or "lm_head" in key
    }
    torch.save(state_dict, ckpt_dir / "state_dict.pt")
    info = {"checkpoint_name": name, "state_dict_keys": sorted(state_dict.keys())}
    write_json(info, ckpt_dir / "checkpoint_info.json")
    return ckpt_dir


def compute_cl_metrics(initial_scores: Dict[str, float], stage_scores: List[Dict[str, float]], order_codes: List[str]) -> Dict[str, float]:
    if not stage_scores:
        return {"FWT": None, "FR": None, "BWT": None}
    diagonal = {code: stage_scores[idx][code] for idx, code in enumerate(order_codes)}
    final_scores = stage_scores[-1]
    fwt_terms = [diagonal[code] - initial_scores[code] for code in order_codes]
    fr_terms = []
    bwt_terms = []
    for idx, code in enumerate(order_codes[:-1]):
        historical = [stage[code] for stage in stage_scores[idx:]]
        fr_terms.append(max(historical) - final_scores[code])
        bwt_terms.append(final_scores[code] - diagonal[code])
    return {
        "FWT": normalize_score(sum(fwt_terms) / len(fwt_terms)) if fwt_terms else None,
        "FR": normalize_score(sum(fr_terms) / len(fr_terms)) if fr_terms else None,
        "BWT": normalize_score(sum(bwt_terms) / len(bwt_terms)) if bwt_terms else None,
    }


def prepare_task_splits(config: Dict[str, Any], order_codes: List[str]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    splits: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for code in order_codes:
        task_name = TASK_CODE_TO_NAME[code]
        records = read_train_records(config["dataset_root"], task_name)
        train_records, dev_records = split_train_dev_records(
            records,
            config["training"]["validation_split_percentage"],
            stable_task_split_seed(config["training"]["seed"], code),
        )
        splits[code] = {"train": train_records, "dev": dev_records}
    return splits


def run_joint_training(model, tokenizer, splits, config, output_dir: Path, order_codes: List[str]) -> Dict[str, Any]:
    train_records: List[Dict[str, Any]] = []
    for code in order_codes:
        train_records.extend(splits[code]["train"])
    train_one_stage(model, tokenizer, train_records, config, output_dir, "joint_training")
    save_checkpoint(model, output_dir, "joint_training_final")

    metrics = make_metrics_template("joint_training", "all_tasks", config)
    department_scores = {}
    for code in order_codes:
        task_metrics = evaluate_records(model, tokenizer, splits[code]["dev"], config, output_dir, code, "joint_final")
        department_scores[code] = task_metrics["rougeL"]
    metrics["department_scores"].update(department_scores)
    metrics["average"] = normalize_score(sum(department_scores.values()) / len(department_scores))
    metrics["FWT"] = "N/A"
    metrics["FR"] = "N/A"
    metrics["BWT"] = "N/A"
    metrics["notes"] = "Joint multi-task upper bound trained on the merged per-task train splits and evaluated on matched dev splits."
    return metrics


def run_single_task_oracle(config: Dict[str, Any], output_dir: Path, order_codes: List[str]) -> Dict[str, Any]:
    tokenizer = build_tokenizer(config["tokenizer"])
    metrics = make_metrics_template("single_task_oracle", "all_tasks", config)
    department_scores = {}
    per_task_dirs = []
    for code in order_codes:
        task_dir = output_dir / code
        task_dir.mkdir(parents=True, exist_ok=True)
        model = load_lora_model(config, task_dir)
        splits = prepare_task_splits(config, [code])
        train_one_stage(model, tokenizer, splits[code]["train"], config, task_dir, f"oracle_{code}")
        save_checkpoint(model, task_dir, f"oracle_{code}_final")
        task_metrics = evaluate_records(model, tokenizer, splits[code]["dev"], config, task_dir, code, "oracle_final")
        department_scores[code] = task_metrics["rougeL"]
        per_task_dirs.append(str(task_dir))
    metrics["department_scores"].update(department_scores)
    metrics["average"] = normalize_score(sum(department_scores.values()) / len(department_scores))
    metrics["FWT"] = "N/A"
    metrics["FR"] = "N/A"
    metrics["BWT"] = "N/A"
    metrics["notes"] = f"Single-task oracle trained independently per department. Task dirs: {per_task_dirs}"
    return metrics


def run_sequential_like(model, tokenizer, splits, config, output_dir: Path, order_codes: List[str], use_replay: bool) -> Dict[str, Any]:
    method = "er_lora" if use_replay else "sequential_ft"
    initial_scores = {}
    for code in order_codes:
        task_metrics = evaluate_records(model, tokenizer, splits[code]["dev"], config, output_dir, code, "initial")
        initial_scores[code] = task_metrics["rougeL"]

    replay_buffer: Dict[str, List[Dict[str, Any]]] = {}
    stage_scores: List[Dict[str, float]] = []
    for idx, code in enumerate(order_codes):
        train_records = list(splits[code]["train"])
        if use_replay:
            for prev_code in order_codes[:idx]:
                train_records.extend(replay_buffer.get(prev_code, []))
        train_one_stage(model, tokenizer, train_records, config, output_dir, f"{method}_{idx}_{code}")
        save_checkpoint(model, output_dir, f"{method}_{idx}_{code}")

        current_scores = {}
        for eval_code in order_codes:
            task_metrics = evaluate_records(model, tokenizer, splits[eval_code]["dev"], config, output_dir, eval_code, f"stage_{idx}")
            current_scores[eval_code] = task_metrics["rougeL"]
        stage_scores.append(current_scores)

        if use_replay:
            replay_buffer[code] = sample_replay_records(
                splits[code]["train"],
                config["replay"]["replay_per_task"],
                config["training"]["seed"] + idx,
            )

    final_scores = stage_scores[-1]
    metrics = make_metrics_template(method, config["order_name"], config)
    metrics["department_scores"].update(final_scores)
    metrics["average"] = normalize_score(sum(final_scores.values()) / len(final_scores))
    metrics.update(compute_cl_metrics(initial_scores, stage_scores, order_codes))
    metrics["notes"] = (
        "Replay baseline uses standard fixed-size experience replay with per-task buffer size "
        f"{config['replay']['replay_per_task']}."
        if use_replay
        else "Sequential fine-tuning lower bound with a single shared LoRA adapter."
    )
    return metrics


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    config = load_config(args.config, args.method, args.order)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(config, output_dir / "config_resolved.yaml")

    set_seed(config["training"]["seed"])
    order_codes = resolve_order_codes(config["continual_learning"]["task_orders"], args.order)
    splits = prepare_task_splits(config, order_codes)

    if args.method == "single_task_oracle":
        metrics = run_single_task_oracle(config, output_dir, order_codes)
    else:
        tokenizer = build_tokenizer(config["tokenizer"])
        model = load_lora_model(config, output_dir)
        trainable_params, total_params = compute_parameter_counts(model)

        if args.method == "joint_training":
            metrics = run_joint_training(model, tokenizer, splits, config, output_dir, order_codes)
        elif args.method == "sequential_ft":
            metrics = run_sequential_like(model, tokenizer, splits, config, output_dir, order_codes, use_replay=False)
        elif args.method == "er_lora":
            metrics = run_sequential_like(model, tokenizer, splits, config, output_dir, order_codes, use_replay=True)
        else:
            raise ValueError(args.method)

        metrics["trainable_params"] = trainable_params
        metrics["total_params"] = total_params

    write_json(metrics, output_dir / "metrics.json")
    write_metrics_csv(metrics, output_dir / "metrics.csv")
    write_json({"status": "ok", "method": args.method, "order": args.order}, output_dir / "fairness_notes.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
