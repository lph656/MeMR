from __future__ import annotations

import csv
import json
import math
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
from transformers import GenerationConfig

TASK_NAMES = ["neike", "waike", "erke", "fuchanke", "nanke", "zhongliuke"]
TASK_TO_ID = {name: idx for idx, name in enumerate(TASK_NAMES)}
TASK_TO_ZH = {
    "neike": "内科",
    "waike": "外科",
    "erke": "儿科",
    "fuchanke": "妇产科",
    "nanke": "男科",
    "zhongliuke": "肿瘤科",
}
PROMPT_TEMPLATE = "你是一位专业的AI医生，请根据患者的提问给出建议。\n\n患者提问：{question}\n\nAI医生回答："
DEFAULT_TOPK_VALUES = (1, 2, 3)


@dataclass
class LoadedModelBundle:
    model: Any
    tokenizer: Any
    query_encoder: Any
    task_list: List[str]
    meta_embeddings_path: str
    checkpoint_dir: str


def ensure_dir(path: os.PathLike[str] | str) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def read_json(path: os.PathLike[str] | str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: os.PathLike[str] | str, data: Any) -> None:
    target = Path(path)
    if target.parent:
        target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_csv(path: os.PathLike[str] | str, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    target = Path(path)
    if target.parent:
        target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def iter_jsonl(path: os.PathLike[str] | str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def append_jsonl(path: os.PathLike[str] | str, record: Dict[str, Any]) -> None:
    target = Path(path)
    if target.parent:
        target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False))
        handle.write("\n")


def find_key_encoder(module):
    if hasattr(module, "key_encoder") and module.key_encoder is not None:
        return module.key_encoder
    if hasattr(module, "base_model"):
        found = find_key_encoder(module.base_model)
        if found is not None:
            return found
    if hasattr(module, "model"):
        return find_key_encoder(module.model)
    return None


def load_checkpoint_task_list(checkpoint_dir: os.PathLike[str] | str) -> List[str]:
    info = read_json(Path(checkpoint_dir) / "checkpoint_info.json")
    task_list = info.get("lora_adapters")
    if not task_list:
        raise ValueError(f"lora_adapters missing in {checkpoint_dir}/checkpoint_info.json")
    return task_list


def load_train_records(task_name: str, dataset_root: os.PathLike[str] | str = "datasets/medical_consult") -> List[Dict[str, Any]]:
    return read_json(Path(dataset_root) / task_name / "train.json")


def load_test_questions(task_name: str, dataset_root: os.PathLike[str] | str = "datasets/medical_consult") -> List[Dict[str, Any]]:
    payload = read_json(Path(dataset_root) / task_name / "test.json")
    questions = payload.get("questions", payload)
    if not isinstance(questions, list):
        raise TypeError(f"Unexpected test format for {task_name}")
    return questions


def build_validation_split(
    task_name: str,
    dataset_root: os.PathLike[str] | str = "datasets/medical_consult",
    validation_ratio: float = 0.1,
    seed: int = 0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    records = list(load_train_records(task_name, dataset_root))
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    val_size = max(1, int(round(len(records) * validation_ratio)))
    val_indices = set(indices[:val_size])
    train_records = [records[idx] for idx in range(len(records)) if idx not in val_indices]
    val_records = [records[idx] for idx in range(len(records)) if idx in val_indices]
    return train_records, val_records


def deterministic_sample(records: Sequence[Dict[str, Any]], sample_size: int, seed: int) -> List[Tuple[int, Dict[str, Any]]]:
    indexed = list(enumerate(records))
    rng = random.Random(seed)
    rng.shuffle(indexed)
    selected = indexed[: min(sample_size, len(indexed))]
    selected.sort(key=lambda item: item[0])
    return selected


def build_metadata_subset(
    source_path: os.PathLike[str] | str,
    task_list: Sequence[str],
    output_path: os.PathLike[str] | str,
) -> str:
    source = torch.load(source_path, map_location="cpu")
    rows = [source[TASK_TO_ID[task_name]].clone() for task_name in task_list]
    subset = torch.stack(rows, dim=0)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(subset, output)
    return str(output)


def compute_dtype_from_name(compute_dtype_name: str):
    if compute_dtype_name == "bfloat16":
        if not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
            raise RuntimeError("Requested bfloat16 but the current CUDA device does not support it.")
        return torch.bfloat16
    if compute_dtype_name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported compute dtype: {compute_dtype_name}")


def load_memr_model(
    base_model_path: str,
    checkpoint_dir: str,
    meta_embeddings_path: str,
    task_list: Optional[Sequence[str]] = None,
    inference_log_dir: str = "/tmp/memr_reviewer_logs",
    compute_dtype_name: str = "float16",
) -> LoadedModelBundle:
    from transformers import BitsAndBytesConfig, LlamaModel, LlamaTokenizer

    from model.causal_lm_llama import LlamaContinualForCausalLM
    from mpeft import KeyEncoderConfig, get_peft_model
    from mpeft.tuners.lora import LoraConfig
    from mpeft.utils import TaskType

    checkpoint_path = Path(checkpoint_dir)
    state_dict_path = checkpoint_path / "state_dict.pt"
    if not state_dict_path.exists():
        raise FileNotFoundError(f"state_dict.pt not found in {checkpoint_dir}")
    if task_list is None:
        task_list = load_checkpoint_task_list(checkpoint_dir)
    task_list = list(task_list)
    compute_dtype = compute_dtype_from_name(compute_dtype_name)

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

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = LlamaContinualForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    query_encoder = LlamaModel.from_pretrained(
        base_model_path,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )
    for param in query_encoder.parameters():
        param.requires_grad = False
    query_encoder.eval()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        mpeft_enabled=True,
    )
    peft_config.mpeft_config = KeyEncoderConfig(
        task_list=task_list,
        meta_embeddings_path=meta_embeddings_path,
        key_dim=query_encoder.config.hidden_size,
        output_dir=inference_log_dir,
    )

    for task_name in task_list:
        model = get_peft_model(model, peft_config, adapter_name=task_name)

    state_dict = torch.load(state_dict_path, map_location="cpu")
    expected_task_count = len(task_list)
    for key, value in list(state_dict.items()):
        if not torch.is_tensor(value):
            continue
        if ("key_encoder.keys" in key or "key_encoder.all_meta_keys" in key) and value.ndim >= 2:
            if value.shape[0] > expected_task_count:
                state_dict[key] = value[:expected_task_count].clone()
            elif value.shape[0] < expected_task_count:
                raise RuntimeError(
                    f"Checkpoint key tensor {key} has only {value.shape[0]} rows, "
                    f"but the requested task list requires {expected_task_count}."
                )
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return LoadedModelBundle(
        model=model,
        tokenizer=tokenizer,
        query_encoder=query_encoder,
        task_list=task_list,
        meta_embeddings_path=meta_embeddings_path,
        checkpoint_dir=checkpoint_dir,
    )


def _query_encoder_device(query_encoder) -> torch.device:
    return next(query_encoder.parameters()).device


def build_query_embedding(
    tokenizer,
    query_encoder,
    question: str,
    query_encoder_type: str = "avg_word_embed",
) -> torch.Tensor:
    prompt = PROMPT_TEMPLATE.format(question=question)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = torch.cat(
        [
            torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long),
            encoded["input_ids"],
        ],
        dim=1,
    )
    attention_mask = torch.ones_like(input_ids)
    input_ids = input_ids.to(_query_encoder_device(query_encoder))
    attention_mask = attention_mask.to(_query_encoder_device(query_encoder))
    with torch.no_grad():
        hidden_states = query_encoder(input_ids, attention_mask=attention_mask)[0]
    if query_encoder_type == "avg_all_embed":
        return hidden_states.mean(dim=1)
    masked_sum = torch.sum(hidden_states * attention_mask.unsqueeze(-1), dim=1)
    token_count = torch.sum(attention_mask, dim=1).unsqueeze(-1)
    return masked_sum / (token_count + 1e-9)


def get_active_adapter(task_list: Sequence[str], active_adapter: Optional[str] = None) -> str:
    return active_adapter if active_adapter is not None else task_list[-1]


def route_query(
    bundle: LoadedModelBundle,
    question: str,
    active_adapter: Optional[str] = None,
) -> Tuple[List[float], List[str]]:
    key_encoder = find_key_encoder(bundle.model)
    if key_encoder is None:
        raise RuntimeError("TaskKeyEncoder not found")
    query_embed = build_query_embedding(bundle.tokenizer, bundle.query_encoder, question)
    key_encoder_device = next(key_encoder.parameters()).device
    query_embed = query_embed.to(key_encoder_device)
    with torch.no_grad():
        weights, _ = key_encoder(
            x_query=query_embed,
            adapter_name=get_active_adapter(bundle.task_list, active_adapter),
            train=False,
            final=True,
        )
    return weights[0].detach().cpu().tolist(), list(bundle.task_list)


def constrain_weights(weights: torch.Tensor, top_k: Optional[int]) -> torch.Tensor:
    if top_k is None or top_k <= 0 or top_k >= weights.shape[-1]:
        return weights
    values, indices = torch.topk(weights, k=top_k, dim=-1)
    constrained = torch.zeros_like(weights)
    constrained.scatter_(dim=-1, index=indices, src=values)
    constrained = constrained / constrained.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return constrained


@contextmanager
def patched_route_mode(key_encoder, top_k: Optional[int]):
    original_forward = key_encoder.forward

    def _patched_forward(*args, **kwargs):
        weights, loss = original_forward(*args, **kwargs)
        return constrain_weights(weights, top_k=top_k), loss

    if top_k is None:
        yield
        return
    key_encoder.forward = _patched_forward
    try:
        yield
    finally:
        key_encoder.forward = original_forward


def parse_route_mode(route_mode: str) -> Optional[int]:
    if route_mode == "full":
        return None
    if route_mode.startswith("top"):
        return int(route_mode[3:])
    raise ValueError(f"Unsupported route mode: {route_mode}")


def generate_answer(
    bundle: LoadedModelBundle,
    question: str,
    route_mode: str = "full",
    active_adapter: Optional[str] = None,
    max_new_tokens: int = 96,
) -> str:
    prompt = PROMPT_TEMPLATE.format(question=question)
    query_embed = build_query_embedding(bundle.tokenizer, bundle.query_encoder, question)
    key_encoder = find_key_encoder(bundle.model)
    if key_encoder is None:
        raise RuntimeError("TaskKeyEncoder not found")
    key_encoder_device = next(key_encoder.parameters()).device
    query_embed = query_embed.to(key_encoder_device)

    inputs = bundle.tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(bundle.model.device) for k, v in inputs.items()}
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.0,
        eos_token_id=bundle.tokenizer.eos_token_id,
        pad_token_id=bundle.tokenizer.pad_token_id,
        do_sample=False,
        use_cache=False,
    )
    top_k = parse_route_mode(route_mode)
    with patched_route_mode(key_encoder, top_k=top_k):
        with torch.no_grad():
            outputs = bundle.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=generation_config,
                active_adapter=get_active_adapter(bundle.task_list, active_adapter),
                query_embed=query_embed,
                train=False,
                final=True,
            )
    input_length = inputs["input_ids"].shape[1]
    return bundle.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()


def compute_multiclass_brier(probabilities: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    total = 0.0
    for probs, label in zip(probabilities, labels):
        row = 0.0
        for idx, value in enumerate(probs):
            target = 1.0 if idx == label else 0.0
            row += (value - target) ** 2
        total += row
    return total / max(len(labels), 1)


def compute_ece(confidences: Sequence[float], correct: Sequence[int], num_bins: int = 10) -> float:
    bins = [[] for _ in range(num_bins)]
    for conf, is_correct in zip(confidences, correct):
        bin_id = min(num_bins - 1, int(conf * num_bins))
        bins[bin_id].append((conf, is_correct))
    total_count = max(len(confidences), 1)
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        avg_conf = sum(item[0] for item in bucket) / len(bucket)
        avg_acc = sum(item[1] for item in bucket) / len(bucket)
        ece += abs(avg_conf - avg_acc) * (len(bucket) / total_count)
    return ece


def compute_entropy(probabilities: Sequence[float]) -> float:
    return -sum(value * math.log(value + 1e-12) for value in probabilities)


def compute_topk_hits(probabilities: Sequence[float], label: int, topk_values: Sequence[int]) -> Dict[str, int]:
    ranked = sorted(range(len(probabilities)), key=lambda idx: probabilities[idx], reverse=True)
    hits = {}
    for k in topk_values:
        hits[f"top{k}_hit"] = int(label in ranked[: min(k, len(ranked))])
    return hits


def summarise_routing_records(records: Sequence[Dict[str, Any]], topk_values: Sequence[int]) -> Dict[str, Any]:
    if not records:
        return {}
    confidences = [row["top1_confidence"] for row in records]
    correct = [row["top1_correct"] for row in records]
    labels = [row["task_id"] for row in records]
    probabilities = [row["weights"] for row in records]
    summary: Dict[str, Any] = {
        "num_samples": len(records),
        "top1_accuracy": sum(correct) / len(correct),
        "ece": compute_ece(confidences, correct),
        "brier_score": compute_multiclass_brier(probabilities, labels),
        "mean_entropy": sum(row["entropy"] for row in records) / len(records),
        "mean_correct_task_weight": sum(row["correct_task_weight"] for row in records) / len(records),
        "mean_top1_confidence": sum(confidences) / len(confidences),
    }
    for k in topk_values:
        summary[f"top{k}_accuracy"] = sum(row[f"top{k}_hit"] for row in records) / len(records)
    return summary


def compute_generation_metrics(
    predictions_by_mode: Dict[str, List[str]],
    references: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    from utils.compute_metrics import compute_metrics

    results: Dict[str, Dict[str, float]] = {}
    for route_mode, predictions in predictions_by_mode.items():
        results[route_mode] = compute_metrics(predictions, list(references))
    return results


def build_record(
    task_name: str,
    source_record: Dict[str, Any],
    sample_id: str,
    split: str,
    original_index: int,
) -> Dict[str, Any]:
    return {
        "sample_id": sample_id,
        "task_name": task_name,
        "task_id": TASK_TO_ID[task_name],
        "question": source_record["question"] if "question" in source_record else source_record["instruction"],
        "reference_answer": source_record.get("output"),
        "split": split,
        "original_index": original_index,
        "original_id": source_record.get("id", original_index),
    }


def load_eval_jsonl(path: os.PathLike[str] | str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    records = list(iter_jsonl(path))
    if max_samples is not None:
        return records[:max_samples]
    return records


def perturb_question(question: str, mode: str, seed: int) -> str:
    if mode == "none":
        return question
    chars = list(question)
    rng = random.Random(seed)
    if mode == "char_delete_10":
        kept = [ch for ch in chars if rng.random() > 0.10 or ch.isspace()]
        return "".join(kept) or question
    if mode == "char_swap_10":
        chars = chars[:]
        swap_count = max(1, int(len(chars) * 0.10))
        for _ in range(swap_count):
            if len(chars) < 2:
                break
            idx = rng.randint(0, len(chars) - 2)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars)
    if mode == "punctuation_10":
        punctuation = ["，", "。", "？", "！", ","]
        out = []
        for ch in chars:
            out.append(ch)
            if rng.random() < 0.10:
                out.append(rng.choice(punctuation))
        return "".join(out)
    if mode == "filler_10":
        fillers = ["请问", "麻烦问下", "就是", "有点", "这个"]
        tokens = []
        for ch in chars:
            if rng.random() < 0.10:
                tokens.append(rng.choice(fillers))
            tokens.append(ch)
        return "".join(tokens)
    raise ValueError(f"Unsupported noise mode: {mode}")
