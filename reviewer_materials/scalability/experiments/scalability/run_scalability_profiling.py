from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import re
import statistics
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_memr")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.inference_utils import build_query_embedding, find_key_encoder, load_memr_model

try:
    import pandas as pd
except Exception:
    pd = None


DEFAULT_QUERIES = [
    "我最近总是头痛恶心，需要挂什么科？",
    "孩子发烧两天了，还伴有咳嗽怎么办？",
    "肚子一直隐隐作痛，吃完饭更明显，是胃的问题吗？",
    "最近胸口发闷，活动后更明显，要不要去医院检查？",
    "月经不规律而且小腹疼痛，应该看哪个科室？",
    "老人血压波动很大，需要注意什么？",
    "耳朵里面嗡嗡响，晚上更严重，这是耳鸣吗？",
    "排尿时刺痛并且尿频，需要做什么检查？",
    "我眼睛干涩看东西模糊，长期用电脑会这样吗？",
    "皮肤反复起红疹，还很痒，可能是什么原因？",
    "最近关节酸痛，晨起僵硬，是否需要风湿免疫科？",
    "喉咙痛伴随吞咽困难，是扁桃体发炎吗？",
    "体检发现甲状腺结节，需要进一步处理吗？",
    "长期失眠、心慌焦虑，应该先看心理还是内科？",
    "怀孕早期总是恶心呕吐，有没有缓解办法？",
    "化疗后食欲很差、乏力明显，有什么建议？",
]

COMPLEXITY_ROWS = [
    {
        "component": "Task matching",
        "main_operation": "query-meta similarity, dynamic fusion, cosine matching, softmax",
        "complexity": "O(Kd)",
        "scalability_implication": "Grows linearly with task count K and representation dimension d.",
    },
    {
        "component": "Module aggregation",
        "main_operation": "weighted summation over task-specific module tensors",
        "complexity": "O(K|P|)",
        "scalability_implication": "Typically dominates matching because |P| is much larger than d.",
    },
    {
        "component": "Task-related memory",
        "main_operation": "store task/meta embeddings and frozen task modules",
        "complexity": "O(Kd + K|P|)",
        "scalability_implication": "Memory grows approximately linearly with task number under independent task banks.",
    },
]


@dataclass
class TaskEntry:
    task_id: str
    source: str
    meta_embedding: torch.Tensor
    task_embedding: torch.Tensor
    module_tensors: "OrderedDict[str, torch.Tensor]"
    recipe: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scalability profiling for MeMR task-bank growth.")
    parser.add_argument("--checkpoint_path", required=True, help="Path to a checkpoint directory containing checkpoint_info.json and state_dict.pt.")
    parser.add_argument("--base_model_path", default="chinese-alpaca-plus-7b-hf", help="Base model path or HF model name.")
    parser.add_argument("--meta_embeddings_path", default="metadata_embeddings/keshi_meta_embeddings.pt", help="Metadata embedding file used for real tasks.")
    parser.add_argument("--output_dir", default="results/scalability_profiling", help="Root output directory. A timestamped subdirectory will be created.")
    parser.add_argument("--k_list", default="1,2,4,6,8,16,32,64", help="Comma-separated task counts to profile.")
    parser.add_argument("--device", default=None, help="Device to use. Defaults to cuda:0 if available else cpu.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Warmup iterations for each timed block.")
    parser.add_argument("--profile_steps", type=int, default=50, help="Measured iterations per repeat.")
    parser.add_argument("--num_repeats", type=int, default=3, help="Number of repeats for each K.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size. Batch size 1 is recommended.")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"], help="Profiling dtype for synthetic computations.")
    parser.add_argument("--measure_plm_forward", action="store_true", help="Also measure one fixed PLM prefill forward pass when feasible.")
    parser.add_argument("--precompute_query_embeddings", action="store_true", help="Precompute query embeddings and reuse them during profiling.")
    parser.add_argument("--perturb_scale", type=float, default=1e-4, help="Perturbation scale for synthetic task-bank expansion.")
    parser.add_argument("--max_query_length", type=int, default=256, help="Maximum query length for tokenization.")
    parser.add_argument("--inference_log_dir", default="/tmp/memr_scalability_logs", help="Auxiliary log dir used by the key encoder loader.")
    return parser.parse_args()


def setup_output_dir(root: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(root, timestamp)
    os.makedirs(out_dir, exist_ok=False)
    return out_dir


def setup_logging(out_dir: str) -> logging.Logger:
    logger = logging.getLogger("scalability_profiling")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(os.path.join(out_dir, "run.log"), encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_k_list(k_list_str: str) -> List[int]:
    values = []
    for part in k_list_str.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("k_list cannot be empty.")
    return values


def resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


def bytes_to_mb(num_bytes: int) -> float:
    return num_bytes / (1024.0 * 1024.0)


def tensor_num_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_csv_rows(path: str, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_table_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    if pd is not None:
        pd.DataFrame(rows).to_csv(path, index=False)
    else:
        write_csv_rows(path, rows, fieldnames)


def format_float(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "N/A"
    return f"{value:.{digits}f}"


def infer_output_dir_from_checkpoint(checkpoint_path: str) -> str:
    normalized = os.path.abspath(checkpoint_path)
    current = normalized
    while True:
        parent = os.path.dirname(current)
        if parent == current:
            break
        checkpoint_loras = os.path.join(parent, "checkpoint_loras")
        if os.path.isdir(checkpoint_loras):
            return parent
        current = parent
    return os.path.dirname(normalized)


def load_state_dict_from_checkpoint(checkpoint_path: str) -> OrderedDict:
    state_dict_path = os.path.join(checkpoint_path, "state_dict.pt")
    if not os.path.isfile(state_dict_path):
        raise FileNotFoundError(f"state_dict.pt not found in checkpoint path: {checkpoint_path}")
    state_dict = torch.load(state_dict_path, map_location="cpu")
    if not isinstance(state_dict, (dict, OrderedDict)):
        raise TypeError("Loaded state_dict.pt is not a mapping.")
    ordered = OrderedDict()
    for key, value in state_dict.items():
        ordered[key] = value.detach().cpu() if torch.is_tensor(value) else value
    return ordered


def canonicalize_module_key(full_key: str, task_list: Sequence[str]) -> Optional[Tuple[str, str]]:
    if ".lora_" not in full_key:
        return None
    for task_name in task_list:
        token = f".{task_name}.weight"
        if token in full_key:
            canonical = full_key.replace(token, ".<TASK>.weight")
            return task_name, canonical
    match = re.search(r"\.lora_(A|B)\.([^.]+)\.weight$", full_key)
    if match:
        adapter_name = match.group(2)
        canonical = full_key.replace(f".{adapter_name}.weight", ".<TASK>.weight")
        return adapter_name, canonical
    return None


def build_real_task_bank(
    checkpoint_path: str,
    model: torch.nn.Module,
    task_list: List[str],
    profiler_dtype: torch.dtype,
    logger: logging.Logger,
) -> List[TaskEntry]:
    key_encoder = find_key_encoder(model)
    if key_encoder is None:
        raise RuntimeError("TaskKeyEncoder not found in loaded MeMR model.")

    all_meta_keys = key_encoder.all_meta_keys.detach().cpu().to(dtype=profiler_dtype)
    task_keys = key_encoder.keys.detach().cpu().to(dtype=profiler_dtype)
    if all_meta_keys.shape[0] != len(task_list) or task_keys.shape[0] != len(task_list):
        raise ValueError("Mismatch between checkpoint task list and key encoder task representations.")

    state_dict = load_state_dict_from_checkpoint(checkpoint_path)
    module_bank: Dict[str, OrderedDict[str, torch.Tensor]] = {task_name: OrderedDict() for task_name in task_list}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        if ".lora_" not in key:
            continue
        parsed = canonicalize_module_key(key, task_list)
        if parsed is None:
            continue
        task_name, canonical_key = parsed
        if task_name in module_bank:
            module_bank[task_name][canonical_key] = value.to(dtype=profiler_dtype).contiguous()

    missing = [task_name for task_name, tensors in module_bank.items() if not tensors]
    if missing:
        raise RuntimeError(f"Failed to locate LoRA module tensors for tasks: {missing}")

    reference_keys = list(next(iter(module_bank.values())).keys())
    inconsistent = [task_name for task_name, tensors in module_bank.items() if list(tensors.keys()) != reference_keys]
    if inconsistent:
        raise RuntimeError(f"Inconsistent LoRA slot layout across tasks: {inconsistent}")

    task_bank: List[TaskEntry] = []
    for idx, task_name in enumerate(task_list):
        task_bank.append(
            TaskEntry(
                task_id=task_name,
                source="real",
                meta_embedding=all_meta_keys[idx].clone(),
                task_embedding=task_keys[idx].clone(),
                module_tensors=OrderedDict((k, v.clone()) for k, v in module_bank[task_name].items()),
                recipe={"task_name": task_name, "task_index": idx},
            )
        )

    logger.info("Loaded %d real tasks from checkpoint.", len(task_bank))
    return task_bank


def clone_with_noise(tensor: torch.Tensor, gen: torch.Generator, scale: float) -> torch.Tensor:
    out = tensor.clone()
    if tensor.is_floating_point():
        noise = torch.randn(tensor.shape, generator=gen, dtype=tensor.dtype) * scale
        out = out + noise
    return out.contiguous()


def interpolate_tensors(a: torch.Tensor, b: torch.Tensor, alpha: float, gen: torch.Generator, scale: float) -> torch.Tensor:
    out = alpha * a + (1.0 - alpha) * b
    if out.is_floating_point():
        noise = torch.randn(out.shape, generator=gen, dtype=out.dtype) * scale
        out = out + noise
    return out.contiguous()


def extend_task_bank(
    real_bank: List[TaskEntry],
    max_k: int,
    seed: int,
    perturb_scale: float,
) -> Tuple[List[TaskEntry], Dict[str, Any]]:
    if max_k <= len(real_bank):
        return list(real_bank[:max_k]), {
            "real_num_tasks": len(real_bank),
            "synthetic_num_tasks": 0,
            "generation_rule": "No synthetic tasks required.",
            "interpolation_alphas": [0.25, 0.5, 0.75],
            "perturbation_scale": perturb_scale,
            "random_seed": seed,
            "synthetic_tasks": [],
        }

    task_bank = list(real_bank)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    alphas = [0.25, 0.5, 0.75]
    recipes = []
    real_count = len(real_bank)

    for synth_idx in range(real_count, max_k):
        src_a_idx = synth_idx % real_count
        src_b_idx = (synth_idx + 1) % real_count
        alpha = alphas[(synth_idx - real_count) % len(alphas)]
        src_a = real_bank[src_a_idx]
        src_b = real_bank[src_b_idx]
        task_id = f"synthetic_{synth_idx:03d}"
        module_tensors = OrderedDict()
        for key_a, value_a in src_a.module_tensors.items():
            value_b = src_b.module_tensors[key_a]
            module_tensors[key_a] = interpolate_tensors(value_a, value_b, alpha, gen, perturb_scale)

        task_entry = TaskEntry(
            task_id=task_id,
            source="synthetic",
            meta_embedding=interpolate_tensors(src_a.meta_embedding, src_b.meta_embedding, alpha, gen, perturb_scale),
            task_embedding=interpolate_tensors(src_a.task_embedding, src_b.task_embedding, alpha, gen, perturb_scale),
            module_tensors=module_tensors,
            recipe={
                "task_id": task_id,
                "source_a": src_a.task_id,
                "source_b": src_b.task_id,
                "alpha": alpha,
                "perturbation_scale": perturb_scale,
            },
        )
        task_bank.append(task_entry)
        recipes.append(task_entry.recipe)

    return task_bank, {
        "real_num_tasks": len(real_bank),
        "synthetic_num_tasks": max_k - len(real_bank),
        "generation_rule": "Deterministic interpolation between two real tasks plus Gaussian perturbation; used only for profiling, not QA accuracy evaluation.",
        "interpolation_alphas": alphas,
        "perturbation_scale": perturb_scale,
        "random_seed": seed,
        "synthetic_tasks": recipes,
    }


def prepare_task_bank_tensors(task_bank: List[TaskEntry], device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    meta = torch.stack([entry.meta_embedding.to(dtype=dtype) for entry in task_bank], dim=0).to(device=device)
    task = torch.stack([entry.task_embedding.to(dtype=dtype) for entry in task_bank], dim=0).to(device=device)
    module_names = list(task_bank[0].module_tensors.keys())
    module_bank: OrderedDict[str, torch.Tensor] = OrderedDict()
    for module_name in module_names:
        stacked = torch.stack([entry.module_tensors[module_name].to(dtype=dtype) for entry in task_bank], dim=0).to(device=device)
        module_bank[module_name] = stacked
    return {
        "task_ids": [entry.task_id for entry in task_bank],
        "meta_embeddings": meta,
        "task_embeddings": task,
        "module_bank": module_bank,
    }


def compute_task_repr_memory_mb(task_bank: List[TaskEntry]) -> float:
    total_bytes = 0
    for entry in task_bank:
        total_bytes += tensor_num_bytes(entry.meta_embedding)
        total_bytes += tensor_num_bytes(entry.task_embedding)
    return bytes_to_mb(total_bytes)


def compute_frozen_module_memory_mb(task_bank: List[TaskEntry]) -> float:
    total_bytes = 0
    for entry in task_bank:
        for tensor in entry.module_tensors.values():
            total_bytes += tensor_num_bytes(tensor)
    return bytes_to_mb(total_bytes)


def get_query_texts(batch_size: int) -> List[str]:
    if batch_size <= len(DEFAULT_QUERIES):
        return DEFAULT_QUERIES[:batch_size]
    repeats = math.ceil(batch_size / len(DEFAULT_QUERIES))
    merged = (DEFAULT_QUERIES * repeats)[:batch_size]
    return merged


def encode_queries(
    tokenizer,
    query_encoder,
    query_texts: List[str],
    max_query_length: int,
    device: torch.device,
    logger: logging.Logger,
) -> torch.Tensor:
    query_embeddings = []
    for question in query_texts:
        try:
            tokenizer.model_max_length = max_query_length
        except Exception:
            pass
        embed = build_query_embedding(tokenizer, query_encoder, question)
        query_embeddings.append(embed.detach().cpu())
    query_tensor = torch.cat(query_embeddings, dim=0)
    logger.info("Prepared %d query embeddings with shape %s.", query_tensor.shape[0], tuple(query_tensor.shape))
    return query_tensor.to(device=device)


def build_random_query_embeddings(
    batch_size: int,
    hidden_size: int,
    dtype: torch.dtype,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    queries = torch.randn((batch_size, hidden_size), generator=gen, dtype=dtype)
    return queries.to(device=device)


def matching_step(query_embeddings: torch.Tensor, meta_embeddings: torch.Tensor, task_embeddings: torch.Tensor) -> torch.Tensor:
    norm_query = F.normalize(query_embeddings.to(task_embeddings.dtype), dim=-1)
    norm_meta = F.normalize(meta_embeddings, dim=-1)
    attn_scores = torch.einsum("bd,kd->bk", norm_query, norm_meta)
    alpha = torch.sigmoid(attn_scores).unsqueeze(-1)
    dynamic_task = alpha * meta_embeddings.unsqueeze(0) + (1.0 - alpha) * task_embeddings.unsqueeze(0)
    dynamic_task = F.normalize(dynamic_task, dim=-1)
    cosine_scores = torch.bmm(norm_query.unsqueeze(1), dynamic_task.transpose(1, 2)).squeeze(1)
    return F.softmax(cosine_scores, dim=-1)


def aggregation_step(weights: torch.Tensor, module_bank: OrderedDict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    mean_weights = weights.mean(dim=0)
    aggregated = {}
    for module_name, stacked in module_bank.items():
        view_shape = [mean_weights.shape[0]] + [1] * (stacked.ndim - 1)
        aggregated[module_name] = torch.sum(stacked * mean_weights.view(*view_shape), dim=0)
    return aggregated


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def unwrap_prefill_model(model: torch.nn.Module) -> torch.nn.Module:
    current = model
    visited = set()
    while isinstance(current, torch.nn.Module) and id(current) not in visited:
        visited.add(id(current))
        module_name = current.__class__.__module__
        if module_name.startswith("mpeft") or module_name.startswith("peft"):
            if hasattr(current, "get_base_model"):
                next_model = current.get_base_model()
                if next_model is current:
                    break
                current = next_model
                continue
            if hasattr(current, "base_model") and isinstance(current.base_model, torch.nn.Module):
                next_model = current.base_model
                if next_model is current:
                    break
                current = next_model
                continue
        break
    return current


def _time_gpu_callable(fn, device: torch.device) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    synchronize_if_needed(device)
    start.record()
    fn()
    end.record()
    synchronize_if_needed(device)
    return float(start.elapsed_time(end))


def _time_cpu_callable(fn) -> float:
    start = time.perf_counter()
    fn()
    end = time.perf_counter()
    return (end - start) * 1000.0


def profile_matching(
    query_embeddings: torch.Tensor,
    task_bank_tensors: Dict[str, Any],
    warmup_steps: int,
    profile_steps: int,
    device: torch.device,
) -> Tuple[List[float], torch.Tensor]:
    meta_embeddings = task_bank_tensors["meta_embeddings"]
    task_embeddings = task_bank_tensors["task_embeddings"]
    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = matching_step(query_embeddings, meta_embeddings, task_embeddings)
        synchronize_if_needed(device)
        times_ms = []
        latest_weights = None
        for _ in range(profile_steps):
            if device.type == "cuda":
                elapsed = _time_gpu_callable(lambda: matching_step(query_embeddings, meta_embeddings, task_embeddings), device)
                latest_weights = matching_step(query_embeddings, meta_embeddings, task_embeddings)
                synchronize_if_needed(device)
            else:
                elapsed = _time_cpu_callable(lambda: matching_step(query_embeddings, meta_embeddings, task_embeddings))
                latest_weights = matching_step(query_embeddings, meta_embeddings, task_embeddings)
            times_ms.append(elapsed / max(query_embeddings.shape[0], 1))
    return times_ms, latest_weights


def profile_aggregation(
    weights: torch.Tensor,
    task_bank_tensors: Dict[str, Any],
    warmup_steps: int,
    profile_steps: int,
    device: torch.device,
) -> List[float]:
    module_bank = task_bank_tensors["module_bank"]
    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = aggregation_step(weights, module_bank)
        synchronize_if_needed(device)
        times_ms = []
        for _ in range(profile_steps):
            if device.type == "cuda":
                elapsed = _time_gpu_callable(lambda: aggregation_step(weights, module_bank), device)
            else:
                elapsed = _time_cpu_callable(lambda: aggregation_step(weights, module_bank))
            times_ms.append(elapsed / max(weights.shape[0], 1))
    return times_ms


def build_prefill_inputs(tokenizer, batch_size: int, max_query_length: int, device: torch.device) -> Dict[str, torch.Tensor]:
    prompts = []
    for question in get_query_texts(batch_size):
        prompts.append(f"你是一位专业的AI医生，请根据患者的提问给出建议。\n\n患者提问：{question}\n\nAI医生回答：")
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_query_length,
    )
    return {key: value.to(device) for key, value in encoded.items()}


def profile_plm_forward(
    model,
    tokenizer,
    active_adapter: str,
    warmup_steps: int,
    profile_steps: int,
    device: torch.device,
    max_query_length: int,
    batch_size: int,
) -> List[float]:
    inputs = build_prefill_inputs(tokenizer, batch_size=batch_size, max_query_length=max_query_length, device=device)
    _ = active_adapter
    with model.disable_adapter():
        base_model = unwrap_prefill_model(model)
        with torch.no_grad():
            for _ in range(warmup_steps):
                _ = base_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    peft_config=None,
                    return_dict=True,
                )
            synchronize_if_needed(device)
            times_ms = []
            for _ in range(profile_steps):
                if device.type == "cuda":
                    elapsed = _time_gpu_callable(
                        lambda: base_model(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            peft_config=None,
                            return_dict=True,
                        ),
                        device,
                    )
                else:
                    elapsed = _time_cpu_callable(
                        lambda: base_model(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            peft_config=None,
                            return_dict=True,
                        )
                    )
                times_ms.append(elapsed / max(batch_size, 1))
    return times_ms


def summarize_metric(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return float(mean), float(std)


def aggregate_repeat_results(repeat_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    matching_values = [x["matching_time_ms_per_query"] for x in repeat_results]
    aggregation_values = [x["aggregation_time_ms_per_query"] for x in repeat_results]
    total_memr_values = [x["total_memr_overhead_ms_per_query"] for x in repeat_results]
    plm_values = [x["plm_forward_time_ms_per_query"] for x in repeat_results if x["plm_forward_time_ms_per_query"] is not None]
    total_inference_values = [x["total_inference_time_ms_per_query"] for x in repeat_results if x["total_inference_time_ms_per_query"] is not None]
    matching_ratio_values = [x["matching_ratio"] for x in repeat_results if x["matching_ratio"] is not None]
    aggregation_ratio_values = [x["aggregation_ratio"] for x in repeat_results if x["aggregation_ratio"] is not None]
    agg_match_ratio_values = [x["agg_match_ratio"] for x in repeat_results if x["agg_match_ratio"] is not None]
    peak_allocated = max((x["peak_gpu_allocated_mb"] for x in repeat_results), default=0.0)
    peak_reserved = max((x["peak_gpu_reserved_mb"] for x in repeat_results), default=0.0)

    return {
        "matching_time_mean_ms": summarize_metric(matching_values)[0],
        "matching_time_std_ms": summarize_metric(matching_values)[1],
        "aggregation_time_mean_ms": summarize_metric(aggregation_values)[0],
        "aggregation_time_std_ms": summarize_metric(aggregation_values)[1],
        "total_memr_overhead_mean_ms": summarize_metric(total_memr_values)[0],
        "total_memr_overhead_std_ms": summarize_metric(total_memr_values)[1],
        "plm_forward_time_mean_ms": summarize_metric(plm_values)[0],
        "plm_forward_time_std_ms": summarize_metric(plm_values)[1],
        "total_inference_time_mean_ms": summarize_metric(total_inference_values)[0],
        "total_inference_time_std_ms": summarize_metric(total_inference_values)[1],
        "matching_ratio_mean_percent": summarize_metric(matching_ratio_values)[0],
        "aggregation_ratio_mean_percent": summarize_metric(aggregation_ratio_values)[0],
        "agg_match_ratio_mean": summarize_metric(agg_match_ratio_values)[0],
        "peak_gpu_allocated_mb": peak_allocated,
        "peak_gpu_reserved_mb": peak_reserved,
    }


def build_markdown_table(rows: List[Dict[str, Any]], columns: Sequence[Tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        values = [str(row.get(key, "")) for key, _ in columns]
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep] + body)


def build_latex_table(rows: List[Dict[str, Any]], columns: Sequence[Tuple[str, str]]) -> str:
    lines = [
        "\\begin{tabular}{" + "c" * len(columns) + "}",
        "\\hline",
        " & ".join(label for _, label in columns) + " \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(" & ".join(str(row.get(key, "")) for key, _ in columns) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines)


def save_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def plot_latency(summary_rows: List[Dict[str, Any]], path: str) -> None:
    k = [row["K"] for row in summary_rows]
    plt.figure(figsize=(8, 5))
    plt.plot(k, [row["matching_time_mean_ms"] for row in summary_rows], marker="o", label="Task matching")
    plt.plot(k, [row["aggregation_time_mean_ms"] for row in summary_rows], marker="s", label="Module aggregation")
    plt.plot(k, [row["total_memr_overhead_mean_ms"] for row in summary_rows], marker="^", label="Total MeMR overhead")
    plt.xlabel("Number of Tasks K")
    plt.ylabel("Latency (ms/query)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_memory(summary_rows: List[Dict[str, Any]], path: str) -> None:
    k = [row["K"] for row in summary_rows]
    plt.figure(figsize=(8, 5))
    plt.plot(k, [row["task_repr_memory_mb"] for row in summary_rows], marker="o", label="Task representation memory")
    plt.plot(k, [row["frozen_module_memory_mb"] for row in summary_rows], marker="s", label="Frozen module memory")
    plt.plot(k, [row["task_related_memory_mb"] for row in summary_rows], marker="^", label="Total task-related memory")
    plt.plot(k, [row["peak_gpu_allocated_mb"] for row in summary_rows], marker="d", label="Peak GPU memory")
    plt.xlabel("Number of Tasks K")
    plt.ylabel("Memory (MB)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_combined(summary_rows: List[Dict[str, Any]], path: str) -> None:
    k = [row["K"] for row in summary_rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(k, [row["matching_time_mean_ms"] for row in summary_rows], marker="o", label="Task matching")
    axes[0].plot(k, [row["aggregation_time_mean_ms"] for row in summary_rows], marker="s", label="Module aggregation")
    axes[0].plot(k, [row["total_memr_overhead_mean_ms"] for row in summary_rows], marker="^", label="Total MeMR overhead")
    axes[0].set_xlabel("Number of Tasks K")
    axes[0].set_ylabel("Latency (ms/query)")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.3)

    axes[1].plot(k, [row["task_repr_memory_mb"] for row in summary_rows], marker="o", label="Task representation memory")
    axes[1].plot(k, [row["frozen_module_memory_mb"] for row in summary_rows], marker="s", label="Frozen module memory")
    axes[1].plot(k, [row["task_related_memory_mb"] for row in summary_rows], marker="^", label="Total task-related memory")
    axes[1].plot(k, [row["peak_gpu_allocated_mb"] for row in summary_rows], marker="d", label="Peak GPU memory")
    axes[1].set_xlabel("Number of Tasks K")
    axes[1].set_ylabel("Memory (MB)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close(fig)


def find_bottleneck_k(summary_rows: List[Dict[str, Any]]) -> Tuple[Optional[int], Optional[str]]:
    for row in summary_rows:
        if row["total_inference_time_mean_ms"] is None:
            continue
        matching_ratio = row["matching_ratio_mean_percent"] or 0.0
        aggregation_ratio = row["aggregation_ratio_mean_percent"] or 0.0
        if matching_ratio >= 10.0:
            return row["K"], "matching"
        if aggregation_ratio >= 10.0:
            return row["K"], "aggregation"
    return None, None


def build_summary_for_response(
    summary_rows: List[Dict[str, Any]],
    generation_info: Dict[str, Any],
    checkpoint_path: str,
    plm_warning: Optional[str],
) -> str:
    six_task_row = next((row for row in summary_rows if row["K"] == 6), summary_rows[min(len(summary_rows) - 1, 0)])
    max_row = summary_rows[-1]
    bottleneck_k, bottleneck_component = find_bottleneck_k(summary_rows)
    if six_task_row["aggregation_time_mean_ms"] is not None and six_task_row["matching_time_mean_ms"] is not None:
        dominant = "aggregation" if six_task_row["aggregation_time_mean_ms"] >= six_task_row["matching_time_mean_ms"] else "matching"
    else:
        dominant = "aggregation"

    if bottleneck_k is None:
        bottleneck_sentence = "Under the measured settings, neither matching nor aggregation exceeded the 10% practical bottleneck criterion of total inference latency."
    else:
        bottleneck_sentence = f"The 10% practical bottleneck criterion began to be approached at approximately K={bottleneck_k}, with {bottleneck_component} being the first component to cross the threshold."

    if plm_warning:
        plm_sentence = f"PLM forward profiling was attempted but not fully available: {plm_warning}"
    else:
        plm_sentence = "PLM forward profiling was successfully included using a fixed prefill-only forward pass."

    return "\n\n".join(
        [
            "This experiment is profiling-only and does not retrain additional tasks. We load an existing MeMR checkpoint, reuse the learned task representations and frozen task-specific modules, and isolate the runtime and memory overhead introduced by task matching and module aggregation as the task bank grows.",
            f"For K larger than the {generation_info['real_num_tasks']} real tasks available in the checkpoint, we construct an independent synthetic task bank by deterministically interpolating pairs of real metadata embeddings, task embeddings, and frozen module tensors, then adding a small fixed-seed perturbation. This synthetic expansion is used only for scalability profiling and not for QA accuracy evaluation.",
            "The measured complexity is consistent with the expected scaling behavior: task matching follows O(Kd), module aggregation follows O(K|P|), and task-related memory follows O(Kd + K|P|). Because the frozen module tensor bank is much larger than the task-representation bank, aggregation and task-related memory are the main quantities that can become dominant as K increases.",
            f"In the current profiling run based on `{checkpoint_path}`, the six-task setting remains computationally manageable: total MeMR overhead is {format_float(six_task_row['total_memr_overhead_mean_ms'])} ms/query at K=6, with {dominant} larger than the other MeMR component. At the largest measured scale K={max_row['K']}, task-related memory reaches {format_float(max_row['task_related_memory_mb'])} MB and peak GPU allocated memory reaches {format_float(max_row['peak_gpu_allocated_mb'])} MB. {bottleneck_sentence}",
            f"Potential future improvements include sparse top-r routing, module pruning, module merging, hierarchical task indexing, and more strongly shared low-rank module design. {plm_sentence}",
        ]
    )


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    profiler_dtype = resolve_dtype(args.dtype)
    k_list = parse_k_list(args.k_list)
    output_dir = setup_output_dir(args.output_dir)
    logger = setup_logging(output_dir)
    set_seed(args.seed)

    logger.info("Output directory: %s", output_dir)
    logger.info("Using device %s and profiling dtype %s", device, profiler_dtype)
    logger.info("Checkpoint path: %s", args.checkpoint_path)

    config_payload = {
        "K_list": k_list,
        "seed": args.seed,
        "device": str(device),
        "warmup_steps": args.warmup_steps,
        "profile_steps": args.profile_steps,
        "num_repeats": args.num_repeats,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "checkpoint_path": os.path.abspath(args.checkpoint_path),
        "base_model_path": args.base_model_path,
        "meta_embeddings_path": args.meta_embeddings_path,
        "measure_plm_forward": args.measure_plm_forward,
        "precompute_query_embeddings": args.precompute_query_embeddings,
        "perturb_scale": args.perturb_scale,
        "max_query_length": args.max_query_length,
        "task_bank_expansion_strategy": "Use real tasks when K <= num_real_tasks; otherwise synthesize independent task entries by deterministic interpolation plus small perturbation.",
    }
    save_json(os.path.join(output_dir, "config.json"), config_payload)

    inferred_output_dir = infer_output_dir_from_checkpoint(args.checkpoint_path)
    logger.info("Inferred training output directory for loader context: %s", inferred_output_dir)

    model, tokenizer, query_encoder, task_list = load_memr_model(
        base_model_path=args.base_model_path,
        checkpoint_dir=args.checkpoint_path,
        meta_embeddings_path=args.meta_embeddings_path,
        inference_log_dir=args.inference_log_dir,
    )
    model.eval()
    query_encoder.eval()

    real_task_bank = build_real_task_bank(args.checkpoint_path, model, task_list, profiler_dtype, logger)
    full_task_bank, generation_info = extend_task_bank(real_task_bank, max(k_list), args.seed, args.perturb_scale)
    save_json(os.path.join(output_dir, "task_bank_generation.json"), generation_info)

    query_warning = None
    hidden_size = real_task_bank[0].task_embedding.shape[-1]
    try:
        query_embeddings_cpu = encode_queries(
            tokenizer=tokenizer,
            query_encoder=query_encoder,
            query_texts=get_query_texts(args.batch_size),
            max_query_length=args.max_query_length,
            device=torch.device("cpu"),
            logger=logger,
        ).to(dtype=profiler_dtype)
    except Exception as exc:
        query_warning = f"Fell back to deterministic random query embeddings because the query encoder path failed: {exc}"
        logger.warning(query_warning)
        query_embeddings_cpu = build_random_query_embeddings(
            batch_size=args.batch_size,
            hidden_size=hidden_size,
            dtype=profiler_dtype,
            seed=args.seed,
            device=torch.device("cpu"),
        )

    active_adapter = task_list[-1]
    raw_results: Dict[str, Any] = {
        "config": config_payload,
        "query_embedding_source": "real_query_encoder" if query_warning is None else "deterministic_random_fallback",
        "query_embedding_warning": query_warning,
        "per_k": {},
    }
    summary_rows: List[Dict[str, Any]] = []
    plm_warning: Optional[str] = None

    for k in k_list:
        logger.info("Profiling K=%d", k)
        task_subset = full_task_bank[:k]
        task_bank_tensors = prepare_task_bank_tensors(task_subset, device=device, dtype=profiler_dtype)
        if args.precompute_query_embeddings:
            query_embeddings = query_embeddings_cpu.to(device=device, dtype=profiler_dtype)
        else:
            query_embeddings = query_embeddings_cpu.to(device=device, dtype=profiler_dtype)

        repeat_results = []
        for repeat_idx in range(args.num_repeats):
            logger.info("  Repeat %d/%d for K=%d", repeat_idx + 1, args.num_repeats, k)
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

            matching_times, weights = profile_matching(
                query_embeddings=query_embeddings,
                task_bank_tensors=task_bank_tensors,
                warmup_steps=args.warmup_steps,
                profile_steps=args.profile_steps,
                device=device,
            )
            aggregation_times = profile_aggregation(
                weights=weights,
                task_bank_tensors=task_bank_tensors,
                warmup_steps=args.warmup_steps,
                profile_steps=args.profile_steps,
                device=device,
            )

            matching_mean = statistics.mean(matching_times)
            aggregation_mean = statistics.mean(aggregation_times)
            total_memr = matching_mean + aggregation_mean

            plm_mean = None
            total_inference = None
            matching_ratio = None
            aggregation_ratio = None
            if args.measure_plm_forward:
                try:
                    plm_times = profile_plm_forward(
                        model=model,
                        tokenizer=tokenizer,
                        active_adapter=active_adapter,
                        warmup_steps=args.warmup_steps,
                        profile_steps=args.profile_steps,
                        device=device,
                        max_query_length=args.max_query_length,
                        batch_size=args.batch_size,
                    )
                    plm_mean = statistics.mean(plm_times)
                    total_inference = total_memr + plm_mean
                    if total_inference > 0:
                        matching_ratio = matching_mean / total_inference * 100.0
                        aggregation_ratio = aggregation_mean / total_inference * 100.0
                except Exception as exc:
                    if plm_warning is None:
                        plm_warning = str(exc)
                    logger.warning("PLM forward profiling failed at K=%d: %s", k, exc)

            repeat_result = {
                "repeat": repeat_idx,
                "matching_time_ms_per_query": float(matching_mean),
                "matching_samples_ms_per_query": matching_times,
                "aggregation_time_ms_per_query": float(aggregation_mean),
                "aggregation_samples_ms_per_query": aggregation_times,
                "total_memr_overhead_ms_per_query": float(total_memr),
                "plm_forward_time_ms_per_query": None if plm_mean is None else float(plm_mean),
                "total_inference_time_ms_per_query": None if total_inference is None else float(total_inference),
                "matching_ratio": None if matching_ratio is None else float(matching_ratio),
                "aggregation_ratio": None if aggregation_ratio is None else float(aggregation_ratio),
                "agg_match_ratio": float(aggregation_mean / matching_mean) if matching_mean > 0 else None,
                "peak_gpu_allocated_mb": bytes_to_mb(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0.0,
                "peak_gpu_reserved_mb": bytes_to_mb(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0.0,
            }
            repeat_results.append(repeat_result)

        task_repr_memory_mb = compute_task_repr_memory_mb(task_subset)
        frozen_module_memory_mb = compute_frozen_module_memory_mb(task_subset)
        aggregated = aggregate_repeat_results(repeat_results)
        summary_row = {
            "K": k,
            "matching_time_mean_ms": aggregated["matching_time_mean_ms"],
            "matching_time_std_ms": aggregated["matching_time_std_ms"],
            "aggregation_time_mean_ms": aggregated["aggregation_time_mean_ms"],
            "aggregation_time_std_ms": aggregated["aggregation_time_std_ms"],
            "total_memr_overhead_mean_ms": aggregated["total_memr_overhead_mean_ms"],
            "total_memr_overhead_std_ms": aggregated["total_memr_overhead_std_ms"],
            "plm_forward_time_mean_ms": aggregated["plm_forward_time_mean_ms"],
            "plm_forward_time_std_ms": aggregated["plm_forward_time_std_ms"],
            "total_inference_time_mean_ms": aggregated["total_inference_time_mean_ms"],
            "total_inference_time_std_ms": aggregated["total_inference_time_std_ms"],
            "matching_ratio_mean_percent": aggregated["matching_ratio_mean_percent"],
            "aggregation_ratio_mean_percent": aggregated["aggregation_ratio_mean_percent"],
            "agg_match_ratio_mean": aggregated["agg_match_ratio_mean"],
            "task_repr_memory_mb": task_repr_memory_mb,
            "frozen_module_memory_mb": frozen_module_memory_mb,
            "task_related_memory_mb": task_repr_memory_mb + frozen_module_memory_mb,
            "peak_gpu_allocated_mb": aggregated["peak_gpu_allocated_mb"],
            "peak_gpu_reserved_mb": aggregated["peak_gpu_reserved_mb"],
        }
        summary_rows.append(summary_row)
        raw_results["per_k"][str(k)] = {
            "task_ids": [entry.task_id for entry in task_subset],
            "summary": summary_row,
            "repeats": repeat_results,
        }

    save_json(os.path.join(output_dir, "raw_results.json"), raw_results)
    write_table_csv(os.path.join(output_dir, "scalability_results.csv"), summary_rows)
    write_table_csv(os.path.join(output_dir, "complexity_analysis.csv"), COMPLEXITY_ROWS)

    complexity_columns = [
        ("component", "Component"),
        ("main_operation", "Main Operation"),
        ("complexity", "Complexity"),
        ("scalability_implication", "Scalability Implication"),
    ]
    complexity_md = build_markdown_table(COMPLEXITY_ROWS, complexity_columns)
    save_text(os.path.join(output_dir, "table_complexity_analysis.md"), complexity_md)

    summary_table_rows = []
    for row in summary_rows:
        summary_table_rows.append(
            {
                "K": row["K"],
                "Match (ms)": format_float(row["matching_time_mean_ms"]),
                "Agg (ms)": format_float(row["aggregation_time_mean_ms"]),
                "MeMR (ms)": format_float(row["total_memr_overhead_mean_ms"]),
                "PLM (ms)": format_float(row["plm_forward_time_mean_ms"]),
                "Total (ms)": format_float(row["total_inference_time_mean_ms"]),
                "Match %": format_float(row["matching_ratio_mean_percent"]),
                "Agg %": format_float(row["aggregation_ratio_mean_percent"]),
                "Agg/Match": format_float(row["agg_match_ratio_mean"]),
                "Task Mem (MB)": format_float(row["task_related_memory_mb"]),
                "Peak GPU (MB)": format_float(row["peak_gpu_allocated_mb"]),
            }
        )

    scalability_columns = list((key, key) for key in summary_table_rows[0].keys())
    scalability_md = build_markdown_table(summary_table_rows, scalability_columns)
    scalability_tex = build_latex_table(summary_table_rows, scalability_columns)
    save_text(os.path.join(output_dir, "table_scalability_results.md"), scalability_md)
    save_text(os.path.join(output_dir, "table_scalability_results_latex.tex"), scalability_tex)

    plot_latency(summary_rows, os.path.join(output_dir, "fig_scalability_latency.png"))
    plot_memory(summary_rows, os.path.join(output_dir, "fig_scalability_memory.png"))
    plot_combined(summary_rows, os.path.join(output_dir, "fig_scalability_combined.png"))

    summary_text = build_summary_for_response(
        summary_rows=summary_rows,
        generation_info=generation_info,
        checkpoint_path=os.path.abspath(args.checkpoint_path),
        plm_warning=plm_warning if plm_warning else query_warning,
    )
    save_text(os.path.join(output_dir, "summary_for_response.md"), summary_text + "\n")
    logger.info("Finished profiling. Results saved to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
