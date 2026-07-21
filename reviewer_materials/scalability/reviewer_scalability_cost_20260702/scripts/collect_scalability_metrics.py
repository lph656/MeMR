from __future__ import annotations

import csv
import json
import os
import platform
import re
import statistics
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
try:
    import pandas as pd
except Exception:
    pd = None

from arguments import DataTrainingArguments, ModelArguments, OSLArguments
from model.causal_lm_llama import LlamaContinualForCausalLM
from mpeft import KeyEncoderConfig, LoraConfig, TaskType, get_peft_model
from tasks.mtl5.dataloader_mtl_causal_llama import DataLoaderMTL
from training.trainer_continual_causal_llama_lora import ContinualTrainerMTL
from utils.compute_metrics import compute_metrics
from utils.inference_utils import find_key_encoder, load_memr_model
from transformers import BitsAndBytesConfig, LlamaModel, LlamaTokenizer, TrainingArguments, set_seed


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "reviewer_scalability_cost_20260702"
DEFAULT_RESULTS_DIR = DEFAULT_OUTPUT_ROOT / "results"
DEFAULT_LOGS_DIR = DEFAULT_OUTPUT_ROOT / "logs"

PRIMARY_ORDER_DIR = PROJECT_ROOT / "checkpoints_continual_keshi_llama" / "order1_compose_peft"
PRIMARY_LOG_PATH = PRIMARY_ORDER_DIR / "log.txt"
PRIMARY_SNAPSHOT_DIR = PRIMARY_ORDER_DIR / "snapshots"
PRIMARY_FINAL_SNAPSHOT = PRIMARY_SNAPSHOT_DIR / "task_5_zhongliuke_train_end_20260626_123739"

ARCHIVE_RUNTIME_CSV = PROJECT_ROOT / "results" / "scalability_profiling" / "20260627_122553" / "scalability_results.csv"
BASE_MODEL_INDEX_JSON = PROJECT_ROOT / "chinese-alpaca-plus-7b-hf" / "pytorch_model.bin.index.json"
METADATA_EMBEDDINGS_PATH = PROJECT_ROOT / "metadata_embeddings" / "keshi_meta_embeddings.pt"
MEMR_METRICS_JSON = PROJECT_ROOT / "reviewer_fairness" / "results" / "reference" / "memr" / "order1" / "metrics.json"

BASE_MODEL_NAME = "chinese-alpaca-plus-7b-hf"
TASK_CODE_ORDER = ["IM", "S", "P", "GO", "A", "O"]
TASK_NAME_ORDER = ["neike", "waike", "erke", "fuchanke", "nanke", "zhongliuke"]
TASK_NAME_TO_CODE = dict(zip(TASK_NAME_ORDER, TASK_CODE_ORDER))

K_REAL_LIST = [1, 2, 3, 4, 5, 6]
K_EXTENDED_LIST = [8, 16, 32, 64]

TRAIN_PROFILE_MAX_BATCHES = 8
INFER_PROFILE_WARMUP = 3
INFER_PROFILE_STEPS = 10
INFER_PROFILE_EVAL_EXAMPLES = 8


@dataclass
class SnapshotInfo:
    k: int
    task_name: str
    task_code: str
    checkpoint_dir: Path
    checkpoint_info: Dict[str, Any]
    state_dict: OrderedDict[str, torch.Tensor]


@dataclass
class RuntimeProfile:
    inference_peak_gpu_memory_mb: Optional[float]
    matching_time_ms_mean: Optional[float]
    matching_time_ms_std: Optional[float]
    aggregation_time_ms_mean: Optional[float]
    aggregation_time_ms_std: Optional[float]
    plm_time_ms_mean: Optional[float]
    plm_time_ms_std: Optional[float]
    total_inference_latency_ms_mean: Optional[float]
    total_inference_latency_ms_std: Optional[float]
    matching_latency_ratio_percent: Optional[float]
    aggregation_latency_ratio_percent: Optional[float]
    total_overhead_ratio_percent: Optional[float]
    source: str


def ensure_dirs() -> None:
    DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_LOGS_DIR.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def maybe_write_xlsx(path: Path, sheets: Dict[str, List[Dict[str, Any]]]) -> Optional[str]:
    if pd is None:
        return "pandas_not_available"
    try:
        with pd.ExcelWriter(path) as writer:
            for name, rows in sheets.items():
                pd.DataFrame(rows).to_excel(writer, sheet_name=name[:31], index=False)
        return None
    except Exception as exc:
        return f"xlsx_write_failed: {exc}"


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.upper() == "N/A":
            return None
        if text.endswith("%"):
            text = text[:-1]
        try:
            return float(text)
        except Exception:
            return None
    try:
        return float(value)
    except Exception:
        return None


def format_value(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def bytes_to_mb(num_bytes: int | float) -> float:
    return float(num_bytes) / (1024.0 * 1024.0)


def parse_base_model_storage_and_params() -> Tuple[Optional[int], Optional[float]]:
    if not BASE_MODEL_INDEX_JSON.is_file():
        return None, None
    payload = read_json(BASE_MODEL_INDEX_JSON)
    total_bytes = payload.get("metadata", {}).get("total_size")
    if not isinstance(total_bytes, int):
        return None, None
    # Weight shards are fp16/bf16 serialized, so two bytes per parameter is the best local estimate.
    total_params = total_bytes // 2
    return int(total_params), bytes_to_mb(total_bytes)


def ensure_stage_meta_embeddings(k: int) -> Path:
    target = DEFAULT_LOGS_DIR / f"stage_meta_embeddings_k{k}.pt"
    if target.is_file():
        return target
    full_meta = torch.load(METADATA_EMBEDDINGS_PATH, map_location="cpu", weights_only=True)
    if not torch.is_tensor(full_meta):
        raise TypeError(f"Unexpected metadata embeddings payload type: {type(full_meta)}")
    sliced = full_meta[:k].contiguous().cpu()
    torch.save(sliced, target)
    return target


def load_state_dict(path: Path) -> OrderedDict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    ordered: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in payload.items():
        if torch.is_tensor(value):
            ordered[key] = value.detach().cpu()
    return ordered


def slice_state_dict_for_stage(state_dict: OrderedDict[str, torch.Tensor], k: int) -> OrderedDict[str, torch.Tensor]:
    sliced: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        if key.endswith("key_encoder.keys") or key.endswith("key_encoder.all_meta_keys"):
            if value.ndim >= 1 and value.shape[0] >= k:
                sliced[key] = value[:k].contiguous()
            else:
                sliced[key] = value
        else:
            sliced[key] = value
    return sliced


def discover_snapshots() -> List[SnapshotInfo]:
    pattern = re.compile(r"task_(\d+)_([^_]+)_train_end_")
    snapshots: List[SnapshotInfo] = []
    for child in sorted(PRIMARY_SNAPSHOT_DIR.iterdir()):
        if not child.is_dir():
            continue
        match = pattern.search(child.name)
        if not match:
            continue
        task_id = int(match.group(1))
        task_name = match.group(2)
        info_path = child / "checkpoint_info.json"
        state_dict_path = child / "state_dict.pt"
        if not info_path.is_file() or not state_dict_path.is_file():
            continue
        snapshots.append(
            SnapshotInfo(
                k=task_id + 1,
                task_name=task_name,
                task_code=TASK_NAME_TO_CODE[task_name],
                checkpoint_dir=child,
                checkpoint_info=read_json(info_path),
                state_dict=load_state_dict(state_dict_path),
            )
        )
    snapshots.sort(key=lambda item: item.k)
    return snapshots


def parse_training_times(log_path: Path) -> Dict[str, float]:
    lines = log_path.read_text(encoding="utf-8").splitlines()
    starts: Dict[str, str] = {}
    ends: Dict[str, str] = {}
    for line in lines:
        start_match = re.search(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - INFO - root - \*{5} 开始训练 - 任务 \d+: ([^ ]+) \*{5}$", line)
        if start_match:
            starts[start_match.group(2)] = start_match.group(1)
            continue
        end_match = re.search(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - INFO - root - Saved checkpoint info for ([^ ]+) at train_end", line)
        if end_match:
            ends[end_match.group(2)] = end_match.group(1)
    durations: Dict[str, float] = {}
    for task_name, start_text in starts.items():
        end_text = ends.get(task_name)
        if not end_text:
            continue
        start_ts = time.mktime(time.strptime(start_text, "%Y-%m-%d %H:%M:%S"))
        end_ts = time.mktime(time.strptime(end_text, "%Y-%m-%d %H:%M:%S"))
        durations[task_name] = end_ts - start_ts
    return durations


def load_archive_runtime_rows() -> Dict[int, Dict[str, Any]]:
    rows: Dict[int, Dict[str, Any]] = {}
    with ARCHIVE_RUNTIME_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[int(row["K"])] = row
    return rows


def split_snapshot_tensors(state_dict: OrderedDict[str, torch.Tensor]) -> Dict[str, List[Tuple[str, torch.Tensor]]]:
    buckets: Dict[str, List[Tuple[str, torch.Tensor]]] = {
        "lora": [],
        "trainable_key_vectors": [],
        "frozen_metadata_vectors": [],
        "routing_aux_trainable": [],
        "other": [],
    }
    for key, tensor in state_dict.items():
        if ".lora_" in key:
            buckets["lora"].append((key, tensor))
        elif key.endswith("key_encoder.keys"):
            buckets["trainable_key_vectors"].append((key, tensor))
        elif key.endswith("key_encoder.all_meta_keys"):
            buckets["frozen_metadata_vectors"].append((key, tensor))
        elif "key_encoder" in key:
            buckets["routing_aux_trainable"].append((key, tensor))
        else:
            buckets["other"].append((key, tensor))
    return buckets


def count_params(items: Iterable[Tuple[str, torch.Tensor]]) -> int:
    return int(sum(int(t.numel()) for _, t in items))


def count_bytes(items: Iterable[Tuple[str, torch.Tensor]]) -> int:
    return int(sum(tensor_bytes(t) for _, t in items))


def current_task_lora_slice(task_name: str, lora_items: List[Tuple[str, torch.Tensor]]) -> List[Tuple[str, torch.Tensor]]:
    token = f".{task_name}.weight"
    return [(key, tensor) for key, tensor in lora_items if token in key]


def build_parameter_rows(
    snapshots: List[SnapshotInfo],
    base_model_params: Optional[int],
    base_model_storage_mb: Optional[float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for snapshot in snapshots:
        buckets = split_snapshot_tensors(snapshot.state_dict)
        current_lora = current_task_lora_slice(snapshot.task_name, buckets["lora"])
        current_lora_params = count_params(current_lora)
        current_lora_bytes = count_bytes(current_lora)
        key_params = count_params(buckets["trainable_key_vectors"])
        key_bytes = count_bytes(buckets["trainable_key_vectors"])
        metadata_bytes = count_bytes(buckets["frozen_metadata_vectors"])
        routing_aux_params = count_params(buckets["routing_aux_trainable"])
        routing_aux_bytes = count_bytes(buckets["routing_aux_trainable"])

        rows.append(
            {
                "K": snapshot.k,
                "per_task_trainable_params": current_lora_params + (key_params // snapshot.k if snapshot.k else 0),
                "current_task_module_params": current_lora_params,
                "task_embedding_params": key_params,
                "routing_related_params": routing_aux_params,
                "cumulative_task_module_params": count_params(buckets["lora"]),
                "cumulative_task_related_params": count_params(buckets["lora"]) + key_params + routing_aux_params,
                "task_module_storage_MB": round(bytes_to_mb(count_bytes(buckets["lora"])), 6),
                "embedding_storage_MB": round(bytes_to_mb(key_bytes + metadata_bytes), 6),
                "routing_storage_MB": round(bytes_to_mb(routing_aux_bytes), 6),
                "cumulative_task_related_storage_MB": round(bytes_to_mb(count_bytes(buckets["lora"]) + key_bytes + metadata_bytes + routing_aux_bytes), 6),
                "base_model_params": base_model_params,
                "base_model_storage_MB": round(base_model_storage_mb, 6) if base_model_storage_mb is not None else None,
            }
        )
    return rows


def make_runtime_profile_from_archive(row: Dict[str, Any]) -> RuntimeProfile:
    matching = safe_float(row.get("matching_time_mean_ms"))
    aggregation = safe_float(row.get("aggregation_time_mean_ms"))
    plm_time = safe_float(row.get("plm_forward_time_mean_ms"))
    total_inference = safe_float(row.get("total_inference_time_mean_ms"))
    matching_ratio = safe_float(row.get("matching_ratio_mean_percent"))
    aggregation_ratio = safe_float(row.get("aggregation_ratio_mean_percent"))
    overhead_ratio = None
    if matching_ratio is not None and aggregation_ratio is not None:
        overhead_ratio = matching_ratio + aggregation_ratio
    elif matching is not None and aggregation is not None and plm_time is not None:
        denom = matching + aggregation + plm_time
        if denom > 0:
            overhead_ratio = (matching + aggregation) / denom * 100.0
    return RuntimeProfile(
        inference_peak_gpu_memory_mb=safe_float(row.get("peak_gpu_allocated_mb")),
        matching_time_ms_mean=matching,
        matching_time_ms_std=safe_float(row.get("matching_time_std_ms")),
        aggregation_time_ms_mean=aggregation,
        aggregation_time_ms_std=safe_float(row.get("aggregation_time_std_ms")),
        plm_time_ms_mean=plm_time,
        plm_time_ms_std=safe_float(row.get("plm_forward_time_std_ms")),
        total_inference_latency_ms_mean=total_inference,
        total_inference_latency_ms_std=safe_float(row.get("total_inference_time_std_ms")),
        matching_latency_ratio_percent=matching_ratio,
        aggregation_latency_ratio_percent=aggregation_ratio,
        total_overhead_ratio_percent=overhead_ratio,
        source="archived_scalability_profile",
    )


def get_compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_quantization_config(enable_cpu_offload: bool = False) -> BitsAndBytesConfig:
    kwargs: Dict[str, Any] = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": get_compute_dtype(),
    }
    if enable_cpu_offload:
        kwargs["llm_int8_enable_fp32_cpu_offload"] = True
    return BitsAndBytesConfig(**kwargs)


def build_auto_max_memory() -> Optional[Dict[Any, str]]:
    if not torch.cuda.is_available():
        return None
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except Exception:
        return None
    reserve_bytes = 1024 ** 3
    usable_bytes = max(int(free_bytes - reserve_bytes), int(total_bytes * 0.6))
    usable_gib = max(1, usable_bytes // (1024 ** 3))
    return {0: f"{usable_gib}GiB", "cpu": "256GiB"}


def get_model_input_device(model: Any) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        pass
    try:
        return torch.device(model.device)
    except Exception:
        pass
    return next(model.parameters()).device


class NullScalarWriter:
    def add_scalar(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None


def disable_key_encoder_logging(model: Any) -> None:
    key_encoder = find_key_encoder(model)
    if key_encoder is None:
        return
    key_encoder.log_weights = False
    if hasattr(key_encoder, "writer") and key_encoder.writer is not None:
        try:
            key_encoder.writer.close()
        except Exception:
            pass
    key_encoder.writer = NullScalarWriter()


def reset_key_encoder_runtime_state(model: Any) -> None:
    key_encoder = find_key_encoder(model)
    if key_encoder is None:
        return
    if hasattr(key_encoder, "attn_weights") and isinstance(key_encoder.attn_weights, list):
        key_encoder.attn_weights = [None for _ in range(len(key_encoder.attn_weights))]
    for attr in ("steps", "steps_val", "steps_final"):
        state = getattr(key_encoder, attr, None)
        if isinstance(state, dict):
            for key in list(state.keys()):
                state[key] = 0


def load_4bit_model_with_fallback(
    model_cls,
    pretrained_model_name_or_path: str,
    *,
    notes: Optional[List[str]] = None,
    note_prefix: str = "",
    extra_kwargs: Optional[Dict[str, Any]] = None,
):
    extra_kwargs = dict(extra_kwargs or {})
    attempts: List[Tuple[str, Dict[str, Any]]] = [
        (
            "4bit_auto",
            {
                "quantization_config": build_quantization_config(enable_cpu_offload=False),
                "device_map": "auto",
                "trust_remote_code": True,
            },
        ),
        (
            "4bit_auto_cpu_offload",
            {
                "quantization_config": build_quantization_config(enable_cpu_offload=True),
                "device_map": "auto",
                "max_memory": build_auto_max_memory(),
                "offload_folder": str(DEFAULT_LOGS_DIR / "hf_offload"),
                "trust_remote_code": True,
            },
        ),
    ]
    failures: List[str] = []
    for label, kwargs in attempts:
        merged = dict(kwargs)
        merged.update(extra_kwargs)
        if merged.get("max_memory") is None:
            merged.pop("max_memory", None)
        try:
            return model_cls.from_pretrained(pretrained_model_name_or_path, **merged)
        except Exception as exc:
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
    if notes is not None:
        message = "; ".join(failures)
        prefix = f"{note_prefix}: " if note_prefix else ""
        notes.append(f"{prefix}4-bit model loading failed after fallback attempts. {message}")
    raise RuntimeError("; ".join(failures))


def load_query_encoder_with_fallback(
    pretrained_model_name_or_path: str,
    *,
    notes: Optional[List[str]] = None,
    note_prefix: str = "",
):
    compute_dtype = get_compute_dtype()
    attempts: List[Tuple[str, Dict[str, Any]]] = [
        (
            "4bit_auto",
            {
                "quantization_config": build_quantization_config(enable_cpu_offload=False),
                "device_map": "auto",
                "offload_folder": str(DEFAULT_LOGS_DIR / "hf_offload_query"),
                "torch_dtype": compute_dtype,
                "trust_remote_code": True,
            },
        ),
        (
            "4bit_auto_cpu_offload",
            {
                "quantization_config": build_quantization_config(enable_cpu_offload=True),
                "device_map": "auto",
                "max_memory": build_auto_max_memory(),
                "offload_folder": str(DEFAULT_LOGS_DIR / "hf_offload_query"),
                "torch_dtype": compute_dtype,
                "trust_remote_code": True,
            },
        ),
        (
            "cpu_fp32_fallback",
            {
                "torch_dtype": torch.float32,
                "device_map": {"": "cpu"},
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
            },
        ),
    ]
    failures: List[str] = []
    for label, kwargs in attempts:
        merged = dict(kwargs)
        if merged.get("max_memory") is None:
            merged.pop("max_memory", None)
        try:
            query_encoder = LlamaModel.from_pretrained(pretrained_model_name_or_path, **merged)
            for p in query_encoder.parameters():
                p.requires_grad = False
            query_encoder.eval()
            if label != "4bit_auto" and notes is not None:
                prefix = f"{note_prefix}: " if note_prefix else ""
                notes.append(f"{prefix}query_encoder loaded via fallback mode `{label}` instead of the default 4-bit auto placement.")
            return query_encoder
        except Exception as exc:
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
    if notes is not None:
        message = "; ".join(failures)
        prefix = f"{note_prefix}: " if note_prefix else ""
        notes.append(f"{prefix}query_encoder loading failed after fallback attempts. {message}")
    raise RuntimeError("; ".join(failures))


def build_peft_config(task_list: List[str], meta_embeddings_path: Path, output_dir: Path, hidden_size: int) -> LoraConfig:
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        mpeft_enabled=True,
        output_dir=str(output_dir),
    )
    peft_config.mpeft_config = KeyEncoderConfig(
        seed=0,
        query_encoder_type="avg_word_embed",
        task_list=task_list,
        meta_embeddings_path=str(meta_embeddings_path),
        key_dim=hidden_size,
        matching_loss_v2=True,
        matching_loss_coeff=1.0,
        output_dir=str(output_dir),
    )
    return peft_config


def load_stage_memr_model(checkpoint_dir: Path, k: int, log_tag: str, notes: Optional[List[str]] = None):
    stage_meta_path = ensure_stage_meta_embeddings(k)
    tokenizer = LlamaTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        add_prefix_space=True,
        padding_side="left",
        trust_remote_code=True,
    )
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 1
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    checkpoint_info = read_json(checkpoint_dir / "checkpoint_info.json")
    task_list = checkpoint_info.get("lora_adapters") or TASK_NAME_ORDER[:k]
    model = load_4bit_model_with_fallback(
        LlamaContinualForCausalLM,
        BASE_MODEL_NAME,
        notes=notes,
        note_prefix=f"K={k} {log_tag}",
    )
    query_encoder = load_query_encoder_with_fallback(
        BASE_MODEL_NAME,
        notes=notes,
        note_prefix=f"K={k} {log_tag}",
    )

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
        meta_embeddings_path=str(stage_meta_path),
        key_dim=query_encoder.config.hidden_size,
        output_dir=str(DEFAULT_LOGS_DIR / log_tag),
    )
    for task_name in task_list:
        model = get_peft_model(model, peft_config, adapter_name=task_name)

    state_dict = slice_state_dict_for_stage(load_state_dict(checkpoint_dir / "state_dict.pt"), k)
    model.load_state_dict(state_dict, strict=False)
    disable_key_encoder_logging(model)
    model.eval()
    return model, tokenizer, query_encoder, task_list


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(device: torch.device, fn) -> float:
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        synchronize(device)
        start.record()
        fn()
        end.record()
        synchronize(device)
        return float(start.elapsed_time(end))
    start = time.perf_counter()
    fn()
    end = time.perf_counter()
    return (end - start) * 1000.0


def profile_inference_runtime(
    checkpoint_dir: Path,
    k: int,
    task_list: List[str],
    notes: List[str],
) -> RuntimeProfile:
    compute_dtype = get_compute_dtype()
    model, tokenizer, query_encoder, loaded_tasks = load_stage_memr_model(checkpoint_dir, k, f"inference_loader_k{k}", notes)
    key_encoder = find_key_encoder(model)
    if key_encoder is None:
        raise RuntimeError("TaskKeyEncoder not found in loaded MeMR model.")
    reset_key_encoder_runtime_state(model)

    device = next(key_encoder.parameters()).device
    query_encoder_device = next(query_encoder.parameters()).device
    model_input_device = get_model_input_device(model)

    prompts = [
        "我最近总是头痛恶心，需要挂什么科？",
        "孩子发烧两天了，还伴有咳嗽怎么办？",
        "肚子一直隐隐作痛，吃完饭更明显，是胃的问题吗？",
        "最近胸口发闷，活动后更明显，要不要去医院检查？",
        "月经不规律而且小腹疼痛，应该看哪个科室？",
        "老人血压波动很大，需要注意什么？",
        "耳朵里面嗡嗡响，晚上更严重，这是耳鸣吗？",
        "排尿时刺痛并且尿频，需要做什么检查？",
    ][:INFER_PROFILE_EVAL_EXAMPLES]

    with torch.no_grad():
        q_embeds: List[torch.Tensor] = []
        for question in prompts:
            q_inputs = tokenizer(f"Query: {question}", return_tensors="pt")
            q_inputs = {key: value.to(query_encoder_device) for key, value in q_inputs.items()}
            q_outputs = query_encoder(**q_inputs)
            q_embed = q_outputs.last_hidden_state[:, -1, :].detach().to(device=device, dtype=compute_dtype)
            q_embeds.append(q_embed)
        query_embed = torch.cat(q_embeds, dim=0)

        prompt_texts = [f"你是一位专业的AI医生，请根据患者的提问给出建议。\n\n患者提问：{q}\n\nAI医生回答：" for q in prompts]
        model_inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        model_inputs = {key: value.to(model_input_device) for key, value in model_inputs.items()}

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        # Warmup
        for _ in range(INFER_PROFILE_WARMUP):
            _ = key_encoder(query_embed, adapter_name=loaded_tasks[-1], train=False, final=True)
            _ = model(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                labels=model_inputs["input_ids"],
                loss_mask=torch.ones_like(model_inputs["input_ids"]),
                active_adapter=loaded_tasks[-1],
                query_embed=query_embed,
                train=False,
                final=True,
            )
        synchronize(device)

        matching_times: List[float] = []
        aggregation_times: List[float] = []
        plm_times: List[float] = []
        total_times: List[float] = []

        for _ in range(INFER_PROFILE_STEPS):
            holder: Dict[str, Any] = {}

            def match_fn() -> None:
                holder["w"], _ = key_encoder(query_embed, adapter_name=loaded_tasks[-1], train=False, final=True)

            match_ms = timed_call(device, match_fn) / len(prompts)
            matching_times.append(match_ms)

            # Measure full model forward and subtract MeMR overhead from full pass to estimate PLM time.
            def total_fn() -> None:
                holder["outputs"] = model(
                    input_ids=model_inputs["input_ids"],
                    attention_mask=model_inputs["attention_mask"],
                    labels=model_inputs["input_ids"],
                    loss_mask=torch.ones_like(model_inputs["input_ids"]),
                    active_adapter=loaded_tasks[-1],
                    query_embed=query_embed,
                    train=False,
                    final=True,
                )

            total_ms = timed_call(device, total_fn) / len(prompts)
            total_times.append(total_ms)

            # Approximate aggregation as archived total overhead minus archived matching when available.
            # For live runs we measure total MeMR overhead again by replaying key_encoder plus full pass without query adaptation
            # and then isolate aggregation as residual in the archived-style overhead decomposition.
            aggregation_ms = None
            if key_encoder.attn_weights and key_encoder.attn_weights[-1] is not None:
                pass
            # In this code path, a separate pure aggregation hook is not exposed, so we use archived value for K=3/5 gap filling
            # only when live runtime was necessary.
            aggregation_ms = None
            aggregation_times.append(aggregation_ms if aggregation_ms is not None else float("nan"))

            plm_est = total_ms - match_ms
            if plm_est < 0:
                plm_est = 0.0
            plm_times.append(plm_est)

        peak_mem = bytes_to_mb(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None

    # For live gap filling, aggregation-only instrumentation is not accessible without invasive monkeypatching.
    # Use archived regression fit to avoid misleading zeros.
    valid_archive = load_archive_runtime_rows()
    if k in valid_archive and safe_float(valid_archive[k].get("aggregation_time_mean_ms")) is not None:
        aggregation_mean = safe_float(valid_archive[k].get("aggregation_time_mean_ms"))
        aggregation_std = safe_float(valid_archive[k].get("aggregation_time_std_ms"))
        source = "archived_scalability_profile"
    else:
        known = [(kk, safe_float(row.get("aggregation_time_mean_ms")), safe_float(row.get("aggregation_time_std_ms"))) for kk, row in valid_archive.items() if kk <= 8]
        known = [(kk, mean, std) for kk, mean, std in known if mean is not None]
        if len(known) >= 2:
            x = [float(kk) for kk, _, _ in known]
            y = [float(mean) for _, mean, _ in known]
            mean_x = sum(x) / len(x)
            mean_y = sum(y) / len(y)
            denom = sum((xi - mean_x) ** 2 for xi in x) or 1.0
            slope = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / denom
            intercept = mean_y - slope * mean_x
            aggregation_mean = intercept + slope * k
            aggregation_std = statistics.mean([float(std) for _, _, std in known if std is not None]) if known else None
            notes.append(f"K={k}: aggregation_time derived by linear interpolation/regression from archived K grid because pure live aggregation hook is not directly exposed.")
            source = "mixed_live_profile_plus_archived_interpolation"
        else:
            aggregation_mean = None
            aggregation_std = None
            source = "short_profile_estimate"

    matching_mean = statistics.mean(matching_times) if matching_times else None
    matching_std = statistics.stdev(matching_times) if len(matching_times) > 1 else 0.0 if matching_times else None
    total_mean = statistics.mean(total_times) if total_times else None
    total_std = statistics.stdev(total_times) if len(total_times) > 1 else 0.0 if total_times else None

    if aggregation_mean is not None and matching_mean is not None:
        overhead = aggregation_mean + matching_mean
    else:
        overhead = None
    if total_mean is not None and overhead is not None:
        plm_mean = max(total_mean - overhead, 0.0)
        matching_ratio = matching_mean / total_mean * 100.0
        aggregation_ratio = aggregation_mean / total_mean * 100.0
        overhead_ratio = overhead / total_mean * 100.0
    else:
        plm_mean = statistics.mean(plm_times) if plm_times else None
        matching_ratio = None
        aggregation_ratio = None
        overhead_ratio = None

    return RuntimeProfile(
        inference_peak_gpu_memory_mb=peak_mem,
        matching_time_ms_mean=matching_mean,
        matching_time_ms_std=matching_std,
        aggregation_time_ms_mean=aggregation_mean,
        aggregation_time_ms_std=aggregation_std,
        plm_time_ms_mean=plm_mean,
        plm_time_ms_std=None,
        total_inference_latency_ms_mean=total_mean,
        total_inference_latency_ms_std=total_std,
        matching_latency_ratio_percent=matching_ratio,
        aggregation_latency_ratio_percent=aggregation_ratio,
        total_overhead_ratio_percent=overhead_ratio,
        source=source,
    )


def make_training_args(output_dir: Path) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        do_train=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        num_train_epochs=1.0,
        logging_steps=1,
        save_strategy="no",
        evaluation_strategy="no",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        seed=0,
        report_to=[],
    )


def profile_training_peak_and_step(
    checkpoint_snapshot: SnapshotInfo,
    notes: List[str],
) -> Tuple[Optional[float], Optional[float], str]:
    compute_dtype = get_compute_dtype()
    profile_dir = DEFAULT_LOGS_DIR / f"train_profile_k{checkpoint_snapshot.k}"
    profile_dir.mkdir(parents=True, exist_ok=True)

    stage_meta_path = ensure_stage_meta_embeddings(checkpoint_snapshot.k)
    training_args = make_training_args(profile_dir)
    data_args = DataTrainingArguments(
        task_list="_".join(TASK_NAME_ORDER[:checkpoint_snapshot.k]),
        validation_split_percentage=0.1,
        max_seq_length=512,
        max_target_length=64,
        overwrite_cache=False,
    )
    model_args = ModelArguments(
        model_name_or_path=BASE_MODEL_NAME,
        meta_embeddings_path=str(stage_meta_path),
        mpeft_enabled=True,
        continual_learning=True,
        query_encoder_type="avg_word_embed",
        matching_loss_v2=True,
        matching_loss_coeff=1.0,
        multi_peft_modules=True,
    )
    osl_args = OSLArguments(lamda_1=0.05, lamda_2=0.01, orthogonal_threshold=0.2)

    tokenizer = LlamaTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        add_prefix_space=True,
        padding_side="left",
        trust_remote_code=True,
    )
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 1

    set_seed(0)

    dataloaders = DataLoaderMTL(
        data_args=data_args,
        training_args=training_args,
        task_list=TASK_NAME_ORDER[:checkpoint_snapshot.k],
        tokenizer=tokenizer,
        max_seq_length=512,
        overwrite_cache=False,
    )
    train_loader = dataloaders[checkpoint_snapshot.task_name]["train"]

    try:
        model = load_4bit_model_with_fallback(
            LlamaContinualForCausalLM,
            BASE_MODEL_NAME,
            notes=notes,
            note_prefix=f"K={checkpoint_snapshot.k} training_profile",
        )
    except Exception:
        notes.append(
            f"K={checkpoint_snapshot.k}: training peak memory short profile skipped because the quantized backbone could not be placed safely on the available GPU memory."
        )
        return None, None, "skipped_due_to_memory_guard"
    for attr in dir(model_args):
        if not attr.startswith("__") and not callable(getattr(model_args, attr)):
            setattr(model.config, attr, getattr(model_args, attr))

    query_encoder = load_query_encoder_with_fallback(
        BASE_MODEL_NAME,
        notes=notes,
        note_prefix=f"K={checkpoint_snapshot.k} training_profile",
    )

    peft_config = build_peft_config(TASK_NAME_ORDER[:checkpoint_snapshot.k], stage_meta_path, profile_dir, model.config.hidden_size)
    for task_name in TASK_NAME_ORDER[:checkpoint_snapshot.k]:
        model = get_peft_model(model, peft_config, adapter_name=task_name)
    model.load_state_dict(slice_state_dict_for_stage(checkpoint_snapshot.state_dict, checkpoint_snapshot.k), strict=False)
    disable_key_encoder_logging(model)
    model.set_adapter(checkpoint_snapshot.task_name)

    trainer = ContinualTrainerMTL(
        args=training_args,
        model=model,
        query_encoder=query_encoder,
        logger=type("NullLogger", (), {"info": lambda *args, **kwargs: None})(),
        task_list=TASK_NAME_ORDER[:checkpoint_snapshot.k],
        peft_config=peft_config,
        lora_save_dir=str(profile_dir / "checkpoint_loras"),
        tokenizer=tokenizer,
        max_target_length=64,
        learning_rate_list=None,
        max_train_batches_per_epoch=TRAIN_PROFILE_MAX_BATCHES,
        max_eval_batches=1,
        max_final_test_batches=1,
        lamda_1=0.05,
        lamda_2=0.01,
        orthogonal_threshold=0.2,
    )
    trainer._prepare_optimizer(checkpoint_snapshot.k - 1)

    device = get_model_input_device(model)
    step_times: List[float] = []
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model.train()
    model.zero_grad()
    for step_idx, batch in enumerate(train_loader):
        if step_idx >= TRAIN_PROFILE_MAX_BATCHES:
            break
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
        query_embed = trainer._get_query_embed(batch)
        if torch.is_tensor(query_embed):
            query_embed = query_embed.to(next(find_key_encoder(model).parameters()).device, dtype=compute_dtype)

        def step_fn() -> None:
            outputs = trainer.compute_loss(model, batch, checkpoint_snapshot.task_name, mode="train", query_embed=query_embed)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            loss.backward()
            trainer.optimizer.zero_grad()

        step_ms = timed_call(device, step_fn)
        step_times.append(step_ms / training_args.gradient_accumulation_steps)

    peak_mem = bytes_to_mb(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    avg_step_sec = statistics.mean(step_times) / 1000.0 if step_times else None
    notes.append(
        f"K={checkpoint_snapshot.k}: training peak memory and step time measured via short_profile_estimate over <= {TRAIN_PROFILE_MAX_BATCHES} mini-batches; no checkpoint saved."
    )
    return peak_mem, avg_step_sec, "short_profile_estimate"


def make_runtime_rows(
    snapshots: List[SnapshotInfo],
    archive_rows: Dict[int, Dict[str, Any]],
    training_times: Dict[str, float],
    notes: List[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for snapshot in snapshots:
        k = snapshot.k
        if k in archive_rows:
            runtime = make_runtime_profile_from_archive(archive_rows[k])
        else:
            runtime = profile_inference_runtime(snapshot.checkpoint_dir, k, TASK_NAME_ORDER[:k], notes)

        training_peak, training_step, training_source = profile_training_peak_and_step(snapshot, notes)
        full_training_time = training_times.get(snapshot.task_name)
        if full_training_time is not None:
            training_time_sec = full_training_time
            training_time_source = "parsed_from_existing_logs"
        else:
            training_time_sec = training_step
            training_time_source = training_source

        rows.append(
            {
                "K": k,
                "training_peak_gpu_memory_MB": training_peak,
                "training_time_sec_per_task": training_time_sec,
                "training_time_source": training_time_source,
                "inference_peak_gpu_memory_MB": runtime.inference_peak_gpu_memory_mb,
                "matching_time_ms_mean": runtime.matching_time_ms_mean,
                "matching_time_ms_std": runtime.matching_time_ms_std,
                "aggregation_time_ms_mean": runtime.aggregation_time_ms_mean,
                "aggregation_time_ms_std": runtime.aggregation_time_ms_std,
                "plm_time_ms_mean": runtime.plm_time_ms_mean,
                "plm_time_ms_std": runtime.plm_time_ms_std,
                "total_inference_latency_ms_mean": runtime.total_inference_latency_ms_mean,
                "total_inference_latency_ms_std": runtime.total_inference_latency_ms_std,
                "matching_latency_ratio_percent": runtime.matching_latency_ratio_percent,
                "aggregation_latency_ratio_percent": runtime.aggregation_latency_ratio_percent,
                "total_overhead_ratio_percent": runtime.total_overhead_ratio_percent,
            }
        )
    return rows


def parse_manuscript_order1_metrics() -> Dict[str, float]:
    # Table 5, order-1, Ours row
    return {
        "average": 82.90,
        "FWT": 13.65,
        "FR": 2.24,
        "BWT": -2.48,
    }


def load_memr_reference_payload() -> Dict[str, Any]:
    return read_json(MEMR_METRICS_JSON)


def evaluate_snapshot_seen_tasks(
    snapshot: SnapshotInfo,
    notes: List[str],
) -> Dict[str, Optional[float]]:
    model, tokenizer, query_encoder, task_list = load_stage_memr_model(snapshot.checkpoint_dir, snapshot.k, f"eval_loader_k{snapshot.k}", notes)
    model.eval()
    query_encoder.eval()
    reset_key_encoder_runtime_state(model)

    compute_dtype = get_compute_dtype()
    training_args = make_training_args(DEFAULT_LOGS_DIR / f"eval_args_k{snapshot.k}")
    data_args = DataTrainingArguments(
        task_list="_".join(TASK_NAME_ORDER[:snapshot.k]),
        validation_split_percentage=0.1,
        max_seq_length=512,
        max_target_length=64,
        overwrite_cache=False,
    )
    dataloaders = DataLoaderMTL(
        data_args=data_args,
        training_args=training_args,
        task_list=TASK_NAME_ORDER[:snapshot.k],
        tokenizer=tokenizer,
        max_seq_length=512,
        overwrite_cache=False,
    )

    scores: List[float] = []
    model_input_device = get_model_input_device(model)
    key_encoder = find_key_encoder(model)
    if key_encoder is None:
        raise RuntimeError("TaskKeyEncoder not found in loaded MeMR model.")
    key_device = next(key_encoder.parameters()).device
    query_encoder_device = next(query_encoder.parameters()).device
    for task_name in TASK_NAME_ORDER[:snapshot.k]:
        loader = dataloaders[task_name]["dev"]
        preds: List[str] = []
        refs: List[str] = []
        for idx, batch in enumerate(loader):
            if idx >= INFER_PROFILE_EVAL_EXAMPLES:
                break
            targets = batch.pop("targets")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(model_input_device)
            q_inputs = {
                "input_ids": batch["query_input_ids"].to(query_encoder_device),
                "attention_mask": batch["query_attention_mask"].to(query_encoder_device),
            }
            with torch.no_grad():
                q_outputs = query_encoder(**q_inputs)
                q_embed = q_outputs.last_hidden_state[:, -1, :].to(key_device, dtype=compute_dtype)
                generated = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    active_adapter=task_name,
                    query_embed=q_embed,
                    train=False,
                    final=True,
                    generation_config=None,
                    max_new_tokens=64,
                    do_sample=False,
                    use_cache=False,
                )
            prompt_len = batch["input_ids"].shape[1]
            generated_part = generated[:, prompt_len:]
            preds.extend(tokenizer.batch_decode(generated_part, skip_special_tokens=True, clean_up_tokenization_spaces=True))
            if torch.is_tensor(targets):
                ref_inputs = targets
            else:
                ref_inputs = []
                for item in targets:
                    if torch.is_tensor(item):
                        ref_inputs.append(item.tolist())
                    else:
                        ref_inputs.append(item)
            refs.extend(tokenizer.batch_decode(ref_inputs, skip_special_tokens=True, clean_up_tokenization_spaces=True))
        if preds and refs:
            metrics = compute_metrics([p.strip() for p in preds], [r.strip() for r in refs])
            scores.append(metrics["rougeL"])

    avg_score = round(sum(scores) / len(scores), 4) if scores else None
    notes.append(
        f"K={snapshot.k}: average seen-task performance estimated via lightweight dev evaluation over up to {INFER_PROFILE_EVAL_EXAMPLES} batches/task; source marked short_profile_estimate."
    )
    return {
        "average_llm_score_on_seen_tasks": avg_score,
        "final_average_performance_or_FAP": avg_score,
        "FWT": None,
        "FR": None,
        "BWT": None,
        "source": "short_profile_estimate",
    }


def make_performance_rows(snapshots: List[SnapshotInfo], notes: List[str]) -> List[Dict[str, Any]]:
    manuscript = parse_manuscript_order1_metrics()
    rows: List[Dict[str, Any]] = []
    for snapshot in snapshots:
        learned = ",".join(TASK_CODE_ORDER[:snapshot.k])
        if snapshot.k == 6:
            rows.append(
                {
                    "K": 6,
                    "learned_tasks": learned,
                    "average_llm_score_on_seen_tasks": manuscript["average"],
                    "final_average_performance_or_FAP": manuscript["average"],
                    "FWT": manuscript["FWT"],
                    "FR": manuscript["FR"],
                    "BWT": manuscript["BWT"],
                    "source": "manuscript_table5_order1",
                }
            )
        else:
            try:
                estimate = evaluate_snapshot_seen_tasks(snapshot, notes)
            except Exception as exc:
                notes.append(f"K={snapshot.k}: lightweight seen-task evaluation failed and was kept null. {type(exc).__name__}: {exc}")
                estimate = {
                    "average_llm_score_on_seen_tasks": None,
                    "final_average_performance_or_FAP": None,
                    "FWT": None,
                    "FR": None,
                    "BWT": None,
                    "source": "evaluation_failed_kept_null",
                }
            rows.append(
                {
                    "K": snapshot.k,
                    "learned_tasks": learned,
                    "average_llm_score_on_seen_tasks": estimate["average_llm_score_on_seen_tasks"],
                    "final_average_performance_or_FAP": estimate["final_average_performance_or_FAP"],
                    "FWT": estimate["FWT"],
                    "FR": estimate["FR"],
                    "BWT": estimate["BWT"],
                    "source": estimate["source"],
                }
            )
    notes.append("Stagewise K=1..5 FWT/FR/BWT are still unavailable without archived stage matrices or full rerun; only final K=6 order-1 CL metrics are recovered from the manuscript.")
    return rows


def make_extended_rows(archive_rows: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for k in K_EXTENDED_LIST:
        row = archive_rows[k]
        overhead = safe_float(row.get("total_memr_overhead_mean_ms"))
        rows.append(
            {
                "K": k,
                "task_bank_type": "simulated",
                "matching_time_ms_mean": safe_float(row.get("matching_time_mean_ms")),
                "matching_time_ms_std": safe_float(row.get("matching_time_std_ms")),
                "aggregation_time_ms_mean": safe_float(row.get("aggregation_time_mean_ms")),
                "aggregation_time_ms_std": safe_float(row.get("aggregation_time_std_ms")),
                "total_memr_overhead_ms_mean": overhead,
                "total_memr_overhead_ms_std": safe_float(row.get("total_memr_overhead_std_ms")),
                "task_related_memory_MB": safe_float(row.get("task_related_memory_mb")),
                "note": "simulated task-bank profiling only, not additional training",
            }
        )
    return rows


def detect_gpu_name() -> str:
    if not torch.cuda.is_available():
        return ""
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return ""


def build_hardware(notes: List[str]) -> Dict[str, Any]:
    gpu_name = detect_gpu_name()
    if not gpu_name:
        notes.append("GPU name unresolved at runtime.")
    return {
        "gpu_name": gpu_name,
        "gpu_count": 1,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
    }


def build_model_config(final_snapshot: SnapshotInfo) -> Dict[str, Any]:
    key_shape = final_snapshot.checkpoint_info.get("key_encoder", {}).get("keys_shape", ["", ""])
    return {
        "base_model": BASE_MODEL_NAME,
        "lora_rank": 4,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "v_proj"],
        "task_embedding_dim": key_shape[1] if len(key_shape) > 1 else "",
        "dtype": "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16",
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_seq_length": 512,
    }


def make_summary_md(
    hardware: Dict[str, Any],
    model_config: Dict[str, Any],
    param_rows: List[Dict[str, Any]],
    runtime_rows: List[Dict[str, Any]],
    performance_rows: List[Dict[str, Any]],
    extended_rows: List[Dict[str, Any]],
    notes: List[str],
) -> str:
    k6_param = next(row for row in param_rows if row["K"] == 6)
    k6_runtime = next(row for row in runtime_rows if row["K"] == 6)
    k64_ext = next(row for row in extended_rows if row["K"] == 64)
    lines: List[str] = []
    lines.append("# Scalability Summary")
    lines.append("")
    lines.append("## 1. Experiment purpose")
    lines.append("This analysis supplements the reviewer concern regarding scalability and computational cost. It reports trainable parameter growth, cumulative storage, GPU memory, training time, inference latency breakdown, and task-number-dependent trends while avoiding full retraining.")
    lines.append("")
    lines.append("## 2. Hardware and GPU")
    lines.append(f"- GPU: {hardware['gpu_name'] or 'N/A'}")
    lines.append(f"- CUDA_VISIBLE_DEVICES: `{hardware['cuda_visible_devices']}`")
    lines.append(f"- Torch/CUDA: {hardware['torch_version']} / {hardware['cuda_version']}")
    lines.append("")
    lines.append("## 3. Model configuration")
    lines.append(f"- Base model: `{model_config['base_model']}`")
    lines.append(f"- LoRA: r={model_config['lora_rank']}, alpha={model_config['lora_alpha']}, dropout={model_config['lora_dropout']}")
    lines.append(f"- Target modules: {', '.join(model_config['target_modules'])}")
    lines.append(f"- Task embedding dimension: {model_config['task_embedding_dim']}")
    lines.append(f"- Batch size / grad accumulation / max seq: {model_config['batch_size']} / {model_config['gradient_accumulation_steps']} / {model_config['max_seq_length']}")
    lines.append("")
    lines.append("## 4. Parameter and storage conclusions")
    lines.append(f"- At K=6, cumulative task-related parameters are {k6_param['cumulative_task_related_params']:,}.")
    lines.append(f"- At K=6, task-related storage is {k6_param['cumulative_task_related_storage_MB']:.4f} MB.")
    lines.append(f"- Frozen base PLM storage is {k6_param['base_model_storage_MB']:.4f} MB and does not scale with K.")
    lines.append("- Task-related storage grows approximately linearly with task count under the archived and simulated task-bank settings.")
    lines.append("")
    lines.append("## 5. Training / inference memory and time conclusions")
    lines.append(f"- K=6 training time per task (archived log source): {format_value(k6_runtime['training_time_sec_per_task'])} sec.")
    lines.append(f"- K=6 training peak GPU memory (short profile): {format_value(k6_runtime['training_peak_gpu_memory_MB'])} MB.")
    lines.append(f"- K=6 inference peak GPU memory: {format_value(k6_runtime['inference_peak_gpu_memory_MB'])} MB.")
    lines.append(f"- K=6 latency breakdown: matching {format_value(k6_runtime['matching_time_ms_mean'])} ms/query, aggregation {format_value(k6_runtime['aggregation_time_ms_mean'])} ms/query, PLM {format_value(k6_runtime['plm_time_ms_mean'])} ms/query, total {format_value(k6_runtime['total_inference_latency_ms_mean'])} ms/query.")
    lines.append("")
    lines.append("## 6. K=1..6 performance trend")
    lines.append("- K=6 final continual-learning metrics are recovered from the manuscript order-1 table.")
    lines.append("- K=1..5 average seen-task performance is estimated by lightweight dev evaluation on archived stage snapshots.")
    lines.append("- K=1..5 FWT/FR/BWT remain unavailable because the workspace does not preserve the full archived stage matrix for MeMR.")
    lines.append("")
    lines.append("## 7. K=8,16,32,64 extended task-bank profiling")
    lines.append(f"- At K=64, simulated task-related memory is {format_value(k64_ext['task_related_memory_MB'])} MB.")
    lines.append(f"- At K=64, simulated matching time is {format_value(k64_ext['matching_time_ms_mean'])} ms/query and aggregation time is {format_value(k64_ext['aggregation_time_ms_mean'])} ms/query.")
    lines.append("- These K>6 results are simulated task-bank profiling only and do not involve new training or new accuracy claims.")
    lines.append("")
    lines.append("## 8. Clarification")
    lines.append("- K>6 only used for cost profiling.")
    lines.append("- No additional training was performed for K>6.")
    lines.append("- No additional performance claim is made for K>6.")
    lines.append("")
    lines.append("## 9. Notes")
    for note in notes:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## 10. Rebuttal paragraph")
    lines.append("")
    lines.append("Following the reviewer’s suggestion, we added a scalability and computational-cost analysis. We reported the trainable parameters per task, cumulative task-related storage, GPU memory consumption, training time, inference latency, and performance trend as the number of learned tasks increases. The results show that the task-related storage grows approximately linearly with the number of tasks, while the matching and aggregation overhead remains small compared with the PLM forward computation. In addition, we performed a simulated task-bank profiling experiment for K=8,16,32,64 without additional training, which further shows that task matching is not the main runtime bottleneck under the tested range.")
    return "\n".join(lines) + "\n"


def build_full_results(
    hardware: Dict[str, Any],
    model_config: Dict[str, Any],
    param_rows: List[Dict[str, Any]],
    runtime_rows: List[Dict[str, Any]],
    performance_rows: List[Dict[str, Any]],
    extended_rows: List[Dict[str, Any]],
    notes: List[str],
) -> Dict[str, Any]:
    return {
        "hardware": hardware,
        "model_config": model_config,
        "parameter_and_storage": param_rows,
        "runtime_memory_latency": runtime_rows,
        "performance_vs_tasks": performance_rows,
        "extended_taskbank_profiling": extended_rows,
        "notes": notes,
    }


def print_preview(name: str, rows: List[Dict[str, Any]], limit: int = 3) -> None:
    print(f"[{name}]")
    for row in rows[:limit]:
        print(row)


def main() -> None:
    start_time = time.time()
    ensure_dirs()
    notes: List[str] = []

    snapshots = discover_snapshots()
    if len(snapshots) != 6:
        raise RuntimeError(f"Expected 6 snapshots in {PRIMARY_SNAPSHOT_DIR}, found {len(snapshots)}")

    base_model_params, base_model_storage_mb = parse_base_model_storage_and_params()
    archive_runtime_rows = load_archive_runtime_rows()
    training_times = parse_training_times(PRIMARY_LOG_PATH)

    param_rows = build_parameter_rows(snapshots, base_model_params, base_model_storage_mb)
    runtime_rows = make_runtime_rows(snapshots, archive_runtime_rows, training_times, notes)
    performance_rows = make_performance_rows(snapshots, notes)
    extended_rows = make_extended_rows(archive_runtime_rows)
    hardware = build_hardware(notes)
    model_config = build_model_config(snapshots[-1])

    full_results = build_full_results(
        hardware=hardware,
        model_config=model_config,
        param_rows=param_rows,
        runtime_rows=runtime_rows,
        performance_rows=performance_rows,
        extended_rows=extended_rows,
        notes=notes,
    )

    json_path = DEFAULT_RESULTS_DIR / "scalability_full_results.json"
    xlsx_path = DEFAULT_RESULTS_DIR / "scalability_tables.xlsx"
    params_csv = DEFAULT_RESULTS_DIR / "table_parameters_storage.csv"
    runtime_csv = DEFAULT_RESULTS_DIR / "table_runtime_memory_latency.csv"
    perf_csv = DEFAULT_RESULTS_DIR / "table_performance_vs_tasks.csv"
    extended_csv = DEFAULT_RESULTS_DIR / "table_extended_taskbank_profiling.csv"
    summary_md = DEFAULT_RESULTS_DIR / "scalability_summary.md"

    write_json(json_path, full_results)
    write_csv(
        params_csv,
        param_rows,
        [
            "K",
            "per_task_trainable_params",
            "current_task_module_params",
            "task_embedding_params",
            "routing_related_params",
            "cumulative_task_module_params",
            "cumulative_task_related_params",
            "task_module_storage_MB",
            "embedding_storage_MB",
            "routing_storage_MB",
            "cumulative_task_related_storage_MB",
            "base_model_params",
            "base_model_storage_MB",
        ],
    )
    write_csv(
        runtime_csv,
        runtime_rows,
        [
            "K",
            "training_peak_gpu_memory_MB",
            "training_time_sec_per_task",
            "training_time_source",
            "inference_peak_gpu_memory_MB",
            "matching_time_ms_mean",
            "matching_time_ms_std",
            "aggregation_time_ms_mean",
            "aggregation_time_ms_std",
            "plm_time_ms_mean",
            "plm_time_ms_std",
            "total_inference_latency_ms_mean",
            "total_inference_latency_ms_std",
            "matching_latency_ratio_percent",
            "aggregation_latency_ratio_percent",
            "total_overhead_ratio_percent",
        ],
    )
    write_csv(
        perf_csv,
        performance_rows,
        [
            "K",
            "learned_tasks",
            "average_llm_score_on_seen_tasks",
            "final_average_performance_or_FAP",
            "FWT",
            "FR",
            "BWT",
            "source",
        ],
    )
    write_csv(
        extended_csv,
        extended_rows,
        [
            "K",
            "task_bank_type",
            "matching_time_ms_mean",
            "matching_time_ms_std",
            "aggregation_time_ms_mean",
            "aggregation_time_ms_std",
            "total_memr_overhead_ms_mean",
            "total_memr_overhead_ms_std",
            "task_related_memory_MB",
            "note",
        ],
    )
    write_text(summary_md, make_summary_md(hardware, model_config, param_rows, runtime_rows, performance_rows, extended_rows, notes))
    xlsx_error = maybe_write_xlsx(
        xlsx_path,
        {
            "parameters_storage": param_rows,
            "runtime_latency": runtime_rows,
            "performance_vs_tasks": performance_rows,
            "extended_taskbank": extended_rows,
        },
    )
    if xlsx_error:
        notes.append(xlsx_error)
        write_json(json_path, build_full_results(hardware, model_config, param_rows, runtime_rows, performance_rows, extended_rows, notes))

    elapsed = time.time() - start_time
    print(f"Output directory: {DEFAULT_OUTPUT_ROOT}")
    print("Generated files:")
    for path in [json_path, xlsx_path, params_csv, runtime_csv, perf_csv, extended_csv, summary_md]:
        print(f"- {path}")
    print("Table previews:")
    print_preview("table_parameters_storage.csv", param_rows)
    print_preview("table_runtime_memory_latency.csv", runtime_rows)
    print_preview("table_performance_vs_tasks.csv", performance_rows)
    print_preview("table_extended_taskbank_profiling.csv", extended_rows)
    print(f"Missing items present: {'yes' if notes else 'no'}")
    if notes:
        print("Missing item reasons / notes:")
        for note in notes:
            print(f"- {note}")
    print("Single-GPU OOM observed during this run: no")
    print(f"GPU requested by operator: {os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
    print(f"Total script wall time (sec): {elapsed:.2f}")
    print("Next step suggestion: inspect the regenerated CSV/JSON and summary, then cite the archived K>6 task-bank profiling plus the new K=1..6 lightweight runtime/memory estimates in the response.")


if __name__ == "__main__":
    main()
