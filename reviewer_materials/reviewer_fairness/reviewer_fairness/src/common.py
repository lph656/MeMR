"""
Utilities for reviewer fairness experiments.

These helpers are intentionally isolated under reviewer_fairness/ so the
original MeMR project workflow and training entrypoints remain unchanged.
"""

from __future__ import annotations

import csv
import json
import os
import random
import traceback
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVIEWER_ROOT = PROJECT_ROOT / "reviewer_fairness"
RESULTS_ROOT = REVIEWER_ROOT / "results"
LOGS_ROOT = REVIEWER_ROOT / "logs"

TASK_CODE_TO_NAME = {
    "IM": "neike",
    "S": "waike",
    "P": "erke",
    "GO": "fuchanke",
    "A": "nanke",
    "O": "zhongliuke",
}
TASK_NAME_TO_CODE = {value: key for key, value in TASK_CODE_TO_NAME.items()}
TASK_CODE_TO_LABEL = {
    "IM": "Internal Medicine",
    "S": "Surgery",
    "P": "Pediatrics",
    "GO": "Gynecology and Obstetrics",
    "A": "Andrology",
    "O": "Oncology",
}
DEFAULT_TASK_ORDER = ["IM", "S", "P", "GO", "A", "O"]
TASK_SPLIT_OFFSETS = {code: idx for idx, code in enumerate(DEFAULT_TASK_ORDER)}

METRIC_FIELDNAMES = [
    "Method",
    "Order",
    "IM",
    "S",
    "P",
    "GO",
    "A",
    "O",
    "Average",
    "FWT",
    "FR",
    "BWT",
    "Trainable Params",
    "Total Params",
    "Notes",
]


def require_yaml():
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - import error path is intentional
        raise RuntimeError(
            "PyYAML is required for reviewer_fairness scripts. "
            "Install it in the experiment environment before running."
        ) from exc
    return yaml


def load_yaml(path: Path | str) -> Dict[str, Any]:
    yaml = require_yaml()
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(data: Dict[str, Any], path: Path | str) -> None:
    yaml = require_yaml()
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def ensure_parent(path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_json(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: Path | str) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_text(path: Path | str, text: str) -> None:
    ensure_parent(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_train_records(dataset_root: Path | str, task_name: str) -> List[Dict[str, Any]]:
    dataset_root = Path(dataset_root)
    return read_json(dataset_root / task_name / "train.json")


def load_test_records(dataset_root: Path | str, task_name: str) -> List[Dict[str, Any]]:
    dataset_root = Path(dataset_root)
    payload = read_json(dataset_root / task_name / "test.json")
    return payload["questions"]


def config_to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return {k: config_to_plain(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): config_to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [config_to_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def make_metrics_template(method: str, order: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "method": method,
        "order": order,
        "department_scores": {code: None for code in DEFAULT_TASK_ORDER},
        "average": None,
        "FWT": None,
        "FR": None,
        "BWT": None,
        "config": config or {},
        "trainable_params": None,
        "total_params": None,
        "notes": "",
    }


def write_metrics_csv(metrics: Dict[str, Any], path: Path | str) -> None:
    row = {
        "Method": metrics.get("method"),
        "Order": metrics.get("order"),
        "IM": metrics.get("department_scores", {}).get("IM"),
        "S": metrics.get("department_scores", {}).get("S"),
        "P": metrics.get("department_scores", {}).get("P"),
        "GO": metrics.get("department_scores", {}).get("GO"),
        "A": metrics.get("department_scores", {}).get("A"),
        "O": metrics.get("department_scores", {}).get("O"),
        "Average": metrics.get("average"),
        "FWT": metrics.get("FWT"),
        "FR": metrics.get("FR"),
        "BWT": metrics.get("BWT"),
        "Trainable Params": metrics.get("trainable_params"),
        "Total Params": metrics.get("total_params"),
        "Notes": metrics.get("notes"),
    }
    ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDNAMES)
        writer.writeheader()
        writer.writerow(row)


def write_table_csv(rows: Iterable[Dict[str, Any]], path: Path | str, fieldnames: List[str]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown_table(rows: List[Dict[str, Any]], path: Path | str, fieldnames: List[str], title: Optional[str] = None) -> None:
    ensure_parent(path)
    lines: List[str] = []
    if title:
        lines.append(f"# {title}")
        lines.append("")
    lines.append("| " + " | ".join(fieldnames) + " |")
    lines.append("| " + " | ".join(["---"] * len(fieldnames)) + " |")
    for row in rows:
        values = [str(row.get(field, "")) for field in fieldnames]
        lines.append("| " + " | ".join(values) + " |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def normalize_score(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 4)


def mean_or_none(values: List[Optional[float]]) -> Optional[float]:
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return None
    return round(sum(filtered) / len(filtered), 4)


def stable_task_split_seed(base_seed: int, task_code_or_name: str) -> int:
    task_code = TASK_NAME_TO_CODE.get(task_code_or_name, task_code_or_name)
    if task_code not in TASK_SPLIT_OFFSETS:
        raise KeyError(f"Unknown task code or name for split seed: {task_code_or_name}")
    return int(base_seed) + TASK_SPLIT_OFFSETS[task_code]


def split_train_dev_records(records: List[Dict[str, Any]], dev_ratio: float, seed: int) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    dev_size = max(1, int(round(len(records) * dev_ratio)))
    dev_indices = set(indices[:dev_size])
    train_records: List[Dict[str, Any]] = []
    dev_records: List[Dict[str, Any]] = []
    for idx, item in enumerate(records):
        if idx in dev_indices:
            dev_records.append(item)
        else:
            train_records.append(item)
    return train_records, dev_records


def has_real_metrics(payload: Dict[str, Any]) -> bool:
    if not isinstance(payload, dict) or not payload.get("method"):
        return False
    average = payload.get("average")
    if isinstance(average, (int, float)):
        return True
    department_scores = payload.get("department_scores", {})
    return any(isinstance(score, (int, float)) for score in department_scores.values())


def stage_matrix_to_cl_metrics(stage_matrix: List[Dict[str, Optional[float]]], task_order_codes: List[str]) -> Dict[str, Optional[float]]:
    """
    Compute simple continual learning metrics from a stage-score matrix.

    stage_matrix[i][task_code] should represent the score after finishing task i.
    Missing entries are ignored and will produce None outputs if insufficient data.
    """
    num_tasks = len(task_order_codes)
    if not stage_matrix or num_tasks < 2:
        return {"FWT": None, "FR": None, "BWT": None}

    diagonal = []
    final_scores = []
    for i, task_code in enumerate(task_order_codes):
        if i >= len(stage_matrix):
            break
        diagonal.append(stage_matrix[i].get(task_code))
        final_scores.append(stage_matrix[-1].get(task_code))

    # We do not have a pre-task zero-shot evaluation matrix in the current project,
    # so FWT is only meaningful when a caller pre-populates `stage_matrix[0]` with it.
    fwt = None
    if len(stage_matrix) > num_tasks:
        zero_shot = stage_matrix[0]
        fwt_terms = []
        for idx, task_code in enumerate(task_order_codes[1:], start=1):
            if diagonal[idx] is not None and zero_shot.get(task_code) is not None:
                fwt_terms.append(diagonal[idx] - float(zero_shot[task_code]))
        if fwt_terms:
            fwt = round(sum(fwt_terms) / len(fwt_terms), 4)

    fr_terms = []
    bwt_terms = []
    for idx, task_code in enumerate(task_order_codes[:-1]):
        historical = [stage.get(task_code) for stage in stage_matrix[idx:]]
        historical = [float(v) for v in historical if v is not None]
        if not historical or diagonal[idx] is None or final_scores[idx] is None:
            continue
        fr_terms.append(max(historical) - float(final_scores[idx]))
        bwt_terms.append(float(final_scores[idx]) - float(diagonal[idx]))

    return {
        "FWT": normalize_score(fwt),
        "FR": normalize_score(sum(fr_terms) / len(fr_terms)) if fr_terms else None,
        "BWT": normalize_score(sum(bwt_terms) / len(bwt_terms)) if bwt_terms else None,
    }


def resolve_order_codes(task_orders: Dict[str, List[str]], order_name: str) -> List[str]:
    if order_name == "all_tasks":
        return DEFAULT_TASK_ORDER
    if order_name not in task_orders:
        raise ValueError(f"Unknown order '{order_name}'. Available: {sorted(task_orders.keys()) + ['all_tasks']}")
    return list(task_orders[order_name])


def order_codes_to_task_names(order_codes: List[str]) -> List[str]:
    return [TASK_CODE_TO_NAME[code] for code in order_codes]


def task_name_to_code(task_name: str) -> str:
    if task_name not in TASK_NAME_TO_CODE:
        raise KeyError(f"Unknown task name: {task_name}")
    return TASK_NAME_TO_CODE[task_name]


def find_existing_memr_snapshot() -> Optional[Path]:
    root = PROJECT_ROOT / "checkpoints_continual_keshi_llama" / "order1_compose_peft" / "snapshots"
    if not root.exists():
        return None
    candidates = sorted(root.glob("task_5_*_train_end_*"))
    return candidates[-1] if candidates else None


def detect_todo_values(config: Dict[str, Any], prefix: str = "") -> List[str]:
    pending: List[str] = []
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            pending.extend(detect_todo_values(value, full_key))
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                if item == "TODO":
                    pending.append(f"{full_key}[{idx}]")
        elif value == "TODO":
            pending.append(full_key)
    return pending


def build_failure_payload(method: str, order: str, message: str, config: Dict[str, Any]) -> Dict[str, Any]:
    metrics = make_metrics_template(method, order, config)
    metrics["notes"] = message
    return metrics


def write_failure_artifacts(output_dir: Path, method: str, order: str, config: Dict[str, Any], exc: BaseException) -> None:
    error_message = f"{type(exc).__name__}: {exc}"
    metrics = build_failure_payload(method, order, error_message, config)
    write_json(metrics, output_dir / "metrics.json")
    write_metrics_csv(metrics, output_dir / "metrics.csv")
    write_json(
        {
            "status": "error",
            "method": method,
            "order": order,
            "error": error_message,
            "traceback": traceback.format_exc(),
            "timestamp": timestamp(),
        },
        output_dir / "fairness_notes.json",
    )
    append_text(output_dir / "error.log", traceback.format_exc())


def format_command(parts: List[str]) -> str:
    return " ".join(parts)
