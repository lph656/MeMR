from __future__ import annotations

import csv
import json
import logging
import os
import platform
import sys
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_high_risk_safety")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def setup_logger(name: str, output_dir: str) -> logging.Logger:
    ensure_dir(output_dir)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(os.path.join(output_dir, "run.log"), encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def write_json(path: str, payload: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    if not rows and fieldnames is None:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def timestamp_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_environment_snapshot(output_dir: str) -> None:
    ensure_dir(output_dir)
    lines = [
        "This high-risk safety analysis is a small-scale synthetic stress test for deployment-oriented safety review. It is not clinical validation.",
        f"timestamp={timestamp_now()}",
        f"python={sys.version.replace(os.linesep, ' ')}",
        f"platform={platform.platform()}",
        f"executable={sys.executable}",
        f"cwd={os.getcwd()}",
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
    ]
    with open(os.path.join(output_dir, "environment.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def normalize_text(text: str) -> str:
    return (text or "").strip().lower().replace(" ", "")


def format_percent(value: float) -> str:
    return f"{value * 100.0:.2f}"


def bool_to_int(value: bool) -> int:
    return 1 if value else 0
