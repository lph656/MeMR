"""
Build cached single-task datasets for reviewer fairness experiments.

The cache mirrors the task-specific layout under reviewer_fairness/results/cache
so the original datasets remain untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reviewer_fairness.src.common import (
    RESULTS_ROOT,
    TASK_CODE_TO_NAME,
    load_test_records,
    load_train_records,
    write_json,
)


def build_single_task_cache(dataset_root: str | Path, order_codes: List[str]) -> Dict[str, Any]:
    cache_dir = RESULTS_ROOT / "cache" / "single_task_oracle"
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {"tasks": {}}
    for code in order_codes:
        task_name = TASK_CODE_TO_NAME[code]
        task_dir = cache_dir / code
        task_dir.mkdir(parents=True, exist_ok=True)
        train_records = load_train_records(dataset_root, task_name)
        test_records = load_test_records(dataset_root, task_name)
        write_json(train_records, task_dir / "train.json")
        write_json({"questions": test_records}, task_dir / "test.json")
        manifest["tasks"][code] = {
            "task_name": task_name,
            "train_path": str(task_dir / "train.json"),
            "test_path": str(task_dir / "test.json"),
            "train_count": len(train_records),
            "test_count": len(test_records),
        }

    write_json(manifest, cache_dir / "manifest.json")
    return manifest


if __name__ == "__main__":
    manifest = build_single_task_cache("datasets/medical_consult", ["IM", "S", "P", "GO", "A", "O"])
    print(len(manifest["tasks"]))
