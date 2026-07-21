"""
Build cached joint-training datasets for reviewer fairness experiments.

This file does not modify the original data directory. It materializes merged
artifacts under reviewer_fairness/results/cache/joint_training/.
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


def build_joint_dataset(dataset_root: str | Path, order_codes: List[str]) -> Dict[str, Any]:
    cache_dir = RESULTS_ROOT / "cache" / "joint_training"
    cache_dir.mkdir(parents=True, exist_ok=True)

    joint_records = []
    manifest = {"task_order": order_codes, "tasks": {}, "joint_train_path": str(cache_dir / "train.json")}
    for code in order_codes:
        task_name = TASK_CODE_TO_NAME[code]
        train_records = load_train_records(dataset_root, task_name)
        test_records = load_test_records(dataset_root, task_name)
        joint_records.extend(train_records)
        manifest["tasks"][code] = {
            "task_name": task_name,
            "train_count": len(train_records),
            "test_count": len(test_records),
        }

    write_json(joint_records, cache_dir / "train.json")
    write_json(manifest, cache_dir / "manifest.json")
    return manifest


if __name__ == "__main__":
    manifest = build_joint_dataset("datasets/medical_consult", ["IM", "S", "P", "GO", "A", "O"])
    print(manifest["joint_train_path"])
