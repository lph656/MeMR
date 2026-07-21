from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .common import TASK_NAMES, ensure_dir, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description="Create a task-label-noise dataset without modifying the original data.")
    parser.add_argument("--source-root", default="datasets/medical_consult")
    parser.add_argument(
        "--output-root",
        default="experiments/reviewer_0629/generated_datasets/noisy_labels_20/medical_consult",
    )
    parser.add_argument("--noise-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260629)
    args = parser.parse_args()

    del args.seed  # deterministic cyclic reassignment is used; keep arg for manifest compatibility
    output_root = ensure_dir(args.output_root)
    original_train = {}
    rotated_payload = {task_name: [] for task_name in TASK_NAMES}
    manifest = {
        "noise_ratio": args.noise_ratio,
        "mapping": {},
        "counts": {},
    }

    for task_name in TASK_NAMES:
        original_train[task_name] = read_json(Path(args.source_root) / task_name / "train.json")

    carry_out = {}
    remain = {}
    for idx, task_name in enumerate(TASK_NAMES):
        next_task = TASK_NAMES[(idx + 1) % len(TASK_NAMES)]
        source_records = original_train[task_name]
        cut = int(round(len(source_records) * args.noise_ratio))
        carry_out[task_name] = source_records[:cut]
        remain[task_name] = source_records[cut:]
        manifest["mapping"][task_name] = next_task

    for idx, task_name in enumerate(TASK_NAMES):
        prev_task = TASK_NAMES[(idx - 1) % len(TASK_NAMES)]
        rotated_payload[task_name] = remain[task_name] + carry_out[prev_task]
        manifest["counts"][task_name] = {
            "original": len(original_train[task_name]),
            "kept": len(remain[task_name]),
            "received_from": prev_task,
            "received_count": len(carry_out[prev_task]),
            "final": len(rotated_payload[task_name]),
        }

    for task_name in TASK_NAMES:
        target_dir = ensure_dir(Path(output_root) / task_name)
        write_json(target_dir / "train.json", rotated_payload[task_name])
        shutil.copy2(Path(args.source_root) / task_name / "test.json", target_dir / "test.json")

    write_json(Path(output_root).parent / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

