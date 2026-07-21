from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import (
    TASK_NAMES,
    build_record,
    build_validation_split,
    deterministic_sample,
    ensure_dir,
    load_test_questions,
)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Build fixed evaluation sets for reviewer experiments.")
    parser.add_argument("--output-dir", default="experiments/reviewer_0629/data")
    parser.add_argument("--reference-per-task", type=int, default=60)
    parser.add_argument("--test-per-task", type=int, default=40)
    parser.add_argument("--holdout-task", default="zhongliuke")
    parser.add_argument("--holdout-reference-count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260629)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    reference_rows = []
    test_rows = []

    for task_name in TASK_NAMES:
        _train_records, val_records = build_validation_split(task_name)
        for original_index, source_record in deterministic_sample(
            val_records,
            sample_size=args.reference_per_task,
            seed=args.seed + 101 * (1 + TASK_NAMES.index(task_name)),
        ):
            reference_rows.append(
                build_record(
                    task_name=task_name,
                    source_record=source_record,
                    sample_id=f"{task_name}_val_{original_index}",
                    split="validation_reference",
                    original_index=original_index,
                )
            )

        test_records = load_test_questions(task_name)
        for original_index, source_record in deterministic_sample(
            test_records,
            sample_size=args.test_per_task,
            seed=args.seed + 313 * (1 + TASK_NAMES.index(task_name)),
        ):
            row = build_record(
                task_name=task_name,
                source_record={"question": source_record["question"], "output": None, "id": source_record.get("id")},
                sample_id=f"{task_name}_test_{original_index}",
                split="test_question_only",
                original_index=original_index,
            )
            test_rows.append(row)

    holdout_rows = []
    _holdout_train_records, holdout_val_records = build_validation_split(args.holdout_task)
    for original_index, source_record in deterministic_sample(
        holdout_val_records,
        sample_size=args.holdout_reference_count,
        seed=args.seed + 7001,
    ):
        holdout_rows.append(
            build_record(
                task_name=args.holdout_task,
                source_record=source_record,
                sample_id=f"{args.holdout_task}_holdout_{original_index}",
                split="holdout_reference",
                original_index=original_index,
            )
        )

    write_jsonl(output_dir / "routing_reference_eval.jsonl", reference_rows)
    write_jsonl(output_dir / "routing_test_eval.jsonl", test_rows)
    write_jsonl(output_dir / f"holdout_{args.holdout_task}_reference_eval.jsonl", holdout_rows)
    manifest = {
        "reference_path": str(output_dir / "routing_reference_eval.jsonl"),
        "test_path": str(output_dir / "routing_test_eval.jsonl"),
        "holdout_path": str(output_dir / f"holdout_{args.holdout_task}_reference_eval.jsonl"),
        "reference_samples": len(reference_rows),
        "test_samples": len(test_rows),
        "holdout_samples": len(holdout_rows),
        "reference_per_task": args.reference_per_task,
        "test_per_task": args.test_per_task,
        "holdout_reference_count": args.holdout_reference_count,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

