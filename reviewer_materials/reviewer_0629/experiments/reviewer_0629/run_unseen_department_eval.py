from __future__ import annotations

import argparse
import json
from collections import defaultdict

from tqdm import tqdm

from .common import (
    append_jsonl,
    build_metadata_subset,
    compute_generation_metrics,
    ensure_dir,
    generate_answer,
    load_eval_jsonl,
    load_memr_model,
    route_query,
    write_csv,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate unseen-department transfer using a checkpoint that has not seen that department.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--full-meta-embeddings-path", default="metadata_embeddings/keshi_meta_embeddings.pt")
    parser.add_argument("--seen-task-list", default="neike,waike,erke,fuchanke,nanke")
    parser.add_argument("--heldout-task", default="zhongliuke")
    parser.add_argument("--eval-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model-path", default="chinese-alpaca-plus-7b-hf")
    parser.add_argument("--route-modes", default="full,top1,top2,top3")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--compute-dtype", default="float16")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    seen_task_list = [item.strip() for item in args.seen_task_list.split(",") if item.strip()]
    route_modes = [item.strip() for item in args.route_modes.split(",") if item.strip()]
    subset_meta_path = output_dir / "cache" / "seen_task_meta_embeddings.pt"
    build_metadata_subset(args.full_meta_embeddings_path, seen_task_list, subset_meta_path)
    bundle = load_memr_model(
        base_model_path=args.base_model_path,
        checkpoint_dir=args.checkpoint_dir,
        meta_embeddings_path=str(subset_meta_path),
        task_list=seen_task_list,
        inference_log_dir=str(output_dir / "tensorboard"),
        compute_dtype_name=args.compute_dtype,
    )
    active_adapter = seen_task_list[-1]
    samples = load_eval_jsonl(args.eval_data, max_samples=args.max_samples)
    predictions_by_mode = {route_mode: [] for route_mode in route_modes}
    references = []
    routing_rows = []
    avg_weight = defaultdict(float)

    result_path = output_dir / "per_sample.jsonl"
    if result_path.exists():
        result_path.unlink()

    for sample in tqdm(samples, desc=f"unseen-department {args.heldout_task}"):
        weights, task_list = route_query(bundle, sample["question"], active_adapter=active_adapter)
        visible_tasks = task_list[: len(weights)]
        references.append(sample["reference_answer"])
        for route_mode in route_modes:
            predictions_by_mode[route_mode].append(
                generate_answer(
                    bundle,
                    question=sample["question"],
                    route_mode=route_mode,
                    active_adapter=active_adapter,
                    max_new_tokens=args.max_new_tokens,
                )
            )
        for task_name, weight in zip(visible_tasks, weights):
            avg_weight[task_name] += weight
        row = {
            "sample_id": sample["sample_id"],
            "question": sample["question"],
            "reference_answer": sample["reference_answer"],
            "weights": weights,
            "visible_tasks": visible_tasks,
            "top1_seen_task": visible_tasks[max(range(len(weights)), key=lambda idx: weights[idx])],
        }
        routing_rows.append(row)
        append_jsonl(result_path, row)

    metrics = compute_generation_metrics(predictions_by_mode, references)
    sample_count = max(len(samples), 1)
    mean_weights = {task_name: avg_weight[task_name] / sample_count for task_name in visible_tasks}
    write_csv(
        output_dir / "mean_seen_task_weights.csv",
        [{"task_name": task_name, "mean_weight": mean_weights[task_name]} for task_name in visible_tasks],
        ["task_name", "mean_weight"],
    )
    summary = {
        "checkpoint_dir": args.checkpoint_dir,
        "heldout_task": args.heldout_task,
        "seen_task_list": seen_task_list,
        "eval_data": args.eval_data,
        "num_samples": len(samples),
        "mean_seen_task_weights": mean_weights,
        "generation_metrics": metrics,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
