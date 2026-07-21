from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from .common import (
    DEFAULT_TOPK_VALUES,
    append_jsonl,
    compute_generation_metrics,
    compute_topk_hits,
    ensure_dir,
    generate_answer,
    load_eval_jsonl,
    load_memr_model,
    perturb_question,
    route_query,
    summarise_routing_records,
    write_csv,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(description="Run routing/top-k/noise diagnostics for MeMR.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--meta-embeddings-path", required=True)
    parser.add_argument("--eval-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model-path", default="chinese-alpaca-plus-7b-hf")
    parser.add_argument("--input-noise-modes", default="none,char_delete_10,char_swap_10,punctuation_10,filler_10")
    parser.add_argument("--route-modes", default="full,top1,top2,top3")
    parser.add_argument("--topk-values", default="1,2,3")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--compute-dtype", default="float16")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    samples = load_eval_jsonl(args.eval_data, max_samples=args.max_samples)
    route_modes = [item.strip() for item in args.route_modes.split(",") if item.strip()]
    topk_values = tuple(int(item) for item in args.topk_values.split(",") if item.strip()) or DEFAULT_TOPK_VALUES
    noise_modes = [item.strip() for item in args.input_noise_modes.split(",") if item.strip()]

    bundle = load_memr_model(
        base_model_path=args.base_model_path,
        checkpoint_dir=args.checkpoint_dir,
        meta_embeddings_path=args.meta_embeddings_path,
        inference_log_dir=str(output_dir / "tensorboard"),
        compute_dtype_name=args.compute_dtype,
    )

    overall_summary = {
        "checkpoint_dir": args.checkpoint_dir,
        "meta_embeddings_path": args.meta_embeddings_path,
        "eval_data": args.eval_data,
        "num_samples": len(samples),
        "noise_modes": noise_modes,
        "route_modes": route_modes,
    }

    for noise_mode in noise_modes:
        per_sample_path = output_dir / f"per_sample_{noise_mode}.jsonl"
        if per_sample_path.exists():
            per_sample_path.unlink()
        routing_records = []
        predictions_by_mode = {route_mode: [] for route_mode in route_modes}
        references = []
        for sample_idx, sample in enumerate(tqdm(samples, desc=f"routing diagnostics {noise_mode}")):
            question = perturb_question(sample["question"], noise_mode, seed=20260629 + sample_idx)
            weights, task_list = route_query(bundle, question)
            predicted_idx = max(range(len(weights)), key=lambda idx: weights[idx])
            record = {
                "sample_id": sample["sample_id"],
                "task_name": sample["task_name"],
                "task_id": sample["task_id"],
                "question": question,
                "weights": weights,
                "predicted_task_id": predicted_idx,
                "predicted_task_name": task_list[predicted_idx],
                "top1_confidence": weights[predicted_idx],
                "top1_correct": int(predicted_idx == sample["task_id"]),
                "correct_task_weight": weights[sample["task_id"]],
                "entropy": -sum(weight * __import__("math").log(weight + 1e-12) for weight in weights),
                "reference_answer": sample.get("reference_answer"),
            }
            record.update(compute_topk_hits(weights, sample["task_id"], topk_values))
            routing_records.append(record)

            if sample.get("reference_answer"):
                references.append(sample["reference_answer"])
                for route_mode in route_modes:
                    predictions_by_mode[route_mode].append(
                        generate_answer(
                            bundle,
                            question=question,
                            route_mode=route_mode,
                            max_new_tokens=args.max_new_tokens,
                        )
                    )
            append_jsonl(per_sample_path, record)

        summary = summarise_routing_records(routing_records, topk_values)
        summary["noise_mode"] = noise_mode
        if references:
            summary["generation_metrics"] = compute_generation_metrics(predictions_by_mode, references)
        write_json(output_dir / f"summary_{noise_mode}.json", summary)

        confusion_rows = []
        counter = defaultdict(int)
        for row in routing_records:
            counter[(row["task_name"], row["predicted_task_name"])] += 1
        for (gold_task, predicted_task), count in sorted(counter.items()):
            confusion_rows.append(
                {
                    "gold_task": gold_task,
                    "predicted_task": predicted_task,
                    "count": count,
                }
            )
        write_csv(
            output_dir / f"confusion_{noise_mode}.csv",
            confusion_rows,
            ["gold_task", "predicted_task", "count"],
        )
        overall_summary[noise_mode] = summary

    write_json(output_dir / "summary.json", overall_summary)
    print(json.dumps(overall_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

