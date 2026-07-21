from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .common import (
    append_jsonl,
    build_metadata_subset,
    ensure_dir,
    find_key_encoder,
    load_eval_jsonl,
    load_memr_model,
    build_query_embedding,
    write_json,
)


def compute_extended_weights(query_embed, key_encoder, heldout_meta, candidate_new_key, seen_count):
    target_device = next(key_encoder.parameters()).device
    query_embed = query_embed.to(target_device)
    heldout_meta = heldout_meta.to(target_device)
    candidate_new_key = candidate_new_key.to(target_device)

    seen_meta = key_encoder.all_meta_keys[:seen_count].to(device=target_device, dtype=query_embed.dtype)
    seen_keys = key_encoder.keys[:seen_count].to(device=target_device, dtype=query_embed.dtype)
    ext_meta = torch.cat([seen_meta, heldout_meta.unsqueeze(0)], dim=0)
    ext_keys = torch.cat([seen_keys, candidate_new_key.unsqueeze(0)], dim=0)
    ext_dynamic = key_encoder.dynamic_attn_layer(query_embed, ext_meta, ext_keys)
    n_q = F.normalize(query_embed.to(ext_dynamic.dtype), dim=-1).detach().unsqueeze(1)
    n_dynamic = F.normalize(ext_dynamic, dim=-1)
    cos_sim = torch.bmm(n_q, n_dynamic.transpose(1, 2)).squeeze(1)
    weights = F.softmax(cos_sim * key_encoder.config.softmax_match_scale, dim=-1)
    return weights[0].detach().cpu().tolist()


def main():
    parser = argparse.ArgumentParser(description="Cold-start task-matching evaluation for a newly introduced department.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--full-meta-embeddings-path", default="metadata_embeddings/keshi_meta_embeddings.pt")
    parser.add_argument("--seen-task-list", default="neike,waike,erke,fuchanke,nanke")
    parser.add_argument("--heldout-task", default="zhongliuke")
    parser.add_argument("--eval-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model-path", default="chinese-alpaca-plus-7b-hf")
    parser.add_argument("--compute-dtype", default="float16")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    seen_task_list = [item.strip() for item in args.seen_task_list.split(",") if item.strip()]
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
    key_encoder = find_key_encoder(bundle.model)
    if key_encoder is None:
        raise RuntimeError("TaskKeyEncoder not found")

    full_meta = torch.load(args.full_meta_embeddings_path, map_location="cpu")
    task_order = ["neike", "waike", "erke", "fuchanke", "nanke", "zhongliuke"]
    heldout_meta = full_meta[task_order.index(args.heldout_task)]
    seen_count = len(seen_task_list)
    seen_norm = key_encoder.keys.detach().cpu()[:seen_count].norm(dim=1).mean().item()
    random_gen = torch.Generator(device="cpu")
    random_gen.manual_seed(20260629)
    random_key = torch.randn(heldout_meta.shape, generator=random_gen)
    random_key = random_key / random_key.norm().clamp_min(1e-12) * seen_norm
    mean_key = key_encoder.keys.detach().cpu()[:seen_count].mean(dim=0)

    samples = load_eval_jsonl(args.eval_data, max_samples=args.max_samples)
    settings = {
        "metadata_init": heldout_meta.clone(),
        "random_init": random_key.clone(),
        "mean_seen_init": mean_key.clone(),
    }

    summary = {}
    for setting_name, candidate_new_key in settings.items():
        result_path = output_dir / f"per_sample_{setting_name}.jsonl"
        if result_path.exists():
            result_path.unlink()
        top1 = 0
        top3 = 0
        mean_last_weight = 0.0
        for sample in tqdm(samples, desc=f"cold-start {setting_name}"):
            query_embed = build_query_embedding(bundle.tokenizer, bundle.query_encoder, sample["question"])
            weights = compute_extended_weights(query_embed, key_encoder, heldout_meta, candidate_new_key, seen_count)
            ranked = sorted(range(len(weights)), key=lambda idx: weights[idx], reverse=True)
            top1 += int(ranked[0] == len(weights) - 1)
            top3 += int((len(weights) - 1) in ranked[:3])
            mean_last_weight += weights[-1]
            append_jsonl(
                result_path,
                {
                    "sample_id": sample["sample_id"],
                    "question": sample["question"],
                    "weights": weights,
                    "predicted_index": ranked[0],
                    "predicted_is_new_task": int(ranked[0] == len(weights) - 1),
                    "new_task_weight": weights[-1],
                },
            )
        sample_count = max(len(samples), 1)
        summary[setting_name] = {
            "num_samples": len(samples),
            "top1_new_task_accuracy": top1 / sample_count,
            "top3_new_task_accuracy": top3 / sample_count,
            "mean_new_task_weight": mean_last_weight / sample_count,
        }

    write_json(
        output_dir / "summary.json",
        {
            "checkpoint_dir": args.checkpoint_dir,
            "seen_task_list": seen_task_list,
            "heldout_task": args.heldout_task,
            "eval_data": args.eval_data,
            "settings": summary,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
