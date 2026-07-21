#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."

for condition in missing_50 noisy_30 stale_coarse institution_mix_40; do
  case "$condition" in
    missing_50)
      checkpoint_dir="metadata_robustness_experiments/outputs/metadata_robustness/missing_50/checkpoints/task_5_zhongliuke_train_end_20260628_000319"
      ;;
    noisy_30)
      checkpoint_dir="metadata_robustness_experiments/outputs/metadata_robustness/noisy_30/checkpoints/task_5_zhongliuke_train_end_20260628_001843"
      ;;
    stale_coarse)
      checkpoint_dir="metadata_robustness_experiments/outputs/metadata_robustness/stale_coarse/checkpoints/task_5_zhongliuke_train_end_20260628_002403"
      ;;
    institution_mix_40)
      checkpoint_dir="metadata_robustness_experiments/outputs/metadata_robustness/institution_mix_40/checkpoints/task_5_zhongliuke_train_end_20260628_001540"
      ;;
    *)
      echo "Unknown condition: $condition" >&2
      exit 1
      ;;
  esac

  CUDA_VISIBLE_DEVICES=2 python -m experiments.reviewer_0629.run_routing_diagnostics \
    --checkpoint-dir "$checkpoint_dir" \
    --meta-embeddings-path "metadata_robustness_experiments/generated_metadata/${condition}/perturbed_meta_embeddings.pt" \
    --eval-data experiments/reviewer_0629/data/routing_reference_eval.jsonl \
    --output-dir "experiments/reviewer_0629/outputs/routing_metadata/${condition}" \
    --input-noise-modes none,char_delete_10,char_swap_10 \
    --route-modes full,top1,top2,top3 \
    --topk-values 1,2,3 \
    --max-samples 180 \
    --max-new-tokens 96
done
