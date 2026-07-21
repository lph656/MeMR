#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."

CUDA_VISIBLE_DEVICES=1 python -m experiments.reviewer_0629.run_routing_diagnostics \
  --checkpoint-dir checkpoints_continual_keshi_llama/order1_compose_peft/snapshots/task_5_zhongliuke_train_end_20260626_123739 \
  --meta-embeddings-path metadata_embeddings/keshi_meta_embeddings.pt \
  --eval-data experiments/reviewer_0629/data/routing_reference_eval.jsonl \
  --output-dir experiments/reviewer_0629/outputs/routing_baseline \
  --input-noise-modes none,char_delete_10,char_swap_10,punctuation_10,filler_10 \
  --route-modes full,top1,top2,top3 \
  --topk-values 1,2,3 \
  --max-samples 180 \
  --max-new-tokens 96

