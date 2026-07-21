#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."

CUDA_VISIBLE_DEVICES=1 python -m experiments.prompt_r3_c2.gate_probe \
  --mode metadata_only \
  --checkpoint-dir checkpoints_continual_keshi_llama/order1_compose_peft/snapshots/task_5_zhongliuke_train_end_20260626_123739 \
  --meta-embeddings-path metadata_embeddings/keshi_meta_embeddings.pt \
  --output-dir experiments/prompt_r3_c2/outputs/metadata_only \
  --cache-dir experiments/prompt_r3_c2/cache/metadata_only \
  --eval-data experiments/reviewer_0629/data/routing_reference_eval.jsonl,experiments/reviewer_0629/data/holdout_zhongliuke_reference_eval.jsonl

