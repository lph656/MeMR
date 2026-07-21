#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."

CUDA_VISIBLE_DEVICES=3 python -m experiments.reviewer_0629.run_cold_start_eval \
  --checkpoint-dir checkpoints_continual_keshi_llama/order1_compose_peft/snapshots/task_4_nanke_train_end_20260626_101856 \
  --full-meta-embeddings-path metadata_embeddings/keshi_meta_embeddings.pt \
  --seen-task-list neike,waike,erke,fuchanke,nanke \
  --heldout-task zhongliuke \
  --eval-data experiments/reviewer_0629/data/holdout_zhongliuke_reference_eval.jsonl \
  --output-dir experiments/reviewer_0629/outputs/cold_start_zhongliuke \
  --max-samples 120

CUDA_VISIBLE_DEVICES=3 python -m experiments.reviewer_0629.run_unseen_department_eval \
  --checkpoint-dir checkpoints_continual_keshi_llama/order1_compose_peft/snapshots/task_4_nanke_train_end_20260626_101856 \
  --full-meta-embeddings-path metadata_embeddings/keshi_meta_embeddings.pt \
  --seen-task-list neike,waike,erke,fuchanke,nanke \
  --heldout-task zhongliuke \
  --eval-data experiments/reviewer_0629/data/holdout_zhongliuke_reference_eval.jsonl \
  --output-dir experiments/reviewer_0629/outputs/unseen_zhongliuke \
  --route-modes full,top1,top2,top3 \
  --max-samples 120 \
  --max-new-tokens 96

