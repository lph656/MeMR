#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."

CHECKPOINT_DIR="experiments/reviewer_0629/outputs/noisy_labels_training/snapshots/task_5_zhongliuke_train_end_20260629_000000"
SNAPSHOT_ROOT="experiments/reviewer_0629/outputs/noisy_labels_training/snapshots"

while [ ! -f "$CHECKPOINT_DIR/state_dict.pt" ]; do
  if [ ! -d "$SNAPSHOT_ROOT" ]; then
    sleep 60
    continue
  fi
  latest="$(find "$SNAPSHOT_ROOT" -maxdepth 1 -type d -name 'task_5_zhongliuke_train_end_*' | sort | tail -n 1 || true)"
  if [ -n "$latest" ] && [ -f "$latest/state_dict.pt" ]; then
    CHECKPOINT_DIR="$latest"
    break
  fi
  sleep 60
done

if [ ! -f "$CHECKPOINT_DIR/state_dict.pt" ]; then
  latest="$(find "$SNAPSHOT_ROOT" -maxdepth 1 -type d -name 'task_5_zhongliuke_train_end_*' | sort | tail -n 1)"
  CHECKPOINT_DIR="$latest"
fi

CUDA_VISIBLE_DEVICES=4 python -m experiments.reviewer_0629.run_routing_diagnostics \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --meta-embeddings-path metadata_embeddings/keshi_meta_embeddings.pt \
  --eval-data experiments/reviewer_0629/data/routing_reference_eval.jsonl \
  --output-dir experiments/reviewer_0629/outputs/routing_noisy_labels \
  --input-noise-modes none,char_delete_10,char_swap_10 \
  --route-modes full,top1,top2,top3 \
  --topk-values 1,2,3 \
  --max-samples 180 \
  --max-new-tokens 96

python -m experiments.reviewer_0629.aggregate_results \
  --output-root experiments/reviewer_0629/outputs \
  --report-path experiments/reviewer_0629/outputs/reviewer_0629_report.md
