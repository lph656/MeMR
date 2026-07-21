#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."

python -m experiments.reviewer_0629.build_eval_sets \
  --output-dir experiments/reviewer_0629/data \
  --reference-per-task 60 \
  --test-per-task 40 \
  --holdout-task zhongliuke \
  --holdout-reference-count 120

python -m experiments.reviewer_0629.make_noisy_label_dataset \
  --source-root datasets/medical_consult \
  --output-root experiments/reviewer_0629/generated_datasets/noisy_labels_20/medical_consult \
  --noise-ratio 0.2

CUDA_VISIBLE_DEVICES=0 python -m experiments.reviewer_0629.train_variant \
  --dataset_root experiments/reviewer_0629/generated_datasets/noisy_labels_20/medical_consult \
  --model_name_or_path chinese-alpaca-plus-7b-hf \
  --task_list neike_waike_erke_fuchanke_nanke_zhongliuke \
  --continual_learning \
  --mpeft_enabled \
  --matching_loss_v2 \
  --meta_embeddings_path ./metadata_embeddings/keshi_meta_embeddings.pt \
  --do_train \
  --padding_strategy longest \
  --max_seq_length 512 \
  --max_target_length 64 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --num_train_epochs 4 \
  --output_dir experiments/reviewer_0629/outputs/noisy_labels_training \
  --overwrite_output_dir \
  --seed 0 \
  --save_strategy no \
  --evaluation_strategy no \
  --validation_split_percentage 0.1 \
  --overwrite_cache True \
  --lamda_1 0.05 \
  --lamda_2 0.01 \
  --orthogonal_threshold 0.2

