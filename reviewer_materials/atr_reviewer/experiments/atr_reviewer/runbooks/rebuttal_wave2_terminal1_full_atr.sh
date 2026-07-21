#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

CUDA_VISIBLE_DEVICES=1 python3 src/run_continual_causal_llama2.py \
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
  --num_train_epochs 1 \
  --max_train_batches_per_epoch 200 \
  --save_strategy no \
  --evaluation_strategy no \
  --validation_split_percentage 0.1 \
  --overwrite_cache True \
  --seed 0 \
  --atr_variant_name full_atr \
  --output_dir checkpoints_continual_keshi_llama/atr_reviewer_suite/full_atr \
  --overwrite_output_dir \
  --lamda_1 0.0 \
  --lamda_2 0.0 \
  --atr_enable_key_ortho \
  --atr_key_ortho_coeff 0.1 \
  --atr_key_ortho_mode unnormalized_soft \
  --atr_key_ortho_threshold 0.1 \
  --atr_key_l2_lambda 0.005

CUDA_VISIBLE_DEVICES=1 python3 experiments/atr_reviewer/evaluate_variant.py \
  --experiment_dir checkpoints_continual_keshi_llama/atr_reviewer_suite/full_atr \
  --per_device_eval_batch_size 1 \
  --max_final_test_batches 50

python3 tools/plot_atr_reviewer_stats.py \
  --data_dir checkpoints_continual_keshi_llama/atr_reviewer_suite/full_atr/atr_reviewer_eval \
  --out_dir checkpoints_continual_keshi_llama/atr_reviewer_suite/full_atr/atr_reviewer_eval/figures
