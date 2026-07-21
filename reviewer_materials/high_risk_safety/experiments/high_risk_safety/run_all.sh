#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_DIR="results/high_risk_safety/${TIMESTAMP}"

python experiments/high_risk_safety/create_safety_cases.py \
  --output experiments/high_risk_safety/safety_cases_zh.jsonl

python experiments/high_risk_safety/discover_checkpoints.py \
  --method_registry experiments/high_risk_safety/method_registry.template.json \
  --output_dir "${RESULT_DIR}"

python experiments/high_risk_safety/run_safety_eval.py \
  --dataset experiments/high_risk_safety/safety_cases_zh.jsonl \
  --method_registry "${RESULT_DIR}/method_registry.resolved.json" \
  --output_dir "${RESULT_DIR}" \
  --gpu 0 \
  --seed 42 \
  --max_new_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --do_sample false

python experiments/high_risk_safety/auto_safety_judge.py \
  --dataset experiments/high_risk_safety/safety_cases_zh.jsonl \
  --responses_dir "${RESULT_DIR}/responses" \
  --output_dir "${RESULT_DIR}"

python experiments/high_risk_safety/aggregate_safety_eval.py \
  --dataset experiments/high_risk_safety/safety_cases_zh.jsonl \
  --scores "${RESULT_DIR}/per_case_scores_auto.csv" \
  --output_dir "${RESULT_DIR}"

python experiments/high_risk_safety/make_figures.py \
  --summary "${RESULT_DIR}/summary_auto.csv" \
  --category_summary "${RESULT_DIR}/summary_by_category.csv" \
  --matching_weights "${RESULT_DIR}/matching_weights/MeMR_matching_weights.csv" \
  --output_dir "${RESULT_DIR}"

echo "High-risk safety experiment finished."
echo "Results saved to: ${RESULT_DIR}"
