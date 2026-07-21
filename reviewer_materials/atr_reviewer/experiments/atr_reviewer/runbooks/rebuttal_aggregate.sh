#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

python3 experiments/atr_reviewer/aggregate_variants.py \
  --root_dir checkpoints_continual_keshi_llama/atr_reviewer_suite
