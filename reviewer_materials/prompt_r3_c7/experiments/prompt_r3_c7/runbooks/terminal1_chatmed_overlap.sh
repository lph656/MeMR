#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
python -m experiments.prompt_r3_c7.audit_dataset_integrity \
  --job chatmed \
  --project-root . \
  --dataset-root datasets/medical_consult \
  --output-root experiments/prompt_r3_c7/outputs

