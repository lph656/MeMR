#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: bash reviewer_fairness/scripts/run_reference_baselines.sh <gpu_id> <method> <order_name> [extra args...]"
  exit 1
fi

GPU_ID="$1"
METHOD="$2"
ORDER_NAME="$3"
shift 3

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

python reviewer_fairness/src/run_reference_experiment.py \
  --method "${METHOD}" \
  --config reviewer_fairness/configs/fairness_default.yaml \
  --order "${ORDER_NAME}" \
  --gpu "${GPU_ID}" \
  "$@"

