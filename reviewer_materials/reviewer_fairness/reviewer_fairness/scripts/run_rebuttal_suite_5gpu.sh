#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash reviewer_fairness/scripts/run_rebuttal_suite_5gpu.sh <slot>"
  echo "  slot1 -> sequential_ft order1"
  echo "  slot2 -> metadata_only_routing order1"
  echo "  slot3 -> er_lora order1"
  echo "  slot4 -> joint_training all_tasks"
  echo "  slot5 -> single_task_oracle all_tasks"
  exit 1
fi

SLOT="$1"
shift || true

case "${SLOT}" in
  slot1)
    exec bash reviewer_fairness/scripts/run_reference_baselines.sh 0 sequential_ft order1 --overwrite "$@"
    ;;
  slot2)
    exec bash reviewer_fairness/scripts/run_reference_baselines.sh 1 metadata_only_routing order1 --overwrite "$@"
    ;;
  slot3)
    exec bash reviewer_fairness/scripts/run_reference_baselines.sh 2 er_lora order1 --overwrite "$@"
    ;;
  slot4)
    exec bash reviewer_fairness/scripts/run_reference_baselines.sh 3 joint_training all_tasks --overwrite "$@"
    ;;
  slot5)
    exec bash reviewer_fairness/scripts/run_reference_baselines.sh 4 single_task_oracle all_tasks --overwrite "$@"
    ;;
  memr)
    exec bash reviewer_fairness/scripts/run_reference_baselines.sh 0 memr order1 --overwrite "$@"
    ;;
  *)
    echo "Unknown slot: ${SLOT}"
    exit 1
    ;;
esac
