#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."

python -m experiments.prompt_r3_c2.aggregate_results \
  --output-root experiments/prompt_r3_c2/outputs \
  --report-path experiments/prompt_r3_c2/outputs/prompt_r3_c2_report.md

python -m experiments.prompt_r3_c2.make_figures

