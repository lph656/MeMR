#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
python -m experiments.prompt_r3_c7.aggregate_results \
  --output-root experiments/prompt_r3_c7/outputs \
  --report-path experiments/prompt_r3_c7/outputs/prompt_r3_c7_report.md \
  --response-path experiments/prompt_r3_c7/outputs/reviewer_response_draft_cn.md

