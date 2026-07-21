#!/usr/bin/env bash
set -euo pipefail

# This script is documentation-oriented. Copy the commands into separate terminals
# in your own environment instead of executing this file blindly.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
META_EMBEDDINGS_PATH="${ROOT_DIR}/metadata_embeddings/keshi_meta_embeddings.pt"
TASK_LIST="neike_waike_erke_fuchanke_nanke_zhongliuke"
MODEL_PATH="chinese-alpaca-plus-7b-hf"

echo "Use one terminal per command below."
echo
echo "Preferred reviewer-facing suite:"
echo "python3 experiments/atr_reviewer/prepare_commands.py --suite rebuttal_minimal --include_lm_off_diagnostic"
echo
echo "Wave 1 / four terminals:"
echo "  GPU 1: wo_atr"
echo "  GPU 2: norm_only"
echo "  GPU 3: hard_orth"
echo "  GPU 4: cosine_soft_orth"
echo
echo "Wave 2 / after one terminal frees up:"
echo "  GPU 1: full_atr"
echo "  GPU 2: lm_off_full_atr (optional but recommended for collapse diagnosis)"
