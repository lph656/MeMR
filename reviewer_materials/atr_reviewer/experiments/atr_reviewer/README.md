# ATR Reviewer Experiments

This directory contains supplementary experiment code for responding to the ATR reviewer comment without changing the default project workflow.

The workflow is:

1. Run one training command per variant with `src/run_continual_causal_llama2.py`.
2. Run the unified evaluation script on each experiment output directory.
3. Run the aggregation script to collect metrics and reviewer-facing statistics.

Expected final artifacts per variant:

- `configs.json`
- `log.txt`
- `osl_losses.log`
- `snapshots/task_5_.../checkpoint_info.json`
- `snapshots/task_5_.../state_dict.pt`
- `atr_reviewer_eval/final_test_metrics.json`
- `atr_reviewer_eval/stage_norm_summary.csv`
- `atr_reviewer_eval/task_norms_by_stage.csv`
- `atr_reviewer_eval/final_task_cosine_matrix.csv`
- `atr_reviewer_eval/final_metadata_cosine_matrix.csv`
- `atr_reviewer_eval/task_vs_metadata_similarity_pairs.csv`
- `atr_reviewer_eval/summary.json`

Preferred reviewer-facing suite:

- `wo_atr`
- `norm_only`
- `hard_orth`
- `cosine_soft_orth`
- `full_atr`

Optional collapse diagnostic:

- `lm_off_full_atr`

Use `python3 experiments/atr_reviewer/prepare_commands.py --suite rebuttal_minimal --include_lm_off_diagnostic`
to generate the current recommended commands. This suite isolates manuscript ATR on task vectors and keeps LoRA-level
OSL regularizers disabled to avoid confounding the reviewer response.

Legacy fast variants kept for backward compatibility:

- `norm_only`
- `hard_orth`
- `cosine_soft_orth`
- `lm_ablation_diagnostic`

The full baseline can reuse the already completed `order1_compose_peft` training run, or be re-run separately if strict budget matching is needed.
