# Reviewer Fairness Experiments

This directory adds reviewer-facing fairness experiments without changing the original MeMR training entrypoints or project structure. All code here is additive and isolated under `reviewer_fairness/`.

These baselines are included to answer the reviewer request for matched reference experiments under the same backbone, tokenizer, prompt style, LoRA budget, task order definitions, and reporting format.

Reference baselines covered here:

- `joint_training`: multi-task upper bound.
- `sequential_ft`: continual-learning lower bound.
- `single_task_oracle`: single-task oracle baseline.
- `metadata_only_routing`: metadata-only routing baseline.
- `er_lora`: standard replay baseline.
- `memr`: full MeMR method under matched setup.

## Layout

```text
reviewer_fairness/
  configs/
  scripts/
  src/
  results/
  logs/
```

All caches are stored under `reviewer_fairness/results/cache/`. The original dataset directory is never modified.

## Configuration

Default config:

- `reviewer_fairness/configs/fairness_default.yaml`
- `reviewer_fairness/configs/task_orders.yaml`

If a required field cannot be confirmed safely, the config or collector marks it as `TODO` or `UNVERIFIED`. Runtime code raises an explicit error for required unresolved `TODO` values instead of silently guessing.

## Running One Experiment

From the project root:

```bash
bash reviewer_fairness/scripts/run_reference_baselines.sh 0 sequential_ft order1
bash reviewer_fairness/scripts/run_reference_baselines.sh 1 metadata_only_routing order1
bash reviewer_fairness/scripts/run_reference_baselines.sh 2 er_lora order1
bash reviewer_fairness/scripts/run_reference_baselines.sh 3 joint_training all_tasks
bash reviewer_fairness/scripts/run_reference_baselines.sh 4 single_task_oracle all_tasks
bash reviewer_fairness/scripts/run_reference_baselines.sh 0 memr order1
```

Optional flags are forwarded to `reviewer_fairness/src/run_reference_experiment.py`, for example:

```bash
bash reviewer_fairness/scripts/run_reference_baselines.sh 0 memr order1 --dry_run
bash reviewer_fairness/scripts/run_reference_baselines.sh 0 memr order1 --overwrite
```

## Five-GPU Parallel Recommendation

Use the reviewer-requested launch pattern:

```bash
bash reviewer_fairness/scripts/run_reference_baselines.sh 0 sequential_ft order1
bash reviewer_fairness/scripts/run_reference_baselines.sh 1 metadata_only_routing order1
bash reviewer_fairness/scripts/run_reference_baselines.sh 2 er_lora order1
bash reviewer_fairness/scripts/run_reference_baselines.sh 3 joint_training all_tasks
bash reviewer_fairness/scripts/run_reference_baselines.sh 4 single_task_oracle all_tasks
```

Equivalent slot launcher for five terminals:

```bash
bash reviewer_fairness/scripts/run_rebuttal_suite_5gpu.sh slot1
bash reviewer_fairness/scripts/run_rebuttal_suite_5gpu.sh slot2
bash reviewer_fairness/scripts/run_rebuttal_suite_5gpu.sh slot3
bash reviewer_fairness/scripts/run_rebuttal_suite_5gpu.sh slot4
bash reviewer_fairness/scripts/run_rebuttal_suite_5gpu.sh slot5
```

After all baseline jobs complete:

```bash
bash reviewer_fairness/scripts/collect_fairness_config.sh
bash reviewer_fairness/scripts/aggregate_results.sh
```

To score the existing MeMR checkpoint under the same protocol:

```bash
bash reviewer_fairness/scripts/run_rebuttal_suite_5gpu.sh memr
```

## Existing Results and Skip Logic

If `reviewer_fairness/results/reference/<method>/<order>/metrics.json` already exists, the entrypoint skips by default.

Only `--overwrite` forces regeneration of that output directory.

The current implementation evaluates `memr` and `metadata_only_routing` from the detected `checkpoints_continual_keshi_llama/order1_compose_peft` snapshot, and launches actual matched-setting LoRA training for `joint_training`, `single_task_oracle`, `sequential_ft`, and `er_lora`.

For `memr` and `metadata_only_routing`, the fairness entrypoint evaluates an existing MeMR checkpoint:

- `memr`: evaluates the existing full MeMR checkpoint.
- `metadata_only_routing`: evaluates the same checkpoint with a runtime metadata-only routing wrapper.

## Output Locations

Per-method outputs:

- `reviewer_fairness/results/reference/<method>/<order>/`

Each run writes:

- `metrics.json`
- `metrics.csv`
- `config_resolved.yaml`
- `fairness_notes.json`
- `train.log`
- `eval.log`

Additional directories are created for compatibility with future runs:

- `checkpoints/`
- `predictions/`

Cache outputs:

- `reviewer_fairness/results/cache/joint_training/`
- `reviewer_fairness/results/cache/single_task_oracle/`
- `reviewer_fairness/results/cache/replay_buffer/`

Summary outputs:

- `reviewer_fairness/results/summary/fairness_config_table.csv`
- `reviewer_fairness/results/summary/fairness_config_table.md`
- `reviewer_fairness/results/summary/reference_baselines_summary.csv`
- `reviewer_fairness/results/summary/reference_baselines_summary.md`
- `reviewer_fairness/results/summary/all_results_summary.csv`
- `reviewer_fairness/results/summary/all_results_summary.md`

## Notes

- This directory is for reviewer fairness experiments only and is not part of the original MeMR method implementation.
- The original project files are not deleted, renamed, or overwritten.
- The original training entrypoint default behavior is unchanged.
- The fairness scripts are wrappers around the existing project code and produce standardized reviewer-facing outputs.
- The local `datasets/medical_consult/*/test.json` files are unlabeled question sets, so the automatic fairness scores here are computed on a fixed held-out dev split from each `train.json` using a task-stable seed shared across all methods.
