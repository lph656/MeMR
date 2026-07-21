# All Results Summary

| Method | Order | IM | S | P | GO | A | O | Average | FWT | FR | BWT | Trainable Params | Total Params | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| er_lora | order1 | None | None | None | None | None | None | None | N/A | N/A | N/A | None | None | Replay cache prepared with replay_per_task=200. Training wrapper not executed in this no-run workflow. |
| joint_training | all_tasks | None | None | None | None | None | None | None | N/A | N/A | N/A | None | None | Joint dataset cache prepared at /home/THJ1/Taohj/Liph/Continual-Learning/MoCL-NAACL-huatuo-main-v3/reviewer_fairness/results/cache/joint_training/train.json. Training wrapper not executed in this no-run workflow. |
| memr | order1 | None | None | None | None | None | None | None | N/A | N/A | N/A | None | None | Dry run only. Resolved command: python reviewer_fairness/src/run_reference_experiment.py --method memr --config reviewer_fairness/configs/fairness_default.yaml --order order1 --gpu 0 --overwrite --dry_run |
| metadata_only_routing | order1 | 14.332 | 14.332 | 14.332 | 14.332 | 14.332 | 14.332 | 14.332 | N/A | N/A | N/A | None | None | Reused reviewer_0629 routing baseline summary. The artifact is routing-centric and does not contain true per-department CL scores, so the generation rougeL summary is replicated as a placeholder. |
| sequential_ft | order1 | 7.6 | 6.837 | 12.4316 | 8.6842 | 0.0 | 2.5778 | 6.3551 | N/A | N/A | N/A | None | None | Reused ATR reviewer 'wo_atr' result as the closest existing sequential fine-tuning lower-bound proxy. FWT/FR/BWT remain unavailable from the archived artifact. |
| single_task_oracle | all_tasks | None | None | None | None | None | None | None | N/A | N/A | N/A | None | None | Single-task cache prepared for 6 tasks. Training wrapper not executed in this no-run workflow. |
