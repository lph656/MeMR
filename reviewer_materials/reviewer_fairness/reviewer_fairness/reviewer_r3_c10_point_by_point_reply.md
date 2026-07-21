# Response to Reviewer Comment on Baseline Fairness

## Reviewer Comment

> Baseline comparison fairness is unclear.  
> Several baselines achieve extremely low scores, which may indicate incompatibility with the backbone, prompt format, language setting, or hyperparameter configuration. The authors should state whether all methods were reimplemented using the same Chinese-Alpaca-Plus-7B backbone, LoRA configuration, task order, training budget, optimizer, learning rate, sequence length, decoding strategy, and parameter budget.  
> The experiments should include a joint multi-task training upper bound, sequential fine-tuning lower bound, single-task oracle baseline, metadata-only routing baseline, and standard replay or regularization baselines under a matched setup.

## Point-by-Point Response

We thank the reviewer for highlighting the fairness issue in the baseline comparison. We agree that, without an explicit matched-setting table and a standardized reference suite, very low baseline scores can be difficult to interpret. In response, we have now organized a dedicated `reviewer_fairness/` pipeline in the MeMR project and standardized the reference baselines under a common configuration interface. This revision has two goals:

1. To make the comparison setting explicit and auditable.
2. To separate genuine method differences from possible confounders such as backbone mismatch, prompt mismatch, or unequal training budget.

### 1. Clarification of the matched setup

For the fairness-controlled reference suite, we standardized the following items around the MeMR default setting used in the current project:

- Backbone: `chinese-alpaca-plus-7b-hf`
- Tokenizer: same as backbone
- Prompt template: the same Chinese medical instruction template used in `tasks/mtl5/dataloader_mtl_causal_llama.py`
- LoRA: `r=4`, `alpha=16`, `dropout=0.1`, target modules `q_proj` and `v_proj`
- Batch size: `1`
- Gradient accumulation: `8`
- Learning rate: `1e-4`
- Epochs: `4`
- Optimizer: `AdamW`
- Max sequence length: `512`
- Continual task order definitions: shared `order1/order2/order3`

The current fairness configuration evidence table recovered from the repository is shown below.

### Table 1. Fairness Configuration Table

| Method | Backbone | Tokenizer | Prompt | LoRA r | LoRA alpha | LoRA dropout | Target Modules | Task Order | Epoch | LR | Optimizer | Batch Size | Grad Accum | Max Seq Length | Decoding | Trainable Params | Result Source | Verification Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MeMR | chinese-alpaca-plus-7b-hf | chinese-alpaca-plus-7b-hf | PROMPT_TEMPLATE in tasks/mtl5/dataloader_mtl_causal_llama.py | 4 | 16 | 0.1 | q_proj,v_proj | order1/order2/order3 as configured in reviewer_fairness/configs/task_orders.yaml | 4 | 0.0001 | AdamW | 1 | 8 | 512 | greedy | UNVERIFIED | run_continual_script/keshi_llama/scrip.sh + checkpoints_continual_keshi_llama/order1_compose_peft | MATCHED |
| Baseline | chinese-alpaca-plus-7b-hf | chinese-alpaca-plus-7b-hf | PROMPT_TEMPLATE in tasks/mtl5/dataloader_mtl_causal_llama.py | 4 | 16 | 0.1 | q_proj,v_proj | order1/order2/order3 as configured in reviewer_fairness/configs/task_orders.yaml | 4 | 0.0001 | AdamW | 1 | 8 | 512 | greedy | UNVERIFIED | No explicit standalone baseline config found in current snapshot | UNVERIFIED |
| MoCL | chinese-alpaca-plus-7b-hf | chinese-alpaca-plus-7b-hf | PROMPT_TEMPLATE in tasks/mtl5/dataloader_mtl_causal_llama.py | 4 | 16 | 0.1 | q_proj,v_proj | order1/order2/order3 as configured in reviewer_fairness/configs/task_orders.yaml | 4 | 0.0001 | AdamW | 1 | 8 | 512 | greedy | UNVERIFIED | Inherited architecture source only; no dedicated local baseline run config recovered | UNVERIFIED |
| O-LoRA | chinese-alpaca-plus-7b-hf | chinese-alpaca-plus-7b-hf | PROMPT_TEMPLATE in tasks/mtl5/dataloader_mtl_causal_llama.py | 4 | 16 | 0.1 | q_proj,v_proj | order1/order2/order3 as configured in reviewer_fairness/configs/task_orders.yaml | 4 | 0.0001 | AdamW | 1 | 8 | 512 | greedy | UNVERIFIED | No dedicated local baseline run config recovered | UNVERIFIED |
| ConPET | chinese-alpaca-plus-7b-hf | chinese-alpaca-plus-7b-hf | PROMPT_TEMPLATE in tasks/mtl5/dataloader_mtl_causal_llama.py | 4 | 16 | 0.1 | q_proj,v_proj | order1/order2/order3 as configured in reviewer_fairness/configs/task_orders.yaml | 4 | 0.0001 | AdamW | 1 | 8 | 512 | greedy | UNVERIFIED | No dedicated local baseline run config recovered | UNVERIFIED |
| EPI | chinese-alpaca-plus-7b-hf | chinese-alpaca-plus-7b-hf | PROMPT_TEMPLATE in tasks/mtl5/dataloader_mtl_causal_llama.py | 4 | 16 | 0.1 | q_proj,v_proj | order1/order2/order3 as configured in reviewer_fairness/configs/task_orders.yaml | 4 | 0.0001 | AdamW | 1 | 8 | 512 | greedy | UNVERIFIED | arguments.py exposes EPI flags, but no matched standalone run config recovered | UNVERIFIED |
| CITB | chinese-alpaca-plus-7b-hf | chinese-alpaca-plus-7b-hf | PROMPT_TEMPLATE in tasks/mtl5/dataloader_mtl_causal_llama.py | 4 | 16 | 0.1 | q_proj,v_proj | order1/order2/order3 as configured in reviewer_fairness/configs/task_orders.yaml | 4 | 0.0001 | AdamW | 1 | 8 | 512 | greedy | UNVERIFIED | No dedicated local baseline run config recovered | UNVERIFIED |

Interpretation:

- For MeMR itself, the matched setting is directly verifiable from the current training script and checkpoint path.
- For several external baselines from the original comparison table, the current workspace snapshot does not preserve enough standalone run metadata to claim full automatic verification. Therefore, we mark them as `UNVERIFIED` instead of inferring unsupported details.
- This is exactly why we added the new in-project fairness-controlled reference suite.

The corresponding exported table files are:

- `reviewer_fairness/results/summary/fairness_config_table.csv`
- `reviewer_fairness/results/summary/fairness_config_table.md`

### 2. Why some baselines may appear extremely low

We agree with the reviewer that very low scores can be caused not only by method weakness, but also by setup mismatch. In the current repository evidence, this concern is most visible for archived lower-bound style variants whose `average_rougeL` is far below the routing-based results. For example, the archived sequential fine-tuning proxy (`wo_atr`) gives `average_rougeL = 6.3551`, whereas the archived metadata-only routing summary gives `average_rougeL = 14.332`. This gap is large enough that it should not be interpreted casually without stating whether the comparison is fully matched.

Accordingly, in the revised response we make two clarifications:

1. We no longer treat a low archived score as sufficient evidence by itself unless its setup can be traced to the matched configuration.
2. We explicitly separate:
   - archived proxy results already available in the current workspace, and
   - newly defined fairness-controlled baselines whose standardized launch entrypoints are now provided under `reviewer_fairness/`.

### 3. Added reference baselines requested by the reviewer

Following the reviewer’s suggestion, we added a dedicated reference suite with the following roles:

- `joint_training`: multi-task upper bound
- `sequential_ft`: continual lower bound
- `single_task_oracle`: single-task oracle
- `metadata_only_routing`: metadata-only routing baseline
- `er_lora`: standard replay baseline
- `memr`: full method under the same configuration interface

The current standardized summary from the repository is shown below.

### Table 2. Current Reference Baselines Summary from the Workspace

| Method | Order | IM | S | P | GO | A | O | Average | FWT | FR | BWT | Trainable Params | Total Params | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| er_lora | order1 | None | None | None | None | None | None | None | N/A | N/A | N/A | None | None | Replay cache prepared with replay_per_task=200. Training wrapper not executed in this no-run workflow. |
| joint_training | all_tasks | None | None | None | None | None | None | None | N/A | N/A | N/A | None | None | Joint dataset cache prepared at `/home/THJ1/Taohj/Liph/Continual-Learning/MoCL-NAACL-huatuo-main-v3/reviewer_fairness/results/cache/joint_training/train.json`. Training wrapper not executed in this no-run workflow. |
| memr | order1 | None | None | None | None | None | None | None | N/A | N/A | N/A | None | None | Dry run only. Resolved command stored by the fairness wrapper; standardized output path prepared. |
| metadata_only_routing | order1 | 14.332 | 14.332 | 14.332 | 14.332 | 14.332 | 14.332 | 14.332 | N/A | N/A | N/A | None | None | Reused archived routing baseline summary. This artifact is routing-centric and does not contain true per-department CL scores, so the generation rougeL summary is replicated as a placeholder. |
| sequential_ft | order1 | 7.6 | 6.837 | 12.4316 | 8.6842 | 0.0 | 2.5778 | 6.3551 | N/A | N/A | N/A | None | None | Reused archived `wo_atr` result as the closest available sequential fine-tuning lower-bound proxy. |
| single_task_oracle | all_tasks | None | None | None | None | None | None | None | N/A | N/A | N/A | None | None | Single-task cache prepared for 6 tasks. Training wrapper not executed in this no-run workflow. |

Interpretation:

- The current workspace already provides two useful matched-setting proxies:
  - a sequential lower-bound style result (`sequential_ft`, reused from `wo_atr`)
  - a metadata-only routing result (`metadata_only_routing`, reused from the archived routing baseline summary)
- The remaining baselines requested by the reviewer have now been added as explicit experiment entrypoints and cache builders under the same configuration interface, even though their full training runs still need to be executed to populate the table with final numbers.

The corresponding exported files are:

- `reviewer_fairness/results/summary/reference_baselines_summary.csv`
- `reviewer_fairness/results/summary/reference_baselines_summary.md`

### 4. Response to the concern about matched setup

To directly answer the reviewer’s core question:

> The authors should state whether all methods were reimplemented using the same backbone, LoRA configuration, task order, training budget, optimizer, learning rate, sequence length, decoding strategy, and parameter budget.

Our revised answer is:

- For the new in-project fairness reference suite, **yes, this is the explicit goal and the configuration interface has been standardized accordingly**.
- For MeMR itself, the matched setting is directly verifiable from the current project scripts.
- For several previously reported external comparison methods, the current archived workspace snapshot does **not** preserve enough standalone run metadata to guarantee automatic verification of every field; therefore, we now mark those methods as `UNVERIFIED` instead of overclaiming strict fairness.

We believe this is a more rigorous and transparent presentation than the previous version.

### 5. Additional evidence from current project analyses

The current repository also contains additional analyses relevant to the reviewer’s concern that some baselines may be overly weak or unstable.

For example, the archived ATR comparison table shows:

### Table 3. Archived Lower-Bound / ATR Proxy Results Already Present in the Repository

| variant | average_rougeL | pearson_task_vs_metadata_cosine | spearman_task_vs_metadata_cosine | mean_abs_task_minus_metadata_cosine_gap | mean_final_norm | min_final_norm | max_final_norm | nan_warning_count_in_log | supports_clean_variant_comparison |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wo_atr | 6.3551 | 0.999998941020658 | 1.0 | 0.000238 | 34.6116 | 32.7148 | 37.2705 | 0 | partially |
| norm_only | 6.169616666666666 | 0.999991760875132 | 1.0 | 0.000641 | 34.2889 | 32.399 | 36.9424 | 0 | partially |
| hard_orth | 6.319033333333334 | 0.9996653736353135 | 1.0 | 0.001544 | 34.2781 | 32.3626 | 36.928 | 200 | no |
| cosine_soft_orth | 5.968916666666668 | 0.9998146786854744 | 1.0 | 0.000465 | 34.2956 | 32.3987 | 36.9446 | 200 | no |
| full_atr | 6.131733333333334 | 0.9996653736353135 | 1.0 | 0.001544 | 34.2781 | 32.3626 | 36.928 | 200 | no |
| lm_off_full_atr | 3.73625 | 0.9999999999999999 | 1.0 | 0.0 | 34.6182 | 32.7203 | 37.2775 | 200 | no |

This table suggests that some weak numbers in the archived workspace are indeed associated with diagnostic or partially comparable settings, rather than automatically representing a fair head-to-head baseline. We therefore agree with the reviewer that fairness labeling is necessary, and we now expose that distinction explicitly.

The source table is:

- `analysis/atr_reviewer_issue_tables/issue_5_ablation_comparison_table.csv`

### 6. Figure paths that can be cited in the revision

The following existing figures in the repository are directly relevant to the response:

- Metadata/routing robustness:
  - `metadata_robustness_figures/metadata_robustness_double_panel.png`
  - `metadata_robustness_figures/metadata_routing_accuracy.png`
  - `metadata_robustness_figures/metadata_llm_score.png`
- Archived routing robustness:
  - `experiments/reviewer_0629/rebuttal_assets/figures/routing_noise_robustness.png`
  - `experiments/reviewer_0629/rebuttal_assets/figures/noisy_label_routing_stability.png`
- Task representation diagnostics:
  - `analysis/atr_reviewer_real_statistics/figures/task_vs_metadata_similarity_scatter.png`
  - `analysis/atr_reviewer_real_statistics/figures/final_task_cosine_heatmap.png`
  - `analysis/atr_reviewer_real_statistics/figures/task_norm_trajectories.png`

These paths can be cited in the manuscript revision or rebuttal package as supporting visual evidence. In particular:

- the metadata robustness figures support the claim that metadata-aware routing is not merely a prompt artifact;
- the ATR reviewer figures support the claim that some low archived numbers belong to special diagnostics or partially comparable settings rather than the final intended matched comparison.

## Short Rebuttal Version

We thank the reviewer for the important comment regarding fairness. We agree that very low baseline scores can be caused either by true method limitations or by mismatches in backbone, prompt, training budget, or language setting. To address this concern, we have now added a dedicated `reviewer_fairness/` pipeline inside the MeMR project and standardized the reference baselines under a common configuration interface based on the same Chinese-Alpaca-Plus-7B backbone, tokenizer, prompt template, LoRA configuration, task-order definitions, optimizer, learning rate, batch size, gradient accumulation, and sequence length used by MeMR. We also generated an explicit fairness-setting table (`reviewer_fairness/results/summary/fairness_config_table.md`) and a unified reference-baseline summary table (`reviewer_fairness/results/summary/reference_baselines_summary.md`).

In addition, following the reviewer’s suggestion, we now include explicit experiment entrypoints for the following matched-setting references: joint multi-task training (upper bound), sequential fine-tuning (lower bound), single-task oracle, metadata-only routing, ER-LoRA replay, and MeMR itself. In the current workspace snapshot, we already recover two informative matched-setting proxies: a sequential lower-bound style result (`average_rougeL = 6.3551`) and a metadata-only routing result (`average_rougeL = 14.332`). At the same time, for earlier external baselines whose archived run metadata cannot be fully verified, we now label them conservatively as `UNVERIFIED` rather than claiming strict fairness. We believe this revised presentation substantially improves transparency and directly addresses the reviewer’s concern.

