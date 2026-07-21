# Issue 1：ATR 相关损失是否会把任务表示压缩到接近 0？

结论：目前结果可以支持“没有观察到向 0 的显著 collapse”，但不能支持“LM-off 时一定出现向 0 collapse”。

使用证据：
- `issue_1_norm_collapse_table.csv`
- `analysis/atr_reviewer_real_statistics/stage_norm_summary.csv`
- `checkpoints_continual_keshi_llama/atr_reviewer_suite/*/atr_reviewer_eval/stage_norm_summary.csv`

解释：
- 目前所有可用变体的最终 task-vector norm 都大致落在 32 到 37 之间，明显不接近 0。
- 原始 baseline 运行 `order1_compose_peft` 在 6 个阶段中也表现出稳定的非零 norm。
- 因此，基于现有实验，可以支持一个较弱但明确的结论：这些运行里学到的任务表示没有表现出向 0 明显塌缩的现象。
- 即使在 `lm_off_full_atr` 中，也没有看到 norm 直接掉到接近 0；它的问题更像是“表示几乎停留在初始 metadata 几何附近，同时文本表现下降”。

局限性：
- 这仍然不能简单等价为“language-model objective 唯一阻止了 collapse”。
- 因为当前 `lm_off_full_atr` 更像是在无 LM loss 时几乎不偏离初始化表示，而不是发生向零塌缩，所以 reviewer 的这一因果问题目前只能得到部分回答。
