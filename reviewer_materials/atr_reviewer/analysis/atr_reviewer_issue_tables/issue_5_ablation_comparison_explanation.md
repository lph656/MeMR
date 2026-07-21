# Issue 5：当前更强的 ATR 消融是否已经足以支持 preferred design？

结论：目前仍不能支持“完整 ATR 设计已经被干净证明为最佳”，但现在已经比上一版多出一条可用信息：`LM-off` 会明显损害文本表现，并使表示更接近初始化 metadata 几何。

使用证据：
- `issue_5_ablation_comparison_table.csv`
- `checkpoints_continual_keshi_llama/atr_reviewer_suite/*/atr_reviewer_eval/` 中各变体的 summary 和 metric 文件

解释：
- 当前这组消融实验足以报告一些初步趋势，例如“没有明显 norm collapse”、“表示仍然与 metadata 高度一致”，以及“关闭 LM loss 会显著降低文本表现，并使表示更贴近初始化 metadata 结构”。
- 但是，它仍然不能干净地支持如下更强的结论：`full_atr` 明显优于 `hard_orth`、`norm_only` 或 `cosine_soft_orth`。

为什么现在还不够：
- `hard_orth`、`cosine_soft_orth` 和 `full_atr` 的训练日志里都出现了大量 `NaN/Inf` 警告，这会削弱它们作为稳定消融结果的证据力度。
- 当前运行中，`full_atr` 和 `hard_orth` 最终得到的 task-vector 状态实际上相同，因此这两者之间的对比目前并不可信。
- 虽然 `lm_off_full_atr` 已经跑完，并且显示 `average_rougeL` 明显下降，但它没有表现出“向 0 collapse”，而是表现为“几乎停留在 metadata 初始化几何附近”。

当前可以使用的 reviewer-facing 表述：
- 现有消融表格可以作为“部分证据”来展示。
- 可以较谨慎地写：去掉 LM objective 会导致表示更新不足、文本性能下降，因此 LM objective 对避免退化是重要的。
- 但仍不应据此写出“完整 ATR 设计已经被当前结果严格证明为最优”的强结论。
