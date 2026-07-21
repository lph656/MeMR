# Issue 3：pairwise cosine similarity matrix 能说明什么？

结论：这些结果可以报告，但它们对不同变体之间的区分能力还比较弱。

使用证据：
- `issue_3_pairwise_cosine_matrices_table.csv`
- 各变体目录下 `atr_reviewer_eval/figures/` 中的 cosine 热力图

解释：
- 当前所有可用变体都表现出较高但并不完全相同的 cosine 结构，这更符合“稳定且有结构的任务表示”，而不是“完全塌缩”的情况。
- 但是，`wo_atr`、`norm_only`、`hard_orth`、`cosine_soft_orth` 和 `full_atr` 之间的 cosine matrix 仍然非常接近。

局限性：
- 这些矩阵足以满足 reviewer 提出的“请报告 cosine similarity structure”这一要求。
- 但它们还不足以强有力地证明完整 ATR 设计在几何结构上明显优于其他变体。
