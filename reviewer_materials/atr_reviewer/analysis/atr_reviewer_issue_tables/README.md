# ATR 评审问题整理包

这个目录不是按“实验文件夹”来组织结果，而是按“评审子问题”来组织当前 ATR 相关证据。

文件列表：
- `issue_1_norm_collapse_table.csv` 与 `issue_1_norm_collapse_explanation.md`
- `issue_2_stage_norm_trajectories_table.csv` 与 `issue_2_stage_norm_trajectories_explanation.md`
- `issue_3_pairwise_cosine_matrices_table.csv` 与 `issue_3_pairwise_cosine_matrices_explanation.md`
- `issue_4_metadata_similarity_alignment_table.csv` 与 `issue_4_metadata_similarity_alignment_explanation.md`
- `issue_5_ablation_comparison_table.csv` 与 `issue_5_ablation_comparison_explanation.md`
- `issue_6_unresolved_points_table.csv` 与 `issue_6_unresolved_points_explanation.md`

阅读方式：
- 每个 CSV 对应一个可以直接查看的数据表。
- 每个 Markdown 文件解释：对应 reviewer 子问题目前是“可以支持”“部分支持”还是“暂时不能支持”。

当前总判断：
- 现在已经可以较完整地回答“没有观察到向 0 collapse”“表示与 metadata 相似性高度一致”“关闭 LM loss 会让表示更接近初始化几何且文本表现下降”。
- 但仍然不能扎实回答“full ATR 明显优于 hard orth”以及“normalized cosine soft orth 是否优于原始设计”，因为这些变体之间的结果区分还不够干净。
