# Issue 4：表示相似度是否与医学领域相似度一致？

结论：这一点目前是有支持证据的。

使用证据：
- `issue_4_metadata_similarity_alignment_table.csv`
- 各变体和 baseline 提取结果中的 `task_vs_metadata_similarity_pairs.csv`

解释：
- 当前所有可用运行都显示 task-vector cosine similarity 与 metadata cosine similarity 之间具有很高的一致性。
- 其中 `lm_off_full_atr` 的一致性几乎达到 1，这说明在去掉 LM loss 后，task vectors 更倾向于停留在初始化的 metadata 几何附近。
- 表中的 `mean_abs_task_minus_metadata_cosine_gap` 也可以直接量化“最终表示相对初始化 metadata 几何偏离了多少”。
- 因此，这一组结果既支持“表示结构与医学领域相似性一致”，也提示“LM objective 的一个作用可能是推动表示偏离纯初始化几何，而不是单纯维持原状”。

局限性：
- 这组证据支持“表示结构学到了医学领域关系”这一点。
- 但它本身并不能单独回答“哪一种 ATR 消融设计最好”。
