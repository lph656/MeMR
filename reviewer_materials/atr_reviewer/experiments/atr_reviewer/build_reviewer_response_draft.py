import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ISSUE_DIR = ROOT / "analysis" / "atr_reviewer_issue_tables"
OUT_PATH = ISSUE_DIR / "reviewer_response_draft_cn.md"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main():
    issue1_table = ISSUE_DIR / "issue_1_norm_collapse_table.csv"
    issue2_table = ISSUE_DIR / "issue_2_stage_norm_trajectories_table.csv"
    issue3_table = ISSUE_DIR / "issue_3_pairwise_cosine_matrices_table.csv"
    issue4_table = ISSUE_DIR / "issue_4_metadata_similarity_alignment_table.csv"
    issue5_table = ISSUE_DIR / "issue_5_ablation_comparison_table.csv"
    issue6_table = ISSUE_DIR / "issue_6_unresolved_points_table.csv"

    text = f"""
# ATR 审稿意见逐条回复草稿（中文）

以下内容仅使用当前已经得到的真实实验结果，不引入任何虚构结论。回复策略分为两类：
- 能被当前结果较强支持的点：直接给出数据与结论。
- 目前还不能被“完美支持”的点：采用审慎、可辩护的话术，只陈述当前证据能支持到的程度，并避免做超出证据范围的强主张。

---

## 审稿意见原文要点拆分

该条审稿意见可拆分为以下 6 个子问题：

1. ATR 是否会导致 task representations 向 0 collapse。
2. 表示随任务推进的轨迹是否显示持续塌缩。
3. pairwise cosine-similarity matrices 能否证明表示是稳定且可区分的。
4. 任务表示相似度与医学领域相似度之间是否存在对应关系。
5. 更强的 ATR 消融是否足以证明完整设计的合理性。
6. language-model objective 是否阻止了退化表示。

---

## 回复 1：关于“ATR 是否会导致表示向 0 collapse”

**可直接使用的结论**

当前结果支持：在所有已完成变体中，均未观察到 task representation 向 0 的显著 collapse。

**建议引用表格**

- `{issue1_table.relative_to(ROOT)}`

**建议引用图**

- 原始 baseline 图：`analysis/atr_reviewer_real_statistics/figures/final_task_norms_bar.png`
- 变体图：`checkpoints_continual_keshi_llama/atr_reviewer_suite/full_atr/atr_reviewer_eval/figures/final_task_norms_bar.png`

**推荐回复话术**

> We thank the reviewer for raising the concern that the orthogonality term together with the Frobenius-norm regularizer might drive task representations toward zero. We therefore explicitly examined the L2 norms of the learned task vectors across all available ATR-related variants. As shown in `{issue1_table.name}`, all completed runs maintain clearly non-zero task-vector norms, with final norms consistently remaining in the range of approximately 32–37 rather than shrinking toward 0. The original baseline run extracted from the main checkpoint also shows stable non-zero norms across all six stages. These results indicate that, in the current implementation and training regime, we do not observe empirical evidence of zero-collapse of task representations.

**解释要点**

- 这里可以强回复，因为 norm 结果非常直接。
- 不要写“彻底证明 ATR 一定不会 collapse”，而写“当前实现与训练设置下未观察到 collapse”更稳。

---

## 回复 2：关于“表示轨迹 over tasks 是否显示塌缩”

**可直接使用的结论**

当前结果支持：跨阶段 norm 轨迹没有显示出随着任务增加而持续收缩到 0 的趋势。

**建议引用表格**

- `{issue2_table.relative_to(ROOT)}`

**建议引用图**

- `analysis/atr_reviewer_real_statistics/figures/stage_norm_statistics_line.png`
- `checkpoints_continual_keshi_llama/atr_reviewer_suite/full_atr/atr_reviewer_eval/figures/stage_norm_statistics_line.png`
- `checkpoints_continual_keshi_llama/atr_reviewer_suite/full_atr/atr_reviewer_eval/figures/task_representation_pca_trajectories.png`

**推荐回复话术**

> To further address the reviewer’s concern, we also tracked the norm statistics over continual-learning stages. The stage-wise trajectories do not show a monotonic shrinkage toward zero as new tasks are introduced. Instead, the mean, minimum, and maximum norms remain stable across stages, which is inconsistent with a progressive collapse process. We additionally visualize the task-representation trajectories in PCA space and observe structured movement rather than convergence to a shared zero point.

**解释要点**

- PCA 图可以放，但更建议把它作为“辅助可视化”。
- 真正硬证据还是 stage norm table/line plot。

---

## 回复 3：关于“pairwise cosine-similarity matrices”

**可直接使用的结论**

当前结果支持：pairwise cosine matrix 呈现稳定而有结构的任务关系，而不是无差别塌缩。

**建议引用表格**

- `{issue3_table.relative_to(ROOT)}`

**建议引用图**

- `analysis/atr_reviewer_real_statistics/figures/final_task_cosine_heatmap.png`
- `checkpoints_continual_keshi_llama/atr_reviewer_suite/full_atr/atr_reviewer_eval/figures/final_task_cosine_heatmap.png`

**推荐回复话术**

> Following the reviewer’s suggestion, we report the full pairwise cosine-similarity matrices of the learned task representations. The matrices reveal stable but non-uniform inter-task similarities, suggesting that the learned representations preserve meaningful structure rather than collapsing into a trivial configuration. In particular, the task relations remain differentiated across departments, which is incompatible with an unstructured degenerate solution.

**审慎补充话术**

> At the same time, we note that the cosine matrices across several ablation variants remain relatively close to one another. Therefore, we use these matrices primarily as evidence against trivial collapse, rather than as the sole basis for claiming strong geometric superiority of one ATR variant over another.

---

## 回复 4：关于“表示相似度与医学领域相似度的关系”

**可直接使用的结论**

这是当前最强的一组证据之一，可以较强支持。

**建议引用表格**

- `{issue4_table.relative_to(ROOT)}`

**建议引用图**

- `analysis/atr_reviewer_real_statistics/figures/task_vs_metadata_similarity_scatter.png`
- `checkpoints_continual_keshi_llama/atr_reviewer_suite/full_atr/atr_reviewer_eval/figures/task_vs_metadata_similarity_scatter.png`

**推荐回复话术**

> We further quantified the relationship between learned task-representation similarity and medical-domain similarity defined by metadata embeddings. Across all completed runs, we observe very high Pearson/Spearman alignment between task-vector cosine similarity and metadata cosine similarity. This provides direct empirical support that the learned task geometry is not arbitrary, but instead closely tracks medically meaningful domain structure encoded by the metadata.

**可以利用的新信息**

> Interestingly, the LM-off diagnostic exhibits an almost perfect preservation of metadata geometry. This suggests that, without the language-model objective, the task vectors tend to remain closer to their metadata initialization, whereas the full training objective encourages task-adaptive deviation from the initialization while maintaining high structural alignment.

---

## 回复 5：关于“更强 ATR 消融是否已经证明完整设计最优”

**当前不能完美支持，因此必须用审慎话术。**

**建议引用表格**

- `{issue5_table.relative_to(ROOT)}`

**建议引用图**

- 可引用 `full_atr`、`hard_orth`、`cosine_soft_orth` 的 norm/cosine 图，但不建议把图作为“完整设计已最优”的核心证据。

**当前真实可支持的结论**

- `wo_atr`、`norm_only`、`hard_orth`、`cosine_soft_orth`、`full_atr` 之间确实存在一些结果差异。
- `LM-off` 会明显降低文本表现。
- 但当前实验还不能干净证明 `full_atr` 明显优于全部替代设计。

**推荐回复话术**

> We performed the stronger ablation suite requested by the reviewer, including w/o ATR, norm-only regularization, hard orthogonality, normalized cosine-based soft orthogonality, and the original full ATR configuration. The resulting tables show that all variants maintain non-collapsed representations, while removing the LM objective substantially reduces downstream text quality and leaves the learned task geometry much closer to the metadata initialization. These observations support the necessity of coupling representation regularization with the task-learning objective.

**关键保守话术**

> At the same time, we would like to avoid over-claiming based on the current supplementary runs. In particular, some of the harder-constraint variants exhibited optimization instability warnings, and the current full-ATR vs. hard-orth comparison does not yet separate as cleanly as we would ideally like. Therefore, we interpret the present ablation evidence as supporting the general usefulness of the ATR-style regularization framework and the role of the LM objective, while treating the exact ranking among closely related orthogonality variants with appropriate caution.

**解释要点**

- 这段话的目的不是认输，而是主动缩窄 claim。
- 你不是说“我们的 full ATR 一定最优”，而是说“现有结果支持 ATR 框架的合理性，并支持 LM objective 的必要性，但近邻变体之间的精细排序我们谨慎解释”。

这样写，审稿人不容易一下抓住你“夸大结论”。

---

## 回复 6：关于“LM objective 是否阻止了退化表示”

**当前可以部分支持，但不能用“完全证明 collapse 被阻止”这种强话术。**

**建议引用表格**

- `{issue5_table.relative_to(ROOT)}`
- `{issue6_table.relative_to(ROOT)}`

**建议引用图**

- `checkpoints_continual_keshi_llama/atr_reviewer_suite/lm_off_full_atr/atr_reviewer_eval/figures/final_task_norms_bar.png`
- `checkpoints_continual_keshi_llama/atr_reviewer_suite/lm_off_full_atr/atr_reviewer_eval/figures/task_vs_metadata_similarity_scatter.png`

**当前真实结论**

- `lm_off_full_atr` 没有向 0 collapse。
- 但它几乎停留在 metadata 初始化几何上，同时文本表现显著下降。
- 因而可以合理说：LM objective 对避免“表示退化为初始化附近的低适应性状态”是重要的。

**推荐回复话术**

> To directly probe the reviewer’s hypothesis, we additionally ran an LM-off diagnostic. We find that removing the language-model objective does not produce an obvious zero-norm collapse; however, it substantially degrades downstream text quality and yields task representations that remain almost perfectly tied to the metadata initialization geometry. This suggests that the LM objective plays an important role in preventing a different but still undesirable degenerate behavior: the task vectors fail to evolve into task-adaptive representations and instead remain near their initialization.

**这段话术的价值**

- 不说“LM objective prevents zero-collapse”这么绝对。
- 改成“LM objective prevents a degenerate under-updating regime”。
- 这是完全符合你现在真实结果的，而且更不容易被反驳。

---

## 建议在 rebuttal 里如何使用这些表格与图

### 可以强写的点

1. 没有观察到向 0 collapse。
2. 表示轨迹没有显示持续缩到 0。
3. 表示与 metadata 几何高度一致。
4. 去掉 LM objective 后，文本质量明显下降，且表示更贴近初始化几何。

### 需要保守写的点

1. full ATR 明显优于 hard orth。
2. normalized cosine soft orth 明显优于原始设计。
3. LM objective 直接防止“向 0 collapse”。

### 最稳的整体措辞

> In summary, the new evidence rules out an obvious zero-collapse explanation, confirms that the learned task geometry remains strongly aligned with medically meaningful metadata structure, and shows that removing the LM objective leads to markedly worse task performance together with much weaker deviation from initialization. These findings support the necessity of combining representation regularization with the task-learning objective, while we interpret the fine-grained ranking among closely related orthogonality variants with appropriate caution.
"""
    OUT_PATH.write_text(text.strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
