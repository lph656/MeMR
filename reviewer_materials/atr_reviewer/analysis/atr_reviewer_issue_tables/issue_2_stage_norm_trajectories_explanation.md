# Issue 2：跨阶段表示轨迹是否显示出持续塌缩？

结论：目前结果可以部分支持“没有观察到明显的 collapse 轨迹”。

使用证据：
- `issue_2_stage_norm_trajectories_table.csv`
- 各变体目录下 `atr_reviewer_eval/figures/` 中的 PCA 轨迹图

解释：
- 在当前所有可用变体中，随着任务阶段推进，stage-wise mean norm 基本保持稳定，并没有随着任务累积而逐步逼近 0。
- 这可以支持一个 reviewer-facing 的结论：表示轨迹没有显示出单调收缩到共同零点的趋势。

局限性：
- 当前这些跨阶段轨迹可以作为“没有明显 collapse”的证据。
- 但它们本身还不足以证明某一种 ATR 设计显著优于另一种设计。
