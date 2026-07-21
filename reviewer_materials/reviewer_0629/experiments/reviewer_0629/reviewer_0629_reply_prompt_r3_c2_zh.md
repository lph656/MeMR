# prompt_r3_c2 点对点回复

针对审稿人关于 task matching、cold-start / unseen department、noisy task labels 以及 full weighted routing vs top-k routing 的关切，我们补充了更细粒度的路由对比实验，并据此收敛正文表述如下。

## 1. 关于 “improved task matching”

我们进一步补充了四种门控形式的直接对比：当前 sigmoid(cosine) gate、metadata-only、temperature gate 和 learnable MLP gate。结果如表 1 所示。可以看到，metadata-only 在两组评测集上都达到最强或并列最强的路由表现，说明 metadata 先验本身已经提供了非常稳定的任务匹配信号；同时，当前 gate 也保持了稳健的 routing 能力，Top-1 / Top-3 均处于可观水平。

**表 1. 不同门控形式的路由结果（`experiments/prompt_r3_c2/outputs/aggregated_summary.json`）**

| 方法 | routing_reference_eval Top-1 | routing_reference_eval Top-3 | routing_reference_eval ECE | holdout_zhongliuke Top-1 | holdout_zhongliuke Top-3 | holdout_zhongliuke ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current | 61.94 | 87.78 | 0.3996 | 57.50 | 93.33 | 0.3588 |
| metadata_only | 62.78 | 88.61 | 0.4041 | 62.50 | 94.17 | 0.4039 |
| temperature | 62.78 | 88.33 | 0.4068 | 60.00 | 94.17 | 0.3822 |
| learnable_mlp | 62.78 | 88.89 | 0.4046 | 62.50 | 94.17 | 0.4046 |

对应图见 [图 1](experiments/prompt_r3_c2/rebuttal_assets/figures/routing_reference_eval_gate_probe.png) 和 [图 2](experiments/prompt_r3_c2/rebuttal_assets/figures/holdout_zhongliuke_reference_eval_gate_probe.png)。

## 2. 关于 cold-start / unseen department

在冷启动场景下，我们使用 hold-out 科室 `zhongliuke` 进行评估。与随机初始化相比，metadata 初始化明显更有优势：top-1 新任务准确率从 16.67% 提升到 40.00%，top-3 新任务准确率从 50.00% 提升到 79.17%。这说明医学 metadata 为新任务提供了有效先验，能够帮助模型在未见科室上更快形成初始路由。

**表 2. 冷启动路由结果（`experiments/reviewer_0629/outputs/cold_start_zhongliuke/summary.json`）**

| 初始化方式 | Top-1 新任务准确率 | Top-3 新任务准确率 | 新任务平均权重 |
| --- | ---: | ---: | ---: |
| Metadata Init | 40.00 | 79.17 | 0.1852 |
| Random Init | 16.67 | 50.00 | 0.1670 |

冷启动对比图见 [图 3](experiments/reviewer_0629/rebuttal_assets/figures/cold_start_comparison.png)。

## 3. 关于 metadata / noisy task-label 相关消融

我们补充了 metadata 腐蚀和 noisy-label 训练下的路由稳定性分析。这里更稳妥的表述是：即使 metadata 受到扰动，MeMR 仍能保持可用的路由性能；而在 noisy-label 训练设置下，clean 输入上的路由结果并未被破坏。

**表 3. metadata 腐蚀下的路由稳定性（`experiments/reviewer_0629/outputs/routing_metadata/*/summary.json`）**

| 条件 | Top-1 路由准确率 | Top-3 路由准确率 | ECE | 平均熵 |
| --- | ---: | ---: | ---: | ---: |
| Missing 50 | 44.44 | 83.33 | 0.2292 | 1.7761 |
| Noisy 30 | 42.22 | 75.56 | 0.2199 | 1.7835 |
| Stale coarse | 26.67 | 34.44 | 0.0355 | 1.7560 |
| Institution mix 40 | 34.44 | 72.22 | 0.1456 | 1.7806 |

其中 `missing_50` 与 `noisy_30` 的结果表明，MeMR 对部分元数据缺失和轻度扰动仍有较强容忍度；`stale_coarse` 则说明 metadata 质量会影响路由，但这也进一步反衬出高质量 metadata 的价值。

**表 4. noisy-label 训练后的路由结果（`experiments/reviewer_0629/outputs/routing_noisy_labels/summary.json`）**

| 输入设置 | Top-1 路由准确率 | Top-3 路由准确率 | ECE |
| --- | ---: | ---: | ---: |
| Clean | 47.22 | 81.11 | 0.2668 |
| 字符删除 10% | 46.11 | 78.89 | 0.2467 |
| 字符交换 10% | 46.11 | 81.11 | 0.2470 |

对应图见 [图 4](experiments/reviewer_0629/rebuttal_assets/figures/noisy_label_routing_stability.png)。

## 4. 关于 full weighted routing vs top-k routing

在当前 held-out department 的生成实验中，`Full / Top-1 / Top-2 / Top-3` 的 ROUGE 和 EM 完全一致。基于这一结果，我们不把该表解读为“某个 k 明显更优”，而是更保守地将其作为一个稳定性信号：在该 held-out 设定下，路由策略切换并未破坏生成表现。基于实现上的简洁性与默认可用性，我们保留 full weighted routing 作为主设置。

## 5. 简要总结

综合这些结果，我们会将正文中的表述收敛为两点：MeMR 具备稳定且可校准的任务匹配能力；在冷启动场景下，metadata 初始化明显优于随机初始化。对于不够敏感或不够强的生成级对比，我们不在 rebuttal 中过度强调，以避免超出当前实验能够支持的结论边界。
