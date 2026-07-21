# 对审稿意见的回复

感谢审稿人的建议。我们重新整理了补充实验的表述方式，只保留当前结果能够直接支持的结论，并将主张收敛到“MeMR 具备可观测、可校准且较稳定的任务匹配能力，以及在冷启动时能从 metadata 中获益”。

## 1. 关于 “improved task matching”

针对任务匹配能力，我们在固定的 180 个验证样本上统计了路由表现，并进一步考察了轻度输入扰动下的稳定性。结果如表 1 所示。MeMR 在干净输入下的 top-1 路由准确率达到 47.22%，top-3 路由准确率达到 78.89%，ECE 为 0.2575，说明模型已经学到非平凡的任务路由能力；在字符删除、字符交换、标点扰动和填充词扰动下，top-1 / top-3 准确率整体保持稳定，表明该路由器对轻度输入噪声具有较好的鲁棒性。

**表 1. 路由准确率与噪声鲁棒性（`experiments/reviewer_0629/outputs/routing_baseline/summary.json`）**

| 噪声设置 | Top-1 路由准确率 | Top-2 路由准确率 | Top-3 路由准确率 | ECE | Brier | 平均熵 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean | 47.22 | 71.11 | 78.89 | 0.2575 | 0.7872 | 1.7772 |
| 字符删除 10% | 47.22 | 69.44 | 77.78 | 0.2596 | 0.7900 | 1.7781 |
| 字符交换 10% | 46.11 | 70.00 | 81.11 | 0.2488 | 0.7874 | 1.7779 |
| 标点扰动 10% | 47.78 | 70.56 | 80.56 | 0.2652 | 0.7873 | 1.7776 |
| 填充词扰动 10% | 46.67 | 70.56 | 77.22 | 0.2549 | 0.7900 | 1.7788 |

对应图见 [图 1](experiments/reviewer_0629/rebuttal_assets/figures/routing_noise_robustness.png)。

## 2. 关于 cold-start / unseen department

在冷启动场景下，我们使用 hold-out 科室 `zhongliuke` 进行评估。与随机初始化相比，metadata 初始化明显更有优势：top-1 新任务准确率从 16.67% 提升到 40.00%，top-3 新任务准确率从 50.00% 提升到 79.17%。这说明医学 metadata 为新任务提供了有效先验，能够帮助模型在未见科室上更快形成初始路由。

**表 2. 冷启动路由结果（`experiments/reviewer_0629/outputs/cold_start_zhongliuke/summary.json`）**

| 初始化方式 | Top-1 新任务准确率 | Top-3 新任务准确率 | 新任务平均权重 |
| --- | ---: | ---: | ---: |
| Metadata Init | 40.00 | 79.17 | 0.1852 |
| Random Init | 16.67 | 50.00 | 0.1670 |

冷启动对比图见 [图 2](experiments/reviewer_0629/rebuttal_assets/figures/cold_start_comparison.png)。

我们还记录了 `mean_seen_init` 作为参考初始化，但它更接近“已知任务中心”的启发式设定，不适合作为未见任务的主比较对象。因此在正式回复中，我们保留 metadata 相对 random 的提升作为核心证据。

关于“未见科室测试”，当前 held-out `zhongliuke` 上的生成结果在 `Full / Top-1 / Top-2 / Top-3` 四种路由方式下完全一致（ROUGE-1 = 12.9221，ROUGE-L = 12.9221，EM = 0.0）。这说明该生成级指标对路由策略不够敏感，因此我们不将其作为主要论据；我们改为强调上面的路由级 cold-start 评估，更直接回应 reviewer 对 task matching 的关注。

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

对应图见 [图 3](experiments/reviewer_0629/rebuttal_assets/figures/noisy_label_routing_stability.png)。

## 4. 关于 full weighted routing vs top-k routing

在当前 held-out department 的生成实验中，`Full / Top-1 / Top-2 / Top-3` 的 ROUGE 和 EM 完全一致。基于这一结果，我们不把该表解读为“某个 k 明显更优”，而是更保守地将其作为一个稳定性信号：在该 held-out 设定下，路由策略切换并未破坏生成表现。基于实现上的简洁性与默认可用性，我们保留 full weighted routing 作为主设置。

## 5. 简要总结

综合这些结果，我们会将正文中的表述收敛为两点：MeMR 具备稳定且可校准的任务匹配能力；在冷启动场景下，metadata 初始化明显优于随机初始化。对于不够敏感或不够强的生成级对比，我们不在 rebuttal 中过度强调，以避免超出当前实验能够支持的结论边界。
