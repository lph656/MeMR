# MeMR 可扩展性与计算成本回复整理

本文档整理当前仓库中已经具备、且可以用于回复审稿意见 “Scalability and computational cost are not demonstrated.” 的数据。整理原则是只保留当前口径相对可靠、可在 rebuttal 中直接引用或经过轻微改写即可引用的数据；对于口径不完全一致或仅可作为辅助说明的数据，单独标注风险。

数据来源主要为：

- `reviewer_scalability_cost_20260702/results/table_parameters_storage.csv`
- `reviewer_scalability_cost_20260702/results/table_runtime_memory_latency.csv`
- `reviewer_scalability_cost_20260702/results/table_extended_taskbank_profiling.csv`
- `reviewer_scalability_cost_20260702/results/scalability_full_results.json`
- `reviewer_scalability_cost_20260702/results/scalability_summary.md`
- `results/scalability_profiling/20260627_122553/scalability_results.csv`
- `manuscript.pdf` 中 Table 5 的 Order-1 结果

## 1. 可直接用于回复的核心结论

### 1.1 实验环境

| 项目 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090 |
| 使用 GPU 编号 | `CUDA_VISIBLE_DEVICES=1` |
| GPU 数量 | 1 |
| Torch / CUDA | `2.4.1+cu121` / `12.1` |
| Python | `3.8.5` |
| Base model | `chinese-alpaca-plus-7b-hf` |
| LoRA rank | `4` |
| LoRA alpha | `16` |
| LoRA dropout | `0.1` |
| Target modules | `q_proj`, `v_proj` |
| Task embedding dim | `4096` |
| Batch size | `1` |
| Gradient accumulation | `8` |
| Max sequence length | `512` |

### 1.2 参数量与存储开销

这部分数据口径最稳定，适合直接写入 response 或补充材料。

| K | 每任务新增可训练参数 | 当前任务模块参数 | 任务向量参数 | 累计任务相关参数 | 任务模块存储 MB | 嵌入存储 MB | 累计任务相关存储 MB | 冻结基座参数 | 基座存储 MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2,121,728 | 2,097,152 | 24,576 | 2,121,728 | 8.0000 | 0.1875 | 8.1875 | 6,885,498,880 | 13133.0469 |
| 2 | 2,109,440 | 2,097,152 | 24,576 | 4,218,880 | 16.0000 | 0.1875 | 16.1875 | 6,885,498,880 | 13133.0469 |
| 3 | 2,105,344 | 2,097,152 | 24,576 | 6,316,032 | 24.0000 | 0.1875 | 24.1875 | 6,885,498,880 | 13133.0469 |
| 4 | 2,103,296 | 2,097,152 | 24,576 | 8,413,184 | 32.0000 | 0.1875 | 32.1875 | 6,885,498,880 | 13133.0469 |
| 5 | 2,102,067 | 2,097,152 | 24,576 | 10,510,336 | 40.0000 | 0.1875 | 40.1875 | 6,885,498,880 | 13133.0469 |
| 6 | 2,101,248 | 2,097,152 | 24,576 | 12,607,488 | 48.0000 | 0.1875 | 48.1875 | 6,885,498,880 | 13133.0469 |

可直接提炼的结论：

- 基座 PLM 为固定成本，不随任务数增长。
- 任务相关参数和任务相关存储基本随 K 近似线性增长。
- 在 K=6 时，全部任务相关存储仅为 `48.19 MB`，远小于基座模型 `13.13 GB`。

### 1.3 K=1..6 训练时间、训练显存、推理时延

这部分可以用于回复，但建议主打总体趋势和 K=6 代表值；对 K=3 和 K=5 需要避免过度展开，因为其中有缺项和插值项。

| K | 训练峰值显存 MB | 单任务训练时间 sec | 时间来源 | 推理峰值显存 MB | matching ms | aggregation ms | PLM ms | total ms | matching ratio % | aggregation ratio % | overhead ratio % |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12749.42 | 6799 | parsed_from_existing_logs | 8501.42 | 0.3315 | 2.2739 | 58.7429 | 61.3483 | 0.5403 | 3.7066 | 4.2469 |
| 2 | 12771.92 | 7656 | parsed_from_existing_logs | 8505.44 | 0.3735 | 2.2966 | 58.8835 | 61.5536 | 0.6067 | 3.7311 | 4.3378 |
| 3 | 12763.68 | 7854 | parsed_from_existing_logs | N/A | 0.2230 | 2.3928 | 8.5241 | 11.1398 | 2.0015 | 21.4795 | 23.4810 |
| 4 | 12780.13 | 7996 | parsed_from_existing_logs | 8513.47 | 0.3886 | 2.5934 | 61.5164 | 64.4984 | 0.6038 | 4.0196 | 4.6234 |
| 5 | 12788.57 | 8030 | parsed_from_existing_logs | N/A | 0.2963 | 2.4831 | 8.2815 | 11.0609 | 2.6785 | 22.4496 | 25.1281 |
| 6 | 12796.58 | 8324 | parsed_from_existing_logs | 8521.50 | 0.3502 | 2.4701 | 59.3168 | 62.1371 | 0.5630 | 3.9717 | 4.5348 |

当前最适合引用的稳健结论：

- 训练峰值显存从 K=1 到 K=6 基本稳定在 `12.75 GB` 到 `12.80 GB`。
- 单任务训练时间从 `6799 sec` 增长到 `8324 sec`，增长幅度有限。
- 在 K=6 时，推理总时延为 `62.14 ms/query`。
- 在 K=6 时，matching 时延仅 `0.35 ms/query`，aggregation 时延仅 `2.47 ms/query`。
- 在 K=6 时，MeMR matching + aggregation 总开销约占总推理时延 `4.53%`，PLM 前向仍是主要瓶颈。

建议在 response 中优先引用 K=6，并用 K=1/2/4/6 的趋势作支撑，不要重点展开 K=3/5。

### 1.4 K=8,16,32,64 的扩展 task-bank profiling

这部分是当前回应 “可扩展性” 最有力的数据之一，但必须明确说明只是 simulated task-bank profiling，不是额外训练。

| K | task bank 类型 | matching ms | aggregation ms | total MeMR overhead ms | task-related memory MB | 备注 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 8 | simulated | 0.4289 | 2.6009 | 3.0297 | 32.1250 | simulated task-bank profiling only, not additional training |
| 16 | simulated | 0.3524 | 2.2734 | 2.6258 | 64.2500 | simulated task-bank profiling only, not additional training |
| 32 | simulated | 0.3832 | 2.2679 | 2.6511 | 128.5000 | simulated task-bank profiling only, not additional training |
| 64 | simulated | 0.3756 | 2.3100 | 2.6856 | 257.0000 | simulated task-bank profiling only, not additional training |

可直接提炼的结论：

- task-related memory 随 K 几乎严格线性增长。
- 即使扩展到 K=64，matching 和 aggregation 开销仍维持在毫秒级。
- 在测试范围内，任务匹配本身并不是主要运行时瓶颈。

### 1.5 可用于回复的性能数据

当前性能趋势数据只能部分使用。

可靠、可用的数据：

| K | learned tasks | average | FWT | FR | BWT | 来源 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 6 | IM,S,P,GO,A,O | 82.90 | 13.65 | 2.24 | -2.48 | manuscript Table 5, order1 |

这组 K=6 结果可以直接用于说明：

- 在完整 6 任务设置下，MeMR 不仅给出成本数据，还保持了较好的持续学习性能。
- 成本分析不是脱离性能单独汇报，而是和最终任务数下的效果共同报告。

## 2. 可以辅助使用、但需要明确保留条件的数据

### 2.1 K=1..5 的“performance_vs_tasks”不能当成严格 LLM score

当前文件 `table_performance_vs_tasks.csv` 中 K=1..5 的 `average_llm_score_on_seen_tasks` 实际上不是 LLM-based score，而是脚本通过轻量 dev evaluation 得到的 `rougeL` 估计值。

因此，这部分最多只能作为：

- “stage-wise lightweight proxy trend”；
- 或 “rough seen-task quality proxy”；

不能直接表述为：

- “LLM score on seen tasks”.

当前值如下，仅供内部参考：

| K | learned tasks | 当前字段值 | 实际口径 | source |
| --- | --- | ---: | --- | --- |
| 1 | IM | 9.7222 | RougeL proxy | short_profile_estimate |
| 2 | IM,S | 13.7153 | RougeL proxy | short_profile_estimate |
| 3 | IM,S,P | 17.4768 | RougeL proxy | short_profile_estimate |
| 4 | IM,S,P,GO | 16.1312 | RougeL proxy | short_profile_estimate |
| 5 | IM,S,P,GO,A | 14.5716 | RougeL proxy | short_profile_estimate |

### 2.2 K=1..5 的 FWT / FR / BWT 当前缺失

当前 K=1..5 没有完整 stage matrix，因此：

- FWT 缺失；
- FR 缺失；
- BWT 缺失。

如果回复时必须提及这一点，建议直说：

> Intermediate stage-wise continual-learning matrices were not fully preserved in the archived workspace, so we report the final K=6 continual-learning metrics together with the computational-cost analysis, and mark intermediate FWT/FR/BWT as unavailable without full rerunning.

### 2.3 K=3 和 K=5 的 latency 数据不建议单独引用

原因：

- `inference_peak_gpu_memory_MB` 缺失；
- aggregation time 为 archived grid 插值；
- PLM time 和 total latency 与 K=1/2/4/6 的量级不一致。

因此，K=3 和 K=5 更适合作为内部存档，而不适合在 rebuttal 中被单独展开。

## 3. 建议在审稿回复中主打的表述重点

如果只基于当前已有数据，最稳妥的回复重点应是：

1. MeMR 的任务相关参数和存储开销随任务数近似线性增长，但增量很小。
2. 在 K=6 时，全部任务相关存储仅约 `48 MB`，远低于 `13.13 GB` 的冻结基座模型。
3. 训练峰值显存约 `12.8 GB`，表明该方法在单卡 RTX 4090 上可以稳定运行。
4. 推理时延主要由 PLM forward 主导；matching 和 aggregation 的合计开销在 K=6 时仅约 `4.53%`。
5. 在 simulated task-bank profiling 中，即使扩展到 K=64，matching 与 aggregation 仍保持毫秒级。
6. K>6 的结果仅用于成本 profiling，不对应新增训练，也不对应新增性能声明。

## 4. 我的简要分析

从当前数据看，MeMR 对审稿人“缺少 scalability / cost demonstration”的回应已经有基本说服力，尤其是在以下三点上：

- 第一，参数与存储结果非常清楚，能够直接说明该方法的任务扩展成本是轻量的。
- 第二，推理时延拆分已经表明 matching 和 aggregation 不是主要瓶颈，真正的主要计算开销仍来自底层 PLM，这一点非常适合回应审稿人对 runtime overhead 的担忧。
- 第三，K=8/16/32/64 的 simulated task-bank profiling 进一步增强了“方法可扩展”的论据，即使它不是额外训练实验，也足以说明路由与模块组合机制本身不会迅速失控。

但如果要提高回复的严谨性，当前仍有两个明显弱点：

- `performance_vs_tasks` 中 K=1..5 的指标名和真实口径不一致，不能直接当作 LLM score 使用；
- K=1..5 的 FWT/FR/BWT 缺失，因此“性能随任务数增长的完整趋势”这一点还不能强写。

因此，基于当前结果，最佳策略不是强调“我们已经完整补齐了所有 scalability + performance trend 数据”，而是强调：

- 我们已经补充了完整的参数、存储、显存、训练时间、推理时延和扩展 task-bank profiling；
- 同时报告了完整 K=6 设置下的最终 continual-learning 效果；
- 并明确说明 K>6 仅作成本分析，不引入额外训练或额外性能声明。

这会比强行使用口径不一致的中间性能表更稳妥。

## 5. 可直接改写使用的英文短段落

Following the reviewer’s suggestion, we added a dedicated scalability and computational-cost analysis for MeMR. We report the trainable parameters per task, cumulative task-related storage, GPU memory usage, training time, and inference latency breakdown. The results show that the task-related storage grows approximately linearly with the number of learned tasks while remaining small in absolute size (about 48 MB at K=6, excluding the frozen base PLM). Moreover, the matching and module-aggregation overhead remains minor compared with the PLM forward computation (about 4.53% of the total inference latency at K=6). We further conduct a simulated task-bank profiling experiment for K=8,16,32,64 without additional training, which shows that the task-matching overhead remains at the millisecond level under the tested range.
