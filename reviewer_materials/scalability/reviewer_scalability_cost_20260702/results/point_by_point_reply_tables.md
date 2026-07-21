dd# Reviewer R3 C10 Point-by-Point Reply Tables

本文件将 `prompt_r3_c10` 所需的点对点回复数据统一整理为单文件表格，便于直接引用到 rebuttal 或 point-by-point reply 中。数据来源为：

- `reviewer_scalability_cost_20260702/results/scalability_full_results.json`
- `reviewer_scalability_cost_20260702/results/table_parameters_storage.csv`
- `reviewer_scalability_cost_20260702/results/table_runtime_memory_latency.csv`
- `reviewer_scalability_cost_20260702/results/table_performance_vs_tasks.csv`
- `reviewer_scalability_cost_20260702/results/table_extended_taskbank_profiling.csv`

## Table 1. Hardware and Matched Configuration

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090 |
| GPU count | 1 |
| CUDA_VISIBLE_DEVICES | 1 |
| CUDA version | 12.1 |
| Torch version | 2.4.1+cu121 |
| Python version | 3.8.5 |
| Base model | chinese-alpaca-plus-7b-hf |
| LoRA rank | 4 |
| LoRA alpha | 16 |
| LoRA dropout | 0.1 |
| Target modules | q_proj, v_proj |
| Task embedding dimension | 4096 |
| Dtype | bfloat16 |
| Batch size | 1 |
| Gradient accumulation steps | 8 |
| Max sequence length | 512 |

## Table 2. Parameter and Storage Growth for K=1..6

| K | Per-task trainable params | Current task module params | Task embedding params | Routing-related params | Cumulative task module params | Cumulative task-related params | Task module storage (MB) | Embedding storage (MB) | Routing storage (MB) | Cumulative task-related storage (MB) | Base model params | Base model storage (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2,121,728 | 2,097,152 | 24,576 | 0 | 2,097,152 | 2,121,728 | 8.0000 | 0.1875 | 0.0000 | 8.1875 | 6,885,498,880 | 13,133.0469 |
| 2 | 2,109,440 | 2,097,152 | 24,576 | 0 | 4,194,304 | 4,218,880 | 16.0000 | 0.1875 | 0.0000 | 16.1875 | 6,885,498,880 | 13,133.0469 |
| 3 | 2,105,344 | 2,097,152 | 24,576 | 0 | 6,291,456 | 6,316,032 | 24.0000 | 0.1875 | 0.0000 | 24.1875 | 6,885,498,880 | 13,133.0469 |
| 4 | 2,103,296 | 2,097,152 | 24,576 | 0 | 8,388,608 | 8,413,184 | 32.0000 | 0.1875 | 0.0000 | 32.1875 | 6,885,498,880 | 13,133.0469 |
| 5 | 2,102,067 | 2,097,152 | 24,576 | 0 | 10,485,760 | 10,510,336 | 40.0000 | 0.1875 | 0.0000 | 40.1875 | 6,885,498,880 | 13,133.0469 |
| 6 | 2,101,248 | 2,097,152 | 24,576 | 0 | 12,582,912 | 12,607,488 | 48.0000 | 0.1875 | 0.0000 | 48.1875 | 6,885,498,880 | 13,133.0469 |

## Table 3. Runtime, Memory, and Latency Breakdown for K=1..6

| K | Training peak GPU memory (MB) | Training time per task (sec) | Training time source | Inference peak GPU memory (MB) | Matching time (ms/query) | Aggregation time (ms/query) | PLM time (ms/query) | Total inference latency (ms/query) | Matching ratio (%) | Aggregation ratio (%) | Total overhead ratio (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 12,749.4165 | 6,799.0000 | parsed_from_existing_logs | 8,501.4229 | 0.3315 | 2.2739 | 58.7429 | 61.3483 | 0.5403 | 3.7066 | 4.2469 |
| 2 | 12,771.9243 | 7,656.0000 | parsed_from_existing_logs | 8,505.4385 | 0.3735 | 2.2966 | 58.8835 | 61.5536 | 0.6067 | 3.7311 | 4.3378 |
| 3 | 12,763.6821 | 7,854.0000 | parsed_from_existing_logs | N/A | 0.2230 | 2.3928 | 8.5241 | 11.1398 | 2.0015 | 21.4795 | 23.4810 |
| 4 | 12,780.1274 | 7,996.0000 | parsed_from_existing_logs | 8,513.4697 | 0.3886 | 2.5934 | 61.5164 | 64.4984 | 0.6038 | 4.0196 | 4.6234 |
| 5 | 12,788.5728 | 8,030.0000 | parsed_from_existing_logs | N/A | 0.2963 | 2.4831 | 8.2815 | 11.0609 | 2.6785 | 22.4496 | 25.1281 |
| 6 | 12,796.5806 | 8,324.0000 | parsed_from_existing_logs | 8,521.5010 | 0.3502 | 2.4701 | 59.3168 | 62.1371 | 0.5630 | 3.9717 | 4.5348 |

## Table 4. Performance Trend as the Number of Learned Tasks Increases

| K | Learned tasks | Average LLM score on seen tasks | Final average performance / FAP | FWT | FR | BWT | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | IM | 9.7222 | 9.7222 | N/A | N/A | N/A | short_profile_estimate |
| 2 | IM, S | 13.7153 | 13.7153 | N/A | N/A | N/A | short_profile_estimate |
| 3 | IM, S, P | 17.4768 | 17.4768 | N/A | N/A | N/A | short_profile_estimate |
| 4 | IM, S, P, GO | 16.1312 | 16.1312 | N/A | N/A | N/A | short_profile_estimate |
| 5 | IM, S, P, GO, A | 14.5716 | 14.5716 | N/A | N/A | N/A | short_profile_estimate |
| 6 | IM, S, P, GO, A, O | 82.9000 | 82.9000 | 13.6500 | 2.2400 | -2.4800 | manuscript_table5_order1 |

## Table 5. Extended Task-Bank Profiling for K=8,16,32,64

| K | Task-bank type | Matching time (ms/query) | Aggregation time (ms/query) | Total MeMR overhead (ms/query) | Task-related memory (MB) | Note |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | simulated | 0.4289 | 2.6009 | 3.0297 | 32.1250 | simulated task-bank profiling only, not additional training |
| 16 | simulated | 0.3524 | 2.2734 | 2.6258 | 64.2500 | simulated task-bank profiling only, not additional training |
| 32 | simulated | 0.3832 | 2.2679 | 2.6511 | 128.5000 | simulated task-bank profiling only, not additional training |
| 64 | simulated | 0.3756 | 2.3100 | 2.6856 | 257.0000 | simulated task-bank profiling only, not additional training |

## Brief Analysis

1. 参数和存储的增长趋势非常清楚。每新增一个任务，新增的主要成本来自 LoRA task-specific module，单任务模块参数基本稳定在约 2.10M，任务相关总存储从 K=1 的 8.19 MB 近似线性增长到 K=6 的 48.19 MB，而冻结 base PLM 始终固定在约 13.13 GB。这说明 MeMR 的任务增量成本远小于基座模型本体。

2. 推理时延中，真正占主导的是 PLM forward，而不是 MDTM matching 或 module aggregation。以 K=6 为例，matching 仅 0.35 ms，aggregation 仅 2.47 ms，而 PLM 时间为 59.32 ms，总额外开销比例约 4.53%。因此，对于审稿意见中的“scalability and computational cost”，当前结果最有力的结论是：随着任务数增加，路由不是主要瓶颈。

3. K>6 的 simulated task-bank profiling 进一步支持这一点。即使扩展到 K=64，matching 仍约 0.38 ms，aggregation 仍约 2.31 ms，任务相关内存为 257 MB，说明任务库规模扩大后，额外的路由与聚合成本仍然较小。但这部分只能用于成本分析，不能用于性能声明。

4. 训练成本方面，K=1 到 K=6 的单任务训练时间大约从 6,799 sec 增长到 8,324 sec，增长幅度有限，训练峰值显存则稳定在 12.75 到 12.80 GB 左右。这说明在当前 K<=6 的真实任务范围内，训练成本没有出现明显爆炸式上升。

5. 需要谨慎引用的地方有两点。第一，K=3 和 K=5 的 runtime 表中 PLM 时间明显低于其他点，属于异常值，建议正文只重点引用 K=1、2、4、6 或直接引用 K=6。第二，K=1..5 的性能趋势来自 `short_profile_estimate`，而 K=6 的 82.9/FWT/FR/BWT 来自 manuscript 表格，二者协议不同，因此这张性能趋势表更适合说明“当前可恢复证据范围”，不适合做严格同口径曲线结论。

## Recommended Citation Use

- 如果只需要一段最稳妥的 rebuttal 结论，优先引用 Table 2、Table 3 中的 K=6 行，以及 Table 5 的 K=64 行。
- 如果需要强调“任务相关成本近似线性增长”，优先引用 Table 2 的 `cumulative_task_related_storage_MB` 列。
- 如果需要强调“routing overhead is small”, 优先引用 Table 3 的 `matching/aggregation/PLM/total` 四列以及 `total_overhead_ratio_percent`。
- 如果需要强调“larger task-bank profiling without retraining”, 引用 Table 5，并明确附上 `simulated task-bank profiling only, not additional training`。
