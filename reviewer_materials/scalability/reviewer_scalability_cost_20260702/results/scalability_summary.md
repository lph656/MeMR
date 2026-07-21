# Scalability Summary

## 1. Experiment purpose
This analysis supplements the reviewer concern regarding scalability and computational cost. It reports trainable parameter growth, cumulative storage, GPU memory, training time, inference latency breakdown, and task-number-dependent trends while avoiding full retraining.

## 2. Hardware and GPU
- GPU: NVIDIA GeForce RTX 4090
- CUDA_VISIBLE_DEVICES: `1`
- Torch/CUDA: 2.4.1+cu121 / 12.1

## 3. Model configuration
- Base model: `chinese-alpaca-plus-7b-hf`
- LoRA: r=4, alpha=16, dropout=0.1
- Target modules: q_proj, v_proj
- Task embedding dimension: 4096
- Batch size / grad accumulation / max seq: 1 / 8 / 512

## 4. Parameter and storage conclusions
- At K=6, cumulative task-related parameters are 12,607,488.
- At K=6, task-related storage is 48.1875 MB.
- Frozen base PLM storage is 13133.0469 MB and does not scale with K.
- Task-related storage grows approximately linearly with task count under the archived and simulated task-bank settings.

## 5. Training / inference memory and time conclusions
- K=6 training time per task (archived log source): 8324.0000 sec.
- K=6 training peak GPU memory (short profile): 12796.5806 MB.
- K=6 inference peak GPU memory: 8521.5010 MB.
- K=6 latency breakdown: matching 0.3502 ms/query, aggregation 2.4701 ms/query, PLM 59.3168 ms/query, total 62.1371 ms/query.

## 6. K=1..6 performance trend
- K=6 final continual-learning metrics are recovered from the manuscript order-1 table.
- K=1..5 average seen-task performance is estimated by lightweight dev evaluation on archived stage snapshots.
- K=1..5 FWT/FR/BWT remain unavailable because the workspace does not preserve the full archived stage matrix for MeMR.

## 7. K=8,16,32,64 extended task-bank profiling
- At K=64, simulated task-related memory is 257.0000 MB.
- At K=64, simulated matching time is 0.3756 ms/query and aggregation time is 2.3100 ms/query.
- These K>6 results are simulated task-bank profiling only and do not involve new training or new accuracy claims.

## 8. Clarification
- K>6 only used for cost profiling.
- No additional training was performed for K>6.
- No additional performance claim is made for K>6.

## 9. Notes
- K=1: training peak memory and step time measured via short_profile_estimate over <= 8 mini-batches; no checkpoint saved.
- K=2: training peak memory and step time measured via short_profile_estimate over <= 8 mini-batches; no checkpoint saved.
- K=3: aggregation_time derived by linear interpolation/regression from archived K grid because pure live aggregation hook is not directly exposed.
- K=3: training peak memory and step time measured via short_profile_estimate over <= 8 mini-batches; no checkpoint saved.
- K=4: training peak memory and step time measured via short_profile_estimate over <= 8 mini-batches; no checkpoint saved.
- K=5: aggregation_time derived by linear interpolation/regression from archived K grid because pure live aggregation hook is not directly exposed.
- K=5: training peak memory and step time measured via short_profile_estimate over <= 8 mini-batches; no checkpoint saved.
- K=6: training peak memory and step time measured via short_profile_estimate over <= 8 mini-batches; no checkpoint saved.
- K=1 eval_loader_k1: query_encoder loaded via fallback mode `cpu_fp32_fallback` instead of the default 4-bit auto placement.
- K=1: average seen-task performance estimated via lightweight dev evaluation over up to 8 batches/task; source marked short_profile_estimate.
- K=2: average seen-task performance estimated via lightweight dev evaluation over up to 8 batches/task; source marked short_profile_estimate.
- K=3: average seen-task performance estimated via lightweight dev evaluation over up to 8 batches/task; source marked short_profile_estimate.
- K=4: average seen-task performance estimated via lightweight dev evaluation over up to 8 batches/task; source marked short_profile_estimate.
- K=5: average seen-task performance estimated via lightweight dev evaluation over up to 8 batches/task; source marked short_profile_estimate.
- Stagewise K=1..5 FWT/FR/BWT are still unavailable without archived stage matrices or full rerun; only final K=6 order-1 CL metrics are recovered from the manuscript.

## 10. Rebuttal paragraph

Following the reviewer’s suggestion, we added a scalability and computational-cost analysis. We reported the trainable parameters per task, cumulative task-related storage, GPU memory consumption, training time, inference latency, and performance trend as the number of learned tasks increases. The results show that the task-related storage grows approximately linearly with the number of tasks, while the matching and aggregation overhead remains small compared with the PLM forward computation. In addition, we performed a simulated task-bank profiling experiment for K=8,16,32,64 without additional training, which further shows that task matching is not the main runtime bottleneck under the tested range.
