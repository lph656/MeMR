Scalability plotting summary

- K=6: matching=0.350 ms, aggregation=2.470 ms, total overhead=2.820 ms, matching ratio=0.563%, aggregation ratio=3.972%, task-related memory=24.094 MB.
- K=64: matching=0.376 ms, aggregation=2.310 ms, total overhead=2.686 ms, matching ratio=0.611%, aggregation ratio=3.755%, task-related memory=257.000 MB.

- Bottleneck threshold check (10.0%): neither matching nor aggregation exceeds the threshold.
- Aggregation larger than matching: yes.
- Task-related memory dominated by frozen modules: yes.
- Interpretation note: The theoretical complexity grows linearly with the number of tasks, while the empirical profiling results show that the measured overhead remains stable within the tested range up to 64 tasks.
- Warning: Do not claim that empirical latency increases strictly linearly with K unless a separate measurement clearly demonstrates that trend.
