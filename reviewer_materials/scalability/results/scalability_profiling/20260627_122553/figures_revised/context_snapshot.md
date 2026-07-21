# Context Snapshot

## config.json
```json
{
  "K_list": [
    1,
    2,
    4,
    6,
    8,
    16,
    32,
    64
  ],
  "seed": 42,
  "device": "cuda:0",
  "warmup_steps": 10,
  "profile_steps": 50,
  "num_repeats": 3,
  "batch_size": 1,
  "dtype": "float16",
  "checkpoint_path": "/home/THJ1/Taohj/Liph/Continual-Learning/MoCL-NAACL-huatuo-main-v3/checkpoints_continual_keshi_llama/order1_compose_peft/snapshots/task_5_zhongliuke_train_end_20260626_123739",
  "base_model_path": "chinese-alpaca-plus-7b-hf",
  "meta_embeddings_path": "metadata_embeddings/keshi_meta_embeddings.pt",
  "measure_plm_forward": true,
  "precompute_query_embeddings": false,
  "perturb_scale": 0.0001,
  "max_query_length": 256,
  "task_bank_expansion_strategy": "Use real tasks when K <= num_real_tasks; otherwise synthesize independent task entries by deterministic interpolation plus small perturbation."
}
```

## task_bank_generation.json
```json
{
  "real_num_tasks": 6,
  "synthetic_num_tasks": 58,
  "generation_rule": "Deterministic interpolation between two real tasks plus Gaussian perturbation; used only for profiling, not QA accuracy evaluation.",
  "interpolation_alphas": [
    0.25,
    0.5,
    0.75
  ],
  "perturbation_scale": 0.0001,
  "random_seed": 42,
  "synthetic_tasks": [
    {
      "task_id": "synthetic_006",
      "source_a": "neike",
      "source_b": "waike",
      "alpha": 0.25,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_007",
      "source_a": "waike",
      "source_b": "erke",
      "alpha": 0.5,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_008",
      "source_a": "erke",
      "source_b": "fuchanke",
      "alpha": 0.75,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_009",
      "source_a": "fuchanke",
      "source_b": "nanke",
      "alpha": 0.25,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_010",
      "source_a": "nanke",
      "source_b": "zhongliuke",
      "alpha": 0.5,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_011",
      "source_a": "zhongliuke",
      "source_b": "neike",
      "alpha": 0.75,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_012",
      "source_a": "neike",
      "source_b": "waike",
      "alpha": 0.25,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_013",
      "source_a": "waike",
      "source_b": "erke",
      "alpha": 0.5,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_014",
      "source_a": "erke",
      "source_b": "fuchanke",
      "alpha": 0.75,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_015",
      "source_a": "fuchanke",
      "source_b": "nanke",
      "alpha": 0.25,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_016",
      "source_a": "nanke",
      "source_b": "zhongliuke",
      "alpha": 0.5,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_017",
      "source_a": "zhongliuke",
      "source_b": "neike",
      "alpha": 0.75,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_018",
      "source_a": "neike",
      "source_b": "waike",
      "alpha": 0.25,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_019",
      "source_a": "waike",
      "source_b": "erke",
      "alpha": 0.5,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_020",
      "source_a": "erke",
      "source_b": "fuchanke",
      "alpha": 0.75,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_021",
      "source_a": "fuchanke",
      "source_b": "nanke",
      "alpha": 0.25,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_022",
      "source_a": "nanke",
      "source_b": "zhongliuke",
      "alpha": 0.5,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_023",
      "source_a": "zhongliuke",
      "source_b": "neike",
      "alpha": 0.75,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_024",
      "source_a": "neike",
      "source_b": "waike",
      "alpha": 0.25,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_025",
      "source_a": "waike",
      "source_b": "erke",
      "alpha": 0.5,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_026",
      "source_a": "erke",
      "source_b": "fuchanke",
      "alpha": 0.75,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_027",
      "source_a": "fuchanke",
      "source_b": "nanke",
      "alpha": 0.25,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_028",
      "source_a": "nanke",
      "source_b": "zhongliuke",
      "alpha": 0.5,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_029",
      "source_a": "zhongliuke",
      "source_b": "neike",
      "alpha": 0.75,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_030",
      "source_a": "neike",
      "source_b": "waike",
      "alpha": 0.25,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_031",
      "source_a": "waike",
      "source_b": "erke",
      "alpha": 0.5,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_032",
      "source_a": "erke",
      "source_b": "fuchanke",
      "alpha": 0.75,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_033",
      "source_a": "fuchanke",
      "source_b": "nanke",
      "alpha": 0.25,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_034",
      "source_a": "nanke",
      "source_b": "zhongliuke",
      "alpha": 0.5,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_035",
      "source_a": "zhongliuke",
      "source_b": "neike",
      "alpha": 0.75,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_036",
      "source_a": "neike",
      "source_b": "waike",
      "alpha": 0.25,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_037",
      "source_a": "waike",
      "source_b": "erke",
      "alpha": 0.5,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_038",
      "source_a": "erke",
      "source_b": "fuchanke",
      "alpha": 0.75,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_039",
      "source_a": "fuchanke",
      "source_b": "nanke",
      "alpha": 0.25,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_040",
      "source_a": "nanke",
      "source_b": "zhongliuke",
      "alpha": 0.5,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_041",
      "source_a": "zhongliuke",
      "source_b": "neike",
      "alpha": 0.75,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_042",
      "source_a": "neike",
      "source_b": "waike",
      "alpha": 0.25,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_043",
      "source_a": "waike",
      "source_b": "erke",
      "alpha": 0.5,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_044",
      "source_a": "erke",
      "source_b": "fuchanke",
      "alpha": 0.75,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_045",
      "source_a": "fuchanke",
      "source_b": "nanke",
      "alpha": 0.25,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_046",
      "source_a": "nanke",
      "source_b": "zhongliuke",
      "alpha": 0.5,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_047",
      "source_a": "zhongliuke",
      "source_b": "neike",
      "alpha": 0.75,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_048",
      "source_a": "neike",
      "source_b": "waike",
      "alpha": 0.25,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_049",
      "source_a": "waike",
      "source_b": "erke",
      "alpha": 0.5,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_050",
      "source_a": "erke",
      "source_b": "fuchanke",
      "alpha": 0.75,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_051",
      "source_a": "fuchanke",
      "source_b": "nanke",
      "alpha": 0.25,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_052",
      "source_a": "nanke",
      "source_b": "zhongliuke",
      "alpha": 0.5,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_053",
      "source_a": "zhongliuke",
      "source_b": "neike",
      "alpha": 0.75,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_054",
      "source_a": "neike",
      "source_b": "waike",
      "alpha": 0.25,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_055",
      "source_a": "waike",
      "source_b": "erke",
      "alpha": 0.5,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_056",
      "source_a": "erke",
      "source_b": "fuchanke",
      "alpha": 0.75,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_057",
      "source_a": "fuchanke",
      "source_b": "nanke",
      "alpha": 0.25,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_058",
      "source_a": "nanke",
      "source_b": "zhongliuke",
      "alpha": 0.5,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_059",
      "source_a": "zhongliuke",
      "source_b": "neike",
      "alpha": 0.75,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_060",
      "source_a": "neike",
      "source_b": "waike",
      "alpha": 0.25,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_061",
      "source_a": "waike",
      "source_b": "erke",
      "alpha": 0.5,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_062",
      "source_a": "erke",
      "source_b": "fuchanke",
      "alpha": 0.75,
      "perturbation_scale": 0.0001
    },
    {
      "task_id": "synthetic_063",
      "source_a": "fuchanke",
      "source_b": "nanke",
      "alpha": 0.25,
      "perturbation_scale": 0.0001
    }
  ]
}
```

## summary_for_response.md
This experiment is profiling-only and does not retrain additional tasks. We load an existing MeMR checkpoint, reuse the learned task representations and frozen task-specific modules, and isolate the runtime and memory overhead introduced by task matching and module aggregation as the task bank grows.

For K larger than the 6 real tasks available in the checkpoint, we construct an independent synthetic task bank by deterministically interpolating pairs of real metadata embeddings, task embeddings, and frozen module tensors, then adding a small fixed-seed perturbation. This synthetic expansion is used only for scalability profiling and not for QA accuracy evaluation.

The measured complexity is consistent with the expected scaling behavior: task matching follows O(Kd), module aggregation follows O(K|P|), and task-related memory follows O(Kd + K|P|). Because the frozen module tensor bank is much larger than the task-representation bank, aggregation and task-related memory are the main quantities that can become dominant as K increases.

In the current profiling run based on `/home/THJ1/Taohj/Liph/Continual-Learning/MoCL-NAACL-huatuo-main-v3/checkpoints_continual_keshi_llama/order1_compose_peft/snapshots/task_5_zhongliuke_train_end_20260626_123739`, the six-task setting remains computationally manageable: total MeMR overhead is 2.820 ms/query at K=6, with aggregation larger than the other MeMR component. At the largest measured scale K=64, task-related memory reaches 257.000 MB and peak GPU allocated memory reaches 8784.220 MB. Under the measured settings, neither matching nor aggregation exceeded the 10% practical bottleneck criterion of total inference latency.

Potential future improvements include sparse top-r routing, module pruning, module merging, hierarchical task indexing, and more strongly shared low-rank module design. PLM forward profiling was successfully included using a fixed prefill-only forward pass.
