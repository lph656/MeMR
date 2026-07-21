# prompt_r3_c2 Gate Probe Suite

This suite is a standalone, non-invasive set of gate probing experiments for the reviewer comment about MDTM.

Variants:

- `current`: existing sigmoid(cosine) gate
- `metadata_only`: alpha fixed to 1
- `temperature`: trainable scalar temperature + bias
- `learnable_mlp`: small learnable gate network

Outputs:

- `experiments/prompt_r3_c2/outputs/<variant>/<eval_name>/summary.json`
- `experiments/prompt_r3_c2/outputs/<variant>/<eval_name>/per_sample.jsonl`
- `experiments/prompt_r3_c2/outputs/<variant>/gate_state.pt` for trained variants
- `experiments/prompt_r3_c2/outputs/aggregated_summary.json`
- `experiments/prompt_r3_c2/outputs/prompt_r3_c2_report.md`
- `experiments/prompt_r3_c2/rebuttal_assets/figures/*.png`

