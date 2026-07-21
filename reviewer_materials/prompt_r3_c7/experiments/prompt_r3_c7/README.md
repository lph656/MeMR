# prompt_r3_c7 Dataset Integrity Audit

This suite is a standalone, read-only audit for the reviewer comment about CMedCL / Cmedia overlap and data-cleaning transparency.

What it does:

- audits CMedCL source splits and validation rules
- checks exact and near-duplicate overlap within CMedCL
- checks overlap against the currently available external medical QA subsets
- exports question / answer length distributions and turn-count statistics
- records unresolved provenance items that are not recoverable from the current snapshot

Outputs:

- `experiments/prompt_r3_c7/outputs/internal/summary.json`
- `experiments/prompt_r3_c7/outputs/chatmed/summary.json`
- `experiments/prompt_r3_c7/outputs/cmedia/summary.json`
- `experiments/prompt_r3_c7/outputs/huatuo/summary.json`
- `experiments/prompt_r3_c7/outputs/aggregated_summary.json`
- `experiments/prompt_r3_c7/outputs/prompt_r3_c7_report.md`
- `experiments/prompt_r3_c7/outputs/reviewer_response_draft_cn.md`

