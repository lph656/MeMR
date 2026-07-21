# prompt_r3_c7 Dataset Integrity Audit

## CMedCL Internal Audit
- total records: 33312
- question length median: 62.0000
- answer length median: 162.0000
- turn count: 1.0000
- validation rule: train_test_split(test_size=validation_split_percentage, seed=training_args.seed) inside tasks/mtl5/dataloader_mtl_causal_llama.py
- department order: neike, waike, erke, fuchanke, nanke, zhongliuke

## External Overlap Checks
### ChatMed_Consult-v0.3_test_500
- exact question overlaps: 0
- near-duplicate candidate hits: 0

### Chinese-medical-dialogue-data_test_500
- exact question overlaps: 0
- near-duplicate candidate hits: 0

### huatuo26M_test_500
- exact question overlaps: 0
- near-duplicate candidate hits: 0

## Unresolved Snapshot Items
- complete data-cleaning prompt: No cleaning prompt is embedded in the current repository snapshot.
- LLM used for cleaning: The snapshot does not record the exact model name/version used for the data-cleaning pass.
- temperature / decoding settings: No cleaning-time decoding hyperparameters are stored in the available repo files.
- human validation protocol: No validation checklist or annotation protocol is available in the snapshot.
- dataset-specific release license: The project license exists, but dataset-specific release terms are not stated in the current files.

