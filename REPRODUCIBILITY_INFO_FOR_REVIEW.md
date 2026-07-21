# Reproducibility Information for Reviewer

## 1. Code Release Version

* Repository: Local repository copy; upstream URL unavailable from current workspace snapshot
* Branch: TBD (local .git metadata unavailable in this workspace snapshot)
* Commit hash: TBD (local .git metadata unavailable in this workspace snapshot)
* Tag / release: No formal release tag found
* Commit date: TBD
* Working tree status: TBD (local .git metadata unavailable in this workspace snapshot)
* Files with uncommitted changes: TBD (cannot enumerate without git metadata)
* Code version description: Code version: local repository snapshot of MeMR / MoCL-NAACL-huatuo-main-v3, exact Git commit and branch TBD because the workspace snapshot does not contain usable .git metadata.

## 2. Data Release Version

Dataset | Source | Version / Commit / Download Date | Local Path | Split | Number of Samples | Preprocessing Script | SHA256 / Notes
--- | --- | --- | --- | --- | --- | --- | ---
CMedCL (neike) | TBD | TBD | datasets/medical_consult/neike/train.json | train | 5238 | tasks/mtl5/dataloader_mtl_causal_llama.py | 193fa0565892ae528dc1d419e2f52f00285ba91db68fa8c0c3208ef9531171ce
CMedCL (neike) | TBD | TBD | datasets/medical_consult/neike/train.json | validation | Script-specified 10% split from train with seed 0; exact realized count TBD in current runtime | tasks/mtl5/dataloader_mtl_causal_llama.py | Generated on the fly via train_test_split(test_size=0.1, seed=0)
CMedCL (neike) | TBD | TBD | datasets/medical_consult/neike/test.json | test | 309 | metadata_robustness_experiments/inference_evaluation/build_pilot50_subset.py (for pilot subset only) | 8692a124d845383f9e4c730318bba895d89fcb5dc594e698a1efdfb5d4342c7e; test split contains questions only in the current local files
CMedCL (waike) | TBD | TBD | datasets/medical_consult/waike/train.json | train | 5239 | tasks/mtl5/dataloader_mtl_causal_llama.py | c818eda0d6d863ad3c217984a9d508ccf7bcb5e25bf6b3fc331d66bc1be85f5e
CMedCL (waike) | TBD | TBD | datasets/medical_consult/waike/train.json | validation | Script-specified 10% split from train with seed 0; exact realized count TBD in current runtime | tasks/mtl5/dataloader_mtl_causal_llama.py | Generated on the fly via train_test_split(test_size=0.1, seed=0)
CMedCL (waike) | TBD | TBD | datasets/medical_consult/waike/test.json | test | 303 | metadata_robustness_experiments/inference_evaluation/build_pilot50_subset.py (for pilot subset only) | 356dd623458d6443152bfc1acbef0267d5b1d84dc71d02cc1640be6435ab6cb9; test split contains questions only in the current local files
CMedCL (erke) | TBD | TBD | datasets/medical_consult/erke/train.json | train | 5234 | tasks/mtl5/dataloader_mtl_causal_llama.py | 1bd65342f6c71f28aa1646ed6662afe8108d3277d52a5c756da7c526aee12604
CMedCL (erke) | TBD | TBD | datasets/medical_consult/erke/train.json | validation | Script-specified 10% split from train with seed 0; exact realized count TBD in current runtime | tasks/mtl5/dataloader_mtl_causal_llama.py | Generated on the fly via train_test_split(test_size=0.1, seed=0)
CMedCL (erke) | TBD | TBD | datasets/medical_consult/erke/test.json | test | 314 | metadata_robustness_experiments/inference_evaluation/build_pilot50_subset.py (for pilot subset only) | 3123ade9e9f570b1f0e37519216c4c7f8cb74a04d29511c09d0d0146afb0c1cb; test split contains questions only in the current local files
CMedCL (fuchanke) | TBD | TBD | datasets/medical_consult/fuchanke/train.json | train | 5237 | tasks/mtl5/dataloader_mtl_causal_llama.py | 648b5716b7a9eb9394cc59c68287f9f5d7d35f0154b8e55bcd5870c7131cf3d6
CMedCL (fuchanke) | TBD | TBD | datasets/medical_consult/fuchanke/train.json | validation | Script-specified 10% split from train with seed 0; exact realized count TBD in current runtime | tasks/mtl5/dataloader_mtl_causal_llama.py | Generated on the fly via train_test_split(test_size=0.1, seed=0)
CMedCL (fuchanke) | TBD | TBD | datasets/medical_consult/fuchanke/test.json | test | 362 | metadata_robustness_experiments/inference_evaluation/build_pilot50_subset.py (for pilot subset only) | acc2f0708d3fcf5a7832747e117f097d93e330ebe830bd075b81253a68107dca; test split contains questions only in the current local files
CMedCL (nanke) | TBD | TBD | datasets/medical_consult/nanke/train.json | train | 5178 | tasks/mtl5/dataloader_mtl_causal_llama.py | c11c4d7a99c7bb8917c28d701bd5fded1eab1089fedc28533fee62a3aaf89c78
CMedCL (nanke) | TBD | TBD | datasets/medical_consult/nanke/train.json | validation | Script-specified 10% split from train with seed 0; exact realized count TBD in current runtime | tasks/mtl5/dataloader_mtl_causal_llama.py | Generated on the fly via train_test_split(test_size=0.1, seed=0)
CMedCL (nanke) | TBD | TBD | datasets/medical_consult/nanke/test.json | test | 341 | metadata_robustness_experiments/inference_evaluation/build_pilot50_subset.py (for pilot subset only) | 536c01a792415379b33b443a206b468ac1393b420558ae3b793304e262d0122e; test split contains questions only in the current local files
CMedCL (zhongliuke) | TBD | TBD | datasets/medical_consult/zhongliuke/train.json | train | 5242 | tasks/mtl5/dataloader_mtl_causal_llama.py | 61f809da4c01bc2227741ba2b0d951b812c69d6100fc48cf6849c9f34f3829be
CMedCL (zhongliuke) | TBD | TBD | datasets/medical_consult/zhongliuke/train.json | validation | Script-specified 10% split from train with seed 0; exact realized count TBD in current runtime | tasks/mtl5/dataloader_mtl_causal_llama.py | Generated on the fly via train_test_split(test_size=0.1, seed=0)
CMedCL (zhongliuke) | TBD | TBD | datasets/medical_consult/zhongliuke/test.json | test | 315 | metadata_robustness_experiments/inference_evaluation/build_pilot50_subset.py (for pilot subset only) | 52352bae3552f333c15b2aad3573a10cae0204825a4a39023f1fdcb5932c7fd8; test split contains questions only in the current local files
Medical knowledge base for RAG | TBD | TBD | datasets/knowledge_base/normalized_knowledge_base.json | full | entities=5991 | test_infer_RAG/build_rag_index.py | 153d4f3985f9f6fe09c17301b9f1621ed6df750f5cb34dae48920910902d1553
ChatMed_Consult-v0.3_test_500 | TBD | TBD | test_infer_RAG/datasets/ChatMed_Consult-v0.3_test_500/ChatMed_Consult-v0.3_test_500.json | test | 500 | test_infer_RAG/generate_answer.py / generate_score_DS.py | Local evaluation subset only
Chinese-medical-dialogue-data_test_500 | TBD | TBD | test_infer_RAG/datasets/Chinese-medical-dialogue-data_test_500/Chinese-medical-dialogue-data_test_500.json | test | 500 | test_infer_RAG/generate_answer.py / generate_score_DS.py | Local evaluation subset only
huatuo26M_test_500 | TBD | TBD | test_infer_RAG/datasets/huatuo26M_test_500/huatuo26M_test_500.json | test | 500 | test_infer_RAG/generate_answer.py / generate_score_DS.py | Local evaluation subset only
keshi_medical_consult_500 | TBD | TBD | test_infer_RAG/datasets/keshi_medical_consult_500/keshi_medical_consult_500.json | test | 500 | test_infer_RAG/generate_answer.py / generate_score_DS.py | Local evaluation subset only

Task/department sample counts in CMedCL train/test files recovered from the local snapshot:

Department | Train file count | Test file count
--- | ---: | ---:
neike | 5238 | 309
waike | 5239 | 303
erke | 5234 | 314
fuchanke | 5237 | 362
nanke | 5178 | 341
zhongliuke | 5242 | 315

Pilot-50 subset task counts (from metadata_robustness_experiments/inference_evaluation/data/pilot50_manifest.json): neike=9, waike=9, erke=8, fuchanke=8, nanke=8, zhongliuke=8; sample_seed=20260627.

## 3. Experimental Environment

Item | Value
--- | ---
OS | CentOS Linux 7 (Core); Linux kernel 5.4.278-1.el7.elrepo.x86_64
Python (current runtime) | 3.12.7 (Anaconda)
Python (declared target env) | 3.8.5 (environment.yml)
CUDA runtime from torch | cu124 via torch 2.6.0+cu124 in current runtime
CUDA toolkit / driver | TBD (nvidia-smi unavailable in current session)
cuDNN | 9.1.0.70 package present in current runtime; training-time cuDNN exact runtime TBD
PyTorch (current runtime) | 2.6.0+cu124
PyTorch (declared target env) | 2.2.2
transformers (current runtime) | 4.55.3
transformers (declared target env) | >=4.39.0,<4.41
datasets | TBD in current runtime import; declared target env 2.18.0
accelerate | Not importable in current runtime; declared target env 0.29.2
peft | Not importable in current runtime; declared target env 0.10.0
deepspeed | Not importable / not declared
numpy | 2.3.2 current runtime; 1.24.4 declared target env
scipy | Not importable in current runtime; 1.9.3 declared target env
scikit-learn | Not importable in current runtime; 1.3.2 declared target env
sentencepiece | 0.2.0 declared target env
bitsandbytes | Declared in environment.yml without pinned version
GPU driver | TBD (nvidia-smi unavailable in current session)

Project-declared environment files scanned: `environment.yml`, `env.txt`, `rouge/requirements.txt`, `rouge/setup.py`.

## 4. Hardware

Component | Value
--- | ---
GPU model | TBD from current session; logs indicate CUDA training was used
Number of GPUs used | Baseline training script uses 1 GPU (CUDA_VISIBLE_DEVICES=0); metadata robustness launcher maps 4 parallel runs to physical GPUs 2/3/4/5
GPU memory | Peak allocated memory in metadata robustness train_summary.json is about 14.10 GB
CPU | Intel(R) Xeon(R) Gold 6430
CPU sockets / cores / threads | 2 sockets / 32 cores per socket / 128 logical CPUs total
RAM | 1.0T total system memory
Distributed setting | No DDP/DeepSpeed found in main training entry; metadata robustness launcher runs one process per GPU

## 5. Random Seeds

Component | Seed Value | Source File / Line | Notes
--- | --- | --- | ---
Global Python / NumPy / PyTorch / HF seed | 0 | src/run_continual_causal_llama2.py:37-38; run_continual_script/keshi_llama/scrip.sh:41 | Set through transformers.set_seed(seed)
Dataset split seed | 0 | tasks/mtl5/dataloader_mtl_causal_llama.py:184 | Used in raw_dataset.train_test_split(test_size=0.1, seed=training_args.seed)
DataLoader worker seed | Not explicitly specified in the current code | tasks/mtl5/dataloader_mtl_causal_llama.py:205-224 | num_workers is passed but worker_init_fn is not set
RandomSampler seed | Not explicitly specified in the current code | tasks/mtl5/dataloader_mtl_causal_llama.py:207-214 | Depends on global torch RNG state
Task key encoder seed | 0 | src/run_continual_causal_llama2.py:188-196 | Passed into KeyEncoderConfig(seed=seed)
Metadata perturbation seed | 20260627 | metadata_robustness_experiments/run_4_metadata_experiments.sh:138-143; metadata_robustness_experiments/runner.py parser | Shared across the four metadata robustness conditions
Pilot-50 subset sampling seed | 20260627 | metadata_robustness_experiments/inference_evaluation/build_pilot50_subset.py:75 | Used to stratify and fix the 50-sample subset

## 6. Model Checkpoints

Model / Module | Source or Path | Version / Revision | File Size | SHA256 | Release Status
--- | --- | --- | --- | --- | ---
Base LLM | chinese-alpaca-plus-7b-hf | HF model id documented as shibing624/chinese-alpaca-plus-7b-hf; local path chinese-alpaca-plus-7b-hf | 13.77 GB across two .bin shards | 75ddade0a97a9592daff2fe764d03000cf7561a7798c1fc054e6f3789fded45a; d1e32eade4c61c029f24e921380812fb698d8faf651944dc11b1ef61efdeffff | Weights present in local copy
Tokenizer | chinese-alpaca-plus-7b-hf/tokenizer.model | bundled with base model | 757972 bytes | 2d967e855b1213a439df6c8ce2791f869c84b4f3b6cfacf22b86440b8192a2f8 | Present in local copy
Original metadata embeddings | metadata_embeddings/keshi_meta_embeddings.pt | local artifact | 99554 bytes | 0ee2393dfe06ddddb846e927fd6d559a667cdcd2c8f07132b45387f9a6497e58 | Present in local copy
Perturbed metadata embeddings: missing_50 | metadata_robustness_experiments/generated_metadata/missing_50/perturbed_meta_embeddings.pt | local artifact | 99638 bytes | 818589dacb9e1cdd1a228ba52bae493fa9bc76baba9e180bc829d5c5f2b11b22 | Present in local copy
Perturbed metadata embeddings: noisy_30 | metadata_robustness_experiments/generated_metadata/noisy_30/perturbed_meta_embeddings.pt | local artifact | 99638 bytes | fcafc1c1d1c7bb34918f949fba4b1b1caa902ba6b69eaedea645dd0ce638f964 | Present in local copy
Perturbed metadata embeddings: stale_coarse | metadata_robustness_experiments/generated_metadata/stale_coarse/perturbed_meta_embeddings.pt | local artifact | 99638 bytes | aeacdf59f8a0eb3f63233fab1e6e867f40c0c787b854996f340b54f34564c92f | Present in local copy
Perturbed metadata embeddings: institution_mix_40 | metadata_robustness_experiments/generated_metadata/institution_mix_40/perturbed_meta_embeddings.pt | local artifact | 99638 bytes | 3b3ddc6790257df21b48cbe2ba65624466fab2a47e830b24d623fc2d522211ee | Present in local copy
Baseline continual checkpoint (final snapshot) | checkpoints_continual_keshi_llama/order1_compose_peft/snapshots/task_5_zhongliuke_train_end_20260626_123739/state_dict.pt | seed=0; adapters=[neike,waike,erke,fuchanke,nanke,zhongliuke] | 460087745 bytes | b00ef35b54e861cee42b9e83e39c0d7387174e4a637d894634140d9583270149 | Present in local copy
Metadata robustness checkpoint: missing_50 | metadata_robustness_experiments/outputs/metadata_robustness/missing_50/checkpoints/task_5_zhongliuke_train_end_20260628_000319/state_dict.pt | seed=0; metadata_seed=20260627 | 460087745 bytes | 0daad22c8b551527a8f1dff9571dc26ef64a9733f97774f1adfc67be6f54d58f | Present in local copy
Metadata robustness checkpoint: noisy_30 | metadata_robustness_experiments/outputs/metadata_robustness/noisy_30/checkpoints/task_5_zhongliuke_train_end_20260628_001843/state_dict.pt | seed=0; metadata_seed=20260627 | 460087745 bytes | e6f7a5adb76b836cd94f9dd2c073d92350822e106b9091513ec46a4d6d58b6de | Present in local copy
Metadata robustness checkpoint: stale_coarse | metadata_robustness_experiments/outputs/metadata_robustness/stale_coarse/checkpoints/task_5_zhongliuke_train_end_20260628_002403/state_dict.pt | seed=0; metadata_seed=20260627 | 460087745 bytes | d4725f7857495762ab777fb25d0afa2a4bdc81db3e504b83e0945d42baa0f67a | Present in local copy
Metadata robustness checkpoint: institution_mix_40 | metadata_robustness_experiments/outputs/metadata_robustness/institution_mix_40/checkpoints/task_5_zhongliuke_train_end_20260628_001540/state_dict.pt | seed=0; metadata_seed=20260627 | 460087745 bytes | 61b4489ab20b759ab12ec7d1323a4f550fea0b5d53b1c0aa1c77425cee0d2315 | Present in local copy

Additional adapter files are also present under `metadata_robustness_experiments/outputs/metadata_robustness/<condition>/final_model/<task>/adapter_model.bin` with per-file size 8,445,706 bytes and condition/task-specific SHA256 values recoverable from the local repository snapshot.

## 7. Key Training and Evaluation Settings

Experiment | Config File | Command | Batch Size | Learning Rate | Epochs | Precision | Notes
--- | --- | --- | --- | --- | --- | --- | ---
Baseline continual training | run_continual_script/keshi_llama/scrip.sh | python3 src/run_continual_causal_llama2.py --model_name_or_path chinese-alpaca-plus-7b-hf --task_list neike_waike_erke_fuchanke_nanke_zhongliuke --continual_learning --mpeft_enabled --matching_loss_v2 --meta_embeddings_path ./metadata_embeddings/keshi_meta_embeddings.pt --do_train --padding_strategy longest --max_seq_length 512 --max_target_length 64 --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --learning_rate 1e-4 --num_train_epochs 4 --output_dir checkpoints_continual_keshi_llama/order1_compose_peft --overwrite_output_dir --seed 0 --save_strategy no --evaluation_strategy no --validation_split_percentage 0.1 --overwrite_cache True --lamda_1 0.05 --lamda_2 0.01 --orthogonal_threshold 0.2 | 1 | 1e-4 | 4 | 4-bit nf4 quantization; bf16 if supported else fp16 | LoRA r=4, alpha=16, dropout=0.1; target modules q_proj/v_proj
Metadata robustness training (4 conditions) | metadata_robustness_experiments/run_4_metadata_experiments.sh + metadata_robustness_experiments/configs/*.yaml | CUDA_VISIBLE_DEVICES=<mapped ordinal> python -m metadata_robustness_experiments.runner --project-root <ROOT> --condition <missing_50|noisy_30|stale_coarse|institution_mix_40> --metadata-seed 20260627 --gpu-id <2|3|4|5> [--resume] | TBD from runner-resolved config (training summary confirms standard run completed) | TBD from resolved runner config | 4 (from train_summary.json) | CUDA run; exact bf16/fp16 depends on torch.cuda.is_bf16_supported() | One process per GPU; training_seed=0 for all four runs
Pilot-50 inference | metadata_robustness_experiments/inference_evaluation/run_pilot50_4gpu.sh | CUDA_VISIBLE_DEVICES=<gpu> python -m metadata_robustness_experiments.inference_evaluation.run_inference --condition <condition> --artifact_registry metadata_robustness_experiments/inference_evaluation/artifact_registry.json --samples metadata_robustness_experiments/inference_evaluation/data/pilot50_samples.jsonl --output_dir metadata_robustness_experiments/inference_evaluation/outputs/pilot_50/<condition> --device cuda:0 | TBD | TBD | N/A | float16 recommended in README example | Loads snapshot checkpoint + corresponding perturbed metadata embeddings
Pilot-50 scoring | metadata_robustness_experiments/inference_evaluation/run_scoring.sh | python -m metadata_robustness_experiments.inference_evaluation.score_pilot50 --condition <condition> --input metadata_robustness_experiments/inference_evaluation/outputs/pilot_50/<condition>/inference_results.jsonl --output_dir metadata_robustness_experiments/inference_evaluation/outputs/pilot_50/<condition>/scoring --max_workers 4 [--resume] | N/A | N/A | N/A | API-based judge | Requires DEEPSEEK_API_KEY

Metadata robustness train summaries recovered from `metadata_robustness_experiments/outputs/metadata_robustness/*/metrics/train_summary.json` indicate: total_steps=14104, total_epochs=4, trainable_parameter_count=409,489,408, total_parameter_count=3,660,099,584, and peak allocated GPU memory about 14.10 GB for all four conditions.

## 8. Reproduction Commands

* Data preprocessing
  `python generate_metadata.py`
  `python test_infer_RAG/build_rag_index.py --kb_path datasets/knowledge_base/normalized_knowledge_base.json --index_path datasets/faiss_index_medical --embedding_model shibing624/text2vec-base-chinese`
  `python -m metadata_robustness_experiments.inference_evaluation.build_pilot50_subset --project-root . --output-dir metadata_robustness_experiments/inference_evaluation/data --sample-seed 20260627`
* Training
  `bash run_continual_script/keshi_llama/scrip.sh`
  `bash metadata_robustness_experiments/run_4_metadata_experiments.sh --fresh`
* Evaluation
  `CHECKPOINT_DIR=<checkpoint_dir> bash test_infer/generate_answer.sh`
  `bash test_infer/generate_score_DS.sh`
  `CHECKPOINT_DIR=<checkpoint_dir> bash test_infer_RAG/generate_answer.sh`
  `bash test_infer_RAG/generate_score_DS.sh`
  `./metadata_robustness_experiments/inference_evaluation/run_pilot50_4gpu.sh`
  `DEEPSEEK_API_KEY=<key> ./metadata_robustness_experiments/inference_evaluation/run_scoring.sh --resume`
* Table/Figure reproduction
  `./metadata_robustness_experiments/inference_evaluation/aggregate_results.sh`
  `python metadata_robustness_experiments/inference_evaluation/outputs/pilot_50/plot_metadata_robustness.py`

## 9. Missing or Unconfirmed Items

* TBD: Exact Git branch, commit hash, latest commit author/date/message, and clean/dirty working-tree status are TBD because the local workspace snapshot contains an empty .git directory and no recoverable Git metadata.
* TBD: Official code release URL / repository remote is TBD for the same reason.
* TBD: Exact public release URL, version, and download date for the CMedCL dataset are TBD; only local file paths and checksums are recoverable from the current snapshot.
* TBD: Exact source/version metadata for datasets/knowledge_base/normalized_knowledge_base.json is TBD; only local path, entity count, and SHA256 are recoverable.
* TBD: Whether Cmedia specifically participates in the current cleaned repo is TBD; it is not explicitly referenced in the accessible local scripts/README scanned here.
* TBD: Exact train/validation split counts after datasets.train_test_split(test_size=0.1, seed=0) are TBD in this runtime because the HuggingFace datasets package is not importable in the current environment.
* TBD: Training-time GPU model, count visible to each process, and NVIDIA driver version are TBD because nvidia-smi is unavailable in the current session.
* TBD: cuDNN runtime version during the original training runs is TBD; only the current Python environment package list suggests nvidia-cudnn-cu12 9.1.0.70.
* TBD: Exact versions actually used during the original paper experiments may differ from the current runtime; environment.yml and env.txt suggest the intended environment, while the current shell uses a different Conda base environment.
* TBD: Base-model revision / commit hash for chinese-alpaca-plus-7b-hf is TBD; the local model README points to Hugging Face model id shibing624/chinese-alpaca-plus-7b-hf but no pinned revision is stored locally.
* TBD: Checkpoint release status for public artifact release is TBD; if these checkpoints are not yet public, authors may state that trained checkpoints will be released upon acceptance.
* TBD: Some evaluation scripts for score generation use external services; the exact API model revision and service-side version are TBD.

## 10. Suggested Text for Manuscript

We will release the code as a repository snapshot of our MeMR implementation together with the corresponding data processing scripts and evaluation pipeline. The cleaned repository includes the six-department CMedCL continual-learning dataset split used in this work (`neike`, `waike`, `erke`, `fuchanke`, `nanke`, and `zhongliuke`), the RAG knowledge base file, the metadata embedding files, and the scripts for training, inference, and pilot-50 robustness evaluation. Our main experiments fine-tune the `chinese-alpaca-plus-7b-hf` base model with MeMR/MDTM/ATR using seed 0, a batch size of 1 with gradient accumulation 8, learning rate 1e-4, and 4 training epochs. Metadata robustness experiments use the same training seed and a metadata perturbation seed of 20260627. We additionally provide file paths, checksums, environment specifications, and checkpoint identifiers in the supplementary reproducibility note for precise replication.

## 11. Suggested Response to Reviewer

Responses:
Thank you for your valuable suggestion. We agree that providing detailed reproducibility information is important for improving the transparency and reliability of our work. In the revised manuscript, we have added detailed information regarding the code and data artifacts, the intended software environment, the available hardware information, the random seeds used in training and metadata perturbation, and the corresponding model checkpoints. Specifically, we now report the local code snapshot information, the six-department CMedCL data files and their checksums, the RAG knowledge-base artifact, the training and inference commands, the environment specification files, the main hyper-parameters, the seed settings (training seed 0 and metadata seed 20260627 for robustness experiments), and the identifiers/checksums of the released checkpoints and metadata embeddings. For items that cannot be unambiguously recovered from the current archived workspace snapshot (e.g., missing Git metadata or non-accessible driver information), we explicitly mark them as TBD rather than infer them.

## Appendix: Dependency Snapshot

Current `pip freeze` and `conda env export --no-builds` outputs were inspected from the active shell environment, while `environment.yml` and `env.txt` were used as the project-declared target environment. The current shell environment differs from the declared training environment and should not be treated as the authoritative training setup without additional confirmation.