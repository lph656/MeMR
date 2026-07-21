# Repository Contents

This file maps the cleaned public-release contents to the paper and reviewer-response materials.

## Core MeMR Implementation

- `src/run_continual_causal_llama2.py`
  Main continual-learning training entry.
- `model/`
  Core model code, including the MeMR routing/key encoder implementation.
- `training/trainer_continual_causal_llama_lora.py`
  Continual trainer and optimization logic.
- `mpeft/`
  Local PEFT/LoRA implementation extended for MeMR.
- `arguments.py`
  Training/config dataclasses and method arguments.

## Data and Metadata

- `datasets/medical_consult/`
  CMedCL six-department dataset used in the paper.
- `datasets/knowledge_base/normalized_knowledge_base.json`
  Knowledge base used by the project and RAG-related scripts.
- `metadata_embeddings/`
  Encoded metadata embeddings and merged metadata json.
- `generate_metadata.py`
  Metadata generation script.

## Paper Files

- `paper/manuscript.pdf`
  Original manuscript snapshot in this workspace.
- `paper/manuscript.docx`
  Original manuscript source.
- `paper/manuscript_highlighted.docx`
  Revised highlighted manuscript.
- `paper/Detailed Response to Reviewers.pdf`
  Point-by-point reviewer response.
- `paper/Detailed Response to Reviewers.docx`
  Editable source of the reviewer response.

## Prompt Files

- `prompts/`
  Prompt files used during rebuttal/revision preparation in this workspace, including:
  - metadata/routing reviewer prompts
  - ATR reviewer prompts
  - scalability reviewer prompts
  - dataset-integrity and safety-related prompts

These are included for transparency because the reviewer response and manuscript revision referenced prompt-driven cleaning, evaluation, and rebuttal drafting workflows.

## Reviewer Materials

### `reviewer_materials/reviewer_0629/`

Supplementary scripts and compact assets for routing diagnostics, cold-start analysis, unseen-department evaluation, and noisy-label studies.

Included:

- scripts
- fixed evaluation subsets
- runbooks
- compact summary outputs
- draft reply files

Excluded:

- checkpoints
- full tensorboard logs
- bulky training artifacts

### `reviewer_materials/prompt_r3_c2/`

Gate-formulation diagnostics for metadata-only, temperature, and learnable-gate variants.

Included:

- scripts
- runbooks
- compact summaries
- report markdown
- figure assets

Excluded:

- cache `.pt` files
- gate states
- full per-sample bulky outputs

### `reviewer_materials/prompt_r3_c7/`

Dataset integrity and overlap-audit materials.

Included:

- audit scripts
- runbooks
- compact summary json and markdown

Excluded:

- large record dumps

### `reviewer_materials/atr_reviewer/`

ATR-focused reviewer-response utilities and explanation tables.

Included:

- variant-evaluation scripts
- response-draft builders
- runbooks
- compact analysis markdown

### `reviewer_materials/high_risk_safety/`

Exploratory high-risk and conflict-aware safety stress-test materials.

Included:

- inference-only evaluation scripts
- synthetic safety-case set
- compact summaries and representative cases
- figures

Excluded:

- large response dumps except compact representative artifacts

### `reviewer_materials/scalability/`

Scalability profiling scripts and compact tables used in the reviewer response.

Included:

- profiling scripts
- compact summary tables
- revised figures

### `reviewer_materials/reviewer_fairness/`

Matched-baseline supplementary scripts, configs, and compact summary tables.

Included:

- scripts and configs
- evaluation/training helpers
- summary markdown/csv

Excluded:

- checkpoints
- caches
- full logs

### `reviewer_materials/llm_judge_reproducibility/`

Compact scripts and data used for LLM-judge reproducibility aggregation.

## Additional Supporting Files

- `REPRODUCIBILITY_INFO_FOR_REVIEW.md`
- `reproducibility_info.json`
- `ADAPTER_COMPOSITION_AUDIT.md`
- `adapter_composition_audit.json`
- `codex_summary_for_rebuttal.md`

These are retained because they help explain implementation details and rebuttal-side reproducibility claims.

## Intentionally Excluded from Public Release

- base-model weight shards
- LoRA checkpoints
- intermediate caches
- tensorboard logs
- large raw experiment outputs
- runtime-generated FAISS indexes
- transient Python caches
