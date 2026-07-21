# MeMR: Metadata-Enhanced Matching and Adaptive Representation for Continual Learning in Medical Consultation

This directory is a cleaned, GitHub-ready release of the MeMR project. It keeps the core source code, dataset files, metadata files, paper documents, prompts, and reviewer-response materials that are appropriate for public release, while excluding large model weights, LoRA checkpoints, tensorboard logs, caches, and other bulky runtime artifacts.

MeMR is a continual-learning framework for Chinese medical consultation built on a LLaMA-style causal language model with multi-adapter LoRA routing. The two main method components are:

- `MDTM`: Metadata-Augmented Dynamic Task Matching
- `ATR`: Adaptive Task Representation

The project also includes `CMedCL`, a six-department Chinese medical continual-learning dataset used in the manuscript.

## What Is Included

- Core training and inference code for MeMR
- The local `mpeft` implementation used by this project
- CMedCL data and the medical knowledge base used by the project
- Metadata embeddings and metadata-generation scripts
- The manuscript, highlighted revision, and point-by-point reviewer response
- Prompts and supplementary reviewer-response materials referenced during revision
- Lightweight summary results, scripts, and figures for reviewer-response experiments

## What Is Excluded

This cleaned release does **not** include:

- Base-model weight shards
- Trained LoRA checkpoints
- Large experiment output directories
- TensorBoard event files
- Intermediate caches
- Runtime-generated indexes and logs

If you want to rerun training or inference, you must supply a compatible local base model and your own trained checkpoints.

## Repository Layout

```text
MeMR-main/
├── src/                         # Main training entry
├── model/                       # Core MeMR model and routing/key encoder code
├── mpeft/                       # Local PEFT/LoRA extensions used by MeMR
├── training/                    # Continual trainer and optimization logic
├── utils/ tools/ rouge/         # Utilities and evaluation helpers
├── tasks/                       # Task loaders
├── test_infer*/                 # Inference and evaluation scripts
├── datasets/
│   ├── medical_consult/         # CMedCL data
│   └── knowledge_base/          # Medical knowledge base
├── metadata_embeddings/         # Metadata embeddings and metadata json
├── run_continual_script/        # Main training shell script
├── paper/                       # Manuscript and reviewer response documents
├── prompts/                     # Prompts used during rebuttal/revision workflow
├── reviewer_materials/          # Supplementary scripts and compact reviewer assets
├── base_model_stub/             # Tokenizer/config stub for the base model
└── REPOSITORY_CONTENTS.md       # Guide to the public-release contents
```

## Environment

The original project uses Python 3.8.x.

```bash
conda create -n memr python=3.8.5
conda activate memr
pip install torch torchvision torchaudio
pip install -r env.txt
```

The local `mpeft/` package should be imported from the repository root.

## Base Model Preparation

The cleaned release contains tokenizer and config files under `base_model_stub/chinese-alpaca-plus-7b-hf/`, but not the large weight shards.

Before running training or inference, either:

1. Place the missing weight files back under a compatible model directory, or
2. Change `--model_name_or_path` to another compatible local or Hugging Face model path.

The original training script expects a model path named `chinese-alpaca-plus-7b-hf`.

## Training

The main continual-learning script is:

```bash
bash run_continual_script/keshi_llama/scrip.sh
```

It trains the six departments in this order:

```text
neike -> waike -> erke -> fuchanke -> nanke -> zhongliuke
```

The script uses:

- metadata embeddings from `metadata_embeddings/keshi_meta_embeddings.pt`
- query-task matching via `--matching_loss_v2`
- ATR-related regularization through `--lamda_1`, `--lamda_2`, and `--orthogonal_threshold`

## Metadata Regeneration

To regenerate department metadata embeddings:

```bash
python generate_metadata.py
```

## Inference

Example without RAG:

```bash
python test_infer/generate_answer.py \
  --checkpoint_dir path/to/checkpoint \
  --input_path datasets/medical_consult/neike/test.json \
  --output_path test_infer/generate_answer/neike.json
```

Example with RAG:

```bash
python test_infer_RAG/build_rag_index.py \
  --kb_path datasets/knowledge_base/normalized_knowledge_base.json \
  --index_path datasets/faiss_index_medical \
  --embedding_model shibing624/text2vec-base-chinese

python test_infer_RAG/generate_answer.py \
  --checkpoint_dir path/to/checkpoint \
  --input_path datasets/medical_consult/neike/test.json \
  --output_path test_infer_RAG/generate_answer_rag/neike.json \
  --index_path datasets/faiss_index_medical \
  --embedding_model shibing624/text2vec-base-chinese
```

## Paper and Reviewer Materials

The paper files are in `paper/`:

- `manuscript.pdf`
- `manuscript.docx`
- `manuscript_highlighted.docx`
- `Detailed Response to Reviewers.pdf`
- `Detailed Response to Reviewers.docx`

The supplementary revision materials are organized in `reviewer_materials/`. These include:

- metadata robustness and cold-start diagnostics
- gate-formulation diagnostics
- dataset integrity audit scripts and summaries
- ATR reviewer diagnostics
- high-risk safety stress-test scripts and compact outputs
- scalability profiling scripts and compact tables
- matched-baseline supplementary scripts and summaries

See `REPOSITORY_CONTENTS.md` for a more detailed map.

## Notes on Scope

- The highlighted manuscript explicitly clarifies that MeMR is evaluated under externally specified task boundaries and closed-world routing over an externally registered task set.
- The safety materials in this release are exploratory behavioral analyses, not clinical validation.
- Some reviewer-response directories contain summary outputs retained for transparency, but not the full raw experimental artifacts.

## License

See `LICENSE` and `3rd-party-licenses.txt`.
