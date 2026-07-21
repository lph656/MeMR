# High-risk Safety Extension for MeMR

This directory adds a lightweight, inference-only extension experiment for reviewer questions about practical deployment and clinical safety considerations. It does not retrain any model and does not modify the original MeMR training, model, dataset, or main experiment code.

## Purpose

The experiment evaluates high-risk and conflict-aware safety behavior on synthetic Chinese medical consultation stress cases. It focuses on:

- unsafe recommendation risk
- appropriate escalation to urgent care
- conflict recognition across departments
- over-certainty under uncertainty
- responsibility-aware response boundaries

This is a small-scale deployment-oriented safety stress test. It is **not clinical validation** and must not be interpreted as a substitute for licensed clinical decision-making.

## Safety Case Source

The dataset `safety_cases_zh.jsonl` contains 60 synthetic high-risk and conflict-aware cases. They are manually specified stress-test prompts, not real patient records.

## Files

- `create_safety_cases.py`: writes the synthetic safety-case JSONL file
- `discover_checkpoints.py`: resolves which checkpoints are available for each method
- `run_safety_eval.py`: runs inference-only safety evaluation on discovered checkpoints
- `auto_safety_judge.py`: computes automatic preliminary safety scores and creates a manual review template
- `aggregate_safety_eval.py`: aggregates scores into summary tables and representative case reports
- `make_figures.py`: creates bar charts, heatmaps, and the deployment pipeline figure
- `run_all.sh`: one-shot launcher for the full pipeline
- `method_registry.template.json`: template for method registration and checkpoint override

## Default Methods

The default method registry includes:

- `SequentialFT`
- `MoCL`
- `MeMR_wo_MDTM`
- `MeMR_wo_ATR`
- `MeMR`

If a checkpoint is missing, the method is skipped and recorded in `missing_checkpoints.json`.

## Checkpoint Resolution

`discover_checkpoints.py` searches recursively under:

- `checkpoints/`
- `checkpoint/`
- `outputs/`
- `output/`
- `runs/`
- `logs/`
- `saved_models/`
- `models/`

It matches directory names using configured aliases and picks the newest valid checkpoint candidate by modification time. If you want to override a checkpoint path manually, edit `method_registry.template.json` and set `checkpoint` explicitly for the target method.

## Running

Always use GPU 0 for this extension experiment:

```bash
bash experiments/high_risk_safety/run_all.sh
```

Or run each stage manually:

```bash
python experiments/high_risk_safety/create_safety_cases.py \
  --output experiments/high_risk_safety/safety_cases_zh.jsonl

python experiments/high_risk_safety/discover_checkpoints.py \
  --method_registry experiments/high_risk_safety/method_registry.template.json \
  --output_dir results/high_risk_safety/<timestamp>

python experiments/high_risk_safety/run_safety_eval.py \
  --dataset experiments/high_risk_safety/safety_cases_zh.jsonl \
  --method_registry results/high_risk_safety/<timestamp>/method_registry.resolved.json \
  --output_dir results/high_risk_safety/<timestamp> \
  --gpu 0
```

## Automatic Scoring and Manual Review

The automatic judge computes five preliminary binary indicators:

- `URR`: Unsafe Recommendation Rate
- `AER`: Appropriate Escalation Rate
- `CRR`: Conflict Recognition Rate
- `OCR`: Over-certainty Rate
- `RRR`: Responsibility-aware Response Rate

These automatic scores are intended to produce an initial analysis and a structured manual review template. They do **not** replace clinician review.

If a filled-in `manual_review_filled.csv` is provided later, `aggregate_safety_eval.py` will prefer manual scores over auto scores.

## Outputs

Each run writes a timestamped directory under:

```text
results/high_risk_safety/<timestamp>/
```

The directory contains:

- run log and environment snapshot
- resolved method registry
- missing checkpoint report
- per-method response JSONL files
- automatic per-case safety scores
- manual review template
- summary tables in CSV / Markdown / LaTeX
- representative case exports
- figures for reviewer response

If MeMR matching weights are available through a non-invasive wrapper/hook, they are saved under:

```text
matching_weights/MeMR_matching_weights.csv
```

If they are not available in the current implementation, the pipeline records this in logs and skips the matching-weight heatmap without failing.

## How to Use in Reviewer Response

Recommended assets for direct reuse:

- `summary_auto.md`: safety summary table
- `summary_auto.tex`: manuscript-ready table
- `summary_by_category.csv`: category breakdown
- `representative_cases.md`: qualitative examples
- `figures/safety_metrics_bar.*`: main safety metric comparison
- `figures/category_safety_heatmap.*`: category-level safety comparison
- `figures/cdss_pipeline.*`: deployment-oriented clinical decision support workflow

When citing these results, clearly state that the experiment is a synthetic safety stress test and not a clinical validation study.
