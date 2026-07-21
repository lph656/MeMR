# Reviewer 0629 Suite

This directory contains a standalone experiment suite that answers the reviewer request in `prompt_0629`
without modifying the original project workflow.

Covered reviewer points:

1. Top-1 and top-k task-routing accuracy.
2. Routing calibration and entropy analysis.
3. Controlled input-noise and metadata-corruption experiments.
4. Cold-start evaluation on a newly introduced task.
5. Unseen medical-department testing.
6. Ablations with noisy metadata and noisy task labels.
7. Full weighted routing versus top-k routing.

Main outputs:

- `experiments/reviewer_0629/data/`: fixed evaluation subsets.
- `experiments/reviewer_0629/generated_datasets/`: label-noise training data.
- `experiments/reviewer_0629/outputs/`: routing, cold-start, unseen-department, and aggregation results.
- `experiments/reviewer_0629/runbooks/`: five terminal scripts for parallel execution.

