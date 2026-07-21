from __future__ import annotations

import argparse
from pathlib import Path

from .common import read_json, write_json


def maybe_read(path: Path):
    return read_json(path) if path.exists() else None


def main():
    parser = argparse.ArgumentParser(description="Aggregate reviewer_0629 experiment outputs into a response-ready summary.")
    parser.add_argument("--output-root", default="experiments/reviewer_0629/outputs")
    parser.add_argument("--report-path", default="experiments/reviewer_0629/outputs/reviewer_0629_report.md")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    baseline = maybe_read(output_root / "routing_baseline" / "summary.json")
    noisy_labels = maybe_read(output_root / "routing_noisy_labels" / "summary.json")
    cold_start = maybe_read(output_root / "cold_start_zhongliuke" / "summary.json")
    unseen = maybe_read(output_root / "unseen_zhongliuke" / "summary.json")
    metadata_conditions = {}
    metadata_root = output_root / "routing_metadata"
    if metadata_root.exists():
        for child in sorted(metadata_root.iterdir()):
            if child.is_dir() and (child / "summary.json").exists():
                metadata_conditions[child.name] = read_json(child / "summary.json")

    summary = {
        "baseline": baseline,
        "noisy_labels": noisy_labels,
        "cold_start": cold_start,
        "unseen_department": unseen,
        "metadata_conditions": metadata_conditions,
    }
    write_json(output_root / "aggregated_summary.json", summary)

    lines = []
    lines.append("# Reviewer 0629 Experiment Summary")
    lines.append("")
    lines.append("## Review Points to Experiment Mapping")
    lines.append("")
    lines.append("- (a) Top-1 / top-k routing accuracy: `routing_baseline/summary.json` and metadata/noisy-label variants.")
    lines.append("- (b) Routing calibration and entropy: `ece`, `brier_score`, and `mean_entropy` in routing summaries.")
    lines.append("- (c) Controlled input-noise and metadata-corruption experiments: noise-mode summaries under baseline and metadata condition subdirectories.")
    lines.append("- (d) Cold-start evaluation on newly introduced tasks: `cold_start_zhongliuke/summary.json`.")
    lines.append("- (e) Unseen medical-department testing: `unseen_zhongliuke/summary.json`.")
    lines.append("- (f) Noisy metadata and noisy task labels: `routing_metadata/*/summary.json` and `routing_noisy_labels/summary.json`.")
    lines.append("- (g) Full weighted routing vs top-k routing: `generation_metrics` fields inside routing summaries.")
    lines.append("")

    if baseline:
        clean = baseline.get("none", {})
        lines.append("## Baseline Clean Routing")
        lines.append("")
        lines.append(f"- Samples: {clean.get('num_samples')}")
        lines.append(f"- Top-1 accuracy: {clean.get('top1_accuracy')}")
        lines.append(f"- Top-3 accuracy: {clean.get('top3_accuracy')}")
        lines.append(f"- ECE: {clean.get('ece')}")
        lines.append(f"- Brier score: {clean.get('brier_score')}")
        lines.append(f"- Mean entropy: {clean.get('mean_entropy')}")
        lines.append("")

    if cold_start:
        lines.append("## Cold-Start Routing")
        lines.append("")
        for setting_name, metrics in cold_start.get("settings", {}).items():
            lines.append(
                f"- {setting_name}: top1={metrics.get('top1_new_task_accuracy')}, "
                f"top3={metrics.get('top3_new_task_accuracy')}, "
                f"mean_new_task_weight={metrics.get('mean_new_task_weight')}"
            )
        lines.append("")

    if unseen:
        lines.append("## Unseen Department Transfer")
        lines.append("")
        lines.append(f"- Held-out department: {unseen.get('heldout_task')}")
        metrics = unseen.get("generation_metrics", {})
        for route_mode, route_metrics in metrics.items():
            lines.append(
                f"- {route_mode}: rougeL={route_metrics.get('rougeL')}, "
                f"rouge1={route_metrics.get('rouge1')}, exact_match={route_metrics.get('exact_match')}"
            )
        lines.append("")

    if metadata_conditions:
        lines.append("## Metadata Corruption Variants")
        lines.append("")
        for condition_name, condition_summary in metadata_conditions.items():
            clean = condition_summary.get("none", {})
            lines.append(
                f"- {condition_name}: top1={clean.get('top1_accuracy')}, "
                f"ece={clean.get('ece')}, "
                f"mean_entropy={clean.get('mean_entropy')}"
            )
        lines.append("")

    if noisy_labels:
        clean = noisy_labels.get("none", {})
        lines.append("## Noisy Task Labels Variant")
        lines.append("")
        lines.append(
            f"- Top-1 accuracy: {clean.get('top1_accuracy')}, "
            f"ECE: {clean.get('ece')}, "
            f"mean entropy: {clean.get('mean_entropy')}"
        )
        lines.append("")

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()

