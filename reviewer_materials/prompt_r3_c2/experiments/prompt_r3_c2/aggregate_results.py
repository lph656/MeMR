from __future__ import annotations

import argparse
from pathlib import Path

from experiments.reviewer_0629.common import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description="Aggregate prompt_r3_c2 gate probe results.")
    parser.add_argument("--output-root", default="experiments/prompt_r3_c2/outputs")
    parser.add_argument("--report-path", default="experiments/prompt_r3_c2/outputs/prompt_r3_c2_report.md")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    variants = ["current", "metadata_only", "temperature", "learnable_mlp"]
    summary = {}
    for variant in variants:
        variant_file = output_root / variant / "summary.json"
        if variant_file.exists():
            summary[variant] = read_json(variant_file)

    write_json(output_root / "aggregated_summary.json", summary)

    lines = ["# prompt_r3_c2 Gate Probe Summary", ""]
    for variant, payload in summary.items():
        lines.append(f"## {variant}")
        lines.append(f"- checkpoint: {payload.get('checkpoint_dir')}")
        lines.append(f"- train samples: {payload.get('train_samples')}")
        lines.append(f"- dev samples: {payload.get('dev_samples')}")
        for eval_name, eval_summary in payload.get("eval_summaries", {}).items():
            lines.append(f"- {eval_name}: top1={eval_summary.get('top1_accuracy')}, top3={eval_summary.get('top3_accuracy')}, ece={eval_summary.get('ece')}, mean_alpha={eval_summary.get('mean_gate_alpha')}")
        lines.append("")

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()

