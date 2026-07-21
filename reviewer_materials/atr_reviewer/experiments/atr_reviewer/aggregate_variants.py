import argparse
import csv
import json
import os
import sys
from pathlib import Path

CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description="Aggregate ATR reviewer experiment outputs.")
    ap.add_argument("--root_dir", required=True, help="Root directory containing variant subdirectories.")
    ap.add_argument("--output_csv", default=None)
    args = ap.parse_args()

    root = Path(args.root_dir).resolve()
    rows = []
    for variant_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        eval_dir = variant_dir / "atr_reviewer_eval"
        metrics_path = eval_dir / "final_test_metrics.json"
        summary_path = eval_dir / "summary.json"
        if not metrics_path.exists() or not summary_path.exists():
            continue
        metrics = read_json(metrics_path)
        summary = read_json(summary_path)
        rows.append(
            {
                "variant": variant_dir.name,
                "average_rougeL": metrics["average_rougeL"],
                "pearson_task_vs_metadata_cosine": summary["pearson_task_vs_metadata_cosine"],
                "spearman_task_vs_metadata_cosine": summary["spearman_task_vs_metadata_cosine"],
                "mean_final_norm": sum(summary["final_norms"]) / len(summary["final_norms"]),
                "min_final_norm": min(summary["final_norms"]),
                "max_final_norm": max(summary["final_norms"]),
                "final_norms": "|".join(str(x) for x in summary["final_norms"]),
                "final_snapshot": metrics["final_snapshot"],
            }
        )

    out_csv = Path(args.output_csv) if args.output_csv else root / "atr_reviewer_variant_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "average_rougeL",
                "pearson_task_vs_metadata_cosine",
                "spearman_task_vs_metadata_cosine",
                "mean_final_norm",
                "min_final_norm",
                "max_final_norm",
                "final_norms",
                "final_snapshot",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
