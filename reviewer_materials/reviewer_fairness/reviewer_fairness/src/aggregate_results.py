"""
Aggregate reviewer fairness experiment results.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reviewer_fairness.src.common import (
    DEFAULT_TASK_ORDER,
    METRIC_FIELDNAMES,
    RESULTS_ROOT,
    mean_or_none,
    read_json,
    write_markdown_table,
    write_table_csv,
)


SUMMARY_ROOT = RESULTS_ROOT / "summary"
REFERENCE_ROOT = RESULTS_ROOT / "reference"


def load_metric_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metrics_path in sorted(REFERENCE_ROOT.glob("**/metrics.json")):
        payload = read_json(metrics_path)
        row = {
            "Method": payload.get("method"),
            "Order": payload.get("order"),
            "IM": payload.get("department_scores", {}).get("IM", "N/A"),
            "S": payload.get("department_scores", {}).get("S", "N/A"),
            "P": payload.get("department_scores", {}).get("P", "N/A"),
            "GO": payload.get("department_scores", {}).get("GO", "N/A"),
            "A": payload.get("department_scores", {}).get("A", "N/A"),
            "O": payload.get("department_scores", {}).get("O", "N/A"),
            "Average": payload.get("average", "N/A"),
            "FWT": payload.get("FWT", "N/A") if payload.get("FWT", None) is not None else "N/A",
            "FR": payload.get("FR", "N/A") if payload.get("FR", None) is not None else "N/A",
            "BWT": payload.get("BWT", "N/A") if payload.get("BWT", None) is not None else "N/A",
            "Trainable Params": payload.get("trainable_params", "N/A"),
            "Total Params": payload.get("total_params", "N/A"),
            "Notes": payload.get("notes", ""),
        }
        rows.append(row)
    return rows


def compute_group_stats(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["Method"]), []).append(row)

    stats_rows: List[Dict[str, Any]] = []
    for method, method_rows in sorted(grouped.items()):
        if len(method_rows) < 2:
            continue
        numeric_cols = ["IM", "S", "P", "GO", "A", "O", "Average", "FWT", "FR", "BWT"]
        mean_row: Dict[str, Any] = {
            "Method": f"{method} (mean)",
            "Order": "mean",
            "Trainable Params": "N/A",
            "Total Params": "N/A",
            "Notes": "Computed across available orders.",
        }
        std_row: Dict[str, Any] = {
            "Method": f"{method} (std)",
            "Order": "std",
            "Trainable Params": "N/A",
            "Total Params": "N/A",
            "Notes": "Computed across available orders.",
        }

        for col in numeric_cols:
            values = [row[col] for row in method_rows if isinstance(row[col], (int, float))]
            if values:
                mean_value = sum(values) / len(values)
                variance = sum((value - mean_value) ** 2 for value in values) / len(values)
                std_value = math.sqrt(variance)
                mean_row[col] = round(mean_value, 4)
                std_row[col] = round(std_value, 4)
            else:
                mean_row[col] = "N/A"
                std_row[col] = "N/A"
        stats_rows.extend([mean_row, std_row])
    return stats_rows


def main() -> int:
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    rows = load_metric_rows()
    stats_rows = compute_group_stats(rows)

    reference_rows = [row for row in rows if row["Method"] in {
        "sequential_ft",
        "joint_training",
        "single_task_oracle",
        "metadata_only_routing",
        "er_lora",
        "memr",
    }]
    all_rows = rows + stats_rows

    write_table_csv(reference_rows, SUMMARY_ROOT / "reference_baselines_summary.csv", METRIC_FIELDNAMES)
    write_markdown_table(reference_rows, SUMMARY_ROOT / "reference_baselines_summary.md", METRIC_FIELDNAMES, "Reference Baselines Summary")

    write_table_csv(all_rows, SUMMARY_ROOT / "all_results_summary.csv", METRIC_FIELDNAMES)
    write_markdown_table(all_rows, SUMMARY_ROOT / "all_results_summary.md", METRIC_FIELDNAMES, "All Results Summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
