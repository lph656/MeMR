from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_high_risk_safety")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common import ensure_dir, read_csv, setup_logger, write_environment_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate figures for high-risk safety evaluation.")
    parser.add_argument("--summary", required=True, help="summary_auto.csv")
    parser.add_argument("--category_summary", required=True, help="summary_by_category.csv")
    parser.add_argument("--matching_weights", required=True, help="MeMR matching weights csv (may be missing).")
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    return parser.parse_args()


def parse_percent(value: str) -> float:
    return float(value)


def save_figure(fig, png_path: str, pdf_path: str) -> None:
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path, dpi=300)
    plt.close(fig)


def plot_safety_metrics(summary_rows: List[Dict[str, str]], figures_dir: str) -> None:
    if not summary_rows:
        return
    methods = [row["Method"] for row in summary_rows]
    metrics = ["URR ↓", "AER ↑", "CRR ↑", "OCR ↓", "RRR ↑"]
    x = np.arange(len(methods))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, metric in enumerate(metrics):
        values = [parse_percent(row[metric]) for row in summary_rows]
        ax.bar(x + (idx - 2) * width, values, width=width, label=metric.replace(" ↓", "").replace(" ↑", ""))

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20)
    ax.set_ylabel("Percentage")
    ax.set_title("Safety Metrics by Method")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    save_figure(fig, os.path.join(figures_dir, "safety_metrics_bar.png"), os.path.join(figures_dir, "safety_metrics_bar.pdf"))


def plot_category_heatmap(category_rows: List[Dict[str, str]], figures_dir: str) -> None:
    if not category_rows:
        return
    methods = sorted({row["Method"] for row in category_rows})
    categories = sorted({row["Category"] for row in category_rows})
    matrix = np.full((len(methods), len(categories)), np.nan)
    for row in category_rows:
        i = methods.index(row["Method"])
        j = categories.index(row["Category"])
        matrix[i, j] = float(row["SafetyScore"])

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(categories, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_title("Category-level Safety Score")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, os.path.join(figures_dir, "category_safety_heatmap.png"), os.path.join(figures_dir, "category_safety_heatmap.pdf"))


def plot_matching_heatmap(matching_rows: List[Dict[str, str]], figures_dir: str) -> None:
    departments = [
        "Internal Medicine",
        "Surgery",
        "Pediatrics",
        "Andrology",
        "Gynecology and Obstetrics",
        "Oncology",
    ]
    selected_rows = matching_rows[:12]
    labels = [row["case_id"] for row in selected_rows]
    matrix = []
    for row in selected_rows:
        matrix.append([float(row[col]) if row[col] not in {"", None} else 0.0 for col in departments])
    matrix_np = np.array(matrix)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix_np, aspect="auto", cmap="OrRd")
    ax.set_xticks(np.arange(len(departments)))
    ax.set_xticklabels(departments, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("MeMR Matching Weights on Representative Cases")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, os.path.join(figures_dir, "matching_weights_heatmap.png"), os.path.join(figures_dir, "matching_weights_heatmap.pdf"))


def plot_cdss_pipeline(figures_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")
    steps = [
        "Patient consultation input",
        "MeMR task matching and response generation",
        "High-risk trigger",
        "Conflict flagging",
        "Uncertainty warning",
        "Human-in-the-loop clinician review",
        "Final clinical decision by licensed clinicians",
        "Audit logging",
        "Version control and rollback",
        "Privacy and compliance review",
    ]
    x_positions = np.linspace(0.05, 0.95, len(steps))
    y = 0.5
    for idx, (x, step) in enumerate(zip(x_positions, steps)):
        ax.text(x, y, step, ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.4", fc="#e8f1fa", ec="#3a6ea5"))
        if idx < len(steps) - 1:
            ax.annotate("", xy=(x_positions[idx + 1] - 0.03, y), xytext=(x + 0.03, y), arrowprops=dict(arrowstyle="->", lw=1.5))
    fig.suptitle("Clinical Decision Support Deployment Pipeline", fontsize=14)
    save_figure(fig, os.path.join(figures_dir, "cdss_pipeline.png"), os.path.join(figures_dir, "cdss_pipeline.pdf"))


def main() -> int:
    args = parse_args()
    ensure_dir(args.output_dir)
    write_environment_snapshot(args.output_dir)
    logger = setup_logger("high_risk_safety.figures", args.output_dir)

    figures_dir = os.path.join(args.output_dir, "figures")
    ensure_dir(figures_dir)
    summary_rows = read_csv(args.summary)
    category_rows = read_csv(args.category_summary)

    if summary_rows:
        plot_safety_metrics(summary_rows, figures_dir)
    else:
        logger.info("Summary CSV is empty; skipping safety metrics bar figure.")
    if category_rows:
        plot_category_heatmap(category_rows, figures_dir)
    else:
        logger.info("Category summary CSV is empty; skipping category heatmap.")
    plot_cdss_pipeline(figures_dir)

    if os.path.isfile(args.matching_weights):
        matching_rows = read_csv(args.matching_weights)
        if matching_rows:
            plot_matching_heatmap(matching_rows, figures_dir)
        else:
            logger.info("Matching weights file exists but is empty; skipping matching heatmap.")
    else:
        logger.info("Matching weights file not found; skipping matching heatmap.")

    logger.info("Saved figures to %s", figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
