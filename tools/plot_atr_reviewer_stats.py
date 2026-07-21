import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_matrix_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    labels = rows[0][1:]
    data = np.array([[float(x) for x in row[1:]] for row in rows[1:]], dtype=float)
    row_labels = [row[0] for row in rows[1:]]
    return row_labels, labels, data


def short_label(name):
    mapping = {
        "Internal Medicine": "Internal",
        "Surgery": "Surgery",
        "Pediatrics": "Pediatrics",
        "Gynecology and Obstetrics": "Gyn-Obs",
        "Andrology": "Andrology",
        "Oncology": "Oncology",
    }
    return mapping.get(name, name)


def set_style():
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_final_norm_bar(task_rows):
    last_stage = max(int(r["stage_index"]) for r in task_rows)
    rows = [r for r in task_rows if int(r["stage_index"]) == last_stage]
    labels = [short_label(r["task_name"]) for r in rows]
    values = np.array([float(r["norm"]) for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    colors = ["#35618f", "#4f8fba", "#7aa6c2", "#b95d47", "#d28d5f", "#8a6f9b"]
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("L2 Norm")
    ax.set_title("Final-Stage Task Representation Norms")
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.12, f"{val:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "final_task_norms_bar.png", bbox_inches="tight")
    plt.close(fig)


def plot_stage_norm_line(stage_rows):
    stages = [int(r["stage_index"]) for r in stage_rows]
    means = np.array([float(r["mean_norm"]) for r in stage_rows], dtype=float)
    stds = np.array([float(r["std_norm"]) for r in stage_rows], dtype=float)
    mins = np.array([float(r["min_norm"]) for r in stage_rows], dtype=float)
    maxs = np.array([float(r["max_norm"]) for r in stage_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(stages, means, color="#1f4e79", marker="o", linewidth=2.0, label="Mean norm")
    ax.fill_between(stages, means - stds, means + stds, color="#8eb6d8", alpha=0.35, label="Mean ± std")
    ax.plot(stages, mins, color="#c66a4a", linestyle="--", linewidth=1.4, label="Min norm")
    ax.plot(stages, maxs, color="#4a8f58", linestyle="--", linewidth=1.4, label="Max norm")
    ax.set_xlabel("Continual Learning Stage")
    ax.set_ylabel("L2 Norm")
    ax.set_title("Norm Statistics Across Continual Learning Stages")
    ax.set_xticks(stages)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "stage_norm_statistics_line.png", bbox_inches="tight")
    plt.close(fig)


def plot_task_norm_trajectories(task_rows):
    rows = sorted(task_rows, key=lambda r: (int(r["task_index"]), int(r["stage_index"])))
    grouped = {}
    for row in rows:
        grouped.setdefault(row["task_name"], []).append(row)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = ["#1f4e79", "#2f6b9a", "#4f86b2", "#aa5a44", "#c98052", "#7d5f98"]
    for color, (task_name, items) in zip(colors, grouped.items()):
        xs = [int(r["stage_index"]) for r in items]
        ys = [float(r["norm"]) for r in items]
        ax.plot(xs, ys, marker="o", linewidth=1.8, color=color, label=short_label(task_name))
    ax.set_xlabel("Continual Learning Stage")
    ax.set_ylabel("L2 Norm")
    ax.set_title("Task-Vector Norm Trajectories")
    ax.set_xticks(sorted({int(r["stage_index"]) for r in rows}))
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "task_norm_trajectories.png", bbox_inches="tight")
    plt.close(fig)


def draw_heatmap(matrix, row_labels, col_labels, out_name, title, cmap="YlOrRd", vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(7.1, 6.1))
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels([short_label(x) for x in col_labels], rotation=35, ha="right")
    ax.set_yticklabels([short_label(x) for x in row_labels])
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if vmin is not None and vmax is not None:
                color = "black" if val < (vmax + vmin) / 2 else "white"
            else:
                color = "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Cosine Similarity", rotation=90)
    fig.tight_layout()
    fig.savefig(OUT_DIR / out_name, bbox_inches="tight")
    plt.close(fig)


def plot_similarity_scatter(pair_rows):
    x = np.array([float(r["metadata_cosine"]) for r in pair_rows], dtype=float)
    y = np.array([float(r["final_task_cosine"]) for r in pair_rows], dtype=float)
    coeff = np.polyfit(x, y, 1)
    fit = np.poly1d(coeff)
    pearson = float(np.corrcoef(x, y)[0, 1])

    fig, ax = plt.subplots(figsize=(6.6, 5.1))
    ax.scatter(x, y, s=46, color="#35618f", edgecolor="black", linewidth=0.5, alpha=0.9)
    xs = np.linspace(x.min() - 0.01, x.max() + 0.01, 100)
    ax.plot(xs, fit(xs), color="#c66a4a", linewidth=2.0)
    for row in pair_rows:
        label = f"{short_label(row['task_i'])} / {short_label(row['task_j'])}"
        ax.annotate(
            label,
            (float(row["metadata_cosine"]), float(row["final_task_cosine"])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.5,
        )
    ax.set_xlabel("Metadata Cosine Similarity")
    ax.set_ylabel("Task-Vector Cosine Similarity")
    ax.set_title(f"Task Similarity vs. Metadata Similarity (Pearson = {pearson:.3f})")
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "task_vs_metadata_similarity_scatter.png", bbox_inches="tight")
    plt.close(fig)


def plot_pca_trajectories(task_rows):
    rows = sorted(task_rows, key=lambda r: (int(r["task_index"]), int(r["stage_index"])))
    grouped = {}
    for row in rows:
        grouped.setdefault(row["task_name"], []).append(row)

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    colors = ["#1f4e79", "#2f6b9a", "#4f86b2", "#aa5a44", "#c98052", "#7d5f98"]
    for color, (task_name, items) in zip(colors, grouped.items()):
        xs = [float(r["pca_x"]) for r in items]
        ys = [float(r["pca_y"]) for r in items]
        ax.plot(xs, ys, marker="o", linewidth=1.5, color=color, alpha=0.95)
        for idx in range(len(items) - 1):
            dx = xs[idx + 1] - xs[idx]
            dy = ys[idx + 1] - ys[idx]
            ax.arrow(
                xs[idx],
                ys[idx],
                dx,
                dy,
                color=color,
                alpha=0.35,
                length_includes_head=True,
                head_width=0.22,
                head_length=0.34,
                linewidth=0.0,
            )
        ax.text(xs[-1] + 0.15, ys[-1], short_label(task_name), color=color, fontsize=9, va="center")

    for row in rows:
        ax.text(float(row["pca_x"]), float(row["pca_y"]), str(row["stage_index"]), fontsize=7, ha="center", va="center", color="white")

    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("PCA Trajectories of Task Representations Across Stages")
    ax.grid(alpha=0.2, linestyle="--")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "task_representation_pca_trajectories.png", bbox_inches="tight")
    plt.close(fig)


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Plot ATR reviewer statistics from a data directory.")
    ap.add_argument("--data_dir", default=str(ROOT / "analysis" / "atr_reviewer_real_statistics"))
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else data_dir / "figures"

    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    out_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    global OUT_DIR
    OUT_DIR = out_dir

    stage_rows = read_csv(data_dir / "stage_norm_summary.csv")
    task_rows = read_csv(data_dir / "task_norms_by_stage.csv")
    pair_rows = read_csv(data_dir / "task_vs_metadata_similarity_pairs.csv")
    row_labels, col_labels, task_cos = read_matrix_csv(data_dir / "final_task_cosine_matrix.csv")
    _, _, meta_cos = read_matrix_csv(data_dir / "final_metadata_cosine_matrix.csv")

    plot_final_norm_bar(task_rows)
    plot_stage_norm_line(stage_rows)
    plot_task_norm_trajectories(task_rows)
    draw_heatmap(task_cos, row_labels, col_labels, "final_task_cosine_heatmap.png", "Final Task-Vector Cosine Similarity", cmap="YlGnBu", vmin=0.84, vmax=1.00)
    draw_heatmap(meta_cos, row_labels, col_labels, "metadata_cosine_heatmap.png", "Metadata Cosine Similarity", cmap="YlOrBr", vmin=float(meta_cos.min()), vmax=1.00)
    plot_similarity_scatter(pair_rows)
    plot_pca_trajectories(task_rows)

    manifest = {
        "figure_dir": str(OUT_DIR.relative_to(ROOT)) if str(OUT_DIR).startswith(str(ROOT)) else str(OUT_DIR),
        "figures": [
            "final_task_norms_bar.png",
            "stage_norm_statistics_line.png",
            "task_norm_trajectories.png",
            "final_task_cosine_heatmap.png",
            "metadata_cosine_heatmap.png",
            "task_vs_metadata_similarity_scatter.png",
            "task_representation_pca_trajectories.png",
        ],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
