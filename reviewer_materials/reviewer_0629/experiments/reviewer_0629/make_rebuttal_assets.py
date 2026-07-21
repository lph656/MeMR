from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("experiments/reviewer_0629")
OUT_DIR = ROOT / "rebuttal_assets"
FIG_DIR = OUT_DIR / "figures"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_fig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_cold_start(summary):
    settings = summary["settings"]
    labels = ["metadata_init", "random_init"]
    top1 = [settings[k]["top1_new_task_accuracy"] * 100 for k in labels]
    top3 = [settings[k]["top3_new_task_accuracy"] * 100 for k in labels]

    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar([i - width / 2 for i in x], top1, width=width, label="Top-1")
    ax.bar([i + width / 2 for i in x], top3, width=width, label="Top-3")
    ax.set_xticks(list(x))
    ax.set_xticklabels(["Metadata Init", "Random Init"])
    ax.set_ylabel("Routing Accuracy (%)")
    ax.set_title("Cold-Start Routing on Held-Out Department")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save_fig(FIG_DIR / "cold_start_comparison.png")


def plot_routing_noise(summary):
    order = ["none", "char_delete_10", "char_swap_10", "punctuation_10", "filler_10"]
    x = range(len(order))
    top1 = [summary[k]["top1_accuracy"] * 100 for k in order]
    top3 = [summary[k]["top3_accuracy"] * 100 for k in order]

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(list(x), top1, marker="o", linewidth=2.0, label="Top-1")
    ax.plot(list(x), top3, marker="s", linewidth=2.0, label="Top-3")
    ax.set_xticks(list(x))
    ax.set_xticklabels(["Clean", "Delete", "Swap", "Punct.", "Filler"])
    ax.set_ylabel("Routing Accuracy (%)")
    ax.set_title("Routing Stability under Input Noise")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    save_fig(FIG_DIR / "routing_noise_robustness.png")


def plot_noisy_label(summary):
    order = ["none", "char_delete_10", "char_swap_10"]
    x = range(len(order))
    top1 = [summary[k]["top1_accuracy"] * 100 for k in order]
    top3 = [summary[k]["top3_accuracy"] * 100 for k in order]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(list(x), top1, marker="o", linewidth=2.0, label="Top-1")
    ax.plot(list(x), top3, marker="s", linewidth=2.0, label="Top-3")
    ax.set_xticks(list(x))
    ax.set_xticklabels(["Clean", "Delete", "Swap"])
    ax.set_ylabel("Routing Accuracy (%)")
    ax.set_title("Noisy-Label Training: Routing Stability")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    save_fig(FIG_DIR / "noisy_label_routing_stability.png")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_cold_start(load_json(ROOT / "outputs" / "cold_start_zhongliuke" / "summary.json"))
    plot_routing_noise(load_json(ROOT / "outputs" / "routing_baseline" / "summary.json"))
    plot_noisy_label(load_json(ROOT / "outputs" / "routing_noisy_labels" / "summary.json"))
    print(FIG_DIR)


if __name__ == "__main__":
    main()

