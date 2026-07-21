from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("experiments/prompt_r3_c2")
OUT = ROOT / "rebuttal_assets" / "figures"


def main():
    summary_path = ROOT / "outputs" / "aggregated_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)

    for eval_name in sorted({name for payload in summary.values() for name in payload.get("eval_summaries", {}).keys()}):
        variants = [v for v in ["current", "metadata_only", "temperature", "learnable_mlp"] if v in summary and eval_name in summary[v].get("eval_summaries", {})]
        if not variants:
            continue
        top1 = [summary[v]["eval_summaries"][eval_name]["top1_accuracy"] * 100 for v in variants]
        top3 = [summary[v]["eval_summaries"][eval_name]["top3_accuracy"] * 100 for v in variants]
        ece = [summary[v]["eval_summaries"][eval_name]["ece"] for v in variants]
        mean_alpha = [summary[v]["eval_summaries"][eval_name].get("mean_gate_alpha", 0.0) for v in variants]

        x = range(len(variants))
        fig, ax = plt.subplots(figsize=(9.0, 4.8))
        ax.plot(list(x), top1, marker="o", label="Top-1")
        ax.plot(list(x), top3, marker="s", label="Top-3")
        ax2 = ax.twinx()
        ax2.plot(list(x), mean_alpha, marker="^", linestyle="--", color="tab:green", label="Mean gate alpha")
        ax.set_xticks(list(x))
        ax.set_xticklabels(variants)
        ax.set_ylim(0, 100)
        ax2.set_ylim(0, 1.05)
        ax.set_ylabel("Routing Accuracy (%)")
        ax2.set_ylabel("Mean Gate Alpha")
        ax.set_title(f"Gate Probe Comparison on {eval_name}")
        ax.grid(alpha=0.25)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="best")
        plt.tight_layout()
        fig.savefig(OUT / f"{eval_name}_gate_probe.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    print(OUT)


if __name__ == "__main__":
    main()

