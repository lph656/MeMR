"""Plot publication-quality figures for saved MeMR scalability profiling results.

This script is plotting-only. It does not modify training, evaluation, model,
dataset, or profiling logic. It reads an existing scalability profiling result
directory, generates revised figures, and saves all new outputs under
`<result_dir>/figures_revised/` by default.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_memr")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLOR_MATCHING = "#6BAED6"
COLOR_AGGREGATION = "#74C476"
COLOR_OVERHEAD = "#FD8D3C"
COLOR_INFERENCE = "#9E9AC8"
COLOR_TASK_REPR = "#9ECAE1"
COLOR_FROZEN = "#A1D99B"
COLOR_TASK_MEMORY = "#FDD0A2"
COLOR_PEAK_GPU = "#BDBDBD"
COLOR_THRESHOLD = "#D95F5F"

FIG1_CAPTION = (
    "Fig. 1. Runtime scalability and bottleneck analysis of MeMR with increasing task numbers. "
    "Panel (a) shows the latency breakdown of task matching, module aggregation, and total MeMR "
    "overhead. Panel (b) shows their ratios to the total inference latency, where the dashed line "
    "indicates the 10% practical bottleneck threshold. Within the tested range up to 64 tasks, "
    "neither task matching nor module aggregation reaches this threshold."
)

FIG2_CAPTION = (
    "Fig. 2. Memory scalability and normalized scalability trend of MeMR. Panel (a) shows that "
    "task-related memory grows with the number of tasks and is dominated by frozen task-specific "
    "modules. Panel (b) normalizes runtime and memory metrics relative to the real six-task CMedCL "
    "setting, showing that measured runtime overhead remains stable while task-related memory "
    "increases with task count."
)

REQUIRED_COLUMNS = ["K"]
RUNTIME_COLUMNS = [
    "matching_time_mean_ms",
    "matching_time_std_ms",
    "aggregation_time_mean_ms",
    "aggregation_time_std_ms",
    "total_memr_overhead_mean_ms",
    "total_memr_overhead_std_ms",
    "total_inference_time_mean_ms",
    "matching_ratio_mean_percent",
    "aggregation_ratio_mean_percent",
]
MEMORY_COLUMNS = [
    "task_repr_memory_mb",
    "frozen_module_memory_mb",
    "task_related_memory_mb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate revised figures from saved scalability profiling results.")
    parser.add_argument("--result_dir", required=True, help="Result directory containing scalability_results.csv.")
    parser.add_argument("--csv_path", default=None, help="Optional explicit path to scalability_results.csv.")
    parser.add_argument("--output_dir", default=None, help="Optional explicit output directory. Defaults to <result_dir>/figures_revised.")
    parser.add_argument("--dpi", type=int, default=300, help="Figure dpi. Default: 300.")
    parser.add_argument("--formats", default="png,pdf,svg", help="Comma-separated figure formats. Default: png,pdf,svg.")
    parser.add_argument("--show_std", default="true", help="Whether to show standard deviation error bars. Default: true.")
    parser.add_argument("--bottleneck_threshold", type=float, default=10.0, help="Practical bottleneck threshold in percent. Default: 10.0.")
    return parser.parse_args()


def str_to_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def load_optional_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_optional_text(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_results(result_dir: str, csv_path: Optional[str]) -> pd.DataFrame:
    resolved_csv = csv_path if csv_path else os.path.join(result_dir, "scalability_results.csv")
    if not os.path.exists(resolved_csv):
        raise FileNotFoundError(f"scalability results CSV not found: {resolved_csv}")
    df = pd.read_csv(resolved_csv)
    if df.empty:
        raise ValueError(f"scalability results CSV is empty: {resolved_csv}")
    df = df.copy()
    df["K"] = pd.to_numeric(df["K"], errors="raise").astype(int)
    df = df.sort_values("K").reset_index(drop=True)
    return df


def ensure_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def compute_normalized_series(values: pd.Series, baseline_mask: pd.Series) -> pd.Series:
    if not baseline_mask.any():
        return pd.Series([np.nan] * len(values), index=values.index, dtype=float)
    baseline_value = values.loc[baseline_mask].iloc[0]
    if pd.isna(baseline_value) or float(baseline_value) == 0.0:
        return pd.Series([np.nan] * len(values), index=values.index, dtype=float)
    return values.astype(float) / float(baseline_value)


def compute_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    derived = df.copy()
    if {"matching_ratio_mean_percent", "aggregation_ratio_mean_percent"}.issubset(derived.columns):
        derived["total_overhead_ratio_percent"] = (
            derived["matching_ratio_mean_percent"].astype(float) + derived["aggregation_ratio_mean_percent"].astype(float)
        )
    else:
        derived["total_overhead_ratio_percent"] = np.nan

    k6_mask = derived["K"] == 6
    if "matching_time_mean_ms" in derived.columns:
        derived["normalized_matching_time_to_K6"] = compute_normalized_series(derived["matching_time_mean_ms"], k6_mask)
    else:
        derived["normalized_matching_time_to_K6"] = np.nan
    if "aggregation_time_mean_ms" in derived.columns:
        derived["normalized_aggregation_time_to_K6"] = compute_normalized_series(derived["aggregation_time_mean_ms"], k6_mask)
    else:
        derived["normalized_aggregation_time_to_K6"] = np.nan
    if "total_memr_overhead_mean_ms" in derived.columns:
        derived["normalized_total_overhead_to_K6"] = compute_normalized_series(derived["total_memr_overhead_mean_ms"], k6_mask)
    else:
        derived["normalized_total_overhead_to_K6"] = np.nan
    if "task_related_memory_mb" in derived.columns:
        derived["normalized_task_memory_to_K6"] = compute_normalized_series(derived["task_related_memory_mb"], k6_mask)
    else:
        derived["normalized_task_memory_to_K6"] = np.nan
    return derived


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D9D9D9",
            "axes.linewidth": 0.8,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.0,
            "grid.color": "#D9D9D9",
            "grid.alpha": 0.35,
            "grid.linewidth": 0.8,
            "font.size": 11,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: plt.Figure, stem: str, output_dir: str, formats: Iterable[str], dpi: int) -> List[str]:
    saved_paths: List[str] = []
    for fmt in formats:
        resolved_fmt = fmt.strip().lower()
        if not resolved_fmt:
            continue
        output_path = os.path.join(output_dir, f"{stem}.{resolved_fmt}")
        fig.savefig(output_path, dpi=dpi)
        saved_paths.append(output_path)
    return saved_paths


def maybe_errorbar(
    ax: plt.Axes,
    x: pd.Series,
    y: pd.Series,
    yerr: Optional[pd.Series],
    label: str,
    color: str,
    show_std: bool,
) -> None:
    line_kwargs = {
        "label": label,
        "color": color,
        "marker": "o",
        "markersize": 5.5,
        "linewidth": 2.0,
    }
    if show_std and yerr is not None and not yerr.isna().all():
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            capsize=3,
            elinewidth=1.0,
            alpha=0.95,
            **line_kwargs,
        )
    else:
        ax.plot(x, y, **line_kwargs)


def finalize_axis(ax: plt.Axes, x_values: Sequence[int], xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(list(x_values))
    ax.grid(True, axis="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_runtime_scalability(
    df: pd.DataFrame,
    output_dir: str,
    dpi: int,
    formats: Sequence[str],
    show_std: bool,
    bottleneck_threshold: float,
) -> List[str]:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6), constrained_layout=True)
    x = df["K"]

    runtime_specs = [
        ("matching_time_mean_ms", "matching_time_std_ms", "Task matching", COLOR_MATCHING),
        ("aggregation_time_mean_ms", "aggregation_time_std_ms", "Module aggregation", COLOR_AGGREGATION),
        ("total_memr_overhead_mean_ms", "total_memr_overhead_std_ms", "Total MeMR overhead", COLOR_OVERHEAD),
    ]
    for mean_col, std_col, label, color in runtime_specs:
        if mean_col not in df.columns:
            continue
        yerr = df[std_col] if std_col in df.columns else None
        maybe_errorbar(axes[0], x, df[mean_col], yerr, label, color, show_std)
    finalize_axis(axes[0], x.tolist(), "Number of Tasks K", "Latency (ms/query)", "(a) Latency Breakdown")
    axes[0].legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=2,
        columnspacing=1.2,
        handletextpad=0.6,
    )

    ratio_specs = [
        ("matching_ratio_mean_percent", "Matching ratio", COLOR_MATCHING),
        ("aggregation_ratio_mean_percent", "Aggregation ratio", COLOR_AGGREGATION),
        ("total_overhead_ratio_percent", "Total MeMR overhead ratio", COLOR_OVERHEAD),
    ]
    for column, label, color in ratio_specs:
        if column not in df.columns:
            continue
        axes[1].plot(
            x,
            df[column],
            label=label,
            color=color,
            marker="o",
            markersize=5.5,
            linewidth=2.0,
        )
    axes[1].axhline(
        y=bottleneck_threshold,
        color=COLOR_THRESHOLD,
        linestyle="--",
        linewidth=1.4,
        label=f"Practical bottleneck threshold ({bottleneck_threshold:.0f}%)",
    )
    finalize_axis(axes[1], x.tolist(), "Number of Tasks K", "Ratio to Total Inference Latency (%)", "(b) Runtime Bottleneck Ratio")
    axes[1].legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.24),
        ncol=2,
        columnspacing=1.2,
        handletextpad=0.6,
    )

    return save_figure(fig, "fig1_runtime_scalability", output_dir, formats, dpi)


def plot_memory_scalability(
    df: pd.DataFrame,
    output_dir: str,
    dpi: int,
    formats: Sequence[str],
) -> List[str]:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), constrained_layout=True)
    x = df["K"]

    memory_specs = [
        ("task_repr_memory_mb", "Task representation memory", COLOR_TASK_REPR),
        ("frozen_module_memory_mb", "Frozen module memory", COLOR_FROZEN),
        ("task_related_memory_mb", "Total task-related memory", COLOR_TASK_MEMORY),
    ]
    for column, label, color in memory_specs:
        if column not in df.columns:
            continue
        axes[0].plot(
            x,
            df[column],
            label=label,
            color=color,
            marker="o",
            markersize=5.5,
            linewidth=2.0,
        )
    finalize_axis(axes[0], x.tolist(), "Number of Tasks K", "Memory (MB)", "(a) Task-Related Memory Growth")
    axes[0].legend(frameon=False, loc="upper left")

    normalized_specs = [
        ("normalized_matching_time_to_K6", "Matching time / K=6", COLOR_MATCHING),
        ("normalized_aggregation_time_to_K6", "Aggregation time / K=6", COLOR_AGGREGATION),
        ("normalized_total_overhead_to_K6", "Total overhead / K=6", COLOR_OVERHEAD),
        ("normalized_task_memory_to_K6", "Task-related memory / K=6", COLOR_TASK_MEMORY),
    ]
    for column, label, color in normalized_specs:
        if column not in df.columns or df[column].isna().all():
            continue
        axes[1].plot(
            x,
            df[column],
            label=label,
            color=color,
            marker="o",
            markersize=5.5,
            linewidth=2.0,
        )
    finalize_axis(axes[1], x.tolist(), "Number of Tasks K", "Normalized Value Relative to K=6", "(b) Normalized Scalability Trend")
    axes[1].legend(frameon=False, loc="upper left")

    return save_figure(fig, "fig2_memory_scalability", output_dir, formats, dpi)


def write_captions(output_dir: str) -> str:
    path = os.path.join(output_dir, "figure_captions.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(FIG1_CAPTION + "\n\n" + FIG2_CAPTION + "\n")
    return path


def format_metric(value: object, digits: int = 3, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{float(value):.{digits}f}{suffix}"


def select_row_by_k(df: pd.DataFrame, k_value: int) -> Optional[pd.Series]:
    matched = df[df["K"] == k_value]
    if matched.empty:
        return None
    return matched.iloc[0]


def write_summary(df: pd.DataFrame, output_dir: str, bottleneck_threshold: float) -> str:
    path = os.path.join(output_dir, "plot_summary_for_response.md")
    row_k6 = select_row_by_k(df, 6)
    row_k64 = select_row_by_k(df, 64)

    max_matching_ratio = float(df["matching_ratio_mean_percent"].max()) if "matching_ratio_mean_percent" in df.columns else np.nan
    max_aggregation_ratio = float(df["aggregation_ratio_mean_percent"].max()) if "aggregation_ratio_mean_percent" in df.columns else np.nan
    threshold_exceeded = (
        (not np.isnan(max_matching_ratio) and max_matching_ratio > bottleneck_threshold)
        or (not np.isnan(max_aggregation_ratio) and max_aggregation_ratio > bottleneck_threshold)
    )

    aggregation_larger = False
    if {"matching_time_mean_ms", "aggregation_time_mean_ms"}.issubset(df.columns):
        comparable = df[["matching_time_mean_ms", "aggregation_time_mean_ms"]].dropna()
        if not comparable.empty:
            aggregation_larger = bool((comparable["aggregation_time_mean_ms"] > comparable["matching_time_mean_ms"]).all())

    frozen_dominates = False
    if {"frozen_module_memory_mb", "task_repr_memory_mb", "task_related_memory_mb"}.issubset(df.columns):
        comparable_mem = df[["frozen_module_memory_mb", "task_repr_memory_mb", "task_related_memory_mb"]].dropna()
        if not comparable_mem.empty:
            frozen_dominates = bool((comparable_mem["frozen_module_memory_mb"] > comparable_mem["task_repr_memory_mb"]).all())

    def row_summary(label: str, row: Optional[pd.Series]) -> List[str]:
        if row is None:
            return [f"- {label}: unavailable in the current CSV."]
        return [
            (
                f"- {label}: matching={format_metric(row.get('matching_time_mean_ms'), suffix=' ms')}, "
                f"aggregation={format_metric(row.get('aggregation_time_mean_ms'), suffix=' ms')}, "
                f"total overhead={format_metric(row.get('total_memr_overhead_mean_ms'), suffix=' ms')}, "
                f"matching ratio={format_metric(row.get('matching_ratio_mean_percent'), suffix='%')}, "
                f"aggregation ratio={format_metric(row.get('aggregation_ratio_mean_percent'), suffix='%')}, "
                f"task-related memory={format_metric(row.get('task_related_memory_mb'), suffix=' MB')}."
            )
        ]

    lines = [
        "Scalability plotting summary",
        "",
        *row_summary("K=6", row_k6),
        *row_summary("K=64", row_k64),
        "",
        (
            f"- Bottleneck threshold check ({bottleneck_threshold:.1f}%): "
            + ("at least one component exceeds the threshold." if threshold_exceeded else "neither matching nor aggregation exceeds the threshold.")
        ),
        f"- Aggregation larger than matching: {'yes' if aggregation_larger else 'not consistently in all measured points'}.",
        f"- Task-related memory dominated by frozen modules: {'yes' if frozen_dominates else 'not established from the available columns'}.",
        "- Interpretation note: The theoretical complexity grows linearly with the number of tasks, while the empirical profiling results show that the measured overhead remains stable within the tested range up to 64 tasks.",
        "- Warning: Do not claim that empirical latency increases strictly linearly with K unless a separate measurement clearly demonstrates that trend.",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def maybe_copy_context_files(result_dir: str, output_dir: str) -> None:
    context_paths = {
        "config.json": load_optional_json(os.path.join(result_dir, "config.json")),
        "task_bank_generation.json": load_optional_json(os.path.join(result_dir, "task_bank_generation.json")),
        "summary_for_response.md": load_optional_text(os.path.join(result_dir, "summary_for_response.md")),
    }
    context_summary_path = os.path.join(output_dir, "context_snapshot.md")
    lines = ["# Context Snapshot", ""]
    for name, payload in context_paths.items():
        lines.append(f"## {name}")
        if payload is None:
            lines.append("Not found.")
        elif isinstance(payload, str):
            lines.append(payload.strip() or "(empty)")
        else:
            lines.append("```json")
            lines.append(json.dumps(payload, ensure_ascii=False, indent=2))
            lines.append("```")
        lines.append("")
    with open(context_summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> int:
    args = parse_args()
    show_std = str_to_bool(args.show_std)
    formats = [item.strip() for item in args.formats.split(",") if item.strip()]

    result_dir = os.path.abspath(args.result_dir)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(result_dir, "figures_revised")
    os.makedirs(output_dir, exist_ok=True)

    apply_plot_style()
    df = load_results(result_dir, args.csv_path)
    ensure_columns(df, REQUIRED_COLUMNS)
    derived_df = compute_derived_metrics(df)

    plot_source_columns = [
        "K",
        "matching_time_mean_ms",
        "matching_time_std_ms",
        "aggregation_time_mean_ms",
        "aggregation_time_std_ms",
        "total_memr_overhead_mean_ms",
        "total_memr_overhead_std_ms",
        "plm_forward_time_mean_ms",
        "plm_forward_time_std_ms",
        "total_inference_time_mean_ms",
        "total_inference_time_std_ms",
        "matching_ratio_mean_percent",
        "aggregation_ratio_mean_percent",
        "agg_match_ratio_mean",
        "task_repr_memory_mb",
        "frozen_module_memory_mb",
        "task_related_memory_mb",
        "peak_gpu_allocated_mb",
        "peak_gpu_reserved_mb",
        "total_overhead_ratio_percent",
        "normalized_matching_time_to_K6",
        "normalized_aggregation_time_to_K6",
        "normalized_total_overhead_to_K6",
        "normalized_task_memory_to_K6",
    ]
    present_columns = [column for column in plot_source_columns if column in derived_df.columns]
    plot_source_path = os.path.join(output_dir, "plot_source_values.csv")
    derived_df[present_columns].to_csv(plot_source_path, index=False)

    saved_files: List[str] = []
    saved_files.extend(plot_runtime_scalability(derived_df, output_dir, args.dpi, formats, show_std, args.bottleneck_threshold))
    saved_files.extend(plot_memory_scalability(derived_df, output_dir, args.dpi, formats))
    caption_path = write_captions(output_dir)
    summary_path = write_summary(derived_df, output_dir, args.bottleneck_threshold)
    maybe_copy_context_files(result_dir, output_dir)

    print(f"[plot] Loaded CSV from: {args.csv_path if args.csv_path else os.path.join(result_dir, 'scalability_results.csv')}")
    print(f"[plot] Output directory: {output_dir}")
    print(f"[plot] Saved plot source CSV: {plot_source_path}")
    print(f"[plot] Saved captions: {caption_path}")
    print(f"[plot] Saved summary: {summary_path}")
    for path in saved_files:
        print(f"[plot] Saved figure: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
