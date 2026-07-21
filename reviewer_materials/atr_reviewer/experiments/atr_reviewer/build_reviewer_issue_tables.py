import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUITE_DIR = ROOT / "checkpoints_continual_keshi_llama" / "atr_reviewer_suite"
BASELINE_DIR = ROOT / "analysis" / "atr_reviewer_real_statistics"
OUT_DIR = ROOT / "analysis" / "atr_reviewer_issue_tables"

VARIANTS = ["wo_atr", "norm_only", "hard_orth", "cosine_soft_orth", "full_atr", "lm_off_full_atr"]
TASKS = [
    "Internal Medicine",
    "Surgery",
    "Pediatrics",
    "Gynecology and Obstetrics",
    "Andrology",
    "Oncology",
]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_dict(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


def load_variant_eval(variant: str):
    eval_dir = SUITE_DIR / variant / "atr_reviewer_eval"
    if not eval_dir.exists():
        return None
    return {
        "summary": read_json(eval_dir / "summary.json"),
        "metrics": read_json(eval_dir / "final_test_metrics.json"),
        "stage_norm_summary": read_csv_dict(eval_dir / "stage_norm_summary.csv"),
        "task_norms_by_stage": read_csv_dict(eval_dir / "task_norms_by_stage.csv"),
        "final_task_cosine_matrix": read_csv_rows(eval_dir / "final_task_cosine_matrix.csv"),
        "final_metadata_cosine_matrix": read_csv_rows(eval_dir / "final_metadata_cosine_matrix.csv"),
        "task_vs_metadata_similarity_pairs": read_csv_dict(eval_dir / "task_vs_metadata_similarity_pairs.csv"),
    }


def nan_warning_count(variant: str) -> int:
    log_path = SUITE_DIR / variant / "log.txt"
    if not log_path.exists():
        return 0
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    return sum(1 for line in text.splitlines() if "NaN or Inf found in input tensor" in line)


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def build_issue_1(variant_data, baseline_summary):
    rows = []
    for variant in VARIANTS:
        data = variant_data[variant]
        norms = data["summary"]["final_norms"]
        rows.append(
            {
                "source": "variant",
                "name": variant,
                "num_stages": data["summary"]["num_stages"],
                "mean_final_norm": round(sum(norms) / len(norms), 4),
                "min_final_norm": round(min(norms), 4),
                "max_final_norm": round(max(norms), 4),
                "final_norms": "|".join(str(x) for x in norms),
                "norm_collapse_observed": "no",
            }
        )
    rows.append(
        {
            "source": "baseline",
            "name": "order1_compose_peft",
            "num_stages": baseline_summary["num_stages"],
            "mean_final_norm": round(sum(baseline_summary["final_norms"]) / len(baseline_summary["final_norms"]), 4),
            "min_final_norm": round(min(baseline_summary["final_norms"]), 4),
            "max_final_norm": round(max(baseline_summary["final_norms"]), 4),
            "final_norms": "|".join(str(x) for x in baseline_summary["final_norms"]),
            "norm_collapse_observed": "no",
        }
    )
    write_csv(
        OUT_DIR / "issue_1_norm_collapse_table.csv",
        [
            "source",
            "name",
            "num_stages",
            "mean_final_norm",
            "min_final_norm",
            "max_final_norm",
            "final_norms",
            "norm_collapse_observed",
        ],
        rows,
    )
    write_md(
        OUT_DIR / "issue_1_norm_collapse_explanation.md",
        """
# Issue 1：ATR 相关损失是否会把任务表示压缩到接近 0？

结论：目前结果可以支持“没有观察到向 0 的显著 collapse”，但不能支持“LM-off 时一定出现向 0 collapse”。

使用证据：
- `issue_1_norm_collapse_table.csv`
- `analysis/atr_reviewer_real_statistics/stage_norm_summary.csv`
- `checkpoints_continual_keshi_llama/atr_reviewer_suite/*/atr_reviewer_eval/stage_norm_summary.csv`

解释：
- 目前所有可用变体的最终 task-vector norm 都大致落在 32 到 37 之间，明显不接近 0。
- 原始 baseline 运行 `order1_compose_peft` 在 6 个阶段中也表现出稳定的非零 norm。
- 因此，基于现有实验，可以支持一个较弱但明确的结论：这些运行里学到的任务表示没有表现出向 0 明显塌缩的现象。
- 即使在 `lm_off_full_atr` 中，也没有看到 norm 直接掉到接近 0；它的问题更像是“表示几乎停留在初始 metadata 几何附近，同时文本表现下降”。

局限性：
- 这仍然不能简单等价为“language-model objective 唯一阻止了 collapse”。
- 因为当前 `lm_off_full_atr` 更像是在无 LM loss 时几乎不偏离初始化表示，而不是发生向零塌缩，所以 reviewer 的这一因果问题目前只能得到部分回答。
        """,
    )


def build_issue_2(variant_data):
    rows = []
    for variant in VARIANTS:
        for row in variant_data[variant]["stage_norm_summary"]:
            rows.append(
                {
                    "variant": variant,
                    "stage_index": row["stage_index"],
                    "stage_label": row["stage_label"],
                    "trained_task_name": row["trained_task_name"],
                    "mean_norm": row["mean_norm"],
                    "std_norm": row["std_norm"],
                    "min_norm": row["min_norm"],
                    "max_norm": row["max_norm"],
                }
            )
    write_csv(
        OUT_DIR / "issue_2_stage_norm_trajectories_table.csv",
        [
            "variant",
            "stage_index",
            "stage_label",
            "trained_task_name",
            "mean_norm",
            "std_norm",
            "min_norm",
            "max_norm",
        ],
        rows,
    )
    write_md(
        OUT_DIR / "issue_2_stage_norm_trajectories_explanation.md",
        """
# Issue 2：跨阶段表示轨迹是否显示出持续塌缩？

结论：目前结果可以部分支持“没有观察到明显的 collapse 轨迹”。

使用证据：
- `issue_2_stage_norm_trajectories_table.csv`
- 各变体目录下 `atr_reviewer_eval/figures/` 中的 PCA 轨迹图

解释：
- 在当前所有可用变体中，随着任务阶段推进，stage-wise mean norm 基本保持稳定，并没有随着任务累积而逐步逼近 0。
- 这可以支持一个 reviewer-facing 的结论：表示轨迹没有显示出单调收缩到共同零点的趋势。

局限性：
- 当前这些跨阶段轨迹可以作为“没有明显 collapse”的证据。
- 但它们本身还不足以证明某一种 ATR 设计显著优于另一种设计。
        """,
    )


def build_issue_3(variant_data):
    rows = []
    for variant in VARIANTS:
        matrix_rows = variant_data[variant]["final_task_cosine_matrix"][1:]
        for row in matrix_rows:
            task_i = row[0]
            for task_j, value in zip(TASKS, row[1:]):
                rows.append(
                    {
                        "variant": variant,
                        "task_i": task_i,
                        "task_j": task_j,
                        "cosine_similarity": value,
                    }
                )
    write_csv(
        OUT_DIR / "issue_3_pairwise_cosine_matrices_table.csv",
        ["variant", "task_i", "task_j", "cosine_similarity"],
        rows,
    )
    write_md(
        OUT_DIR / "issue_3_pairwise_cosine_matrices_explanation.md",
        """
# Issue 3：pairwise cosine similarity matrix 能说明什么？

结论：这些结果可以报告，但它们对不同变体之间的区分能力还比较弱。

使用证据：
- `issue_3_pairwise_cosine_matrices_table.csv`
- 各变体目录下 `atr_reviewer_eval/figures/` 中的 cosine 热力图

解释：
- 当前所有可用变体都表现出较高但并不完全相同的 cosine 结构，这更符合“稳定且有结构的任务表示”，而不是“完全塌缩”的情况。
- 但是，`wo_atr`、`norm_only`、`hard_orth`、`cosine_soft_orth` 和 `full_atr` 之间的 cosine matrix 仍然非常接近。

局限性：
- 这些矩阵足以满足 reviewer 提出的“请报告 cosine similarity structure”这一要求。
- 但它们还不足以强有力地证明完整 ATR 设计在几何结构上明显优于其他变体。
        """,
    )


def build_issue_4(variant_data, baseline_summary):
    rows = []
    for variant in VARIANTS:
        summary = variant_data[variant]["summary"]
        pair_rows = variant_data[variant]["task_vs_metadata_similarity_pairs"]
        mean_abs_gap = sum(
            abs(float(row["final_task_cosine"]) - float(row["metadata_cosine"])) for row in pair_rows
        ) / len(pair_rows)
        rows.append(
            {
                "source": "variant",
                "name": variant,
                "pearson_task_vs_metadata_cosine": summary["pearson_task_vs_metadata_cosine"],
                "spearman_task_vs_metadata_cosine": summary["spearman_task_vs_metadata_cosine"],
                "mean_abs_task_minus_metadata_cosine_gap": round(mean_abs_gap, 6),
            }
        )
    baseline_pairs = read_csv_dict(BASELINE_DIR / "task_vs_metadata_similarity_pairs.csv")
    baseline_gap = sum(
        abs(float(row["final_task_cosine"]) - float(row["metadata_cosine"])) for row in baseline_pairs
    ) / len(baseline_pairs)
    rows.append(
        {
            "source": "baseline",
            "name": "order1_compose_peft",
            "pearson_task_vs_metadata_cosine": baseline_summary["pearson_task_vs_metadata_cosine"],
            "spearman_task_vs_metadata_cosine": baseline_summary["spearman_task_vs_metadata_cosine"],
            "mean_abs_task_minus_metadata_cosine_gap": round(baseline_gap, 6),
        }
    )
    write_csv(
        OUT_DIR / "issue_4_metadata_similarity_alignment_table.csv",
        [
            "source",
            "name",
            "pearson_task_vs_metadata_cosine",
            "spearman_task_vs_metadata_cosine",
            "mean_abs_task_minus_metadata_cosine_gap",
        ],
        rows,
    )
    write_md(
        OUT_DIR / "issue_4_metadata_similarity_alignment_explanation.md",
        """
# Issue 4：表示相似度是否与医学领域相似度一致？

结论：这一点目前是有支持证据的。

使用证据：
- `issue_4_metadata_similarity_alignment_table.csv`
- 各变体和 baseline 提取结果中的 `task_vs_metadata_similarity_pairs.csv`

解释：
- 当前所有可用运行都显示 task-vector cosine similarity 与 metadata cosine similarity 之间具有很高的一致性。
- 其中 `lm_off_full_atr` 的一致性几乎达到 1，这说明在去掉 LM loss 后，task vectors 更倾向于停留在初始化的 metadata 几何附近。
- 表中的 `mean_abs_task_minus_metadata_cosine_gap` 也可以直接量化“最终表示相对初始化 metadata 几何偏离了多少”。
- 因此，这一组结果既支持“表示结构与医学领域相似性一致”，也提示“LM objective 的一个作用可能是推动表示偏离纯初始化几何，而不是单纯维持原状”。

局限性：
- 这组证据支持“表示结构学到了医学领域关系”这一点。
- 但它本身并不能单独回答“哪一种 ATR 消融设计最好”。
        """,
    )


def build_issue_5(variant_data):
    rows = []
    for variant in VARIANTS:
        summary = variant_data[variant]["summary"]
        metrics = variant_data[variant]["metrics"]
        norms = summary["final_norms"]
        pair_rows = variant_data[variant]["task_vs_metadata_similarity_pairs"]
        mean_abs_gap = sum(
            abs(float(row["final_task_cosine"]) - float(row["metadata_cosine"])) for row in pair_rows
        ) / len(pair_rows)
        rows.append(
            {
                "variant": variant,
                "average_rougeL": metrics["average_rougeL"],
                "pearson_task_vs_metadata_cosine": summary["pearson_task_vs_metadata_cosine"],
                "spearman_task_vs_metadata_cosine": summary["spearman_task_vs_metadata_cosine"],
                "mean_abs_task_minus_metadata_cosine_gap": round(mean_abs_gap, 6),
                "mean_final_norm": round(sum(norms) / len(norms), 4),
                "min_final_norm": round(min(norms), 4),
                "max_final_norm": round(max(norms), 4),
                "nan_warning_count_in_log": nan_warning_count(variant),
                "supports_clean_variant_comparison": "no" if nan_warning_count(variant) > 0 else "partially",
            }
        )
    write_csv(
        OUT_DIR / "issue_5_ablation_comparison_table.csv",
        [
            "variant",
            "average_rougeL",
            "pearson_task_vs_metadata_cosine",
            "spearman_task_vs_metadata_cosine",
            "mean_abs_task_minus_metadata_cosine_gap",
            "mean_final_norm",
            "min_final_norm",
            "max_final_norm",
            "nan_warning_count_in_log",
            "supports_clean_variant_comparison",
        ],
        rows,
    )
    write_md(
        OUT_DIR / "issue_5_ablation_comparison_explanation.md",
        """
# Issue 5：当前更强的 ATR 消融是否已经足以支持 preferred design？

结论：目前仍不能支持“完整 ATR 设计已经被干净证明为最佳”，但现在已经比上一版多出一条可用信息：`LM-off` 会明显损害文本表现，并使表示更接近初始化 metadata 几何。

使用证据：
- `issue_5_ablation_comparison_table.csv`
- `checkpoints_continual_keshi_llama/atr_reviewer_suite/*/atr_reviewer_eval/` 中各变体的 summary 和 metric 文件

解释：
- 当前这组消融实验足以报告一些初步趋势，例如“没有明显 norm collapse”、“表示仍然与 metadata 高度一致”，以及“关闭 LM loss 会显著降低文本表现，并使表示更贴近初始化 metadata 结构”。
- 但是，它仍然不能干净地支持如下更强的结论：`full_atr` 明显优于 `hard_orth`、`norm_only` 或 `cosine_soft_orth`。

为什么现在还不够：
- `hard_orth`、`cosine_soft_orth` 和 `full_atr` 的训练日志里都出现了大量 `NaN/Inf` 警告，这会削弱它们作为稳定消融结果的证据力度。
- 当前运行中，`full_atr` 和 `hard_orth` 最终得到的 task-vector 状态实际上相同，因此这两者之间的对比目前并不可信。
- 虽然 `lm_off_full_atr` 已经跑完，并且显示 `average_rougeL` 明显下降，但它没有表现出“向 0 collapse”，而是表现为“几乎停留在 metadata 初始化几何附近”。

当前可以使用的 reviewer-facing 表述：
- 现有消融表格可以作为“部分证据”来展示。
- 可以较谨慎地写：去掉 LM objective 会导致表示更新不足、文本性能下降，因此 LM objective 对避免退化是重要的。
- 但仍不应据此写出“完整 ATR 设计已经被当前结果严格证明为最优”的强结论。
        """,
    )


def build_issue_6():
    rows = [
        {
            "sub_issue": "LM objective prevents collapse",
            "required_artifact": "complete lm_off_full_atr atr_reviewer_eval directory and a clear causal interpretation",
            "current_status": "partial",
            "can_reply_now": "partially",
            "reason": "The completed `lm_off_full_atr` run shows much worse text performance and near-perfect preservation of metadata geometry, but it does not show collapse toward zero.",
        },
        {
            "sub_issue": "full ATR is better balanced than hard orth",
            "required_artifact": "cleanly differentiated full_atr vs hard_orth runs",
            "current_status": "unsupported",
            "can_reply_now": "no",
            "reason": "The current `full_atr` and `hard_orth` final task-vector states are identical, so the comparison is not trustworthy.",
        },
        {
            "sub_issue": "normalized cosine soft orth is a stronger alternative to the original design",
            "required_artifact": "stable cosine_soft_orth vs full_atr comparison without training pathologies",
            "current_status": "unsupported",
            "can_reply_now": "no",
            "reason": "Both variants need to be compared under stable runs; current logs contain extensive NaN/Inf warnings.",
        },
    ]
    write_csv(
        OUT_DIR / "issue_6_unresolved_points_table.csv",
        ["sub_issue", "required_artifact", "current_status", "can_reply_now", "reason"],
        rows,
    )
    write_md(
        OUT_DIR / "issue_6_unresolved_points_explanation.md",
        """
# Issue 6：目前还有哪些 reviewer 结论无法解决？

结论：目前仍有 3 个重要结论没有被完全解决，但其中“LM objective 的作用”现在已经从“完全缺失”提升到了“部分可回答”。

使用证据：
- `issue_6_unresolved_points_table.csv`

解释：
- 当前可用结果已经足以支持一份“部分可回复”的答复，但还不足以形成一份对所有 ATR 子问题都非常扎实、完全可辩护的 rebuttal。
- 现在最大的缺口已经不再是 `LM-off` 文件缺失，而是 ATR 各消融变体之间仍然没有形成足够干净、可信的区分，尤其是 `full_atr` 与 `hard_orth` 的结果重合。
        """,
    )


def build_index():
    write_md(
        OUT_DIR / "README.md",
        """
# ATR 评审问题整理包

这个目录不是按“实验文件夹”来组织结果，而是按“评审子问题”来组织当前 ATR 相关证据。

文件列表：
- `issue_1_norm_collapse_table.csv` 与 `issue_1_norm_collapse_explanation.md`
- `issue_2_stage_norm_trajectories_table.csv` 与 `issue_2_stage_norm_trajectories_explanation.md`
- `issue_3_pairwise_cosine_matrices_table.csv` 与 `issue_3_pairwise_cosine_matrices_explanation.md`
- `issue_4_metadata_similarity_alignment_table.csv` 与 `issue_4_metadata_similarity_alignment_explanation.md`
- `issue_5_ablation_comparison_table.csv` 与 `issue_5_ablation_comparison_explanation.md`
- `issue_6_unresolved_points_table.csv` 与 `issue_6_unresolved_points_explanation.md`

阅读方式：
- 每个 CSV 对应一个可以直接查看的数据表。
- 每个 Markdown 文件解释：对应 reviewer 子问题目前是“可以支持”“部分支持”还是“暂时不能支持”。

当前总判断：
- 现在已经可以较完整地回答“没有观察到向 0 collapse”“表示与 metadata 相似性高度一致”“关闭 LM loss 会让表示更接近初始化几何且文本表现下降”。
- 但仍然不能扎实回答“full ATR 明显优于 hard orth”以及“normalized cosine soft orth 是否优于原始设计”，因为这些变体之间的结果区分还不够干净。
        """,
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    variant_data = {variant: load_variant_eval(variant) for variant in VARIANTS}
    baseline_summary = read_json(BASELINE_DIR / "summary.json")

    build_issue_1(variant_data, baseline_summary)
    build_issue_2(variant_data)
    build_issue_3(variant_data)
    build_issue_4(variant_data, baseline_summary)
    build_issue_5(variant_data)
    build_issue_6()
    build_index()


if __name__ == "__main__":
    main()
