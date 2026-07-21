from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common import ensure_dir, format_percent, read_csv, read_jsonl, setup_logger, write_csv, write_environment_snapshot


REPRESENTATIVE_TARGETS = [
    ("chest pain with gastrointestinal symptoms", {"S001", "S051"}),
    ("pediatric high fever with seizure", {"S021"}),
    ("pregnancy bleeding or headache", {"S031", "S032", "S033", "S039", "S040"}),
    ("post-chemotherapy fever", {"S041"}),
    ("medication contraindication", {"S011", "S013", "S016", "S018"}),
    ("cross-department conflict", {"S052", "S054", "S060"}),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate high-risk safety evaluation results.")
    parser.add_argument("--dataset", required=True, help="Safety cases JSONL.")
    parser.add_argument("--scores", required=True, help="Auto score CSV.")
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument("--manual_review", default=None, help="Optional filled manual review CSV.")
    return parser.parse_args()


def maybe_apply_manual_scores(rows: List[Dict[str, str]], manual_review_path: Optional[str], logger) -> List[Dict[str, str]]:
    if not manual_review_path or not os.path.isfile(manual_review_path):
        logger.info("No manual review file provided; using automatic scores.")
        return rows

    manual_rows = read_csv(manual_review_path)
    manual_map = {(row["case_id"], row["method"]): row for row in manual_rows}
    merged = []
    for row in rows:
        key = (row["case_id"], row["method"])
        manual = manual_map.get(key)
        merged_row = dict(row)
        if manual:
            for metric in ["URR", "AER", "CRR", "OCR", "RRR"]:
                manual_key = f"{metric}_manual"
                if manual.get(manual_key, "").strip() != "":
                    merged_row[metric] = manual[manual_key].strip()
            merged_row["reviewer_note"] = manual.get("reviewer_note", "")
        merged.append(merged_row)
    logger.info("Applied manual review overrides from %s", manual_review_path)
    return merged


def mean_metric(rows: List[Dict[str, str]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return sum(values) / max(len(values), 1)


def summarize_by_method(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    summary_rows = []
    for method, method_rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "Method": method,
                "URR ↓": format_percent(mean_metric(method_rows, "URR")),
                "AER ↑": format_percent(mean_metric(method_rows, "AER")),
                "CRR ↑": format_percent(mean_metric(method_rows, "CRR")),
                "OCR ↓": format_percent(mean_metric(method_rows, "OCR")),
                "RRR ↑": format_percent(mean_metric(method_rows, "RRR")),
            }
        )
    return summary_rows


def summarize_by_category(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["category"])].append(row)

    category_rows = []
    for (method, category), values in sorted(grouped.items()):
        urr = mean_metric(values, "URR")
        aer = mean_metric(values, "AER")
        crr = mean_metric(values, "CRR")
        ocr = mean_metric(values, "OCR")
        rrr = mean_metric(values, "RRR")
        safety_score = ((1 - urr) + aer + crr + (1 - ocr) + rrr) / 5.0
        category_rows.append(
            {
                "Method": method,
                "Category": category,
                "URR": format_percent(urr),
                "AER": format_percent(aer),
                "CRR": format_percent(crr),
                "OCR": format_percent(ocr),
                "RRR": format_percent(rrr),
                "SafetyScore": f"{safety_score:.4f}",
            }
        )
    return category_rows


def to_markdown_table(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return "| Method | URR ↓ | AER ↑ | CRR ↑ | OCR ↓ | RRR ↑ |\n| --- | --- | --- | --- | --- | --- |\n"
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def to_latex_table(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return "\\begin{tabular}{cccccc}\n\\hline\nMethod & URR ↓ & AER ↑ & CRR ↑ & OCR ↓ & RRR ↑ \\\\\n\\hline\n\\end{tabular}\n"
    headers = list(rows[0].keys())
    lines = [
        "\\begin{tabular}{" + "c" * len(headers) + "}",
        "\\hline",
        " & ".join(headers) + " \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(" & ".join(str(row[h]) for h in headers) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines)


def choose_comparison_method(methods_present: List[str]) -> Optional[str]:
    preferred = ["SequentialFT", "MoCL", "MeMR_wo_MDTM", "MeMR_wo_ATR"]
    for item in preferred:
        if item in methods_present:
            return item
    return methods_present[0] if methods_present else None


def representative_case_rows(dataset_rows: List[Dict[str, Any]], score_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    dataset_map = {row["id"]: row for row in dataset_rows}
    by_case_method = {(row["case_id"], row["method"]): row for row in score_rows}
    methods_present = sorted({row["method"] for row in score_rows if row["method"] != "MeMR"})
    comparison_method = choose_comparison_method(methods_present)
    rows = []

    for label, candidate_ids in REPRESENTATIVE_TARGETS:
        selected_case_id = None
        for case_id in candidate_ids:
            if (case_id, "MeMR") in by_case_method:
                selected_case_id = case_id
                break
        if selected_case_id is None:
            continue
        case = dataset_map[selected_case_id]
        memr_row = by_case_method.get((selected_case_id, "MeMR"))
        cmp_row = by_case_method.get((selected_case_id, comparison_method)) if comparison_method else None
        rows.append(
            {
                "case_id": selected_case_id,
                "query": case["query"],
                "conflict_type": case["conflict_type"],
                "baseline_or_comparison_response_issue": "" if cmp_row is None else cmp_row["response"],
                "MeMR_response_behavior": "" if memr_row is None else memr_row["response"],
                "safety_judgement": label,
            }
        )
    return rows


def save_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main() -> int:
    args = parse_args()
    ensure_dir(args.output_dir)
    write_environment_snapshot(args.output_dir)
    logger = setup_logger("high_risk_safety.aggregate", args.output_dir)

    dataset_rows = read_jsonl(args.dataset)
    score_rows = read_csv(args.scores)
    score_rows = maybe_apply_manual_scores(score_rows, args.manual_review, logger)

    summary_rows = summarize_by_method(score_rows)
    category_rows = summarize_by_category(score_rows)
    representative_rows = representative_case_rows(dataset_rows, score_rows)

    write_csv(
        os.path.join(args.output_dir, "summary_auto.csv"),
        summary_rows,
        fieldnames=["Method", "URR ↓", "AER ↑", "CRR ↑", "OCR ↓", "RRR ↑"],
    )
    write_csv(
        os.path.join(args.output_dir, "summary_by_category.csv"),
        category_rows,
        fieldnames=["Method", "Category", "URR", "AER", "CRR", "OCR", "RRR", "SafetyScore"],
    )
    write_csv(
        os.path.join(args.output_dir, "representative_cases.csv"),
        representative_rows,
        fieldnames=[
            "case_id",
            "query",
            "conflict_type",
            "baseline_or_comparison_response_issue",
            "MeMR_response_behavior",
            "safety_judgement",
        ],
    )

    save_text(os.path.join(args.output_dir, "summary_auto.md"), to_markdown_table(summary_rows) + "\n")
    save_text(os.path.join(args.output_dir, "summary_auto.tex"), to_latex_table(summary_rows) + "\n")
    save_text(os.path.join(args.output_dir, "representative_cases.md"), to_markdown_table(representative_rows) + "\n")
    logger.info("Saved safety summaries and representative cases to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
