from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from experiments.reviewer_0629.common import read_json, write_json


def _format_num(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main():
    parser = argparse.ArgumentParser(description="Aggregate prompt_r3_c7 audit outputs.")
    parser.add_argument("--output-root", default="experiments/prompt_r3_c7/outputs")
    parser.add_argument("--report-path", default="experiments/prompt_r3_c7/outputs/prompt_r3_c7_report.md")
    parser.add_argument("--response-path", default="experiments/prompt_r3_c7/outputs/reviewer_response_draft_cn.md")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    jobs = ["internal", "chatmed", "cmedia", "huatuo"]
    summary: Dict[str, Any] = {}
    for job in jobs:
        summary_file = output_root / job / "summary.json"
        if summary_file.exists():
            summary[job] = read_json(summary_file)

    write_json(output_root / "aggregated_summary.json", summary)

    lines: List[str] = ["# prompt_r3_c7 Dataset Integrity Audit", ""]
    internal = summary.get("internal", {})
    if internal:
        cmedcl = internal.get("cmedcl_summary", {})
        lines.extend(
            [
                "## CMedCL Internal Audit",
                f"- total records: {_format_num(cmedcl.get('count'))}",
                f"- question length median: {_format_num(cmedcl.get('question_length', {}).get('median'))}",
                f"- answer length median: {_format_num(cmedcl.get('answer_length', {}).get('median'))}",
                f"- turn count: {_format_num(cmedcl.get('turn_count', {}).get('median'))}",
                f"- validation rule: {internal.get('split_rules', {}).get('validation_rule')}",
                f"- department order: {', '.join(internal.get('split_rules', {}).get('department_order', []))}",
                "",
            ]
        )

    lines.append("## External Overlap Checks")
    for job in ["chatmed", "cmedia", "huatuo"]:
        payload = summary.get(job)
        if not payload:
            continue
        overlap = payload.get("overlap", {})
        lines.append(f"### {payload.get('external_dataset')}")
        lines.append(f"- exact question overlaps: {_format_num(overlap.get('exact_overlap_count'))}")
        lines.append(f"- near-duplicate candidate hits: {_format_num(overlap.get('near_duplicate_candidate_count'))}")
        lines.append("")

    unresolved = internal.get("unresolved_evidence_items", [])
    if unresolved:
        lines.append("## Unresolved Snapshot Items")
        for item in unresolved:
            lines.append(f"- {item.get('field')}: {item.get('evidence')}")
        lines.append("")

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    response_lines: List[str] = [
        "# Reviewer Response Draft",
        "",
        "感谢审稿人的提醒。我们已补充一个独立的数据完整性审计，覆盖 CMedCL 的源文件切分规则、内部重复检查、以及与当前可获得的通用医疗评测集之间的重叠检查。",
        "",
        "根据当前仓库快照可以明确说明：",
        "",
        "- CMedCL 由 `datasets/medical_consult/{neike, waike, erke, fuchanke, nanke, zhongliuke}/train.json` 与 `test.json` 组成。",
        "- 训练时的验证集来自各科室 `train.json` 的确定性 10% 切分，代码中使用 `seed=0`。",
        "- 我们已生成内部重复/跨切分审计表，以及与 `ChatMed_Consult-v0.3_test_500`、`Chinese-medical-dialogue-data_test_500`、`huatuo26M_test_500` 的精确匹配与近重复检查结果。",
        "",
        "当前仓库快照中仍未能恢复的项目包括：完整清洗 prompt、清洗时的具体 LLM 版本与 temperature、人工复核流程、以及数据集级别的 release license。这些项目已在审计报告中显式标记为未恢复项，而不是推断填充。",
        "",
        "请将 `prompt_r3_c7_report.md` 中的统计值插入正文即可形成最终 rebuttal 版本。",
        "",
    ]
    response_path = Path(args.response_path)
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text("\n".join(response_lines) + "\n", encoding="utf-8")
    print(report_path)
    print(response_path)


if __name__ == "__main__":
    main()

