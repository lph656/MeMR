from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

from experiments.reviewer_0629.common import ensure_dir, write_csv, write_json

from .common import (
    EXTERNAL_DATASETS,
    TASK_NAMES,
    AuditRecord,
    build_exact_index,
    build_ngram_index,
    build_source_inventory,
    candidate_matches,
    collect_cmedcl,
    department_label_table,
    load_external_records,
    normalize_text,
    qa_key,
    question_key,
    unresolved_evidence_items,
    sha256_file,
    split_rule_table,
    summarize_records,
    text_length_stats,
)


def _records_to_rows(records: Sequence[AuditRecord]) -> List[Dict[str, Any]]:
    return [record.to_dict() for record in records]


def _duplicate_groups(records: Sequence[AuditRecord], use_answer: bool = False) -> List[Dict[str, Any]]:
    index = build_exact_index(records, use_answer=use_answer)
    rows: List[Dict[str, Any]] = []
    for key, indices in index.items():
        if len(indices) <= 1:
            continue
        exemplar = records[indices[0]]
        rows.append(
            {
                "exact_key": key,
                "count": len(indices),
                "dataset_name": exemplar.dataset_name,
                "task_name": exemplar.task_name,
                "split_name": exemplar.split_name,
                "question": exemplar.question,
                "answer": exemplar.answer,
                "indices": json.dumps(indices, ensure_ascii=False),
            }
        )
    rows.sort(key=lambda row: row["count"], reverse=True)
    return rows


def _split_overlap_matrix(records: Sequence[AuditRecord]) -> List[Dict[str, Any]]:
    groups: Dict[tuple[str, str], List[AuditRecord]] = defaultdict(list)
    for record in records:
        groups[(record.task_name, record.split_name)].append(record)

    rows: List[Dict[str, Any]] = []
    keys = sorted(groups.keys())
    for left_task, left_split in keys:
        left_records = groups[(left_task, left_split)]
        left_index = build_exact_index(left_records, use_answer=False)
        left_qa_index = build_exact_index(left_records, use_answer=True)
        for right_task, right_split in keys:
            if (left_task, left_split) >= (right_task, right_split):
                continue
            right_records = groups[(right_task, right_split)]
            right_question_keys = {question_key(record.question): record for record in right_records if question_key(record.question)}
            right_qa_keys = {qa_key(record.question, record.answer): record for record in right_records if qa_key(record.question, record.answer)}
            question_overlap = 0
            qa_overlap = 0
            for key, indices in left_index.items():
                if key in right_question_keys:
                    question_overlap += len(indices)
            for key, indices in left_qa_index.items():
                if key in right_qa_keys:
                    qa_overlap += len(indices)
            rows.append(
                {
                    "left_task": left_task,
                    "left_split": left_split,
                    "right_task": right_task,
                    "right_split": right_split,
                    "question_overlap_count": question_overlap,
                    "qa_overlap_count": qa_overlap,
                }
            )
    return rows


def _external_overlap(reference_records: Sequence[AuditRecord], query_records: Sequence[AuditRecord]) -> Dict[str, Any]:
    exact_index = build_exact_index(reference_records, use_answer=False)
    gram_index = build_ngram_index(reference_records, n=3)

    exact_rows: List[Dict[str, Any]] = []
    near_rows: List[Dict[str, Any]] = []
    exact_count = 0
    near_count = 0

    for query in query_records:
        q_key = question_key(query.question)
        if q_key and q_key in exact_index:
            exact_count += 1
            ref_idx = exact_index[q_key][0]
            ref = reference_records[ref_idx]
            exact_rows.append(
                {
                    "query_dataset": query.dataset_name,
                    "query_index": query.index,
                    "query_id": query.record_id,
                    "query_question": query.question,
                    "ref_dataset": ref.dataset_name,
                    "ref_task": ref.task_name,
                    "ref_split": ref.split_name,
                    "ref_index": ref.index,
                    "ref_id": ref.record_id,
                    "ref_question": ref.question,
                    "match_type": "exact_question",
                }
            )

        candidates = candidate_matches(reference_records, query, gram_index, min_shared_ngrams=2, top_k=5, min_seq_ratio=0.80, min_jaccard=0.45)
        for cand in candidates:
            near_count += 1
            cand["match_type"] = "near_duplicate"
            near_rows.append(cand)

    return {
        "exact_overlap_count": exact_count,
        "near_duplicate_candidate_count": near_count,
        "exact_overlap_examples": exact_rows[:50],
        "near_duplicate_examples": near_rows[:100],
    }


def _length_rows(records: Sequence[AuditRecord], dataset_name: str) -> List[Dict[str, Any]]:
    rows = []
    for record in records:
        rows.append(
            {
                "dataset_name": dataset_name,
                "task_name": record.task_name,
                "split_name": record.split_name,
                "record_id": record.record_id,
                "question_len": len(normalize_text(record.question)),
                "answer_len": len(normalize_text(record.answer)) if record.answer else 0,
                "turn_count": record.turn_count,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Dataset integrity audit for prompt_r3_c7.")
    parser.add_argument("--job", choices=["internal", "chatmed", "cmedia", "huatuo"], required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-root", default="experiments/prompt_r3_c7/outputs")
    parser.add_argument("--dataset-root", default="datasets/medical_consult")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_root = ensure_dir(args.output_root)
    job_dir = ensure_dir(output_root / args.job)

    cmedcl_records = collect_cmedcl(Path(args.dataset_root))
    internal_summary: Dict[str, Any] = {}
    cmedcl_summary = summarize_records(cmedcl_records)
    internal_summary["cmedcl_summary"] = cmedcl_summary
    internal_summary["split_rules"] = split_rule_table(project_root)
    internal_summary["department_labels"] = department_label_table()
    internal_summary["source_inventory"] = build_source_inventory(project_root)
    internal_summary["unresolved_evidence_items"] = unresolved_evidence_items()

    write_json(job_dir / "cmedcl_records.json", _records_to_rows(cmedcl_records))
    write_json(job_dir / "summary.json", internal_summary)
    write_csv(job_dir / "cmedcl_lengths.csv", _length_rows(cmedcl_records, "CMedCL"), ["dataset_name", "task_name", "split_name", "record_id", "question_len", "answer_len", "turn_count"])
    write_csv(job_dir / "cmedcl_duplicate_groups_question.csv", _duplicate_groups(cmedcl_records, use_answer=False), ["exact_key", "count", "dataset_name", "task_name", "split_name", "question", "answer", "indices"])
    write_csv(job_dir / "cmedcl_duplicate_groups_qa.csv", _duplicate_groups(cmedcl_records, use_answer=True), ["exact_key", "count", "dataset_name", "task_name", "split_name", "question", "answer", "indices"])
    write_csv(job_dir / "cmedcl_split_overlap_matrix.csv", _split_overlap_matrix(cmedcl_records), ["left_task", "left_split", "right_task", "right_split", "question_overlap_count", "qa_overlap_count"])

    if args.job == "internal":
        return

    external_records = load_external_records(project_root, args.job)
    overlap = _external_overlap(cmedcl_records, external_records)
    summary = {
        "reference_dataset": "CMedCL",
        "external_dataset": external_records[0].dataset_name if external_records else EXTERNAL_DATASETS[args.job]["label"],
        "reference_summary": cmedcl_summary,
        "external_summary": summarize_records(external_records),
        "overlap": overlap,
        "split_rules": split_rule_table(project_root),
        "department_labels": department_label_table(),
        "source_inventory": build_source_inventory(project_root),
        "unresolved_evidence_items": unresolved_evidence_items(),
    }
    write_json(job_dir / "external_records.json", _records_to_rows(external_records))
    write_json(job_dir / "summary.json", summary)
    write_csv(job_dir / "external_lengths.csv", _length_rows(external_records, summary["external_dataset"]), ["dataset_name", "task_name", "split_name", "record_id", "question_len", "answer_len", "turn_count"])
    write_csv(job_dir / "exact_overlap_examples.csv", overlap["exact_overlap_examples"], ["query_dataset", "query_index", "query_id", "query_question", "ref_dataset", "ref_task", "ref_split", "ref_index", "ref_id", "ref_question", "match_type"])
    write_csv(job_dir / "near_duplicate_examples.csv", overlap["near_duplicate_examples"], ["query_dataset", "query_split", "query_task", "query_index", "query_id", "query_question", "ref_dataset", "ref_split", "ref_task", "ref_index", "ref_id", "ref_question", "shared_ngrams", "seq_ratio", "jaccard_3gram", "match_type"])


if __name__ == "__main__":
    main()

