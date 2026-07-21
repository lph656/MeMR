from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from experiments.reviewer_0629.common import ensure_dir, read_json, write_csv, write_json

TASK_NAMES = ["neike", "waike", "erke", "fuchanke", "nanke", "zhongliuke"]
TASK_TO_ZH = {
    "neike": "内科",
    "waike": "外科",
    "erke": "儿科",
    "fuchanke": "妇产科",
    "nanke": "男科",
    "zhongliuke": "肿瘤科",
}

EXTERNAL_DATASETS = {
    "chatmed": {
        "label": "ChatMed_Consult-v0.3_test_500",
        "path": Path("test_infer_RAG/datasets/ChatMed_Consult-v0.3_test_500/ChatMed_Consult-v0.3_test_500.json"),
    },
    "cmedia": {
        "label": "Chinese-medical-dialogue-data_test_500",
        "path": Path("test_infer_RAG/datasets/Chinese-medical-dialogue-data_test_500/Chinese-medical-dialogue-data_test_500.json"),
    },
    "huatuo": {
        "label": "huatuo26M_test_500",
        "path": Path("test_infer_RAG/datasets/huatuo26M_test_500/huatuo26M_test_500.json"),
    },
}


@dataclass(frozen=True)
class AuditRecord:
    dataset_name: str
    split_name: str
    task_name: str
    source_path: str
    index: int
    question: str
    answer: Optional[str]
    record_id: Optional[str] = None

    @property
    def turn_count(self) -> int:
        return 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "split_name": self.split_name,
            "task_name": self.task_name,
            "source_path": self.source_path,
            "index": self.index,
            "question": self.question,
            "answer": self.answer,
            "record_id": self.record_id,
            "turn_count": self.turn_count,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_any(path: Path) -> Any:
    return read_json(path)


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
    return text


def question_key(question: Any) -> str:
    return normalize_text(question)


def qa_key(question: Any, answer: Any) -> str:
    q = normalize_text(question)
    a = normalize_text(answer)
    return f"{q}||{a}" if a else q


def extract_text(record: Dict[str, Any], candidates: Sequence[str]) -> str:
    for key in candidates:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def load_cmedcl_records(dataset_root: Path, task_name: str) -> List[AuditRecord]:
    train_path = dataset_root / task_name / "train.json"
    test_path = dataset_root / task_name / "test.json"

    train_payload = load_json_any(train_path)
    test_payload = load_json_any(test_path)
    test_records_raw = test_payload.get("questions", test_payload) if isinstance(test_payload, dict) else test_payload

    records: List[AuditRecord] = []
    for idx, row in enumerate(train_payload):
        records.append(
            AuditRecord(
                dataset_name="CMedCL",
                split_name="train",
                task_name=task_name,
                source_path=str(train_path),
                index=idx,
                question=extract_text(row, ("instruction", "question", "query")),
                answer=extract_text(row, ("output", "answer", "response")) or None,
                record_id=str(row.get("id")) if row.get("id") is not None else None,
            )
        )
    for idx, row in enumerate(test_records_raw):
        records.append(
            AuditRecord(
                dataset_name="CMedCL",
                split_name="test",
                task_name=task_name,
                source_path=str(test_path),
                index=idx,
                question=extract_text(row, ("question", "instruction", "query")),
                answer=extract_text(row, ("answer", "output", "response")) or None,
                record_id=str(row.get("id")) if isinstance(row, dict) and row.get("id") is not None else None,
            )
        )
    return records


def load_external_records(root: Path, dataset_key: str) -> List[AuditRecord]:
    entry = EXTERNAL_DATASETS[dataset_key]
    path = root / entry["path"]
    payload = load_json_any(path)
    questions = payload.get("questions", payload) if isinstance(payload, dict) else payload
    records: List[AuditRecord] = []
    for idx, row in enumerate(questions):
        records.append(
            AuditRecord(
                dataset_name=entry["label"],
                split_name="test",
                task_name=dataset_key,
                source_path=str(path),
                index=idx,
                question=extract_text(row, ("question", "instruction", "query")),
                answer=extract_text(row, ("answer", "output", "response")) or None,
                record_id=str(row.get("id")) if isinstance(row, dict) and row.get("id") is not None else None,
            )
        )
    return records


def collect_cmedcl(dataset_root: Path) -> List[AuditRecord]:
    records: List[AuditRecord] = []
    for task_name in TASK_NAMES:
        records.extend(load_cmedcl_records(dataset_root, task_name))
    return records


def char_ngrams(text: str, n: int = 3) -> set[str]:
    normalized = normalize_text(text)
    if not normalized:
        return set()
    if len(normalized) <= n:
        return {normalized}
    return {normalized[i : i + n] for i in range(len(normalized) - n + 1)}


def sequence_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def jaccard_similarity(a: str, b: str, n: int = 3) -> float:
    grams_a = char_ngrams(a, n=n)
    grams_b = char_ngrams(b, n=n)
    if not grams_a and not grams_b:
        return 1.0
    if not grams_a or not grams_b:
        return 0.0
    return len(grams_a & grams_b) / len(grams_a | grams_b)


def text_length_stats(values: Sequence[int]) -> Dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    def percentile(p: float) -> float:
        if len(ordered) == 1:
            return float(ordered[0])
        pos = (len(ordered) - 1) * p
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return float(ordered[int(pos)])
        return float(ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo))

    return {
        "count": len(values),
        "mean": round(float(mean(values)), 4),
        "median": round(float(median(values)), 4),
        "std": round(float(pstdev(values)) if len(values) > 1 else 0.0, 4),
        "min": int(min(values)),
        "p25": round(percentile(0.25), 4),
        "p50": round(percentile(0.5), 4),
        "p75": round(percentile(0.75), 4),
        "p90": round(percentile(0.9), 4),
        "p95": round(percentile(0.95), 4),
        "max": int(max(values)),
    }


def summarize_records(records: Sequence[AuditRecord]) -> Dict[str, Any]:
    question_lengths = [len(normalize_text(record.question)) for record in records]
    answer_lengths = [len(normalize_text(record.answer)) for record in records if record.answer is not None]
    turn_counts = [record.turn_count for record in records]
    nonempty_answers = sum(1 for record in records if record.answer)
    nonempty_inputs = sum(1 for record in records if record.answer is not None and record.answer.strip())
    return {
        "count": len(records),
        "question_length": text_length_stats(question_lengths),
        "answer_length": text_length_stats(answer_lengths),
        "turn_count": text_length_stats(turn_counts),
        "records_with_answers": nonempty_answers,
        "records_with_nonempty_answers": nonempty_inputs,
        "single_turn_ratio": round(sum(1 for _ in records) / len(records), 6) if records else 0.0,
    }


def build_exact_index(records: Sequence[AuditRecord], use_answer: bool = False) -> Dict[str, List[int]]:
    index: Dict[str, List[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        key = qa_key(record.question, record.answer) if use_answer else question_key(record.question)
        if key:
            index[key].append(idx)
    return index


def build_ngram_index(records: Sequence[AuditRecord], n: int = 3) -> Dict[str, List[int]]:
    index: Dict[str, List[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        grams = char_ngrams(record.question, n=n)
        for gram in grams:
            index[gram].append(idx)
    return index


def candidate_matches(
    reference_records: Sequence[AuditRecord],
    query_record: AuditRecord,
    ngram_index: Dict[str, List[int]],
    min_shared_ngrams: int = 2,
    top_k: int = 5,
    min_seq_ratio: float = 0.75,
    min_jaccard: float = 0.45,
) -> List[Dict[str, Any]]:
    query_grams = char_ngrams(query_record.question)
    candidate_counts: Dict[int, int] = defaultdict(int)
    for gram in query_grams:
        for ref_idx in ngram_index.get(gram, []):
            candidate_counts[ref_idx] += 1

    scored_rows: List[Dict[str, Any]] = []
    for ref_idx, shared in candidate_counts.items():
        if shared < min_shared_ngrams:
            continue
        ref = reference_records[ref_idx]
        seq = sequence_ratio(query_record.question, ref.question)
        jac = jaccard_similarity(query_record.question, ref.question)
        if seq < min_seq_ratio and jac < min_jaccard:
            continue
        scored_rows.append(
            {
                "query_dataset": query_record.dataset_name,
                "query_split": query_record.split_name,
                "query_task": query_record.task_name,
                "query_index": query_record.index,
                "query_id": query_record.record_id,
                "query_question": query_record.question,
                "ref_dataset": ref.dataset_name,
                "ref_split": ref.split_name,
                "ref_task": ref.task_name,
                "ref_index": ref.index,
                "ref_id": ref.record_id,
                "ref_question": ref.question,
                "shared_ngrams": shared,
                "seq_ratio": round(seq, 6),
                "jaccard_3gram": round(jac, 6),
            }
        )
    scored_rows.sort(key=lambda row: (row["seq_ratio"], row["jaccard_3gram"], row["shared_ngrams"]), reverse=True)
    return scored_rows[:top_k]


def build_source_inventory(project_root: Path) -> List[Dict[str, Any]]:
    paths = []
    for task_name in TASK_NAMES:
        for split in ("train", "test"):
            source_path = project_root / "datasets" / "medical_consult" / task_name / f"{split}.json"
            if source_path.exists():
                paths.append(
                    {
                        "name": f"CMedCL-{task_name}-{split}",
                        "path": str(source_path),
                        "sha256": sha256_file(source_path),
                        "size_bytes": source_path.stat().st_size,
                    }
                )
    for dataset_key, payload in EXTERNAL_DATASETS.items():
        source_path = project_root / payload["path"]
        if source_path.exists():
            paths.append(
                {
                    "name": payload["label"],
                    "path": str(source_path),
                    "sha256": sha256_file(source_path),
                    "size_bytes": source_path.stat().st_size,
                }
            )
    return paths


def split_rule_table(project_root: Path) -> Dict[str, Any]:
    train_script = project_root / "tasks" / "mtl5" / "dataloader_mtl_causal_llama.py"
    return {
        "department_order": TASK_NAMES,
        "department_order_zh": [TASK_TO_ZH[name] for name in TASK_NAMES],
        "train_test_layout": {task: {"train": f"datasets/medical_consult/{task}/train.json", "test": f"datasets/medical_consult/{task}/test.json"} for task in TASK_NAMES},
        "validation_rule": "train_test_split(test_size=validation_split_percentage, seed=training_args.seed) inside tasks/mtl5/dataloader_mtl_causal_llama.py",
        "validation_default": 0.1,
        "validation_seed": 0,
        "validation_source": str(train_script),
    }


def department_label_table() -> List[Dict[str, Any]]:
    return [{"task_name": task, "department_zh": TASK_TO_ZH[task]} for task in TASK_NAMES]


def unresolved_evidence_items() -> List[Dict[str, Any]]:
    return [
        {
            "field": "complete data-cleaning prompt",
            "status": "not_recovered",
            "evidence": "No cleaning prompt is embedded in the current repository snapshot.",
        },
        {
            "field": "LLM used for cleaning",
            "status": "not_recovered",
            "evidence": "The snapshot does not record the exact model name/version used for the data-cleaning pass.",
        },
        {
            "field": "temperature / decoding settings",
            "status": "not_recovered",
            "evidence": "No cleaning-time decoding hyperparameters are stored in the available repo files.",
        },
        {
            "field": "human validation protocol",
            "status": "not_recovered",
            "evidence": "No validation checklist or annotation protocol is available in the snapshot.",
        },
        {
            "field": "dataset-specific release license",
            "status": "not_recovered",
            "evidence": "The project license exists, but dataset-specific release terms are not stated in the current files.",
        },
    ]

