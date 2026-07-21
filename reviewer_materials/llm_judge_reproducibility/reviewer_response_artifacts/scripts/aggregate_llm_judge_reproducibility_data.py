#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "reviewer_response_artifacts" / "llm_judge_reproducibility_data.json"
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_SEED = 20260629


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def extract_assignment(text: str, name: str) -> Optional[str]:
    triple = re.search(rf"{re.escape(name)}\s*=\s*\"\"\"(.*?)\"\"\"", text, re.S)
    if triple:
        return triple.group(1).strip()
    single = re.search(rf'{re.escape(name)}\s*=\s*"([^"]*)"', text, re.S)
    if single:
        return single.group(1).strip()
    return None


def bootstrap_mean_ci(values: Sequence[float], seed: int = BOOTSTRAP_SEED) -> Optional[Dict[str, Any]]:
    if not values:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = 1.0 - BOOTSTRAP_CONFIDENCE
    lo_idx = max(0, min(len(means) - 1, int((alpha / 2.0) * len(means))))
    hi_idx = max(0, min(len(means) - 1, int((1.0 - alpha / 2.0) * len(means)) - 1))
    return {
        "mean": sum(values) / len(values),
        "lower_ci": means[lo_idx],
        "upper_ci": means[hi_idx],
        "n_samples": len(values),
        "bootstrap_seed": seed,
        "bootstrap_resampling": BOOTSTRAP_RESAMPLES,
        "confidence_level": BOOTSTRAP_CONFIDENCE,
    }


def binom_pmf(n: int, k: int, p: float = 0.5) -> float:
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def sign_test_pvalue(win_count: int, lose_count: int) -> Optional[float]:
    n = win_count + lose_count
    if n == 0:
        return None
    k = min(win_count, lose_count)
    cumulative = sum(binom_pmf(n, i, 0.5) for i in range(0, k + 1))
    return min(1.0, 2.0 * cumulative)


def build_score_protocol(common_text: str, ds_text: str, pilot_config: Dict[str, Any]) -> Dict[str, Any]:
    prompt = extract_assignment(common_text, "SCORE_PROMPT_TEMPLATE")
    return {
        "protocol_name": "single_answer_llm_judge",
        "judge_prompts": {
            "safety_professionalism_fluency_prompt": prompt,
            "pairwise_win_rate_prompt": None,
        },
        "evaluator_models": [
            {
                "name": "deepseek-chat",
                "version": None,
                "source": "test_infer/generate_score_DS.py and metadata_robustness_experiments/inference_evaluation outputs",
            },
            {
                "name": "deepseek-ai/DeepSeek-V3",
                "version": None,
                "source": "test_infer/generate_score_modaDS.py",
            },
        ],
        "evaluator_api_or_checkpoint": {
            "base_url": pilot_config.get("base_url"),
            "api_provider": "DeepSeek-compatible OpenAI API",
            "evaluation_date": None,
            "local_checkpoint": None,
        },
        "temperature": pilot_config.get("temperature"),
        "top_p": None,
        "max_tokens": None,
        "output_format": {
            "subitem_scale": "0-100",
            "dimension_scale": "0-300",
            "total_scale": "0-900",
            "required_response_shape": [
                "- 安全性 / 专业性 / 流畅性 sections",
                "- 每个 section 3 个子项",
                "- 小计",
                "- 总分",
            ],
        },
        "scoring_weights": {
            "Safety": 300,
            "Professionalism": 300,
            "Fluency": 300,
            "normalized": {
                "Safety": 1 / 3,
                "Professionalism": 1 / 3,
                "Fluency": 1 / 3,
            },
        },
        "number_of_evaluation_repetitions": 1,
        "answer_ordering": None,
        "blinding_procedure": None,
        "tie_rule": None,
        "score_scale": {
            "subitem": "0-100",
            "dimension_subtotal": "0-300",
            "overall_total": "0-900",
        },
        "prompts_and_raw_outputs_released_or_saved": {
            "legacy_scripts": {
                "prompts_saved": False,
                "parsed_scores_saved": True,
                "raw_outputs_saved": False,
            },
            "pilot50_scoring_pipeline": {
                "prompts_saved": False,
                "parsed_scores_saved": True,
                "raw_outputs_saved": True,
            },
        },
        "notes": [
            "Legacy scoring scripts do not set temperature/top_p/max_tokens explicitly in the API call.",
            "The on-disk reproducible raw judge outputs are available for the pilot50 metadata-robustness evaluation only.",
            "Main-manuscript GPT-4 / Gemini / Grok / DeepSeek multi-judge table protocols are not recoverable from raw on-disk artifacts in this workspace snapshot.",
        ],
    }


def build_pairwise_protocol(compare_text: str) -> Dict[str, Any]:
    system_prompt = extract_assignment(compare_text, "LLM_SYSTEM_PROMPT")
    criteria_prompt = extract_assignment(compare_text, "LLM_CRITERIA_PROMPT")
    user_snippet_match = re.search(
        r'user_content\s*=\s*f?"""(.*?)"""',
        compare_text,
        re.S,
    )
    user_template = user_snippet_match.group(1).strip() if user_snippet_match else None
    return {
        "protocol_name": "pairwise_win_rate_llm_judge",
        "judge_prompts": {
            "system_prompt": system_prompt,
            "criteria_prompt": criteria_prompt,
            "user_prompt_template": user_template,
        },
        "evaluator_models": [
            {
                "name": "deepseek-chat",
                "version": None,
                "source": "test_infer_compare/compare_medical_qa.py",
            }
        ],
        "evaluator_api_or_checkpoint": {
            "base_url": "https://api.deepseek.com",
            "api_provider": "DeepSeek-compatible OpenAI API",
            "evaluation_date": None,
            "local_checkpoint": None,
        },
        "temperature": 0.0,
        "top_p": None,
        "max_tokens": None,
        "output_format": 'Only "Winner: A", "Winner: B", or "Winner: Draw".',
        "scoring_weights": None,
        "number_of_evaluation_repetitions": 1,
        "answer_ordering": "Deterministic. Folder A is always Answer A; Folder B is always Answer B.",
        "blinding_procedure": "Model names are hidden from the judge; only Answer A and Answer B are shown.",
        "tie_rule": "Judge returns Draw when the two answers are equally good or cannot be clearly distinguished.",
        "score_scale": None,
        "prompts_and_raw_outputs_released_or_saved": {
            "prompt_saved": False,
            "raw_outputs_saved": False,
            "aggregate_text_summary_saved": True,
        },
        "notes": [
            "The available script performs blinded pairwise comparison but does not randomize answer order.",
            "The script saves only aggregate percentages to a text file and does not persist per-sample raw judge outputs.",
        ],
    }


def load_metadata_robustness_raw(missing_fields: List[Dict[str, Any]]) -> Dict[str, Any]:
    base = PROJECT_ROOT / "metadata_robustness_experiments" / "inference_evaluation" / "outputs" / "pilot_50"
    conditions = ["institution_mix_40", "stale_coarse", "noisy_30", "missing_50"]
    condition_records: Dict[str, Dict[str, Any]] = {}
    all_records: List[Dict[str, Any]] = []
    score_cis: Dict[str, Any] = {}

    for condition in conditions:
        inference_path = base / condition / "inference_results.jsonl"
        scores_path = base / condition / "scoring" / "scores.jsonl"
        raw_path = base / condition / "scoring" / "raw_judge_responses.jsonl"
        config_path = base / condition / "scoring" / "scoring_config.json"
        summary_path = base / condition / "scoring" / "score_summary.json"

        inference_by_id = {row["sample_id"]: row for row in iter_jsonl(inference_path)}
        raw_by_id = {row["sample_id"]: row for row in iter_jsonl(raw_path)}
        scores = list(iter_jsonl(scores_path))
        config = read_json(config_path)
        summary = read_json(summary_path)

        per_condition_records = []
        llm_scores = []
        for row in scores:
            sample_id = row["sample_id"]
            inf = inference_by_id.get(sample_id, {})
            raw = raw_by_id.get(sample_id, {})
            record = {
                "sample_id": sample_id,
                "dataset": "pilot_50",
                "department/task": inf.get("task_name"),
                "question": inf.get("question"),
                "model_name": f"MeMR[{condition}]",
                "generated_answer": inf.get("generated_answer"),
                "judge_model": row.get("judge_model"),
                "repetition_id": 1,
                "safety_score": row.get("safety_score"),
                "professionalism_score": row.get("professionalism_score"),
                "fluency_score": row.get("fluency_score"),
                "weighted_average_score": row.get("average_llm_score"),
                "pairwise_compared_model": None,
                "winner": None,
                "loser": None,
                "tie": None,
                "raw_judge_output": raw.get("raw_response"),
                "answer_order": None,
                "blinded_label": None,
                "condition": condition,
                "prompt_hash": row.get("prompt_hash"),
                "generated_answer_hash": row.get("generated_answer_hash"),
                "raw_response_reference": row.get("raw_response_reference"),
                "scoring_config": config,
            }
            per_condition_records.append(record)
            all_records.append(record)
            if row.get("average_llm_score") is not None:
                llm_scores.append(float(row["average_llm_score"]))

        score_cis[condition] = bootstrap_mean_ci(llm_scores)
        condition_records[condition] = {
            "scoring_config": config,
            "score_summary": summary,
            "n_records": len(per_condition_records),
            "per_sample_records": per_condition_records,
        }

    missing_fields.append(
        {
            "field": "per-sample raw LLM-judge results for main manuscript tables (Baseline / Baseline+MDTM / Baseline+ATR / MeMR and generalization multi-judge tables)",
            "reason": "Not present as on-disk raw judge outputs in the current workspace snapshot. Only pilot50 metadata-robustness raw judge artifacts are recoverable.",
            "recoverable_with_existing_scripts": True,
            "evidence_paths": [
                "metadata_robustness_experiments/inference_evaluation/outputs/pilot_50/*/scoring/raw_judge_responses.jsonl",
                "test_infer/generate_score_DS.py",
                "test_infer/generate_score_modaDS.py",
                "test_infer_compare/compare_medical_qa.py",
            ],
        }
    )

    return {
        "available_existing_raw_judge_results": condition_records,
        "all_per_sample_records": all_records,
        "existing_raw_result_groups": list(condition_records.keys()),
        "available_score_confidence_intervals": score_cis,
        "recovery_commands_not_run": [
            "python test_infer/generate_score_DS.py --input_json <input_result_json> --output_json <output_score_json>",
            "python test_infer/generate_score_modaDS.py --input_json <input_result_json> --output_json <output_score_json>",
            "python test_infer_compare/compare_medical_qa.py <folder_a> <folder_b> $DEEPSEEK_API_KEY",
        ],
    }


def load_generalization_outputs() -> List[Dict[str, Any]]:
    datasets = [
        ("ChatMed_Consult-v0.3_test_500", PROJECT_ROOT / "test_infer_RAG" / "datasets" / "ChatMed_Consult-v0.3_test_500" / "result.json"),
        ("Chinese-medical-dialogue-data_test_500", PROJECT_ROOT / "test_infer_RAG" / "datasets" / "Chinese-medical-dialogue-data_test_500" / "result.json"),
        ("huatuo26M_test_500", PROJECT_ROOT / "test_infer_RAG" / "datasets" / "huatuo26M_test_500" / "result.json"),
        ("keshi_medical_consult_500", PROJECT_ROOT / "test_infer_RAG" / "datasets" / "keshi_medical_consult_500" / "result.json"),
    ]
    outputs = []
    for dataset_name, path in datasets:
        payload = read_json(path)
        results = payload["results"] if isinstance(payload, dict) else payload
        outputs.append(
            {
                "dataset": dataset_name,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "num_outputs": len(results),
                "sample_fields": list(results[0].keys()) if results else [],
            }
        )
    return outputs


def build_average_score_ci_section(
    available_cis: Dict[str, Any],
    missing_fields: List[Dict[str, Any]],
) -> Dict[str, Any]:
    requested_missing = {
        "Baseline": None,
        "Baseline+MDTM": None,
        "Baseline+ATR": None,
        "MeMR": None,
    }
    missing_fields.append(
        {
            "field": "95% bootstrap CI for main-manuscript Average LLM Score ablation table (Baseline / Baseline+MDTM / Baseline+ATR / MeMR)",
            "reason": "Per-sample LLM-judge score files for these exact compared systems are not available on disk in this workspace snapshot.",
            "recoverable_with_existing_scripts": True,
            "evidence_paths": [
                "test_infer/generate_score_DS.py",
                "test_infer/generate_score_modaDS.py",
                "test_infer_RAG/datasets/*/result.json",
            ],
        }
    )
    return {
        "requested_main_ablation": requested_missing,
        "available_existing_scored_conditions": available_cis,
        "notes": [
            "Bootstrap settings follow the requested configuration: 1000 resamples, 95% confidence, seed 20260629.",
            "Only the pilot50 metadata-robustness scoring artifacts currently provide recoverable per-sample LLM-judge scores.",
        ],
    }


def build_win_rate_statistics(missing_fields: List[Dict[str, Any]]) -> Dict[str, Any]:
    comparisons = [
        "MeMR vs Baseline",
        "MeMR vs Baseline+MDTM",
        "MeMR vs Baseline+ATR",
        "MeMR vs O-LoRA",
        "MeMR vs MoCL",
        "MeMR vs CITB",
    ]
    results = {}
    for name in comparisons:
        results[name] = {
            "win_count": None,
            "lose_count": None,
            "tie_count": None,
            "win_rate": None,
            "lose_rate": None,
            "tie_rate": None,
            "win_rate_bootstrap_ci": None,
            "binomial_sign_test_p_value": None,
            "n_effective": None,
            "n_total": None,
            "status": "missing_raw_pairwise_results",
        }
    missing_fields.append(
        {
            "field": "Raw pairwise win/lose/tie comparison outputs for requested MeMR vs baseline comparisons",
            "reason": "The repository contains the pairwise comparison script but no archived per-sample comparison outputs or aggregate comparison text files for these requested pairs.",
            "recoverable_with_existing_scripts": True,
            "evidence_paths": [
                "test_infer_compare/compare_medical_qa.py",
                "test_infer_compare/run_comparison.sh",
            ],
        }
    )
    return {
        "requested_comparisons": results,
        "notes": [
            "No pairwise raw comparison artifacts were found, so bootstrap CI and sign-test p-values cannot be computed without rerunning the judge.",
        ],
    }


def build_agreement_statistics(missing_fields: List[Dict[str, Any]]) -> Dict[str, Any]:
    missing_fields.append(
        {
            "field": "LLM judge agreement statistics (Spearman/Kendall between judges, pairwise winner agreement, kappa)",
            "reason": "Available on-disk scored artifacts use a single judge model with a single repetition per sample, which is insufficient for inter-judge agreement estimation.",
            "recoverable_with_existing_scripts": False,
            "evidence_paths": [
                "metadata_robustness_experiments/inference_evaluation/outputs/pilot_50/*/scoring/scoring_config.json",
            ],
        }
    )
    return {
        "status": "insufficient_data",
        "detected_judge_models": ["deepseek-chat"],
        "detected_repetitions_per_sample": 1,
        "spearman_correlation_between_judges": None,
        "kendall_correlation_between_judges": None,
        "pairwise_winner_agreement_rate": None,
        "fleiss_or_cohen_kappa": None,
        "reason": "Only one judge model and one repetition are archived.",
    }


def load_high_risk_safety() -> Dict[str, Any]:
    readme_path = PROJECT_ROOT / "experiments" / "high_risk_safety" / "README.md"
    cases_path = PROJECT_ROOT / "experiments" / "high_risk_safety" / "safety_cases_zh.jsonl"
    latest_dir = PROJECT_ROOT / "results" / "high_risk_safety" / "20260627_171024"

    metric_definitions = {
        "URR": "Unsafe Recommendation Rate",
        "AER": "Appropriate Escalation Rate",
        "CRR": "Conflict Recognition Rate",
        "OCR": "Over-certainty Rate",
        "RRR": "Responsibility-aware Response Rate",
    }

    cases = list(iter_jsonl(cases_path))
    categories = sorted({row.get("category") for row in cases})

    summary_rows = []
    with (latest_dir / "summary_auto.csv").open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            summary_rows.append(row)

    category_rows = []
    with (latest_dir / "summary_by_category.csv").open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            category_rows.append(row)

    per_case = list(iter_jsonl(latest_dir / "per_case_scores_auto.jsonl"))
    resolved_registry = read_json(latest_dir / "method_registry.resolved.json")

    return {
        "safety_set_size": len(cases),
        "categories": categories,
        "metric_definitions": metric_definitions,
        "existing_project_results": {
            "run_dir": str(latest_dir.relative_to(PROJECT_ROOT)),
            "methods": summary_rows,
            "per_category_metrics": category_rows,
            "per_case_scores_preview": per_case[:10],
            "resolved_method_registry": resolved_registry,
        },
        "supplementary_reported_metrics_from_user_prompt": [
            {"method": "SequentialFT", "URR": 30.62, "AER": 76.31, "CRR": 5.39, "OCR": 3.83, "RRR": 6.83, "source": "prompt_r3_c8"},
            {"method": "MoCL", "URR": 28.44, "AER": 75.30, "CRR": 6.50, "OCR": 4.30, "RRR": 6.03, "source": "prompt_r3_c8"},
            {"method": "MeMR w/o MDTM", "URR": 26.76, "AER": 74.67, "CRR": 7.45, "OCR": 0.76, "RRR": 6.25, "source": "prompt_r3_c8"},
            {"method": "MeMR w/o ATR", "URR": 30.43, "AER": 74.79, "CRR": 8.60, "OCR": 0.78, "RRR": 7.69, "source": "prompt_r3_c8"},
            {"method": "MeMR", "URR": 25.91, "AER": 75.18, "CRR": 9.36, "OCR": 0.52, "RRR": 6.42, "source": "prompt_r3_c8"},
        ],
        "notes": [
            "Existing on-disk project artifacts contain only the MeMR safety run in the selected latest directory.",
            "The multi-method safety numbers requested by the user are preserved with explicit provenance as user-provided reported values because matching raw project files are not present in this workspace snapshot.",
        ],
    }


def main() -> None:
    os.makedirs(OUTPUT_PATH.parent / "scripts", exist_ok=True)

    score_common = read_text(PROJECT_ROOT / "metadata_robustness_experiments" / "inference_evaluation" / "common.py")
    legacy_score_script = read_text(PROJECT_ROOT / "test_infer" / "generate_score_DS.py")
    pairwise_script = read_text(PROJECT_ROOT / "test_infer_compare" / "compare_medical_qa.py")
    pilot_config = read_json(
        PROJECT_ROOT
        / "metadata_robustness_experiments"
        / "inference_evaluation"
        / "outputs"
        / "pilot_50"
        / "institution_mix_40"
        / "scoring"
        / "scoring_config.json"
    )

    missing_fields: List[Dict[str, Any]] = []
    raw_summary = load_metadata_robustness_raw(missing_fields)
    generalization_outputs = load_generalization_outputs()
    average_score_cis = build_average_score_ci_section(
        raw_summary["available_score_confidence_intervals"],
        missing_fields,
    )
    win_rate_stats = build_win_rate_statistics(missing_fields)
    agreement_stats = build_agreement_statistics(missing_fields)
    high_risk = load_high_risk_safety()

    files_scanned = [
        "test_infer/generate_score_DS.py",
        "test_infer/generate_score_modaDS.py",
        "test_infer_compare/compare_medical_qa.py",
        "metadata_robustness_experiments/inference_evaluation/common.py",
        "metadata_robustness_experiments/inference_evaluation/score_pilot50.py",
        "metadata_robustness_experiments/inference_evaluation/outputs/pilot_50/institution_mix_40/scoring/scoring_config.json",
        "metadata_robustness_experiments/inference_evaluation/outputs/pilot_50/institution_mix_40/scoring/scores.jsonl",
        "metadata_robustness_experiments/inference_evaluation/outputs/pilot_50/institution_mix_40/scoring/raw_judge_responses.jsonl",
        "metadata_robustness_experiments/inference_evaluation/outputs/pilot_50/stale_coarse/scoring/scores.jsonl",
        "metadata_robustness_experiments/inference_evaluation/outputs/pilot_50/noisy_30/scoring/scores.jsonl",
        "metadata_robustness_experiments/inference_evaluation/outputs/pilot_50/missing_50/scoring/scores.jsonl",
        "metadata_robustness_experiments/inference_evaluation/outputs/pilot_50/*/inference_results.jsonl",
        "test_infer_RAG/datasets/ChatMed_Consult-v0.3_test_500/result.json",
        "test_infer_RAG/datasets/Chinese-medical-dialogue-data_test_500/result.json",
        "test_infer_RAG/datasets/huatuo26M_test_500/result.json",
        "test_infer_RAG/datasets/keshi_medical_consult_500/result.json",
        "experiments/high_risk_safety/README.md",
        "experiments/high_risk_safety/safety_cases_zh.jsonl",
        "results/high_risk_safety/20260627_171024/summary_auto.csv",
        "results/high_risk_safety/20260627_171024/summary_by_category.csv",
        "results/high_risk_safety/20260627_171024/per_case_scores_auto.jsonl",
        "results/high_risk_safety/20260627_171024/method_registry.resolved.json",
        "reviewer_fairness/results/reference/memr/order1/metrics.json",
        "reviewer_fairness/results/reference/metadata_only_routing/order1/metrics.json",
        "reviewer_fairness/results/reference/joint_training/all_tasks/metrics.json",
        "reviewer_fairness/results/reference/single_task_oracle/all_tasks/metrics.json",
        "reviewer_fairness/results/summary/reference_baselines_summary.csv",
        "reviewer_fairness/results/summary/all_results_summary.csv",
    ]

    output = {
        "project_root": str(PROJECT_ROOT),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "gpu_policy": {
            "required_gpu_count": 1,
            "cuda_visible_devices": "1",
            "note": "No training is performed. GPU is only used if response generation is necessary.",
        },
        "files_scanned": files_scanned,
        "evaluation_protocol": {
            "single_answer_scoring": build_score_protocol(score_common, legacy_score_script, pilot_config),
            "pairwise_win_rate": build_pairwise_protocol(pairwise_script),
            "manuscript_level_unrecovered_protocol_fields": {
                "judge_models_reported_in_text": ["DeepSeek", "GPT4", "Grok3", "Gemini"],
                "versions": None,
                "raw_protocol_details": None,
                "reason": "No raw on-disk judge configs or outputs were found for the manuscript-level multi-judge comparison tables.",
            },
        },
        "raw_evaluation_summary": {
            **raw_summary,
            "existing_model_output_files_without_archived_scores": generalization_outputs,
        },
        "average_score_confidence_intervals": average_score_cis,
        "win_rate_statistics": win_rate_stats,
        "judge_agreement_statistics": agreement_stats,
        "high_risk_safety_evaluation": high_risk,
        "missing_fields": missing_fields,
        "commands_run": [
            "python reviewer_response_artifacts/scripts/aggregate_llm_judge_reproducibility_data.py",
        ],
        "reproducibility_notes": [
            "This aggregation uses only existing on-disk artifacts and does not call any external LLM judge API.",
            "No GPU was used while generating this JSON.",
            "Main-manuscript LLM-judge ablation/generalization raw outputs are not recoverable from the current workspace snapshot; only pilot50 metadata-robustness judge artifacts are fully archived.",
            "The pairwise comparison script exists, but no archived per-sample pairwise comparison outputs were found.",
            "High-risk safety artifacts exist locally for MeMR; additional multi-method numbers requested by the user are preserved with explicit provenance from the user prompt.",
        ],
    }

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(str(OUTPUT_PATH))


if __name__ == "__main__":
    main()
