"""
Unified entrypoint for reviewer fairness reference baselines.

This script is intentionally additive and does not modify the original MeMR
training entrypoints. It standardizes outputs under reviewer_fairness/results.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reviewer_fairness.src.build_joint_dataset import build_joint_dataset
from reviewer_fairness.src.build_replay_buffer import build_replay_cache
from reviewer_fairness.src.build_single_task_dataset import build_single_task_cache
from reviewer_fairness.src.common import (
    DEFAULT_TASK_ORDER,
    LOGS_ROOT,
    RESULTS_ROOT,
    TASK_CODE_TO_NAME,
    TASK_NAME_TO_CODE,
    append_text,
    build_failure_payload,
    config_to_plain,
    detect_todo_values,
    dump_yaml,
    find_existing_memr_snapshot,
    format_command,
    has_real_metrics,
    load_yaml,
    make_metrics_template,
    order_codes_to_task_names,
    read_json,
    resolve_order_codes,
    task_name_to_code,
    write_failure_artifacts,
    write_json,
    write_metrics_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reviewer fairness reference experiments.")
    parser.add_argument("--method", required=True, choices=[
        "sequential_ft",
        "joint_training",
        "single_task_oracle",
        "metadata_only_routing",
        "er_lora",
        "memr",
    ])
    parser.add_argument("--config", default="reviewer_fairness/configs/fairness_default.yaml")
    parser.add_argument("--order", required=True, help="order1/order2/order3/all_tasks")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--replay_per_task", type=int, default=None)
    return parser.parse_args()


def load_resolved_config(config_path: str, order_name: str, method: str, replay_override: Optional[int]) -> Dict[str, Any]:
    config = load_yaml(config_path)
    todo_fields = detect_todo_values(config)
    allowed_todos = {
        "decoding.temperature",
        "decoding.top_p",
        "decoding.num_beams",
    }
    unresolved = [item for item in todo_fields if item not in allowed_todos]
    if unresolved:
        raise RuntimeError(
            "fairness_default.yaml still contains unresolved TODO values that are required for execution: "
            + ", ".join(unresolved)
        )

    config = config_to_plain(config)
    config["method"] = method
    config["order_name"] = order_name
    if replay_override is not None:
        config.setdefault("replay", {})["replay_per_task"] = replay_override
    return config


def output_dir_for(method: str, order_name: str) -> Path:
    return RESULTS_ROOT / "reference" / method / order_name


def metrics_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = read_json(path)
    except Exception:
        return False
    return has_real_metrics(payload)


def prepare_output_dir(method: str, order_name: str, overwrite: bool) -> Path:
    out_dir = output_dir_for(method, order_name)
    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ["checkpoints", "predictions"]:
        (out_dir / name).mkdir(parents=True, exist_ok=True)
    return out_dir


def run_subprocess(command: List[str], log_path: Path) -> None:
    ensure_dir = log_path.parent
    ensure_dir.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(command) + "\n")
        log_file.flush()
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        assert process.stdout is not None
        while True:
            chunk = process.stdout.read(1024)
            if not chunk:
                break
            if isinstance(chunk, bytes):
                text = chunk.decode("utf-8", errors="replace")
            else:
                text = chunk
            sys.stdout.write(text)
            sys.stdout.flush()
            log_file.write(text)
            log_file.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Subprocess failed with exit code {return_code}: {' '.join(command)}")


def write_standard_artifacts(out_dir: Path, metrics: Dict[str, Any], config: Dict[str, Any], notes: Dict[str, Any]) -> None:
    write_json(metrics, out_dir / "metrics.json")
    write_metrics_csv(metrics, out_dir / "metrics.csv")
    dump_yaml(config, out_dir / "config_resolved.yaml")
    write_json(notes, out_dir / "fairness_notes.json")


def build_reuse_notes(method: str, source: str, verified: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = {
        "status": "ok",
        "method": method,
        "result_source": source,
        "verification_status": verified,
    }
    if extra:
        payload.update(extra)
    return payload


def collect_parameter_info_from_checkpoint(snapshot_dir: Path) -> Dict[str, Any]:
    info_path = snapshot_dir / "checkpoint_info.json"
    if not info_path.exists():
        return {"trainable_params": None, "total_params": None}
    info = read_json(info_path)
    return {
        "trainable_params": None,
        "total_params": None,
        "checkpoint_info_keys": sorted(info.keys()),
    }


def build_existing_memr_metrics(config: Dict[str, Any], order_name: str) -> Dict[str, Any]:
    metrics = make_metrics_template("memr", order_name, config)
    snapshot_dir = find_existing_memr_snapshot()
    if snapshot_dir is None:
        metrics["notes"] = "No existing MeMR snapshot was found for matched-setting reuse."
        return metrics

    param_info = collect_parameter_info_from_checkpoint(snapshot_dir)
    metrics["trainable_params"] = param_info.get("trainable_params")
    metrics["total_params"] = param_info.get("total_params")
    metrics["notes"] = (
        "Existing MeMR snapshot detected and reused for fairness bookkeeping, "
        "but no standardized per-department metrics.json was found in the current workspace snapshot."
    )
    return metrics


def build_existing_sequential_ft_metrics(config: Dict[str, Any], order_name: str) -> Dict[str, Any]:
    metrics = make_metrics_template("sequential_ft", order_name, config)
    wo_atr_path = PROJECT_ROOT / "checkpoints_continual_keshi_llama" / "atr_reviewer_suite" / "wo_atr" / "atr_reviewer_eval" / "final_test_metrics.json"
    if not wo_atr_path.exists():
        metrics["notes"] = "No existing wo_atr final_test_metrics.json was found for sequential fine-tuning reuse."
        return metrics

    payload = read_json(wo_atr_path)
    task_results = payload.get("task_results", {})
    department_scores = {}
    for task_name, result in task_results.items():
        if task_name in TASK_NAME_TO_CODE:
            department_scores[TASK_NAME_TO_CODE[task_name]] = result.get("rougeL")
    metrics["department_scores"].update(department_scores)
    metrics["average"] = payload.get("average_rougeL")
    metrics["notes"] = (
        "Reused ATR reviewer 'wo_atr' result as the closest existing sequential fine-tuning lower-bound proxy. "
        "FWT/FR/BWT remain unavailable from the archived artifact."
    )
    return metrics


def build_existing_metadata_only_metrics(config: Dict[str, Any], order_name: str) -> Dict[str, Any]:
    metrics = make_metrics_template("metadata_only_routing", order_name, config)
    path = PROJECT_ROOT / "experiments" / "reviewer_0629" / "outputs" / "routing_baseline" / "summary.json"
    if not path.exists():
        metrics["notes"] = "No existing metadata-only routing summary.json was found."
        return metrics
    payload = read_json(path)
    none_block = payload.get("none", {})
    generation = none_block.get("generation_metrics", {}).get("full", {})
    shared_score = generation.get("rougeL")
    if shared_score is not None:
        for code in DEFAULT_TASK_ORDER:
            metrics["department_scores"][code] = shared_score
        metrics["average"] = shared_score
    metrics["notes"] = (
        "Reused reviewer_0629 routing baseline summary. The artifact is routing-centric and does not contain "
        "true per-department CL scores, so the generation rougeL summary is replicated as a placeholder."
    )
    return metrics


def build_not_run_metrics(method: str, order_name: str, config: Dict[str, Any], note: str) -> Dict[str, Any]:
    metrics = make_metrics_template(method, order_name, config)
    metrics["notes"] = note
    return metrics


def write_logs_for_dry_run(out_dir: Path, method: str, command: str) -> None:
    append_text(out_dir / "train.log", f"[{method}] dry-run command\n{command}\n")
    append_text(out_dir / "eval.log", f"[{method}] dry-run command\n{command}\n")


def run_method(args: argparse.Namespace, config: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    task_orders = config["continual_learning"]["task_orders"]
    order_codes = resolve_order_codes(task_orders, args.order)
    task_names = order_codes_to_task_names(order_codes)

    if args.method == "memr":
        existing_snapshot = find_existing_memr_snapshot()
        if existing_snapshot is None:
            raise RuntimeError("No existing MeMR snapshot found. Please train MeMR first or provide a matched checkpoint.")
        eval_cmd = [
            "python",
            "reviewer_fairness/src/evaluate_checkpoint.py",
            "--method",
            "memr",
            "--config",
            args.config,
            "--checkpoint_dir",
            str(existing_snapshot),
            "--output_dir",
            str(out_dir),
            "--order",
            args.order,
        ]
        run_subprocess(eval_cmd, out_dir / "eval.log")
        task_scores = read_json(out_dir / "task_scores.json")
        metrics = make_metrics_template("memr", args.order, config)
        metrics["department_scores"].update(task_scores)
        valid_scores = [value for value in task_scores.values() if isinstance(value, (int, float))]
        metrics["average"] = round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else None
        metrics["notes"] = f"Evaluated existing MeMR checkpoint at {existing_snapshot}."
        return metrics
    if args.method == "sequential_ft":
        train_cmd = [
            "python",
            "reviewer_fairness/src/train_lora_reference.py",
            "--method",
            "sequential_ft",
            "--config",
            args.config,
            "--order",
            args.order,
            "--output_dir",
            str(out_dir),
            "--gpu",
            str(args.gpu),
        ]
        run_subprocess(train_cmd, out_dir / "train.log")
        return read_json(out_dir / "metrics.json")
    if args.method == "metadata_only_routing":
        existing_snapshot = find_existing_memr_snapshot()
        if existing_snapshot is None:
            raise RuntimeError("No existing MeMR snapshot found for metadata-only routing evaluation.")
        eval_cmd = [
            "python",
            "reviewer_fairness/src/evaluate_checkpoint.py",
            "--method",
            "metadata_only_routing",
            "--config",
            args.config,
            "--checkpoint_dir",
            str(existing_snapshot),
            "--output_dir",
            str(out_dir),
            "--order",
            args.order,
        ]
        run_subprocess(eval_cmd, out_dir / "eval.log")
        task_scores = read_json(out_dir / "task_scores.json")
        metrics = make_metrics_template("metadata_only_routing", args.order, config)
        metrics["department_scores"].update(task_scores)
        valid_scores = [value for value in task_scores.values() if isinstance(value, (int, float))]
        metrics["average"] = round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else None
        metrics["notes"] = f"Evaluated metadata-only routing by wrapping existing MeMR checkpoint at {existing_snapshot}."
        return metrics

    if args.method == "joint_training":
        build_joint_dataset(config["dataset_root"], order_codes)
        train_cmd = [
            "python",
            "reviewer_fairness/src/train_lora_reference.py",
            "--method",
            "joint_training",
            "--config",
            args.config,
            "--order",
            args.order,
            "--output_dir",
            str(out_dir),
            "--gpu",
            str(args.gpu),
        ]
        run_subprocess(train_cmd, out_dir / "train.log")
        return read_json(out_dir / "metrics.json")

    if args.method == "single_task_oracle":
        build_single_task_cache(config["dataset_root"], order_codes)
        train_cmd = [
            "python",
            "reviewer_fairness/src/train_lora_reference.py",
            "--method",
            "single_task_oracle",
            "--config",
            args.config,
            "--order",
            args.order,
            "--output_dir",
            str(out_dir),
            "--gpu",
            str(args.gpu),
        ]
        run_subprocess(train_cmd, out_dir / "train.log")
        return read_json(out_dir / "metrics.json")

    if args.method == "er_lora":
        replay_per_task = config["replay"]["replay_per_task"]
        build_replay_cache(config["dataset_root"], order_codes, replay_per_task=replay_per_task, seed=config["training"]["seed"])
        train_cmd = [
            "python",
            "reviewer_fairness/src/train_lora_reference.py",
            "--method",
            "er_lora",
            "--config",
            args.config,
            "--order",
            args.order,
            "--output_dir",
            str(out_dir),
            "--gpu",
            str(args.gpu),
        ]
        run_subprocess(train_cmd, out_dir / "train.log")
        return read_json(out_dir / "metrics.json")

    raise ValueError(f"Unsupported method: {args.method}")


def main() -> int:
    args = parse_args()
    out_dir = output_dir_for(args.method, args.order)
    if metrics_complete(out_dir / "metrics.json") and not args.overwrite:
        print(f"Existing metrics found at {out_dir / 'metrics.json'}, skipping.")
        return 0

    out_dir = prepare_output_dir(args.method, args.order, args.overwrite)
    config = load_resolved_config(args.config, args.order, args.method, args.replay_per_task)

    command = format_command([
        "python",
        "reviewer_fairness/src/run_reference_experiment.py",
        "--method",
        args.method,
        "--config",
        args.config,
        "--order",
        args.order,
        "--gpu",
        str(args.gpu),
    ] + (["--overwrite"] if args.overwrite else []) + (["--dry_run"] if args.dry_run else []))

    try:
        if args.dry_run:
            metrics = build_not_run_metrics(
                args.method,
                args.order,
                config,
                f"Dry run only. Resolved command: {command}",
            )
            notes = build_reuse_notes(args.method, "dry_run", "UNVERIFIED", {"command": command})
            write_standard_artifacts(out_dir, metrics, config, notes)
            write_logs_for_dry_run(out_dir, args.method, command)
            return 0

        metrics = run_method(args, config, out_dir)
        result_source = "existing_memr_checkpoint_eval" if args.method in {"memr", "metadata_only_routing"} else "matched_training_run"
        notes = build_reuse_notes(
            args.method,
            result_source,
            "MATCHED" if metrics.get("average") is not None else "UNVERIFIED",
            {"gpu": args.gpu, "resolved_order": args.order},
        )
        write_standard_artifacts(out_dir, metrics, config, notes)
        append_text(out_dir / "train.log", f"[{args.method}] {command}\n")
        append_text(out_dir / "eval.log", f"[{args.method}] standardized artifacts written\n")
    except Exception as exc:
        dump_yaml(config, out_dir / "config_resolved.yaml")
        write_failure_artifacts(out_dir, args.method, args.order, config, exc)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
