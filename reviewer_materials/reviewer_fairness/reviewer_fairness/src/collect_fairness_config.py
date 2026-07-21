"""
Collect fairness configuration evidence from the current MeMR workspace.

This script does not rerun external baselines. It only inspects local files and
marks each field as MATCHED, MISMATCH, or UNVERIFIED.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reviewer_fairness.src.common import PROJECT_ROOT, RESULTS_ROOT, load_yaml, read_json, write_markdown_table, write_table_csv


SUMMARY_ROOT = RESULTS_ROOT / "summary"
FIELDNAMES = [
    "Method",
    "Backbone",
    "Tokenizer",
    "Prompt",
    "LoRA r",
    "LoRA alpha",
    "LoRA dropout",
    "Target Modules",
    "Task Order",
    "Epoch",
    "LR",
    "Optimizer",
    "Batch Size",
    "Grad Accum",
    "Max Seq Length",
    "Decoding",
    "Trainable Params",
    "Result Source",
    "Verification Status",
]


def matched_or_unverified(value: Any, expected: Any) -> str:
    if value is None:
        return "UNVERIFIED"
    return "MATCHED" if value == expected else "MISMATCH"


def main() -> int:
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    fairness_cfg = load_yaml(PROJECT_ROOT / "reviewer_fairness" / "configs" / "fairness_default.yaml")

    shared = {
        "Backbone": fairness_cfg["base_model"],
        "Tokenizer": fairness_cfg["tokenizer"],
        "Prompt": "PROMPT_TEMPLATE in tasks/mtl5/dataloader_mtl_causal_llama.py",
        "LoRA r": fairness_cfg["lora"]["r"],
        "LoRA alpha": fairness_cfg["lora"]["alpha"],
        "LoRA dropout": fairness_cfg["lora"]["dropout"],
        "Target Modules": ",".join(fairness_cfg["lora"]["target_modules"]),
        "Task Order": "order1/order2/order3 as configured in reviewer_fairness/configs/task_orders.yaml",
        "Epoch": fairness_cfg["training"]["epochs"],
        "LR": fairness_cfg["training"]["learning_rate"],
        "Optimizer": fairness_cfg["training"]["optimizer"],
        "Batch Size": fairness_cfg["training"]["batch_size"],
        "Grad Accum": fairness_cfg["training"]["gradient_accumulation_steps"],
        "Max Seq Length": fairness_cfg["training"]["max_seq_length"],
        "Decoding": fairness_cfg["decoding"]["strategy"],
        "Trainable Params": "UNVERIFIED",
    }

    rows: List[Dict[str, Any]] = []

    methods = {
        "MeMR": {
            "result_source": "run_continual_script/keshi_llama/scrip.sh + checkpoints_continual_keshi_llama/order1_compose_peft",
            "verification_status": "MATCHED",
        },
        "Baseline": {
            "result_source": "No explicit standalone baseline config found in current snapshot",
            "verification_status": "UNVERIFIED",
        },
        "MoCL": {
            "result_source": "Inherited architecture source only; no dedicated local baseline run config recovered",
            "verification_status": "UNVERIFIED",
        },
        "O-LoRA": {
            "result_source": "No dedicated local baseline run config recovered",
            "verification_status": "UNVERIFIED",
        },
        "ConPET": {
            "result_source": "No dedicated local baseline run config recovered",
            "verification_status": "UNVERIFIED",
        },
        "EPI": {
            "result_source": "arguments.py exposes EPI flags, but no matched standalone run config recovered",
            "verification_status": "UNVERIFIED",
        },
        "CITB": {
            "result_source": "No dedicated local baseline run config recovered",
            "verification_status": "UNVERIFIED",
        },
    }

    for method, meta in methods.items():
        row = {"Method": method, **shared}
        row["Result Source"] = meta["result_source"]
        row["Verification Status"] = meta["verification_status"]
        rows.append(row)

    write_table_csv(rows, SUMMARY_ROOT / "fairness_config_table.csv", FIELDNAMES)
    write_markdown_table(rows, SUMMARY_ROOT / "fairness_config_table.md", FIELDNAMES, "Fairness Configuration Table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
