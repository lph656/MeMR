"""
Replay-buffer builders for reviewer fairness experiments.

The implementation is intentionally lightweight and only writes cached JSON
artifacts under reviewer_fairness/results/cache/replay_buffer/.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Dict, List

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reviewer_fairness.src.common import RESULTS_ROOT, TASK_CODE_TO_NAME, load_train_records, write_json


def sample_replay_records(records: List[Dict[str, Any]], replay_per_task: int, seed: int) -> List[Dict[str, Any]]:
    if len(records) <= replay_per_task:
        return list(records)
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    selected = sorted(indices[:replay_per_task])
    return [records[idx] for idx in selected]


def build_replay_cache(
    dataset_root: str | Path,
    order_codes: List[str],
    replay_per_task: int = 200,
    seed: int = 0,
) -> Dict[str, Any]:
    cache_dir = RESULTS_ROOT / "cache" / "replay_buffer"
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "task_order": order_codes,
        "replay_per_task": replay_per_task,
        "seed": seed,
        "buffers": {},
        "stage_mixtures": {},
    }
    prior_records: Dict[str, List[Dict[str, Any]]] = {}

    for stage_idx, code in enumerate(order_codes):
        task_name = TASK_CODE_TO_NAME[code]
        train_records = load_train_records(dataset_root, task_name)
        replay_records = sample_replay_records(train_records, replay_per_task, seed + stage_idx)
        task_dir = cache_dir / code
        task_dir.mkdir(parents=True, exist_ok=True)
        write_json(replay_records, task_dir / "buffer.json")
        prior_records[code] = replay_records
        manifest["buffers"][code] = {
            "task_name": task_name,
            "buffer_path": str(task_dir / "buffer.json"),
            "buffer_count": len(replay_records),
        }

        if stage_idx == 0:
            continue
        mixture = []
        for previous_code in order_codes[:stage_idx]:
            mixture.extend(prior_records[previous_code])
        stage_file = cache_dir / f"stage_{stage_idx:02d}_{code}_mixture.json"
        write_json(mixture, stage_file)
        manifest["stage_mixtures"][code] = {
            "stage_index": stage_idx,
            "mixture_path": str(stage_file),
            "mixture_count": len(mixture),
        }

    write_json(manifest, cache_dir / "manifest.json")
    return manifest


if __name__ == "__main__":
    manifest = build_replay_cache("datasets/medical_consult", ["IM", "S", "P", "GO", "A", "O"])
    print(manifest["replay_per_task"])
