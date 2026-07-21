from __future__ import annotations

import json
from pathlib import Path


def main():
    project_root = Path.cwd()
    final_checkpoint = project_root / "checkpoints_continual_keshi_llama" / "order1_compose_peft" / "snapshots" / "task_5_zhongliuke_train_end_20260626_123739"
    stage4_checkpoint = project_root / "checkpoints_continual_keshi_llama" / "order1_compose_peft" / "snapshots" / "task_4_nanke_train_end_20260626_101856"
    commands = {
        "terminal_0": "bash experiments/reviewer_0629/runbooks/terminal0_train_noisy_labels.sh",
        "terminal_1": "bash experiments/reviewer_0629/runbooks/terminal1_baseline_routing.sh",
        "terminal_2": "bash experiments/reviewer_0629/runbooks/terminal2_metadata_routing.sh",
        "terminal_3": "bash experiments/reviewer_0629/runbooks/terminal3_coldstart_unseen.sh",
        "terminal_4": "bash experiments/reviewer_0629/runbooks/terminal4_noisy_eval_and_aggregate.sh",
        "resolved_checkpoints": {
            "final_checkpoint": str(final_checkpoint),
            "stage4_checkpoint": str(stage4_checkpoint),
        },
    }
    target = project_root / "experiments" / "reviewer_0629" / "runbooks" / "commands.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(commands, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(commands, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

