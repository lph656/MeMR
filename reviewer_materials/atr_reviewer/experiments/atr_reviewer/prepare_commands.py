import argparse
import json
import os
import sys
from pathlib import Path

CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)


ROOT = Path(__file__).resolve().parents[2]


def train_cmd(variant, gpu_id, output_dir, extra_args, num_train_epochs, max_train_batches_per_epoch):
    base = [
        f"CUDA_VISIBLE_DEVICES={gpu_id}",
        "python3",
        "src/run_continual_causal_llama2.py",
        "--model_name_or_path", "chinese-alpaca-plus-7b-hf",
        "--task_list", "neike_waike_erke_fuchanke_nanke_zhongliuke",
        "--continual_learning",
        "--mpeft_enabled",
        "--matching_loss_v2",
        "--meta_embeddings_path", "./metadata_embeddings/keshi_meta_embeddings.pt",
        "--do_train",
        "--padding_strategy", "longest",
        "--max_seq_length", "512",
        "--max_target_length", "64",
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "8",
        "--learning_rate", "1e-4",
        "--num_train_epochs", str(num_train_epochs),
        "--max_train_batches_per_epoch", str(max_train_batches_per_epoch),
        "--save_strategy", "no",
        "--evaluation_strategy", "no",
        "--validation_split_percentage", "0.1",
        "--overwrite_cache", "True",
        "--seed", "0",
        "--atr_variant_name", variant,
        "--output_dir", output_dir,
        "--overwrite_output_dir",
    ]
    return " ".join(base + extra_args)


def eval_cmd(gpu_id, output_dir, max_final_test_batches):
    return " ".join(
        [
            f"CUDA_VISIBLE_DEVICES={gpu_id}",
            "python3",
            "experiments/atr_reviewer/evaluate_variant.py",
            "--experiment_dir", output_dir,
            "--per_device_eval_batch_size", "1",
            "--max_final_test_batches", str(max_final_test_batches),
        ]
    )


def plot_cmd(output_dir):
    return " ".join(
        [
            "python3",
            "tools/plot_atr_reviewer_stats.py",
            "--data_dir", f"{output_dir}/atr_reviewer_eval",
            "--out_dir", f"{output_dir}/atr_reviewer_eval/figures",
        ]
    )


def main():
    ap = argparse.ArgumentParser(description="Prepare ATR reviewer experiment commands.")
    ap.add_argument("--root_output", default="checkpoints_continual_keshi_llama/atr_reviewer_suite")
    ap.add_argument(
        "--suite",
        default="rebuttal_minimal",
        choices=["rebuttal_minimal", "legacy_fast"],
        help="Experiment suite to prepare. `rebuttal_minimal` is the reviewer-facing suite aligned with ATR in the manuscript.",
    )
    ap.add_argument("--num_train_epochs", type=int, default=1)
    ap.add_argument("--max_train_batches_per_epoch", type=int, default=200)
    ap.add_argument("--max_final_test_batches", type=int, default=50)
    ap.add_argument(
        "--include_lm_off_diagnostic",
        action="store_true",
        help="Add one extra regularizer-only diagnostic run to test whether removing LM loss causes shrinkage.",
    )
    args = ap.parse_args()

    root_output = Path(args.root_output)
    if args.suite == "legacy_fast":
        variants = [
            {
                "name": "norm_only",
                "gpu": "1",
                "extra": [
                    "--lamda_1", "0.0",
                    "--lamda_2", "0.01",
                    "--atr_enable_key_ortho",
                    "--atr_key_ortho_coeff", "0.0",
                    "--atr_key_l2_lambda", "0.005",
                ],
            },
            {
                "name": "hard_orth",
                "gpu": "2",
                "extra": [
                    "--lamda_1", "0.05",
                    "--lamda_2", "0.01",
                    "--orthogonal_threshold", "0.0",
                    "--atr_enable_key_ortho",
                    "--atr_key_ortho_coeff", "0.1",
                    "--atr_key_ortho_mode", "hard",
                    "--atr_key_ortho_threshold", "0.0",
                    "--atr_key_l2_lambda", "0.005",
                ],
            },
            {
                "name": "cosine_soft_orth",
                "gpu": "3",
                "extra": [
                    "--lamda_1", "0.05",
                    "--lamda_2", "0.01",
                    "--orthogonal_threshold", "0.2",
                    "--atr_enable_key_ortho",
                    "--atr_key_ortho_coeff", "0.1",
                    "--atr_key_ortho_mode", "cosine_soft",
                    "--atr_key_ortho_threshold", "0.1",
                    "--atr_key_l2_lambda", "0.005",
                ],
            },
            {
                "name": "lm_ablation_diagnostic",
                "gpu": "4",
                "extra": [
                    "--lamda_1", "0.05",
                    "--lamda_2", "0.01",
                    "--orthogonal_threshold", "0.2",
                    "--atr_enable_key_ortho",
                    "--atr_key_ortho_coeff", "0.1",
                    "--atr_key_ortho_mode", "unnormalized_soft",
                    "--atr_key_ortho_threshold", "0.1",
                    "--atr_key_l2_lambda", "0.005",
                    "--lm_loss_weight", "0.0",
                ],
            },
        ]
    else:
        # Reviewer-facing minimal suite: isolate manuscript ATR on task vectors only.
        # Keep LoRA-level OSL regularizers off to avoid confounding the response.
        variants = [
            {
                "name": "wo_atr",
                "gpu": "1",
                "extra": [
                    "--lamda_1", "0.0",
                    "--lamda_2", "0.0",
                    "--atr_key_ortho_coeff", "0.0",
                    "--atr_key_l2_lambda", "0.0",
                ],
            },
            {
                "name": "norm_only",
                "gpu": "2",
                "extra": [
                    "--lamda_1", "0.0",
                    "--lamda_2", "0.0",
                    "--atr_key_ortho_coeff", "0.0",
                    "--atr_key_l2_lambda", "0.005",
                ],
            },
            {
                "name": "hard_orth",
                "gpu": "3",
                "extra": [
                    "--lamda_1", "0.0",
                    "--lamda_2", "0.0",
                    "--atr_enable_key_ortho",
                    "--atr_key_ortho_coeff", "0.1",
                    "--atr_key_ortho_mode", "hard",
                    "--atr_key_ortho_threshold", "0.0",
                    "--atr_key_l2_lambda", "0.005",
                ],
            },
            {
                "name": "cosine_soft_orth",
                "gpu": "4",
                "extra": [
                    "--lamda_1", "0.0",
                    "--lamda_2", "0.0",
                    "--atr_enable_key_ortho",
                    "--atr_key_ortho_coeff", "0.1",
                    "--atr_key_ortho_mode", "cosine_soft",
                    "--atr_key_ortho_threshold", "0.1",
                    "--atr_key_l2_lambda", "0.005",
                ],
            },
            {
                "name": "full_atr",
                "gpu": "1",
                "extra": [
                    "--lamda_1", "0.0",
                    "--lamda_2", "0.0",
                    "--atr_enable_key_ortho",
                    "--atr_key_ortho_coeff", "0.1",
                    "--atr_key_ortho_mode", "unnormalized_soft",
                    "--atr_key_ortho_threshold", "0.1",
                    "--atr_key_l2_lambda", "0.005",
                ],
            },
        ]
        if args.include_lm_off_diagnostic:
            variants.append(
                {
                    "name": "lm_off_full_atr",
                    "gpu": "2",
                    "extra": [
                        "--lamda_1", "0.0",
                        "--lamda_2", "0.0",
                        "--atr_enable_key_ortho",
                        "--atr_key_ortho_coeff", "0.1",
                        "--atr_key_ortho_mode", "unnormalized_soft",
                        "--atr_key_ortho_threshold", "0.1",
                        "--atr_key_l2_lambda", "0.005",
                        "--lm_loss_weight", "0.0",
                    ],
                }
            )

    commands = []
    for variant in variants:
        out_dir = str(root_output / variant["name"])
        commands.append(
            {
                "variant": variant["name"],
                "gpu": variant["gpu"],
                "train": train_cmd(
                    variant["name"],
                    variant["gpu"],
                    out_dir,
                    variant["extra"],
                    args.num_train_epochs,
                    args.max_train_batches_per_epoch,
                ),
                "eval": eval_cmd(variant["gpu"], out_dir, args.max_final_test_batches),
                "plot": plot_cmd(out_dir),
            }
        )

    aggregate = " ".join(
        [
            "python3",
            "experiments/atr_reviewer/aggregate_variants.py",
            "--root_dir", str(root_output),
        ]
    )
    output = {
        "root_output": str(root_output),
        "suite": args.suite,
        "num_train_epochs": args.num_train_epochs,
        "max_train_batches_per_epoch": args.max_train_batches_per_epoch,
        "max_final_test_batches": args.max_final_test_batches,
        "commands": commands,
        "aggregate": aggregate,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
