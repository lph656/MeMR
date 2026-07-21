import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import HfArgumentParser, LlamaTokenizer

CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from arguments import DataTrainingArguments, ModelArguments, OSLArguments
from tasks.mtl5.dataloader_mtl_causal_llama import DataLoaderMTL
from training.trainer_continual_causal_llama_lora import ContinualTrainerMTL
from utils.inference_utils import load_memr_model


TASK_NAMES = [
    "Internal Medicine",
    "Surgery",
    "Pediatrics",
    "Gynecology and Obstetrics",
    "Andrology",
    "Oncology",
]


def find_key_tensor(state_dict):
    for key, value in state_dict.items():
        if key.endswith("key_encoder.keys"):
            return key, value.float()
    raise KeyError("No key_encoder.keys tensor found.")


def cosine_matrix(x):
    x = x.float()
    x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return x @ x.T


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


def pca_2d(rows):
    x = np.asarray(rows, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    comps = vt[:2].T
    return x @ comps


def infer_latest_snapshot(snapshot_root: Path) -> Path:
    snapshots = sorted(snapshot_root.glob("task_*_train_end_*"))
    if not snapshots:
        raise FileNotFoundError(f"No task snapshots found under {snapshot_root}")
    return snapshots[-1]


def dump_reviewer_stats(snapshot_root: Path, meta_embeddings_path: Path, out_dir: Path):
    snapshots = sorted(snapshot_root.glob("task_*_train_end_*"))
    stage_rows = []
    stage_vectors = []
    final_keys = None

    for snapshot in snapshots:
        info = json.loads((snapshot / "checkpoint_info.json").read_text(encoding="utf-8"))
        state_dict = torch.load(snapshot / "state_dict.pt", map_location="cpu")
        key_name, keys = find_key_tensor(state_dict)
        norms = keys.norm(dim=1).tolist()
        task_id = int(info["task_id"])
        stage_rows.append(
            {
                "stage_index": task_id + 1,
                "stage_label": f"Task {task_id + 1}",
                "trained_task_name": info["task_name"],
                "tensor_key": key_name,
                "mean_norm": float(np.mean(norms)),
                "std_norm": float(np.std(norms)),
                "min_norm": float(np.min(norms)),
                "max_norm": float(np.max(norms)),
            }
        )
        for idx, (task_name, norm, vec) in enumerate(zip(TASK_NAMES, norms, keys)):
            stage_vectors.append(
                {
                    "stage_index": task_id + 1,
                    "stage_label": f"Task {task_id + 1}",
                    "task_index": idx + 1,
                    "task_name": task_name,
                    "norm": float(norm),
                    "vector": vec.numpy(),
                }
            )
        if task_id == len(snapshots) - 1:
            final_keys = keys

    meta = torch.load(meta_embeddings_path, map_location="cpu").float()
    meta_cos = cosine_matrix(meta)
    final_cos = cosine_matrix(final_keys)

    pair_rows = []
    task_cos_vals, meta_cos_vals = [], []
    for i in range(len(TASK_NAMES)):
        for j in range(i + 1, len(TASK_NAMES)):
            task_val = float(final_cos[i, j].item())
            meta_val = float(meta_cos[i, j].item())
            task_cos_vals.append(task_val)
            meta_cos_vals.append(meta_val)
            pair_rows.append(
                {
                    "task_i": TASK_NAMES[i],
                    "task_j": TASK_NAMES[j],
                    "final_task_cosine": task_val,
                    "metadata_cosine": meta_val,
                }
            )

    coords = pca_2d([row["vector"] for row in stage_vectors])
    for row, coord in zip(stage_vectors, coords):
        row["pca_x"] = float(coord[0])
        row["pca_y"] = float(coord[1])
        del row["vector"]

    with (out_dir / "stage_norm_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["stage_index", "stage_label", "trained_task_name", "tensor_key", "mean_norm", "std_norm", "min_norm", "max_norm"],
        )
        writer.writeheader()
        writer.writerows(stage_rows)

    with (out_dir / "task_norms_by_stage.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["stage_index", "stage_label", "task_index", "task_name", "norm", "pca_x", "pca_y"],
        )
        writer.writeheader()
        writer.writerows(stage_vectors)

    with (out_dir / "final_task_cosine_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task_name"] + TASK_NAMES)
        for task_name, row in zip(TASK_NAMES, final_cos.tolist()):
            writer.writerow([task_name] + [float(v) for v in row])

    with (out_dir / "final_metadata_cosine_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task_name"] + TASK_NAMES)
        for task_name, row in zip(TASK_NAMES, meta_cos.tolist()):
            writer.writerow([task_name] + [float(v) for v in row])

    with (out_dir / "task_vs_metadata_similarity_pairs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task_i", "task_j", "final_task_cosine", "metadata_cosine"],
        )
        writer.writeheader()
        writer.writerows(pair_rows)

    summary = {
        "num_stages": len(stage_rows),
        "num_tasks": len(TASK_NAMES),
        "final_norms": [round(v["norm"], 4) for v in stage_vectors if v["stage_index"] == len(stage_rows)],
        "pearson_task_vs_metadata_cosine": pearson(task_cos_vals, meta_cos_vals),
        "spearman_task_vs_metadata_cosine": spearman(task_cos_vals, meta_cos_vals),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def build_arg_namespace(config_path: Path):
    parser = HfArgumentParser((DataTrainingArguments, ModelArguments, OSLArguments))
    text = config_path.read_text(encoding="utf-8")
    data_args = DataTrainingArguments()
    model_args = ModelArguments()
    osl_args = OSLArguments()

    # best-effort recovery from saved configs
    if "task_list='" in text:
        data_args.task_list = text.split("task_list='", 1)[1].split("'", 1)[0]
    if "max_target_length=" in text:
        data_args.max_target_length = int(text.split("max_target_length=", 1)[1].split(",", 1)[0])
    if "max_seq_length=" in text:
        data_args.max_seq_length = int(text.split("max_seq_length=", 1)[1].split(",", 1)[0])
    if "validation_split_percentage=" in text:
        data_args.validation_split_percentage = float(text.split("validation_split_percentage=", 1)[1].split(",", 1)[0])
    if "model_name_or_path='" in text:
        model_args.model_name_or_path = text.split("model_name_or_path='", 1)[1].split("'", 1)[0]
    if "meta_embeddings_path='" in text:
        model_args.meta_embeddings_path = text.split("meta_embeddings_path='", 1)[1].split("'", 1)[0]

    return data_args, model_args, osl_args


def main():
    ap = argparse.ArgumentParser(description="Evaluate one ATR reviewer variant output directory.")
    ap.add_argument("--experiment_dir", required=True, help="Training output directory of one variant.")
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--max_final_test_batches", type=int, default=None)
    ap.add_argument("--inference_log_dir", default=None)
    args = ap.parse_args()

    exp_dir = Path(args.experiment_dir).resolve()
    eval_dir = exp_dir / "atr_reviewer_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    snapshot_root = exp_dir / "snapshots"
    final_snapshot = infer_latest_snapshot(snapshot_root)
    config_path = exp_dir / "configs.json"
    data_args, model_args, osl_args = build_arg_namespace(config_path)

    model, tokenizer, query_encoder, task_list = load_memr_model(
        base_model_path=model_args.model_name_or_path,
        checkpoint_dir=str(final_snapshot),
        meta_embeddings_path=model_args.meta_embeddings_path,
        inference_log_dir=args.inference_log_dir or str(eval_dir / "inference_logs"),
    )
    # Align inference-time config with the training entrypoint behavior.
    model.config.query_encoder_type = model_args.query_encoder_type
    model.config.multi_peft_modules = getattr(model_args, "multi_peft_modules", True)
    model.config.disentangle_modules = getattr(model_args, "disentangle_modules", False)

    class EvalArgs:
        device = model.device
        seed = 0
        output_dir = str(eval_dir)
        num_train_epochs = 1
        do_train = False
        per_device_train_batch_size = 1
        per_device_eval_batch_size = args.per_device_eval_batch_size
        dataloader_drop_last = False
        dataloader_num_workers = 0
        dataloader_pin_memory = False
        weight_decay = 0.0
        adam_beta1 = 0.9
        adam_beta2 = 0.999
        adam_epsilon = 1e-8
        learning_rate = 1e-4
        lr_scheduler_type = "linear"

        def get_warmup_steps(self, num_training_steps):
            return 0

    tokenizer = LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        add_prefix_space=True,
        padding_side="left",
        trust_remote_code=True,
    )
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 1

    task_list = data_args.task_list.split("_")
    dataloaders = DataLoaderMTL(
        data_args=data_args,
        training_args=EvalArgs,
        task_list=task_list,
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length,
        overwrite_cache=False,
    )
    test_dataloaders = {task: dataloaders[task]["dev"] for task in task_list}

    trainer = ContinualTrainerMTL(
        args=EvalArgs,
        model=model,
        query_encoder=query_encoder,
        logger=type("DummyLogger", (), {"info": lambda *a, **k: None})(),
        task_list=task_list,
        label_list=None,
        peft_config=None,
        lora_save_dir=str(eval_dir / "noop"),
        early_stopping_patience=-1,
        tokenizer=tokenizer,
        max_target_length=data_args.max_target_length,
        learning_rate_list=None,
        max_train_batches_per_epoch=None,
        max_eval_batches=None,
        max_final_test_batches=args.max_final_test_batches,
        lamda_1=0.0,
        lamda_2=0.0,
        orthogonal_threshold=0.0,
    )

    final_results = {}
    for task in task_list:
        final_results[task] = trainer.eval(
            loader=[test_dataloaders[task], iter(test_dataloaders[task])],
            batch=len(test_dataloaders[task]),
            task=task,
            mode="test",
            final=True,
        )

    avg_rougeL = float(np.mean([v["rougeL"] for v in final_results.values()]))
    metrics = {
        "task_results": final_results,
        "average_rougeL": avg_rougeL,
        "final_snapshot": str(final_snapshot),
    }
    (eval_dir / "final_test_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    dump_reviewer_stats(snapshot_root, Path(model_args.meta_embeddings_path), eval_dir)


if __name__ == "__main__":
    main()
