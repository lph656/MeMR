import csv
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT / "checkpoints_continual_keshi_llama" / "order1_compose_peft" / "snapshots"
META_PATH = ROOT / "metadata_embeddings" / "keshi_meta_embeddings.pt"
OUT_DIR = ROOT / "analysis" / "atr_reviewer_real_statistics"

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
    raise KeyError("No key_encoder.keys tensor found in state_dict")


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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    snapshots = sorted(SNAPSHOT_DIR.glob("task_*_train_end_*"))
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found under {SNAPSHOT_DIR}")

    stage_rows = []
    stage_vectors = []
    final_keys = None

    for snapshot in snapshots:
        info = json.loads((snapshot / "checkpoint_info.json").read_text(encoding="utf-8"))
        state_dict = torch.load(snapshot / "state_dict.pt", map_location="cpu")
        key_name, keys = find_key_tensor(state_dict)
        norms = keys.norm(dim=1).tolist()
        task_id = int(info["task_id"])
        stage_label = f"Task {task_id + 1}"

        stage_rows.append(
            {
                "stage_index": task_id + 1,
                "stage_label": stage_label,
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
                    "stage_label": stage_label,
                    "task_index": idx + 1,
                    "task_name": task_name,
                    "norm": float(norm),
                    "vector": vec.numpy(),
                }
            )

        if task_id == len(snapshots) - 1:
            final_keys = keys

    meta = torch.load(META_PATH, map_location="cpu").float()
    meta_cos = cosine_matrix(meta)
    final_cos = cosine_matrix(final_keys)

    pair_rows = []
    task_cos_vals = []
    meta_cos_vals = []
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

    with (OUT_DIR / "stage_norm_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["stage_index", "stage_label", "trained_task_name", "tensor_key", "mean_norm", "std_norm", "min_norm", "max_norm"],
        )
        writer.writeheader()
        writer.writerows(stage_rows)

    with (OUT_DIR / "task_norms_by_stage.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["stage_index", "stage_label", "task_index", "task_name", "norm", "pca_x", "pca_y"],
        )
        writer.writeheader()
        writer.writerows(stage_vectors)

    with (OUT_DIR / "final_task_cosine_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task_name"] + TASK_NAMES)
        for task_name, row in zip(TASK_NAMES, final_cos.tolist()):
            writer.writerow([task_name] + [float(v) for v in row])

    with (OUT_DIR / "final_metadata_cosine_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task_name"] + TASK_NAMES)
        for task_name, row in zip(TASK_NAMES, meta_cos.tolist()):
            writer.writerow([task_name] + [float(v) for v in row])

    with (OUT_DIR / "task_vs_metadata_similarity_pairs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task_i", "task_j", "final_task_cosine", "metadata_cosine"],
        )
        writer.writeheader()
        writer.writerows(pair_rows)

    summary = {
        "source_snapshot_dir": str(SNAPSHOT_DIR.relative_to(ROOT)),
        "source_meta_path": str(META_PATH.relative_to(ROOT)),
        "num_stages": len(stage_rows),
        "num_tasks": len(TASK_NAMES),
        "final_norms": [round(v["norm"], 4) for v in stage_vectors if v["stage_index"] == len(stage_rows)],
        "pearson_task_vs_metadata_cosine": pearson(task_cos_vals, meta_cos_vals),
        "spearman_task_vs_metadata_cosine": spearman(task_cos_vals, meta_cos_vals),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
