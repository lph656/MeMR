from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from experiments.reviewer_0629.common import (
    TASK_NAMES,
    build_query_embedding,
    build_validation_split,
    compute_entropy,
    compute_ece,
    compute_multiclass_brier,
    deterministic_sample,
    ensure_dir,
    load_eval_jsonl,
    load_memr_model,
    load_train_records,
    read_json,
    write_csv,
    write_json,
)


@dataclass
class Sample:
    sample_id: str
    task_name: str
    task_id: int
    question: str


def _task_name_to_id(task_name: str) -> int:
    return TASK_NAMES.index(task_name)


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return x.unsqueeze(0)
    if x.ndim == 3 and x.shape[1] == 1:
        return x.squeeze(1)
    return x


def _module_device(module: Optional[nn.Module]) -> Optional[torch.device]:
    if module is None:
        return None
    try:
        return next(module.parameters()).device
    except StopIteration:
        return None


def _build_samples_from_task_records(
    task_list: Sequence[str],
    dataset_root: str,
    split: str,
    max_samples_per_task: Optional[int],
    seed: int,
) -> List[Sample]:
    samples: List[Sample] = []
    for task_index, task_name in enumerate(task_list):
        train_records, dev_records = build_validation_split(task_name, dataset_root=dataset_root, seed=seed)
        records = train_records if split == "train" else dev_records
        if max_samples_per_task is not None:
            records = [record for _, record in deterministic_sample(records, max_samples_per_task, seed + task_index * 13)]
        for idx, record in enumerate(records):
            question = record.get("instruction", record.get("question"))
            if question is None:
                raise KeyError(f"Missing instruction/question field in {task_name} {split} record")
            samples.append(
                Sample(
                    sample_id=f"{task_name}_{split}_{idx}",
                    task_name=task_name,
                    task_id=_task_name_to_id(task_name),
                    question=question,
                )
            )
    return samples


def _build_eval_samples(path: str) -> List[Sample]:
    samples: List[Sample] = []
    for record in load_eval_jsonl(path):
        samples.append(
            Sample(
                sample_id=record["sample_id"],
                task_name=record["task_name"],
                task_id=int(record["task_id"]),
                question=record["question"],
            )
        )
    return samples


def _cache_path(cache_dir: Path, split_name: str) -> Path:
    return cache_dir / f"{split_name}.pt"


def _precompute_query_embeddings(
    bundle,
    samples: Sequence[Sample],
    cache_file: Path,
    query_encoder_type: str = "avg_word_embed",
) -> Dict[str, torch.Tensor]:
    if cache_file.exists():
        payload = torch.load(cache_file, map_location="cpu")
        payload["embeddings"] = _ensure_2d(payload["embeddings"])
        return payload

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    embeddings = []
    task_ids = []
    sample_ids = []
    task_names = []
    questions = []
    for sample in tqdm(samples, desc=f"cache {cache_file.stem}"):
        q_emb = build_query_embedding(
            bundle.tokenizer,
            bundle.query_encoder,
            sample.question,
            query_encoder_type=query_encoder_type,
        )
        embeddings.append(_ensure_2d(q_emb.detach().cpu()).squeeze(0))
        task_ids.append(sample.task_id)
        sample_ids.append(sample.sample_id)
        task_names.append(sample.task_name)
        questions.append(sample.question)

    payload = {
        "embeddings": torch.stack(embeddings, dim=0),
        "task_ids": torch.tensor(task_ids, dtype=torch.long),
        "sample_ids": sample_ids,
        "task_names": task_names,
        "questions": questions,
    }
    torch.save(payload, cache_file)
    return payload


class TemperatureGate(nn.Module):
    def __init__(self, init_temperature: float = 1.0, init_bias: float = 0.0):
        super().__init__()
        init_temperature = max(float(init_temperature), 1e-3)
        self.raw_temperature = nn.Parameter(torch.tensor(math.log(math.exp(init_temperature) - 1.0)))
        self.bias = nn.Parameter(torch.tensor(float(init_bias)))

    def forward(self, cos_meta: torch.Tensor) -> torch.Tensor:
        temperature = F.softplus(self.raw_temperature) + 1e-4
        temperature = temperature.to(cos_meta.device)
        bias = self.bias.to(cos_meta.device)
        return torch.sigmoid((cos_meta + bias) / temperature)

    def extra_state(self) -> Dict[str, float]:
        temperature = float((F.softplus(self.raw_temperature) + 1e-4).detach().cpu().item())
        return {"temperature": temperature, "bias": float(self.bias.detach().cpu().item())}


class LearnableGateNet(nn.Module):
    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(features).squeeze(-1))


def _compute_alpha(
    variant: str,
    q: torch.Tensor,
    meta: torch.Tensor,
    task: torch.Tensor,
    gate_model: Optional[nn.Module],
) -> torch.Tensor:
    q = _ensure_2d(q)
    q_n = _normalize(q)
    meta_n = _normalize(meta)
    task_n = _normalize(task)
    cos_q_meta = (q_n.unsqueeze(1) * meta_n).sum(dim=-1)
    cos_q_task = (q_n.unsqueeze(1) * task_n).sum(dim=-1)
    cos_meta_task = (meta_n * task_n).sum(dim=-1)

    if variant == "current":
        return torch.sigmoid(cos_q_meta)
    if variant == "metadata_only":
        return torch.ones_like(cos_q_meta)
    if variant == "temperature":
        assert gate_model is not None
        return gate_model(cos_q_meta)
    if variant == "learnable_mlp":
        assert gate_model is not None
        q_expanded = q_n.unsqueeze(1).expand_as(meta_n)
        features = torch.stack(
            [
                cos_q_meta,
                cos_q_task,
                cos_meta_task,
                (q_expanded - meta_n).norm(dim=-1),
                (q_expanded - task_n).norm(dim=-1),
                (meta_n - task_n).norm(dim=-1),
            ],
            dim=-1,
        )
        return gate_model(features)
    raise ValueError(f"Unsupported variant: {variant}")


def _compute_scores(
    variant: str,
    q: torch.Tensor,
    meta: torch.Tensor,
    task: torch.Tensor,
    softmax_scale: float,
    gate_model: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    target_device = _module_device(gate_model) or meta.device
    q = _ensure_2d(q).to(target_device)
    q_n = _normalize(q)
    meta = meta.to(q.device, dtype=q.dtype)
    task = task.to(q.device, dtype=q.dtype)
    meta_b = meta.unsqueeze(0).expand(q.shape[0], -1, -1)
    task_b = task.unsqueeze(0).expand(q.shape[0], -1, -1)

    alpha = _compute_alpha(variant, q, meta_b, task_b, gate_model)
    dynamic = alpha.unsqueeze(-1) * meta_b + (1.0 - alpha).unsqueeze(-1) * task_b
    scores = torch.bmm(q_n.unsqueeze(1), _normalize(dynamic).transpose(1, 2)).squeeze(1) * softmax_scale
    return scores, alpha


def _summarise_records(records: Sequence[Dict[str, object]], alpha_values: Sequence[float]) -> Dict[str, float]:
    confidences = [float(row["top1_confidence"]) for row in records]
    correct = [int(row["top1_correct"]) for row in records]
    labels = [int(row["task_id"]) for row in records]
    probabilities = [row["weights"] for row in records]
    summary = {
        "num_samples": len(records),
        "top1_accuracy": sum(correct) / max(len(correct), 1),
        "top3_accuracy": sum(int(row["top3_hit"]) for row in records) / max(len(records), 1),
        "ece": compute_ece(confidences, correct),
        "brier_score": compute_multiclass_brier(probabilities, labels),
        "mean_entropy": sum(float(row["entropy"]) for row in records) / max(len(records), 1),
        "mean_correct_task_weight": sum(float(row["correct_task_weight"]) for row in records) / max(len(records), 1),
        "mean_top1_confidence": sum(confidences) / max(len(confidences), 1),
    }
    if alpha_values:
        alpha_tensor = torch.tensor(alpha_values, dtype=torch.float32)
        summary.update(
            {
                "mean_gate_alpha": float(alpha_tensor.mean().item()),
                "std_gate_alpha": float(alpha_tensor.std(unbiased=False).item()),
                "min_gate_alpha": float(alpha_tensor.min().item()),
                "max_gate_alpha": float(alpha_tensor.max().item()),
            }
        )
    return summary


def _evaluate_split(
    variant: str,
    eval_samples: Sequence[Sample],
    embeddings: torch.Tensor,
    task_ids: torch.Tensor,
    task_meta: torch.Tensor,
    task_keys: torch.Tensor,
    softmax_scale: float,
    output_dir: Path,
    gate_model: Optional[nn.Module] = None,
) -> Dict[str, object]:
    records: List[Dict[str, object]] = []
    alpha_values: List[float] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = output_dir / "per_sample.jsonl"
    if per_sample_path.exists():
        per_sample_path.unlink()

    batch_size = 32
    for start in tqdm(range(0, len(eval_samples), batch_size), desc=f"eval {variant}"):
        end = min(start + batch_size, len(eval_samples))
        batch_q = embeddings[start:end]
        batch_labels = task_ids[start:end]
        with torch.no_grad():
            scores, alpha = _compute_scores(variant, batch_q, task_meta, task_keys, softmax_scale, gate_model)
            probs = F.softmax(scores, dim=-1)
        for i in range(end - start):
            prob = probs[i].detach().cpu()
            ranked = torch.argsort(prob, descending=True)
            alpha_row = alpha[i].detach().cpu()
            task_id = int(batch_labels[i].item())
            record = {
                "sample_id": eval_samples[start + i].sample_id,
                "task_name": eval_samples[start + i].task_name,
                "task_id": task_id,
                "question": eval_samples[start + i].question,
                "weights": prob.tolist(),
                "predicted_task_id": int(ranked[0].item()),
                "predicted_task_name": TASK_NAMES[int(ranked[0].item())],
                "top1_confidence": float(prob[ranked[0]].item()),
                "top1_correct": int(int(ranked[0].item()) == task_id),
                "correct_task_weight": float(prob[task_id].item()),
                "entropy": compute_entropy(prob.tolist()),
                "top3_hit": int(task_id in ranked[:3].tolist()),
                "alpha_mean": float(alpha_row.mean().item()),
                "alpha_correct_task": float(alpha_row[task_id].item()),
                "alpha_min": float(alpha_row.min().item()),
                "alpha_max": float(alpha_row.max().item()),
            }
            records.append(record)
            alpha_values.extend(alpha_row.tolist())
            with per_sample_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = _summarise_records(records, alpha_values)
    write_json(output_dir / "summary.json", summary)
    write_csv(
        output_dir / "summary_table.csv",
        [{"metric": k, "value": v} for k, v in summary.items()],
        ["metric", "value"],
    )
    return summary


def _train_gate(
    variant: str,
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    dev_embeddings: torch.Tensor,
    dev_labels: torch.Tensor,
    task_meta: torch.Tensor,
    task_keys: torch.Tensor,
    softmax_scale: float,
    device: torch.device,
    num_epochs: int,
    batch_size: int,
    lr: float,
    temperature_init: float,
    hidden_size: int,
    output_dir: Path,
) -> nn.Module:
    if variant == "temperature":
        gate_model: nn.Module = TemperatureGate(init_temperature=temperature_init).to(device)
    elif variant == "learnable_mlp":
        gate_model = LearnableGateNet(hidden_size=hidden_size).to(device)
    else:
        raise ValueError(f"Variant {variant} is not trainable")

    optimizer = torch.optim.AdamW(gate_model.parameters(), lr=lr)
    train_ds = TensorDataset(train_embeddings, train_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_state = None
    best_dev_acc = -1.0
    history = []

    task_meta = task_meta.to(device)
    task_keys = task_keys.to(device)

    for epoch in range(num_epochs):
        gate_model.train()
        total_loss = 0.0
        total_count = 0
        for batch_q, batch_y in train_loader:
            batch_q = batch_q.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            scores, _ = _compute_scores(variant, batch_q, task_meta, task_keys, softmax_scale, gate_model)
            loss = F.cross_entropy(scores, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * batch_q.size(0)
            total_count += batch_q.size(0)
        dev_metrics = _eval_accuracy_only(variant, dev_embeddings, dev_labels, task_meta, task_keys, softmax_scale, gate_model, device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total_loss / max(total_count, 1),
                **dev_metrics,
            }
        )
        if dev_metrics["top1_accuracy"] > best_dev_acc:
            best_dev_acc = dev_metrics["top1_accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in gate_model.state_dict().items()}

    if best_state is not None:
        gate_model.load_state_dict(best_state)
    write_json(output_dir / "train_history.json", history)
    write_json(output_dir / "best_dev.json", {"best_dev_top1": best_dev_acc})
    torch.save(gate_model.state_dict(), output_dir / "gate_state.pt")
    return gate_model


def _eval_accuracy_only(
    variant: str,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    task_meta: torch.Tensor,
    task_keys: torch.Tensor,
    softmax_scale: float,
    gate_model: Optional[nn.Module],
    device: torch.device,
) -> Dict[str, float]:
    with torch.no_grad():
        scores, alpha = _compute_scores(variant, embeddings.to(device), task_meta.to(device), task_keys.to(device), softmax_scale, gate_model)
        probs = F.softmax(scores, dim=-1)
        preds = probs.argmax(dim=-1)
        top1 = (preds.cpu() == labels).float().mean().item()
        top3 = (torch.topk(probs, k=min(3, probs.shape[-1]), dim=-1).indices.cpu() == labels.unsqueeze(-1)).any(dim=-1).float().mean().item()
    return {"top1_accuracy": top1, "top3_accuracy": top3}


def main():
    parser = argparse.ArgumentParser(description="Probe MDTM gate formulations without changing core project code.")
    parser.add_argument("--mode", choices=["current", "metadata_only", "temperature", "learnable_mlp"], required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--meta-embeddings-path", default="metadata_embeddings/keshi_meta_embeddings.pt")
    parser.add_argument("--dataset-root", default="datasets/medical_consult")
    parser.add_argument("--task-list", default="neike,waike,erke,fuchanke,nanke,zhongliuke")
    parser.add_argument(
        "--eval-data",
        default="experiments/reviewer_0629/data/routing_reference_eval.jsonl,experiments/reviewer_0629/data/holdout_zhongliuke_reference_eval.jsonl",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--base-model-path", default="chinese-alpaca-plus-7b-hf")
    parser.add_argument("--compute-dtype", default="float16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-samples-per-task", type=int, default=400)
    parser.add_argument("--max-dev-samples-per-task", type=int, default=100)
    parser.add_argument("--num-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--temperature-init", type=float, default=1.0)
    parser.add_argument("--mlp-hidden-size", type=int, default=32)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    args = parser.parse_args()

    output_root = ensure_dir(args.output_dir)
    cache_root = ensure_dir(args.cache_dir)
    task_list = [item.strip() for item in args.task_list.split(",") if item.strip()]

    bundle = load_memr_model(
        base_model_path=args.base_model_path,
        checkpoint_dir=args.checkpoint_dir,
        meta_embeddings_path=args.meta_embeddings_path,
        task_list=task_list,
        inference_log_dir=str(output_root / "tensorboard"),
        compute_dtype_name=args.compute_dtype,
    )
    key_encoder = bundle.model.base_model.model.key_encoder
    task_meta = key_encoder.all_meta_keys.detach().cpu()[: len(task_list)].float()
    task_keys = key_encoder.keys.detach().cpu()[: len(task_list)].float()
    softmax_scale = float(key_encoder.config.softmax_match_scale)
    query_encoder_type = getattr(key_encoder.config, "query_encoder_type", "avg_word_embed")

    train_samples = _build_samples_from_task_records(
        task_list=task_list,
        dataset_root=args.dataset_root,
        split="train",
        max_samples_per_task=args.max_train_samples_per_task,
        seed=args.seed,
    )
    dev_samples = _build_samples_from_task_records(
        task_list=task_list,
        dataset_root=args.dataset_root,
        split="dev",
        max_samples_per_task=args.max_dev_samples_per_task,
        seed=args.seed,
    )

    train_cache = _precompute_query_embeddings(
        bundle,
        train_samples,
        _cache_path(cache_root, "train"),
        query_encoder_type=query_encoder_type,
    )
    dev_cache = _precompute_query_embeddings(
        bundle,
        dev_samples,
        _cache_path(cache_root, "dev"),
        query_encoder_type=query_encoder_type,
    )

    train_embeddings = train_cache["embeddings"].float()
    train_labels = train_cache["task_ids"].long()
    dev_embeddings = dev_cache["embeddings"].float()
    dev_labels = dev_cache["task_ids"].long()

    device = train_embeddings.device if train_embeddings.is_cuda else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gate_model: Optional[nn.Module] = None
    variant_output_dir = output_root
    if args.mode in {"temperature", "learnable_mlp"}:
        gate_model = _train_gate(
            variant=args.mode,
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            dev_embeddings=dev_embeddings,
            dev_labels=dev_labels,
            task_meta=task_meta,
            task_keys=task_keys,
            softmax_scale=softmax_scale,
            device=device,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            temperature_init=args.temperature_init,
            hidden_size=args.mlp_hidden_size,
            output_dir=variant_output_dir,
        )

    eval_paths = [item.strip() for item in args.eval_data.split(",") if item.strip()]
    eval_summaries = {}
    for eval_path in eval_paths:
        eval_name = Path(eval_path).stem
        eval_samples = _build_eval_samples(eval_path)
        if args.max_eval_samples is not None:
            eval_samples = eval_samples[: args.max_eval_samples]
        eval_cache = _precompute_query_embeddings(
            bundle,
            eval_samples,
            _cache_path(cache_root, eval_name),
            query_encoder_type=query_encoder_type,
        )
        summary = _evaluate_split(
            variant=args.mode,
            eval_samples=eval_samples,
            embeddings=eval_cache["embeddings"].float(),
            task_ids=eval_cache["task_ids"].long(),
            task_meta=task_meta,
            task_keys=task_keys,
            softmax_scale=softmax_scale,
            output_dir=variant_output_dir / eval_name,
            gate_model=gate_model,
        )
        if args.mode in {"temperature", "learnable_mlp"} and gate_model is not None:
            if hasattr(gate_model, "extra_state"):
                summary["gate_extra_state"] = gate_model.extra_state()
        eval_summaries[eval_name] = summary

    write_json(
        variant_output_dir / "summary.json",
        {
            "variant": args.mode,
            "checkpoint_dir": args.checkpoint_dir,
            "meta_embeddings_path": args.meta_embeddings_path,
            "eval_data": eval_paths,
            "train_samples": len(train_samples),
            "dev_samples": len(dev_samples),
            "eval_summaries": eval_summaries,
        },
    )
    print(json.dumps(eval_summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
