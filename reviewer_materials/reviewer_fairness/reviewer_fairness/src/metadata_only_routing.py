"""
Metadata-only routing utilities for reviewer fairness experiments.

This module wraps existing MeMR components without changing the original
MeMR implementation. It overrides TaskKeyEncoder.forward at runtime so that
task weights are computed only from query-to-metadata cosine similarity.
"""

from __future__ import annotations

from types import MethodType
from typing import Tuple

import torch
import torch.nn.functional as F


def _metadata_only_forward(self, x_query, adapter_name, train=True, final=False):
    task_id = self.task_list.index(adapter_name)
    if train and not final:
        self.steps[task_id] += 1
    elif not train and not final:
        self.steps_val[task_id] += 1
    else:
        self.steps_final[task_id] += 1

    target_device = x_query.device
    target_dtype = x_query.dtype
    k_meta = self.all_meta_keys[: task_id + 1].to(device=target_device, dtype=target_dtype)
    norm_query = F.normalize(x_query, dim=-1).unsqueeze(1)
    norm_meta = F.normalize(k_meta, dim=-1).unsqueeze(0)
    cos_sim = torch.sum(norm_query * norm_meta, dim=-1)

    for idx in range(cos_sim.shape[-1]):
        if idx < len(self.keys_mask) and self.keys_mask[idx]:
            cos_sim = cos_sim.clone()
            cos_sim[:, idx] = -1e8

    weights = F.softmax(cos_sim * self.config.softmax_match_scale, dim=-1)
    loss = torch.tensor(0.0, device=x_query.device)
    self._log_weights(weights, task_id, train, final)
    return weights, loss


def enable_metadata_only_routing(model) -> Tuple[bool, str]:
    key_encoder = getattr(model, "key_encoder", None)
    if key_encoder is None and hasattr(model, "base_model"):
        key_encoder = getattr(model.base_model, "key_encoder", None)
    if key_encoder is None and hasattr(model, "model"):
        key_encoder = getattr(model.model, "key_encoder", None)
    if key_encoder is None:
        return False, "TaskKeyEncoder was not found; metadata-only routing was not enabled."

    key_encoder.forward = MethodType(_metadata_only_forward, key_encoder)
    if hasattr(key_encoder, "keys"):
        key_encoder.keys.requires_grad_(False)
    return True, "Metadata-only routing enabled by runtime wrapper."
