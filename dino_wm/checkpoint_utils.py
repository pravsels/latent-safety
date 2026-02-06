from __future__ import annotations

from typing import Iterable, Mapping, Tuple

import torch


def filter_state_dict_by_shape(
    model_state: Mapping[str, torch.Tensor],
    ckpt_state: Mapping[str, torch.Tensor],
) -> Tuple[dict[str, torch.Tensor], list[str], list[str], list[str]]:
    """Filter checkpoint weights to those matching model keys and shapes.

    Returns:
        filtered_state: key -> tensor (shape matches model)
        missing_keys: keys in model not found in checkpoint
        unexpected_keys: keys in checkpoint not found in model
        mismatched_keys: keys in checkpoint found in model but with shape mismatch
    """
    filtered_state: dict[str, torch.Tensor] = {}
    missing_keys: list[str] = []
    unexpected_keys: list[str] = []
    mismatched_keys: list[str] = []

    model_keys = set(model_state.keys())
    ckpt_keys = set(ckpt_state.keys())

    for key in sorted(model_keys):
        if key not in ckpt_state:
            missing_keys.append(key)
            continue
        if model_state[key].shape != ckpt_state[key].shape:
            mismatched_keys.append(key)
            missing_keys.append(key)
            continue
        filtered_state[key] = ckpt_state[key]

    for key in sorted(ckpt_keys - model_keys):
        unexpected_keys.append(key)

    return filtered_state, missing_keys, unexpected_keys, mismatched_keys
