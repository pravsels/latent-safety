#!/usr/bin/env python3
"""
Shared utility functions for WAN World Model inference scripts.
"""

import json
import os
import sys
from typing import Tuple, Dict, Any

import numpy as np
import torch
from einops import rearrange

# Ensure this repo's dino_wm takes precedence over any installed version
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dino_wm.config import WAN_CONFIG, MODEL_CONFIG
from dino_wm.dino_models import VideoTransformer, normalize_acs, normalize_states, unnormalize_states


def load_stats(stats_path: str, device) -> dict:
    """
    Load normalization stats from a JSON file.

    Expected keys: action_min, action_max, state_min, state_max
    Optional keys: action_delta_q02, action_delta_q98, state_q02, state_q98

    Returns a dict of tensors on `device` plus `state_dim` and `action_dim` as ints.
    """
    with open(stats_path, "r") as f:
        raw = json.load(f)

    stats = {}
    for key, val in raw.items():
        stats[key] = torch.tensor(val, dtype=torch.float32, device=device)

    stats["action_dim"] = int(stats["action_min"].shape[-1])
    stats["state_dim"] = int(stats["state_min"].shape[-1])
    return stats


def load_wan_vae(model_id: str, subfolder: str, device: str, dtype_str: str):
    """
    Load AutoencoderKLWan from diffusers.

    dtype_str: one of "bf16", "fp16", "fp32".
    If device is "cuda" but CUDA is unavailable, falls back to CPU + fp32.

    Returns (vae, dev, model_dtype).
    """
    from diffusers import AutoencoderKLWan

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if device == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")
        model_dtype = torch.float32
    else:
        dev = torch.device(device)
        model_dtype = dtype_map[dtype_str]

    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder=subfolder,
        torch_dtype=model_dtype,
    ).to(dev).eval()

    return vae, dev, model_dtype


def load_wm(
    checkpoint_path: str,
    state_dim: int,
    action_dim: int,
    num_patches: int,
    latent_dim: int,
    sequence_length: int,
    action_horizon: int,
    device,
) -> VideoTransformer:
    """
    Load a VideoTransformer world model from a checkpoint.

    Sets MODEL_CONFIG["dim"] = latent_dim before constructing the model.
    """
    MODEL_CONFIG["dim"] = latent_dim

    model = VideoTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        num_frames=sequence_length - 1,
        action_horizon=action_horizon,
        backbone="wan",
        dino_version="v3",
        num_patches=num_patches,
        **MODEL_CONFIG,
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def encode_frame(
    vae,
    frame_hwc_uint8: np.ndarray,
    device,
    model_dtype: torch.dtype,
    input_size: int = None,
) -> torch.Tensor:
    """
    Encode a single HWC uint8 numpy frame to a (1, num_patches, latent_dim) float32 tensor.

    Steps:
      - Normalize to [-1, 1]
      - Interpolate to input_size x input_size
      - Add batch+time dims -> (1, C, 1, H, W)
      - Encode with VAE -> (1, C, T, latent_h, latent_w)
      - Take [:, :, 0, :, :] and rearrange "b c h w -> b (h w) c"
      - Return as float32
    """
    if input_size is None:
        input_size = WAN_CONFIG["input_size"]

    x = torch.from_numpy(frame_hwc_uint8).to(device=device, dtype=torch.float32)
    x = x.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    x = x.div(127.5).sub(1.0)
    x = torch.nn.functional.interpolate(
        x, size=(input_size, input_size), mode="bilinear", align_corners=False
    )
    x = x.unsqueeze(2).to(dtype=model_dtype)  # (1, C, 1, H, W)

    z = vae.encode(x).latent_dist.mode()  # (1, C, T, latent_h, latent_w)
    z2d = z[:, :, 0, :, :]               # (1, C, latent_h, latent_w)
    patches = rearrange(z2d, "b c h w -> b (h w) c")  # (1, num_patches, latent_dim)
    return patches.float()


@torch.no_grad()
def decode_latent(
    vae,
    latent: torch.Tensor,
    latent_h: int,
    latent_w: int,
    device,
    model_dtype: torch.dtype,
) -> np.ndarray:
    """
    Decode a (1, num_patches, latent_dim) latent to an HWC uint8 numpy array.

    Steps:
      - Rearrange "b (h w) c -> b c 1 h w"
      - Decode with VAE -> (1, 3, 1, H, W) in [-1, 1]
      - Clamp, add 1, mul 127.5
      - Take [0, :, 0], permute to HWC, convert to uint8
    """
    z = rearrange(latent.to(device=device, dtype=model_dtype), "b (h w) c -> b c 1 h w", h=latent_h, w=latent_w)
    out = vae.decode(z).sample  # (1, 3, 1, H, W)
    out = out.clamp(-1.0, 1.0).add(1.0).mul(127.5)
    frame = out[0, :, 0].permute(1, 2, 0)  # (H, W, 3)
    return frame.cpu().to(torch.uint8).numpy()


def load_lerobot_episode(dataset_id: str, episode_idx: int):
    """
    Load a LeRobot dataset and return (dataset, start_idx, end_idx) for the given episode.

    Handles v2.1 (episode_data_index) and v3 (meta.episodes) APIs.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(dataset_id, video_backend="pyav")

    start_idx, end_idx = _get_episode_range(dataset, episode_idx)
    return dataset, start_idx, end_idx


def _get_episode_range(dataset, ep_idx: int) -> Tuple[int, int]:
    """Return (start_idx, end_idx) for an episode, handling v2.1 and v3 APIs."""
    if hasattr(dataset, "episode_data_index"):
        start_idx = dataset.episode_data_index["from"][ep_idx].item()
        end_idx = dataset.episode_data_index["to"][ep_idx].item()
        return int(start_idx), int(end_idx)

    if hasattr(dataset, "meta") and hasattr(dataset.meta, "episodes"):
        episodes_meta = dataset.meta.episodes
        if isinstance(episodes_meta, dict):
            ep_info = episodes_meta.get(ep_idx)
        else:
            if ep_idx >= len(episodes_meta):
                raise IndexError(f"Episode index {ep_idx} out of range")
            ep_info = episodes_meta[ep_idx]
        if not ep_info:
            raise KeyError(f"Episode metadata missing for index {ep_idx}")
        start_idx = ep_info.get("dataset_from_index")
        end_idx = ep_info.get("dataset_to_index")
        if start_idx is None or end_idx is None:
            raise KeyError("Episode metadata missing dataset_from_index/dataset_to_index")
        return int(start_idx), int(end_idx)

    raise AttributeError("Dataset missing episode index metadata")


def get_episode_frames_and_data(dataset, start_idx: int, end_idx: int, episode_idx: int) -> dict:
    """
    Extract all frames, states, and actions for an episode.

    Returns a dict with:
      - front_frames: list of (H, W, 3) uint8 numpy arrays
      - wrist_frames: list of (H, W, 3) uint8 numpy arrays
      - states: (T, state_dim) float32 numpy array
      - actions: (T, action_dim) float32 numpy array
    """
    batch = dataset.hf_dataset.with_format(None)[start_idx:end_idx]

    # Build timestamps for video query
    ts_raw = batch["timestamp"]
    batch_timestamps = [
        float(t.item() if hasattr(t, "item") else t) for t in ts_raw
    ]

    query = {k: batch_timestamps for k in dataset.meta.video_keys}

    orig_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        video_frames = dataset._query_videos(query, episode_idx)
    finally:
        torch.set_default_dtype(orig_dtype)

    def _to_hwc_uint8_list(tensor: torch.Tensor):
        """Convert (T, C, H, W) float tensor to list of (H, W, C) uint8 arrays."""
        if tensor.max() > 1.0:
            arr = tensor.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        else:
            arr = (tensor.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
        return [arr[i] for i in range(arr.shape[0])]

    # Front frames
    if "observation.images.front" in video_frames:
        front_frames = _to_hwc_uint8_list(video_frames["observation.images.front"])
    else:
        raise ValueError("Missing observation.images.front in video frames")

    # Wrist frames
    if "observation.images.wrist" in video_frames:
        wrist_frames = _to_hwc_uint8_list(video_frames["observation.images.wrist"])
    else:
        raise ValueError("Missing observation.images.wrist in video frames")

    # States
    state_key = None
    for candidate in ["observation.state", "observation.state.pos", "observation.state.eef_pose"]:
        if candidate in batch:
            state_key = candidate
            break
    if state_key is None:
        raise KeyError("Missing state key in batch")
    states_raw = batch[state_key]
    if isinstance(states_raw, torch.Tensor):
        states = states_raw.float().numpy()
    elif isinstance(states_raw, list):
        states = np.array(states_raw, dtype=np.float32)
    else:
        states = np.asarray(states_raw, dtype=np.float32)

    # Actions
    action_key = None
    for candidate in ["action", "action.pos", "action.position", "action.eef_pose"]:
        if candidate in batch:
            action_key = candidate
            break
    if action_key is None:
        raise KeyError("Missing action key in batch")
    actions_raw = batch[action_key]
    if isinstance(actions_raw, torch.Tensor):
        actions = actions_raw.float().numpy()
    elif isinstance(actions_raw, list):
        actions = np.array(actions_raw, dtype=np.float32)
    else:
        actions = np.asarray(actions_raw, dtype=np.float32)

    return {
        "front_frames": front_frames,
        "wrist_frames": wrist_frames,
        "states": states,
        "actions": actions,
    }
