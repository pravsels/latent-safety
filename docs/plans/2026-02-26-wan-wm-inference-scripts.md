# WAN World Model Inference Scripts Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create two evaluation scripts for the WAN World Model that load episodes from LeRobot datasets directly (no pre-computed HDF5 embeddings needed).

**Architecture:** Both scripts share a common setup utility (`scripts/wan_wm_utils.py`) that handles loading the WAN VAE, WM checkpoint, dataset stats, and a single LeRobot episode. Script 1 (`wan_wm_state_eval.py`) measures 50-step-ahead state prediction accuracy. Script 2 (`wan_wm_rollout.py`) generates autoregressive rollout comparison videos.

**Tech Stack:** PyTorch, diffusers (AutoencoderKLWan), lerobot, einops, imageio, matplotlib

---

## Background: Key Codebase Facts

- **WAN VAE encode**: `vae.encode(x).latent_dist.mode()` where `x` is `(B, C, 1, H, W)` float in `[-1, 1]` at `WAN_CONFIG["input_size"]` (224px). Returns `(B, C, T, latent_h, latent_w)`. Flatten to `(B, num_patches, C)` via `rearrange(z[:,:,0,:,:], "b c h w -> b (h w) c")`.
- **WAN VAE decode**: `vae.decode(z).sample` where `z` is `(B, C, T, latent_h, latent_w)`. Output is `(B, 3, T, H_pixel, W_pixel)` in `[-1, 1]`. Map to `[0,1]` via `.clamp(-1,1).add(1).mul(0.5)`.
- **VideoTransformer.forward(video1, video2, states, actions, future_actions)** returns `(pred1, pred2, state_preds, failure_preds)` — all `(B, T, num_patches, dim)` for latents, `(B, T, state_dim)` for states.
  - `video1/2`: `(B, H, num_patches, latent_dim)` — H context frames
  - `states`: `(B, H, state_dim)` — normalized
  - `actions`: `(B, H, action_dim)` — normalized, context actions
  - `future_actions`: `(B, future_len, action_dim)` — normalized, actions after context window
  - Predictions are for the **next frame after the last context frame + future_len steps ahead**. `pred_state[:, -1]` is the predicted state at `t + future_len`.
- **Normalization**: `normalize_states(states, state_min, state_max, q02, q98)` and `normalize_acs(acs, action_min, action_max, q02, action_q98)` from `dino_wm.dino_models`.
- **LeRobot episode keys**: `observation.images.front` (CHW float), `observation.images.wrist` (CHW float), `observation.state`, `action`, `timestamp`. Access via `dataset.hf_dataset[start_idx:end_idx]` with `_query_videos()` for image frames.
- **Dataset stats JSON**: keys `action_min`, `action_max`, `state_min`, `state_max`, optionally `action_delta_q02`, `action_delta_q98`, `state_q02`, `state_q98`.
- **WM checkpoint**: `torch.load(path)["model_state_dict"]` or raw state dict.
- **WAN_CONFIG**: `from dino_wm.config import WAN_CONFIG` — has `input_size` (224) and `latent_side` (28).

---

## Task 1: Shared Utility Module

**Files:**
- Create: `scripts/wan_wm_utils.py`

### Step 1: Write the module

```python
"""
Shared utilities for WAN World Model inference scripts.
Handles: VAE loading, WM loading, LeRobot episode loading, stats loading, frame encoding.
"""
import json
import os
import sys

import numpy as np
import torch
from einops import rearrange

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dino_wm.config import WAN_CONFIG
from dino_wm.dino_models import (
    VideoTransformer,
    normalize_acs,
    normalize_states,
    unnormalize_states,
)
from dino_wm.config import MODEL_CONFIG


def load_stats(stats_path: str, device: str) -> dict:
    """Load normalization stats JSON, return dict of tensors on device."""
    with open(stats_path) as f:
        data = json.load(f)
    keys = ["action_min", "action_max", "state_min", "state_max"]
    optional = ["action_delta_q02", "action_delta_q98", "state_q02", "state_q98"]
    stats = {k: torch.tensor(data[k]).float().to(device) for k in keys}
    for k in optional:
        if k in data:
            stats[k] = torch.tensor(data[k]).float().to(device)
    stats["state_dim"] = len(data["state_min"])
    stats["action_dim"] = len(data["action_min"])
    return stats


def load_wan_vae(model_id: str, subfolder: str, device: str, dtype_str: str):
    """Load WAN VAE (AutoencoderKLWan) and return (vae, device, model_dtype)."""
    from diffusers import AutoencoderKLWan
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    model_dtype = dtype_map[dtype_str] if dev.type == "cuda" else torch.float32
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder=subfolder, torch_dtype=model_dtype
    ).to(dev).eval()
    return vae, dev, model_dtype


def load_wm(checkpoint_path: str, state_dim: int, action_dim: int,
            num_patches: int, latent_dim: int, sequence_length: int,
            action_horizon: int, device: str) -> VideoTransformer:
    """Load VideoTransformer WM checkpoint."""
    MODEL_CONFIG["dim"] = latent_dim
    MODEL_CONFIG["image_size"] = (224, 224)
    model = VideoTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        num_frames=sequence_length - 1,
        action_horizon=action_horizon,
        backbone="wan",
        dino_version="v3",
        num_patches=num_patches,
        **MODEL_CONFIG,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def encode_frame(vae, frame_hwc_uint8: np.ndarray, device, model_dtype,
                 input_size: int = None) -> torch.Tensor:
    """
    Encode a single HWC uint8 frame to WAN latent.
    Returns: (1, num_patches, latent_dim) float32 tensor.
    """
    if input_size is None:
        input_size = WAN_CONFIG["input_size"]
    x = torch.from_numpy(frame_hwc_uint8).float().to(device)  # (H, W, C)
    x = x.permute(2, 0, 1).div(127.5).sub(1.0)  # (C, H, W) in [-1, 1]
    x = torch.nn.functional.interpolate(
        x.unsqueeze(0), size=(input_size, input_size), mode="bilinear", align_corners=False
    )  # (1, C, H, W)
    x = x.unsqueeze(2).to(dtype=model_dtype)  # (1, C, 1, H, W)
    z = vae.encode(x).latent_dist.mode()  # (1, C, T, latent_h, latent_w)
    z2d = z[:, :, 0, :, :]  # (1, C, latent_h, latent_w)
    return rearrange(z2d, "b c h w -> b (h w) c").float()  # (1, num_patches, latent_dim)


@torch.no_grad()
def decode_latent(vae, latent: torch.Tensor, latent_h: int, latent_w: int,
                  device, model_dtype) -> np.ndarray:
    """
    Decode a single latent to HWC uint8 image.
    latent: (1, num_patches, latent_dim)
    Returns: (H, W, 3) uint8 numpy array.
    """
    z = rearrange(latent, "b (h w) c -> b c 1 h w", h=latent_h, w=latent_w).to(
        device=device, dtype=model_dtype
    )
    y = vae.decode(z).sample  # (1, 3, 1, H, W)
    y = y.clamp(-1, 1).add(1.0).mul(127.5)
    frame = y[0, :, 0, :, :].float().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return frame


def load_lerobot_episode(dataset_id: str, episode_idx: int):
    """
    Load a single LeRobot episode. Returns (dataset, start_idx, end_idx).
    Handles v2.1 and v3 LeRobot datasets.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(dataset_id, video_backend="pyav")
    if hasattr(dataset, "episode_data_index"):
        start_idx = int(dataset.episode_data_index["from"][episode_idx].item())
        end_idx = int(dataset.episode_data_index["to"][episode_idx].item())
    elif hasattr(dataset, "meta") and hasattr(dataset.meta, "episodes"):
        ep = dataset.meta.episodes[episode_idx]
        start_idx = int(ep["dataset_from_index"])
        end_idx = int(ep["dataset_to_index"])
    else:
        raise AttributeError("Cannot find episode index in dataset")
    return dataset, start_idx, end_idx


def get_episode_frames_and_data(dataset, start_idx: int, end_idx: int, episode_idx: int):
    """
    Extract all frames, states, and actions for an episode.
    Returns dict with keys:
      front_frames: list of (H, W, 3) uint8 numpy arrays
      wrist_frames: list of (H, W, 3) uint8 numpy arrays  
      states: (T, state_dim) float32 numpy array
      actions: (T, action_dim) float32 numpy array
    """
    import torch
    batch = dataset.hf_dataset.with_format(None)[start_idx:end_idx]
    timestamps = [
        float(t.item() if hasattr(t, "item") else t)
        for t in batch["timestamp"]
    ]
    query = {k: timestamps for k in dataset.meta.video_keys}
    orig_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        video_frames = dataset._query_videos(query, episode_idx)
    finally:
        torch.set_default_dtype(orig_dtype)

    def _to_hwc_uint8(t):
        # t: (T, C, H, W) float [0, 255] or [0, 1]
        if isinstance(t, torch.Tensor):
            t = t.float()
        else:
            t = torch.as_tensor(t).float()
        if t.max() <= 1.0:
            t = t * 255.0
        t = t.clamp(0, 255).byte()
        return [t[i].permute(1, 2, 0).numpy() for i in range(t.shape[0])]

    front_frames = _to_hwc_uint8(video_frames["observation.images.front"])
    wrist_frames = _to_hwc_uint8(video_frames["observation.images.wrist"])

    # Actions and states
    action_key = next(k for k in ["action", "action.pos", "action.position"] if k in batch)
    state_key = next(k for k in ["observation.state", "observation.state.pos"] if k in batch)

    def _to_numpy(v):
        if isinstance(v, torch.Tensor):
            return v.float().numpy()
        if isinstance(v, list):
            return np.array(v, dtype=np.float32)
        return np.asarray(v, dtype=np.float32)

    return {
        "front_frames": front_frames,
        "wrist_frames": wrist_frames,
        "states": _to_numpy(batch[state_key]),
        "actions": _to_numpy(batch[action_key]),
    }
```

### Step 2: Commit

```bash
git add scripts/wan_wm_utils.py
git commit -m "feat: add shared WAN WM inference utility module"
```

---

## Task 2: Script 1 — State Evaluation (`wan_wm_state_eval.py`)

**Files:**
- Create: `scripts/wan_wm_state_eval.py`

**What it does:**
- Loads a LeRobot episode by dataset ID + episode index
- Encodes context frames on-the-fly with WAN VAE
- At each step `t` (striding by `future_action_steps`):
  - Feed WM: context latents `[t-H:t]` + context states + context actions + future actions `[t:t+future_action_steps]`
  - WM predicts state at `t + future_action_steps`
  - Compare vs `state_gt[t + future_action_steps]`
- Prints L2 state diff per evaluation step to stdout
- Saves a state diff plot to output dir

### Step 1: Write the script

```python
#!/usr/bin/env python3
"""
WAN World Model state prediction evaluation.

Evaluates how accurately the WM predicts state N steps ahead,
given context frames + a chunk of N future actions — mirroring
real-world safety classifier usage.

Usage:
    python scripts/wan_wm_state_eval.py \
        --dataset villekuosmanen/fail_bil_pick_capsules_drop_on_table \
        --episode 0 \
        --wm-checkpoint wan_wm_checkpoints/best_wm.pth \
        --dataset-stats dataset_stats.json \
        --future-action-steps 50 \
        --context-length 3 \
        --sequence-length 4 \
        --action-horizon 100 \
        --output-dir outputs/state_eval
"""
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.wan_wm_utils import (
    load_stats, load_wan_vae, load_wm,
    load_lerobot_episode, get_episode_frames_and_data,
    encode_frame, decode_latent,
)
from dino_wm.config import WAN_CONFIG
from dino_wm.dino_models import normalize_states, normalize_acs, unnormalize_states


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--wm-checkpoint", required=True)
    p.add_argument("--dataset-stats", required=True)
    p.add_argument("--wan-vae-model", default="ByteDance/Video-As-Prompt-Wan2.1-14B")
    p.add_argument("--wan-vae-subfolder", default="vae")
    p.add_argument("--wan-vae-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--future-action-steps", type=int, default=50)
    p.add_argument("--context-length", type=int, default=3)
    p.add_argument("--sequence-length", type=int, default=4)
    p.add_argument("--action-horizon", type=int, default=100)
    p.add_argument("--output-dir", default="outputs/state_eval")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    H = args.context_length
    N = args.future_action_steps
    latent_side = WAN_CONFIG["latent_side"]
    num_patches = latent_side * latent_side

    print(f"Loading stats from {args.dataset_stats}")
    stats = load_stats(args.dataset_stats, device)
    state_dim = stats["state_dim"]
    action_dim = stats["action_dim"]

    print(f"Loading WAN VAE from {args.wan_vae_model}")
    vae, vae_device, vae_dtype = load_wan_vae(
        args.wan_vae_model, args.wan_vae_subfolder, device, args.wan_vae_dtype
    )

    # Infer latent_dim by encoding a dummy frame
    dummy = np.zeros((224, 224, 3), dtype=np.uint8)
    dummy_latent = encode_frame(vae, dummy, vae_device, vae_dtype)
    latent_dim = dummy_latent.shape[-1]
    print(f"Latent: num_patches={num_patches}, dim={latent_dim}")

    print(f"Loading WM from {args.wm_checkpoint}")
    wm = load_wm(
        args.wm_checkpoint, state_dim, action_dim,
        num_patches, latent_dim, args.sequence_length,
        args.action_horizon, device
    )

    print(f"Loading episode {args.episode} from {args.dataset}")
    dataset, start_idx, end_idx = load_lerobot_episode(args.dataset, args.episode)
    ep = get_episode_frames_and_data(dataset, start_idx, end_idx, args.episode)
    T = len(ep["front_frames"])
    print(f"Episode length: {T} frames")

    # Encode all frames upfront
    print("Encoding frames with WAN VAE...")
    front_latents = []
    wrist_latents = []
    for i in range(T):
        front_latents.append(encode_frame(vae, ep["front_frames"][i], vae_device, vae_dtype))
        wrist_latents.append(encode_frame(vae, ep["wrist_frames"][i], vae_device, vae_dtype))
    # Stack: (T, 1, num_patches, latent_dim)
    front_latents = torch.cat(front_latents, dim=0).unsqueeze(0)  # (1, T, num_patches, D)
    wrist_latents = torch.cat(wrist_latents, dim=0).unsqueeze(0)

    states_raw = torch.tensor(ep["states"]).float().to(device)   # (T, state_dim)
    actions_raw = torch.tensor(ep["actions"]).float().to(device) # (T, action_dim)

    states_norm = normalize_states(
        states_raw.unsqueeze(0),  # (1, T, state_dim)
        stats["state_min"], stats["state_max"],
        q02=stats.get("state_q02"), q98=stats.get("state_q98"),
    )  # (1, T, state_dim)

    actions_norm = normalize_acs(
        actions_raw.unsqueeze(0),  # (1, T, action_dim)
        stats["action_min"], stats["action_max"],
        q02=stats.get("action_delta_q02"), q98=stats.get("action_delta_q98"),
    )  # (1, T, action_dim)

    os.makedirs(args.output_dir, exist_ok=True)

    eval_steps = []
    l2_diffs = []

    print(f"\n{'Step':>6}  {'GT State':>40}  {'Pred State':>40}  {'L2 Diff':>10}")
    print("-" * 100)

    with torch.no_grad():
        # Stride: context starts at t, future actions [t:t+N], predict state at t+N
        # We need t-H:t for context and t:t+N for future, and t+N as target
        # So t ranges from H to T-N-1
        t = H
        while t + N < T:
            ctx_front = front_latents[:, t - H:t]    # (1, H, num_patches, D)
            ctx_wrist = wrist_latents[:, t - H:t]
            ctx_states = states_norm[:, t - H:t]      # (1, H, state_dim)
            ctx_actions = actions_norm[:, t - H:t]    # (1, H, action_dim)
            future_actions = actions_norm[:, t:t + N] # (1, N, action_dim)

            _, _, state_preds, _ = wm(ctx_front, ctx_wrist, ctx_states, ctx_actions, future_actions)
            pred_state_norm = state_preds[:, -1]  # (1, state_dim) — prediction for t+N

            pred_state = unnormalize_states(
                pred_state_norm,
                stats["state_min"], stats["state_max"],
                q02=stats.get("state_q02"), q98=stats.get("state_q98"),
            ).squeeze(0)  # (state_dim,)

            gt_state = states_raw[t + N]  # (state_dim,)
            l2 = (pred_state - gt_state).norm().item()

            print(f"{t:>6}  {str(gt_state.cpu().numpy().round(3)):>40}  "
                  f"{str(pred_state.cpu().numpy().round(3)):>40}  {l2:>10.4f}")

            eval_steps.append(t)
            l2_diffs.append(l2)
            t += N

    # Save plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(eval_steps, l2_diffs, marker="o")
    ax.set_xlabel(f"Context start step (predicting {N} steps ahead)")
    ax.set_ylabel("L2 state diff")
    ax.set_title(f"WM {N}-step state prediction error — {args.dataset} ep{args.episode}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(args.output_dir, f"state_diff_ep{args.episode}.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nMean L2: {np.mean(l2_diffs):.4f}  |  Max L2: {np.max(l2_diffs):.4f}")
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
```

### Step 2: Smoke test (dry run check)

```bash
python scripts/wan_wm_state_eval.py --help
```
Expected: prints usage without errors.

### Step 3: Commit

```bash
git add scripts/wan_wm_state_eval.py
git commit -m "feat: add WAN WM state prediction evaluation script"
```

---

## Task 3: Script 2 — Autoregressive Rollout Video (`wan_wm_rollout.py`)

**Files:**
- Create: `scripts/wan_wm_rollout.py`

**What it does:**
- Loads a LeRobot episode
- Encodes context frames with WAN VAE
- Autoregressively rolls forward: at each step, feeds predicted latents as new context (rolling window), GT actions 1-at-a-time, 0 future actions
- Decodes predicted latents → pixel frames
- Saves comparison video: GT (top row: front | wrist) vs Predicted (bottom row: front | wrist)

### Step 1: Write the script

```python
#!/usr/bin/env python3
"""
WAN World Model autoregressive rollout video.

Runs the WM autoregressively from a context window, feeding predicted
latents back as input at each step. Produces a side-by-side GT vs
predicted video for visual quality assessment.

Usage:
    python scripts/wan_wm_rollout.py \
        --dataset villekuosmanen/fail_bil_pick_capsules_drop_on_table \
        --episode 0 \
        --wm-checkpoint wan_wm_checkpoints/best_wm.pth \
        --dataset-stats dataset_stats.json \
        --context-length 3 \
        --horizon 50 \
        --output-dir outputs/rollout
"""
import argparse
import os
import sys

import numpy as np
import torch
import imageio.v3 as iio

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.wan_wm_utils import (
    load_stats, load_wan_vae, load_wm,
    load_lerobot_episode, get_episode_frames_and_data,
    encode_frame, decode_latent,
)
from dino_wm.config import WAN_CONFIG
from dino_wm.dino_models import normalize_states, normalize_acs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--wm-checkpoint", required=True)
    p.add_argument("--dataset-stats", required=True)
    p.add_argument("--wan-vae-model", default="ByteDance/Video-As-Prompt-Wan2.1-14B")
    p.add_argument("--wan-vae-subfolder", default="vae")
    p.add_argument("--wan-vae-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--context-length", type=int, default=3)
    p.add_argument("--sequence-length", type=int, default=4)
    p.add_argument("--action-horizon", type=int, default=100)
    p.add_argument("--horizon", type=int, default=50,
                   help="Number of autoregressive rollout steps")
    p.add_argument("--render-size", type=int, default=224,
                   help="Output frame size (square)")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--output-dir", default="outputs/rollout")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def resize_frame(frame_hwc: np.ndarray, size: int) -> np.ndarray:
    """Resize HWC uint8 frame to (size, size, 3)."""
    from PIL import Image
    return np.array(Image.fromarray(frame_hwc).resize((size, size), Image.LANCZOS))


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    H = args.context_length
    latent_side = WAN_CONFIG["latent_side"]
    num_patches = latent_side * latent_side
    render_size = args.render_size

    print(f"Loading stats from {args.dataset_stats}")
    stats = load_stats(args.dataset_stats, device)
    state_dim = stats["state_dim"]
    action_dim = stats["action_dim"]

    print(f"Loading WAN VAE from {args.wan_vae_model}")
    vae, vae_device, vae_dtype = load_wan_vae(
        args.wan_vae_model, args.wan_vae_subfolder, device, args.wan_vae_dtype
    )

    dummy = np.zeros((224, 224, 3), dtype=np.uint8)
    dummy_latent = encode_frame(vae, dummy, vae_device, vae_dtype)
    latent_dim = dummy_latent.shape[-1]
    print(f"Latent: num_patches={num_patches}, dim={latent_dim}")

    print(f"Loading WM from {args.wm_checkpoint}")
    wm = load_wm(
        args.wm_checkpoint, state_dim, action_dim,
        num_patches, latent_dim, args.sequence_length,
        args.action_horizon, device
    )

    print(f"Loading episode {args.episode} from {args.dataset}")
    dataset, start_idx, end_idx = load_lerobot_episode(args.dataset, args.episode)
    ep = get_episode_frames_and_data(dataset, start_idx, end_idx, args.episode)
    T = len(ep["front_frames"])
    total_needed = H + args.horizon
    if T < total_needed:
        raise ValueError(f"Episode too short: {T} frames, need {total_needed}")

    print("Encoding context frames with WAN VAE...")
    front_latents = []
    wrist_latents = []
    for i in range(H):
        front_latents.append(encode_frame(vae, ep["front_frames"][i], vae_device, vae_dtype))
        wrist_latents.append(encode_frame(vae, ep["wrist_frames"][i], vae_device, vae_dtype))

    # (1, H, num_patches, latent_dim)
    ctx_front = torch.cat(front_latents, dim=0).unsqueeze(0)
    ctx_wrist = torch.cat(wrist_latents, dim=0).unsqueeze(0)

    states_raw = torch.tensor(ep["states"]).float().to(device)
    actions_raw = torch.tensor(ep["actions"]).float().to(device)

    states_norm = normalize_states(
        states_raw.unsqueeze(0),
        stats["state_min"], stats["state_max"],
        q02=stats.get("state_q02"), q98=stats.get("state_q98"),
    )
    actions_norm = normalize_acs(
        actions_raw.unsqueeze(0),
        stats["action_min"], stats["action_max"],
        q02=stats.get("action_delta_q02"), q98=stats.get("action_delta_q98"),
    )

    ctx_states = states_norm[:, :H]    # (1, H, state_dim)
    ctx_actions = actions_norm[:, :H]  # (1, H, action_dim)

    # Collect GT frames (all at once)
    gt_front_frames = [resize_frame(ep["front_frames"][i], render_size) for i in range(total_needed)]
    gt_wrist_frames = [resize_frame(ep["wrist_frames"][i], render_size) for i in range(total_needed)]

    # Context predicted frames = GT (no prediction for context)
    pred_front_frames = [resize_frame(ep["front_frames"][i], render_size) for i in range(H)]
    pred_wrist_frames = [resize_frame(ep["wrist_frames"][i], render_size) for i in range(H)]

    print(f"Running autoregressive rollout for {args.horizon} steps...")
    with torch.no_grad():
        for k in range(args.horizon):
            t = H + k
            pred1, pred2, pred_state, _ = wm(ctx_front, ctx_wrist, ctx_states, ctx_actions)

            # Decode predicted latents
            pred_front_latent = pred1[:, -1].unsqueeze(1)  # (1, 1, num_patches, D)
            pred_wrist_latent = pred2[:, -1].unsqueeze(1)

            front_frame = decode_latent(
                vae, pred1[:, -1], latent_side, latent_side, vae_device, vae_dtype
            )
            wrist_frame = decode_latent(
                vae, pred2[:, -1], latent_side, latent_side, vae_device, vae_dtype
            )
            pred_front_frames.append(resize_frame(front_frame, render_size))
            pred_wrist_frames.append(resize_frame(wrist_frame, render_size))

            # Roll context window: drop oldest, add predicted
            ctx_front = torch.cat([ctx_front[:, 1:], pred1[:, -1:].unsqueeze(1)
                                    if pred1[:, -1].dim() == 2 else pred1[:, -1:]], dim=1)
            ctx_wrist = torch.cat([ctx_wrist[:, 1:], pred2[:, -1:]], dim=1)
            ctx_states = torch.cat([ctx_states[:, 1:], pred_state[:, -1:]], dim=1)
            if t < actions_norm.shape[1]:
                next_action = actions_norm[:, t:t + 1]
            else:
                next_action = actions_norm[:, -1:]
            ctx_actions = torch.cat([ctx_actions[:, 1:], next_action], dim=1)

    # Build comparison video
    sep = np.ones((render_size, 4, 3), dtype=np.uint8) * 200  # vertical separator
    h_sep = np.ones((4, render_size * 2 + 4, 3), dtype=np.uint8) * 200  # horizontal separator

    frames = []
    for i in range(total_needed):
        gt_row = np.concatenate([gt_front_frames[i], sep, gt_wrist_frames[i]], axis=1)
        pred_row = np.concatenate([pred_front_frames[i], sep, pred_wrist_frames[i]], axis=1)
        frame = np.concatenate([gt_row, h_sep, pred_row], axis=0)
        frames.append(frame)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"rollout_ep{args.episode}.mp4")
    iio.imwrite(out_path, np.stack(frames), fps=args.fps, codec="libx264", pixelformat="yuv420p")
    print(f"Saved: {out_path}")
    print(f"Layout: [GT Front | GT Wrist] (top) / [Pred Front | Pred Wrist] (bottom)")
    print(f"Context frames (0..{H-1}) show GT in both rows.")


if __name__ == "__main__":
    main()
```

### Step 2: Smoke test

```bash
python scripts/wan_wm_rollout.py --help
```
Expected: prints usage without errors.

### Step 3: Commit

```bash
git add scripts/wan_wm_rollout.py
git commit -m "feat: add WAN WM autoregressive rollout video script"
```

---

## Usage Examples

### State eval (50-step-ahead prediction):
```bash
python scripts/wan_wm_state_eval.py \
  --dataset villekuosmanen/fail_bil_pick_capsules_drop_on_table \
  --episode 0 \
  --wm-checkpoint wan_wm_checkpoints/best_wm.pth \
  --dataset-stats train_data/dataset_stats.json \
  --future-action-steps 50 \
  --context-length 3
```

### Autoregressive rollout video:
```bash
python scripts/wan_wm_rollout.py \
  --dataset villekuosmanen/fail_bil_pick_capsules_drop_on_table \
  --episode 0 \
  --wm-checkpoint wan_wm_checkpoints/best_wm.pth \
  --dataset-stats train_data/dataset_stats.json \
  --context-length 3 \
  --horizon 50
```

---

## Notes

- Both scripts require the WAN VAE to be accessible (local path or HuggingFace hub). If cached locally, pass the local path to `--wan-vae-model`.
- `unnormalize_states` is imported from `dino_wm.dino_models` — verify it exists before running (grep for it; if missing, implement as inverse of `normalize_states`).
- The `ctx_front` rolling window concatenation in Script 2 needs careful shape handling — `pred1[:, -1]` is `(1, num_patches, D)`, needs `.unsqueeze(1)` to be `(1, 1, num_patches, D)` before cat.
- LeRobot dataset loading requires `lerobot` and optionally `robocandywrapper` packages to be installed in the environment.
