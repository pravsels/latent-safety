#!/usr/bin/env python3
"""
Evaluate WAN world model quality vs future-action horizon (K=1..max_horizon).

For each horizon K:
  - Feed H context WAN latents + K future actions to WM
  - Compare predicted latent/state at target step against ground truth
  - Record latent MSE (front+wrist average) and state MSE

This mirrors scripts/eval_horizon_sweep.py but uses WAN latents produced on-the-fly
from raw LeRobot episode frames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

DEFAULT_DATASET_STATS = "arx5_datasets_6Feb_26_stats.json"

from dino_wm.config import WAN_CONFIG
from dino_wm.dino_models import normalize_acs, normalize_states
from scripts.wan_wm_utils import (
    encode_frame,
    load_lerobot_episode,
    load_stats,
    load_wan_vae,
    load_wm,
)


def load_episode_wan_latents(
    repo_id: str,
    episode: int,
    *,
    vae: Any,
    vae_device: torch.device,
    vae_dtype: torch.dtype,
    device: str,
    video_query_batch: int = 32,
) -> dict[str, torch.Tensor]:
    """Load one episode and encode front/wrist frames into WAN latents."""
    dataset, start_idx, end_idx = load_lerobot_episode(repo_id, episode)
    batch = dataset.hf_dataset.with_format(None)[start_idx:end_idx]
    T = len(batch["timestamp"])
    if T == 0:
        raise ValueError(f"Episode {episode} from {repo_id} has no frames.")
    if video_query_batch <= 0:
        raise ValueError("--video-query-batch must be >= 1.")

    # Reuse the same key fallback logic used across WAN eval scripts.
    state_key = None
    for candidate in ["observation.state", "observation.state.pos", "observation.state.eef_pose"]:
        if candidate in batch:
            state_key = candidate
            break
    if state_key is None:
        raise KeyError("Missing state key in episode batch.")

    action_key = None
    for candidate in ["action", "action.pos", "action.position", "action.eef_pose"]:
        if candidate in batch:
            action_key = candidate
            break
    if action_key is None:
        raise KeyError("Missing action key in episode batch.")

    states_raw = batch[state_key]
    if isinstance(states_raw, torch.Tensor):
        states_np = states_raw.float().numpy()
    elif isinstance(states_raw, list):
        states_np = np.array(states_raw, dtype=np.float32)
    else:
        states_np = np.asarray(states_raw, dtype=np.float32)

    actions_raw = batch[action_key]
    if isinstance(actions_raw, torch.Tensor):
        actions_np = actions_raw.float().numpy()
    elif isinstance(actions_raw, list):
        actions_np = np.array(actions_raw, dtype=np.float32)
    else:
        actions_np = np.asarray(actions_raw, dtype=np.float32)

    batch_timestamps = [float(t.item() if hasattr(t, "item") else t) for t in batch["timestamp"]]

    def _to_hwc_uint8_list(tensor: torch.Tensor) -> list[np.ndarray]:
        if tensor.max() > 1.0:
            arr = tensor.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        else:
            arr = (tensor.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
        return [arr[i] for i in range(arr.shape[0])]

    front_latents = []
    wrist_latents = []

    for i in range(0, T, video_query_batch):
        j = min(i + video_query_batch, T)
        chunk_ts = batch_timestamps[i:j]
        query = {k: chunk_ts for k in dataset.meta.video_keys}

        orig_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            video_chunk = dataset._query_videos(query, episode)
        finally:
            torch.set_default_dtype(orig_dtype)

        front_frames = _to_hwc_uint8_list(video_chunk["observation.images.front"])
        wrist_frames = _to_hwc_uint8_list(video_chunk["observation.images.wrist"])
        for k in range(len(front_frames)):
            front_latents.append(
                encode_frame(vae, front_frames[k], vae_device, vae_dtype).squeeze(0).cpu()
            )
            wrist_latents.append(
                encode_frame(vae, wrist_frames[k], vae_device, vae_dtype).squeeze(0).cpu()
            )
        if j % 50 == 0 or j == T:
            print(f"  WAN-encoded frames {j}/{T}")

        del video_chunk, front_frames, wrist_frames
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "front_latent": torch.stack(front_latents, dim=0),  # (T, P, D) on CPU
        "wrist_latent": torch.stack(wrist_latents, dim=0),  # (T, P, D) on CPU
        "actions": torch.tensor(actions_np, dtype=torch.float32, device=device),  # (T, A)
        "states": torch.tensor(states_np, dtype=torch.float32, device=device),  # (T, S)
    }


@torch.no_grad()
def evaluate_horizon_wan(
    transition,
    episode_data: dict[str, torch.Tensor],
    stats: dict[str, torch.Tensor],
    *,
    context_length: int,
    pred_step: int,
    max_horizon: int,
    stride: int,
    device: str,
    no_future_actions: bool = False,
) -> dict[str, np.ndarray]:
    """Sweep K=1..max_horizon and return mean/std latent/state MSE curves."""
    front_cpu = episode_data["front_latent"]  # (T, P, D) on CPU
    wrist_cpu = episode_data["wrist_latent"]  # (T, P, D) on CPU
    actions = episode_data["actions"]  # (T, A) on device
    states = episode_data["states"]  # (T, S) on device
    T = int(front_cpu.shape[0])

    action_q02 = stats.get("action_delta_q02")
    action_q98 = stats.get("action_delta_q98")
    state_q02 = stats.get("state_q02")
    state_q98 = stats.get("state_q98")

    norm_actions = normalize_acs(
        actions,
        stats["action_min"],
        stats["action_max"],
        q02=action_q02,
        q98=action_q98,
    )
    norm_states = normalize_states(
        states,
        stats["state_min"],
        stats["state_max"],
        q02=state_q02,
        q98=state_q98,
    )

    # Context indices are spaced in raw-frame space by pred_step.
    ctx_idx_dev = torch.arange(context_length, device=device, dtype=torch.long) * pred_step
    ctx_idx_cpu = ctx_idx_dev.cpu()
    t = int(ctx_idx_dev[-1].item())
    action_dim = int(norm_actions.shape[-1])

    transition.eval()
    latent_mses: list[list[float]] = []
    state_mses: list[list[float]] = []

    for K in range(1, max_horizon + 1):
        target_idx = t + K + 1
        min_len = target_idx + 1
        k_latent: list[float] = []
        k_state: list[float] = []

        for offset in range(0, T - min_len + 1, stride):
            in_front = front_cpu[offset + ctx_idx_cpu].unsqueeze(0).to(device)  # (1,H,P,D)
            in_wrist = wrist_cpu[offset + ctx_idx_cpu].unsqueeze(0).to(device)  # (1,H,P,D)
            in_state = norm_states[offset + ctx_idx_dev].unsqueeze(0)  # (1,H,S)
            in_acs = norm_actions[offset + ctx_idx_dev].unsqueeze(0)  # (1,H,A)

            fa_start = offset + t + 1
            fa_end = offset + t + 1 + K
            if no_future_actions:
                future_actions = torch.zeros((1, K, action_dim), device=device, dtype=in_acs.dtype)
            else:
                future_actions = norm_actions[fa_start:fa_end].unsqueeze(0)  # (1,K,A)

            pred_front, pred_wrist, pred_state, _ = transition(
                in_front,
                in_wrist,
                in_state,
                in_acs,
                future_actions,
            )

            gt_front = front_cpu[offset + target_idx].to(device)
            gt_wrist = wrist_cpu[offset + target_idx].to(device)
            gt_state = norm_states[offset + target_idx]

            mse_front = torch.mean((pred_front[0, -1] - gt_front) ** 2).item()
            mse_wrist = torch.mean((pred_wrist[0, -1] - gt_wrist) ** 2).item()
            mse_state = torch.mean((pred_state[0, -1] - gt_state) ** 2).item()

            k_latent.append((mse_front + mse_wrist) / 2.0)
            k_state.append(mse_state)

        latent_mses.append(k_latent)
        state_mses.append(k_state)
        if K % 10 == 0 or K == 1:
            n = len(k_latent)
            if n:
                print(f"  K={K:3d}  windows={n}  latent_mse={np.mean(k_latent):.6f}")
            else:
                print(f"  K={K:3d}  no valid windows")

    horizons = np.arange(1, max_horizon + 1)
    return {
        "horizons": horizons,
        "latent_mse_mean": np.array([np.mean(v) if v else np.nan for v in latent_mses]),
        "latent_mse_std": np.array([np.std(v) if v else np.nan for v in latent_mses]),
        "state_mse_mean": np.array([np.mean(v) if v else np.nan for v in state_mses]),
        "state_mse_std": np.array([np.std(v) if v else np.nan for v in state_mses]),
        "num_windows": [len(v) for v in latent_mses],
    }


def _annotate_endpoints(ax, horizons: np.ndarray, values: np.ndarray, color: str) -> None:
    start = float(values[0])
    end = float(values[-1])
    ax.annotate(
        f"{start:.4f}",
        xy=(horizons[0], start),
        xytext=(horizons[0] + 3, start),
        fontsize=10,
        fontweight="bold",
        color=color,
        arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
        va="center",
    )
    ax.annotate(
        f"{end:.4f}",
        xy=(horizons[-1], end),
        xytext=(horizons[-1] - 3, end),
        fontsize=10,
        fontweight="bold",
        color=color,
        ha="right",
        va="center",
        arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
    )


def plot_sweep(
    results: dict[str, np.ndarray],
    out_dir: Path,
    *,
    title_suffix: str = "",
    tag: str = "",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    h = results["horizons"]

    fig, ax = plt.subplots(figsize=(10, 5))
    mean = results["latent_mse_mean"]
    std = results["latent_mse_std"]
    ax.plot(h, mean, color="tab:blue", linewidth=1.5)
    ax.fill_between(h, mean - std, mean + std, alpha=0.25, color="tab:blue")
    _annotate_endpoints(ax, h, mean, "tab:blue")
    ax.set_xlabel("Future Action Horizon (K)")
    ax.set_ylabel("Latent MSE (WAN latent space)")
    ax.set_title(f"WAN Latent MSE vs Future Action Horizon{title_suffix}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"latent_mse_vs_horizon{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    fig, ax = plt.subplots(figsize=(10, 5))
    mean = results["state_mse_mean"]
    std = results["state_mse_std"]
    ax.plot(h, mean, color="tab:orange", linewidth=1.5)
    ax.fill_between(h, mean - std, mean + std, alpha=0.25, color="tab:orange")
    _annotate_endpoints(ax, h, mean, "tab:orange")
    ax.set_xlabel("Future Action Horizon (K)")
    ax.set_ylabel("State MSE (normalized)")
    ax.set_title(f"WAN State MSE vs Future Action Horizon{title_suffix}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"state_mse_vs_horizon{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate WAN WM across future-action horizons")
    p.add_argument("--hf-repo", required=True, help="LeRobot dataset repo id")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--wm-checkpoint", required=True)
    p.add_argument("--dataset-stats", default=DEFAULT_DATASET_STATS)
    p.add_argument("--context-length", type=int, default=3)
    p.add_argument("--pred-step", type=int, default=5)
    p.add_argument("--max-horizon", type=int, default=100)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--no-future-actions", action="store_true")
    p.add_argument("--wan-vae-model", type=str, default="ByteDance/Video-As-Prompt-Wan2.1-14B")
    p.add_argument("--wan-vae-subfolder", type=str, default="vae")
    p.add_argument("--wan-vae-dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--sequence-length", type=int, default=4)
    p.add_argument("--action-horizon", type=int, default=100)
    p.add_argument("--video-query-batch", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out-dir", type=str, default="outputs/horizon_sweep")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"Loading dataset stats from {args.dataset_stats}")
    stats = load_stats(args.dataset_stats, device)
    state_dim = int(stats["state_dim"])
    action_dim = int(stats["action_dim"])
    latent_side = int(WAN_CONFIG["latent_side"])
    num_patches = latent_side * latent_side

    print("Loading WAN VAE ...")
    vae, vae_device, vae_dtype = load_wan_vae(
        args.wan_vae_model,
        args.wan_vae_subfolder,
        device,
        args.wan_vae_dtype,
    )

    print("Inferring latent dimension ...")
    dummy = np.zeros((224, 224, 3), dtype=np.uint8)
    latent_dim = int(encode_frame(vae, dummy, vae_device, vae_dtype).shape[-1])
    print(
        f"  state_dim={state_dim}, action_dim={action_dim}, "
        f"num_patches={num_patches}, latent_dim={latent_dim}"
    )

    print(f"Loading WAN world model from {args.wm_checkpoint}")
    transition = load_wm(
        args.wm_checkpoint,
        state_dim=state_dim,
        action_dim=action_dim,
        num_patches=num_patches,
        latent_dim=latent_dim,
        sequence_length=args.sequence_length,
        action_horizon=args.action_horizon,
        device=device,
    )

    print(f"Loading and WAN-encoding episode {args.episode} from {args.hf_repo}")
    episode_data = load_episode_wan_latents(
        args.hf_repo,
        args.episode,
        vae=vae,
        vae_device=vae_device,
        vae_dtype=vae_dtype,
        device=device,
        video_query_batch=args.video_query_batch,
    )
    T = int(episode_data["front_latent"].shape[0])
    print(f"  Episode has {T} frames")

    print(
        f"Running WAN horizon sweep K=1..{args.max_horizon} "
        f"(context_length={args.context_length}, pred_step={args.pred_step}, stride={args.stride}) ..."
    )
    if args.no_future_actions:
        print("ABLATION MODE: future actions zeroed out")

    results = evaluate_horizon_wan(
        transition=transition,
        episode_data=episode_data,
        stats=stats,
        context_length=args.context_length,
        pred_step=args.pred_step,
        max_horizon=args.max_horizon,
        stride=args.stride,
        device=device,
        no_future_actions=args.no_future_actions,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "no_future_actions" if args.no_future_actions else "with_future_actions"
    json_path = out_dir / f"horizon_sweep_results_wan_{tag}.json"
    payload = {
        "hf_repo": args.hf_repo,
        "episode": args.episode,
        "context_length": args.context_length,
        "pred_step": args.pred_step,
        "max_horizon": args.max_horizon,
        "stride": args.stride,
        "horizons": results["horizons"].tolist(),
        "latent_mse_mean": results["latent_mse_mean"].tolist(),
        "latent_mse_std": results["latent_mse_std"].tolist(),
        "state_mse_mean": results["state_mse_mean"].tolist(),
        "state_mse_std": results["state_mse_std"].tolist(),
        "num_windows": results["num_windows"],
    }
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved: {json_path}")

    mode_label = "no future actions (ablation)" if args.no_future_actions else "with future actions"
    title_suffix = f"  (ep {args.episode}, pred_step={args.pred_step}, {mode_label})"
    plot_sweep(results, out_dir, title_suffix=title_suffix, tag=f"wan_{tag}")

    print(f"\nSummary (K=1 → K={args.max_horizon}):")
    print(f"  Latent MSE: {results['latent_mse_mean'][0]:.6f} → {results['latent_mse_mean'][-1]:.6f}")
    print(f"  State  MSE: {results['state_mse_mean'][0]:.6f} → {results['state_mse_mean'][-1]:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
