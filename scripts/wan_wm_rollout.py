#!/usr/bin/env python3
"""
WAN World Model autoregressive rollout video.

Runs the WM autoregressively from a context window, feeding predicted
latents back as input at each step. Ground-truth frames and WAN latents are
fetched only when needed for context, resets, and visualization.

Quickstart:
    python scripts/wan_wm_rollout.py \
        --wm-checkpoint checkpoints/wan_wm_checkpoints/best_wm_from_hpc_2026-03-16.pth \
        --wan-vae-model weights/Video-As-Prompt-Wan2.1-14B \
        --dataset villekuosmanen/bin_pick_pack_coffee_capsules_eval \
        --episode 0 \
        --context-length 3 \
        --full-episode \
        --reset-interval 10 \
        --future-action-steps 1 \
        --video-query-batch 8 \
        --output-dir outputs/wan_replay_full_ep_reset10_k1 \
        --device cuda:0
"""
import argparse
import os
import sys
from collections import OrderedDict

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

DEFAULT_DATASET_STATS = "arx5_datasets_6Feb_26_stats.json"
DEFAULT_WAN_VAE_MODEL = "weights/Video-As-Prompt-Wan2.1-14B"

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.wan_wm_utils import (
    decode_latent,
    encode_frame,
    load_lerobot_episode,
    load_stats,
    load_wan_vae,
    load_wm,
    unnormalize_states,
)
from dino_wm.config import WAN_CONFIG
from dino_wm.dino_models import normalize_acs, normalize_states


def parse_args():
    p = argparse.ArgumentParser(description="WAN WM autoregressive rollout video")
    p.add_argument("--dataset", required=True, help="LeRobot dataset ID or local path")
    p.add_argument("--episode", type=int, default=0, help="Episode index")
    p.add_argument("--wm-checkpoint", default="wan_wm_checkpoints/best_wm.pth",
                   help="Path to WM checkpoint .pth")
    p.add_argument("--dataset-stats", default=DEFAULT_DATASET_STATS,
                   help="Path to normalization stats JSON")
    p.add_argument("--wan-vae-model", default=DEFAULT_WAN_VAE_MODEL,
                   help="WAN VAE model ID or local path")
    p.add_argument("--wan-vae-subfolder", default="vae")
    p.add_argument("--wan-vae-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--context-length", type=int, default=3,
                   help="H: number of GT context frames before rollout begins")
    p.add_argument("--sequence-length", type=int, default=4,
                   help="WM sequence_length parameter (num_frames = sequence_length - 1)")
    p.add_argument("--action-horizon", type=int, default=100,
                   help="WM action_horizon parameter")
    p.add_argument("--horizon", type=int, default=50,
                   help="Number of autoregressive rollout steps")
    p.add_argument("--future-action-steps", type=int, default=0,
                   help="GT action chunk size passed to WM at each rollout step (0 = none, recommended)")
    p.add_argument("--reset-interval", type=int, default=0,
                   help="If >0, every K prediction steps reset WM context with fresh GT context.")
    p.add_argument("--full-episode", action="store_true",
                   help="Ignore --horizon and roll from context to end of episode.")
    p.add_argument("--video-query-batch", type=int, default=16,
                   help="Frames per video backend query chunk (lower = less memory).")
    p.add_argument("--render-size", type=int, default=224,
                   help="Output frame size in pixels (square)")
    p.add_argument("--fps", type=int, default=20, help="Video FPS")
    p.add_argument("--output-dir", default="outputs/rollout",
                   help="Directory to save the output video")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def _resize_frame(frame_hwc: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image

    return np.array(Image.fromarray(frame_hwc).resize((size, size), Image.LANCZOS))


def _to_hwc_uint8_list(tensor: torch.Tensor) -> list[np.ndarray]:
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"Expected video tensor with 4 dims (T,C,H,W), got shape={tuple(tensor.shape)}")
    if tensor.max() > 1.0:
        arr = tensor.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    else:
        arr = (tensor.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
    return [arr[i] for i in range(arr.shape[0])]


class WanEpisodeProvider:
    def __init__(
        self,
        *,
        dataset_id: str,
        episode: int,
        max_frames: int,
        video_query_batch: int,
        vae,
        vae_device: torch.device,
        vae_dtype: torch.dtype,
    ):
        dataset, start_idx, end_idx = load_lerobot_episode(dataset_id, episode)
        batch = dataset.hf_dataset.with_format(None)[start_idx:end_idx]
        total_episode_frames = len(batch["timestamp"])
        if total_episode_frames == 0:
            raise ValueError(f"Episode {episode} has no frames.")
        if video_query_batch <= 0:
            raise ValueError("--video-query-batch must be >= 1.")

        state_key = next((k for k in ["observation.state", "observation.state.pos", "observation.state.eef_pose"] if k in batch), None)
        action_key = next((k for k in ["action", "action.pos", "action.position", "action.eef_pose"] if k in batch), None)
        if state_key is None:
            raise KeyError("Missing state key in dataset batch.")
        if action_key is None:
            raise KeyError("Missing action key in dataset batch.")

        self.dataset = dataset
        self.episode = int(episode)
        self.total_frames = min(int(max_frames), int(total_episode_frames))
        self.episode_total_frames = int(total_episode_frames)
        self.video_query_batch = int(video_query_batch)
        self.vae = vae
        self.vae_device = vae_device
        self.vae_dtype = vae_dtype

        self.timestamps = [float(t.item() if hasattr(t, "item") else t) for t in batch["timestamp"][: self.total_frames]]
        self.states = torch.tensor(np.asarray(batch[state_key][: self.total_frames], dtype=np.float32), dtype=torch.float32)
        self.actions = torch.tensor(np.asarray(batch[action_key][: self.total_frames], dtype=np.float32), dtype=torch.float32)

        self._frame_chunk_cache: OrderedDict[int, tuple[list[np.ndarray], list[np.ndarray], int]] = OrderedDict()
        self._latent_cache: OrderedDict[int, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._max_cached_chunks = 3
        self._max_cached_latents = max(8, 4 * self.video_query_batch)

    def _load_chunk_containing(self, frame_idx: int) -> None:
        if not (0 <= frame_idx < self.total_frames):
            raise IndexError(f"Frame index {frame_idx} out of range for total_frames={self.total_frames}")
        chunk_start = (frame_idx // self.video_query_batch) * self.video_query_batch
        if chunk_start in self._frame_chunk_cache:
            self._frame_chunk_cache.move_to_end(chunk_start)
            return

        chunk_end = min(chunk_start + self.video_query_batch, self.total_frames)
        query = {k: self.timestamps[chunk_start:chunk_end] for k in self.dataset.meta.video_keys}
        orig_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            video_chunk = self.dataset._query_videos(query, self.episode)
        finally:
            torch.set_default_dtype(orig_dtype)

        if "observation.images.front" not in video_chunk or "observation.images.wrist" not in video_chunk:
            raise ValueError("Missing front/wrist image streams in queried video chunk.")
        self._frame_chunk_cache[chunk_start] = (
            _to_hwc_uint8_list(video_chunk["observation.images.front"]),
            _to_hwc_uint8_list(video_chunk["observation.images.wrist"]),
            chunk_end,
        )
        if len(self._frame_chunk_cache) > self._max_cached_chunks:
            self._frame_chunk_cache.popitem(last=False)
        print(f"  loaded frames {chunk_end}/{self.total_frames}")

    def get_frame_pair(self, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        self._load_chunk_containing(frame_idx)
        chunk_start = (frame_idx // self.video_query_batch) * self.video_query_batch
        front_cache, wrist_cache, _ = self._frame_chunk_cache[chunk_start]
        local_idx = frame_idx - chunk_start
        return front_cache[local_idx], wrist_cache[local_idx]

    def get_latent_pair(self, frame_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if frame_idx in self._latent_cache:
            self._latent_cache.move_to_end(frame_idx)
            return self._latent_cache[frame_idx]

        front_frame, wrist_frame = self.get_frame_pair(frame_idx)
        front_latent = encode_frame(self.vae, front_frame, self.vae_device, self.vae_dtype).squeeze(0).cpu()
        wrist_latent = encode_frame(self.vae, wrist_frame, self.vae_device, self.vae_dtype).squeeze(0).cpu()
        self._latent_cache[frame_idx] = (front_latent, wrist_latent)
        if len(self._latent_cache) > self._max_cached_latents:
            self._latent_cache.popitem(last=False)
        return front_latent, wrist_latent


def _make_comparison_frame(
    gt_front: np.ndarray,
    gt_wrist: np.ndarray,
    pred_front: np.ndarray,
    pred_wrist: np.ndarray,
) -> np.ndarray:
    sep_v = np.ones((gt_front.shape[0], 4, 3), dtype=np.uint8) * 180
    sep_h = np.ones((4, gt_front.shape[1] * 2 + 4, 3), dtype=np.uint8) * 180
    gt_row = np.concatenate([gt_front, sep_v, gt_wrist], axis=1)
    pred_row = np.concatenate([pred_front, sep_v, pred_wrist], axis=1)
    return np.concatenate([gt_row, sep_h, pred_row], axis=0)


def plot_state_rollout(gt_states: torch.Tensor, pred_states: torch.Tensor, output_path: str):
    gt = gt_states.detach().cpu().numpy()
    pred = pred_states.detach().cpu().numpy()
    _, dims = gt.shape
    err = np.linalg.norm(gt - pred, axis=1)

    num_plots = dims + 1
    ncols = 3
    nrows = int(np.ceil(num_plots / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 2.8 * nrows), squeeze=False)
    axes = axes.flatten()
    t = np.arange(gt.shape[0])

    for i in range(dims):
        axes[i].plot(t, gt[:, i], label="gt", linewidth=1.2)
        axes[i].plot(t, pred[:, i], label="pred", linewidth=1.2, alpha=0.8)
        axes[i].set_title(f"state[{i}]")
        axes[i].grid(True, alpha=0.3)

    axes[dims].plot(t, err, color="tab:red", linewidth=1.2)
    axes[dims].set_title("L2 error")
    axes[dims].grid(True, alpha=0.3)

    for i in range(dims + 1, len(axes)):
        axes[i].axis("off")

    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    H = args.context_length
    latent_side = WAN_CONFIG["latent_side"]
    num_patches = latent_side * latent_side
    render_size = int(args.render_size)
    if args.reset_interval < 0:
        raise ValueError("--reset-interval must be >= 0.")
    if args.future_action_steps < 0:
        raise ValueError("--future-action-steps must be >= 0.")
    if args.horizon < 0:
        raise ValueError("--horizon must be >= 0.")

    print(f"Loading stats from {args.dataset_stats}")
    stats = load_stats(args.dataset_stats, device)
    state_dim = stats["state_dim"]
    action_dim = stats["action_dim"]
    print(f"  state_dim={state_dim}, action_dim={action_dim}")

    print(f"Loading WAN VAE from {args.wan_vae_model}")
    vae, vae_device, vae_dtype = load_wan_vae(
        args.wan_vae_model, args.wan_vae_subfolder, device, args.wan_vae_dtype
    )

    dummy = np.zeros((224, 224, 3), dtype=np.uint8)
    dummy_latent = encode_frame(vae, dummy, vae_device, vae_dtype)
    latent_dim = dummy_latent.shape[-1]
    print(f"  latent: num_patches={num_patches}, latent_dim={latent_dim}")

    print(f"Loading WM from {args.wm_checkpoint}")
    wm = load_wm(
        args.wm_checkpoint,
        state_dim=state_dim,
        action_dim=action_dim,
        num_patches=num_patches,
        latent_dim=latent_dim,
        sequence_length=args.sequence_length,
        action_horizon=args.action_horizon,
        device=device,
    )

    print(f"Loading episode {args.episode} from {args.dataset}")
    dataset, start_idx, end_idx = load_lerobot_episode(args.dataset, args.episode)
    episode_total = int(end_idx - start_idx)
    if args.full_episode:
        total_frames = episode_total
        rollout_steps = max(0, total_frames - H)
        print(f"  full-episode mode active: total_frames={total_frames}, rollout_steps={rollout_steps}")
    else:
        total_frames = H + args.horizon
        rollout_steps = args.horizon

    provider = WanEpisodeProvider(
        dataset_id=args.dataset,
        episode=args.episode,
        max_frames=total_frames,
        video_query_batch=int(args.video_query_batch),
        vae=vae,
        vae_device=vae_device,
        vae_dtype=vae_dtype,
    )
    if provider.total_frames < total_frames:
        raise ValueError(
            f"Episode too short ({provider.episode_total_frames} frames), need context={H} + horizon={args.horizon}={total_frames}"
        )

    states_raw = provider.states.to(device)
    actions_raw = provider.actions.to(device)
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

    ctx_front = []
    ctx_wrist = []
    context_gt_front = []
    context_gt_wrist = []
    for i in range(H):
        front_latent, wrist_latent = provider.get_latent_pair(i)
        front_frame, wrist_frame = provider.get_frame_pair(i)
        ctx_front.append(front_latent)
        ctx_wrist.append(wrist_latent)
        context_gt_front.append(_resize_frame(front_frame, render_size))
        context_gt_wrist.append(_resize_frame(wrist_frame, render_size))

    ctx_front = torch.stack(ctx_front, dim=0).unsqueeze(0).to(device)
    ctx_wrist = torch.stack(ctx_wrist, dim=0).unsqueeze(0).to(device)
    ctx_states = states_norm[:, :H]
    ctx_actions = actions_norm[:, :H]
    pred_states = [states_raw[:H].detach().cpu()]

    print(
        f"Running rollout for {rollout_steps} steps "
        f"(future_action_steps={args.future_action_steps}, reset_interval={args.reset_interval})..."
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"rollout_ep{args.episode}.mp4")
    with imageio.get_writer(
        out_path,
        fps=args.fps,
        codec="libx264",
        pixelformat="yuv420p",
    ) as writer:
        for gt_front, gt_wrist in zip(context_gt_front, context_gt_wrist):
            writer.append_data(_make_comparison_frame(gt_front, gt_wrist, gt_front, gt_wrist))

        with torch.no_grad():
            for k in range(rollout_steps):
                t = H + k

                if args.reset_interval > 0 and k > 0 and (k % args.reset_interval == 0):
                    reset_start = t - H
                    reset_end = t
                    reset_front = []
                    reset_wrist = []
                    for i in range(reset_start, reset_end):
                        front_latent, wrist_latent = provider.get_latent_pair(i)
                        reset_front.append(front_latent)
                        reset_wrist.append(wrist_latent)
                    ctx_front = torch.stack(reset_front, dim=0).unsqueeze(0).to(device)
                    ctx_wrist = torch.stack(reset_wrist, dim=0).unsqueeze(0).to(device)
                    ctx_states = states_norm[:, reset_start:reset_end]
                    ctx_actions = actions_norm[:, reset_start:reset_end]
                    print(f"  reset @ step {k}: GT context frames [{reset_start}:{reset_end}]")

                if args.future_action_steps > 0:
                    end = min(t + args.future_action_steps, actions_norm.shape[1])
                    fut_actions = actions_norm[:, t:end]
                    if fut_actions.shape[1] == 0:
                        fut_actions = None
                else:
                    fut_actions = None

                pred1, pred2, pred_state, _ = wm(
                    ctx_front, ctx_wrist, ctx_states, ctx_actions, fut_actions
                )

                pred_front_frame = decode_latent(vae, pred1[:, -1], latent_side, latent_side, vae_device, vae_dtype)
                pred_wrist_frame = decode_latent(vae, pred2[:, -1], latent_side, latent_side, vae_device, vae_dtype)
                gt_front_frame, gt_wrist_frame = provider.get_frame_pair(t)
                writer.append_data(
                    _make_comparison_frame(
                        _resize_frame(gt_front_frame, render_size),
                        _resize_frame(gt_wrist_frame, render_size),
                        _resize_frame(pred_front_frame, render_size),
                        _resize_frame(pred_wrist_frame, render_size),
                    )
                )

                ctx_front = torch.cat([ctx_front[:, 1:], pred1[:, -1:]], dim=1)
                ctx_wrist = torch.cat([ctx_wrist[:, 1:], pred2[:, -1:]], dim=1)
                ctx_states = torch.cat([ctx_states[:, 1:], pred_state[:, -1:]], dim=1)
                pred_state_raw = unnormalize_states(
                    pred_state[:, -1],
                    stats["state_min"],
                    stats["state_max"],
                    q02=stats.get("state_q02"),
                    q98=stats.get("state_q98"),
                )
                pred_states.append(pred_state_raw.squeeze(0).unsqueeze(0).detach().cpu())

                act_idx = min(t, actions_norm.shape[1] - 1)
                next_act = actions_norm[:, act_idx:act_idx + 1]
                ctx_actions = torch.cat([ctx_actions[:, 1:], next_act], dim=1)

                if (k + 1) % 10 == 0 or (k + 1) == rollout_steps:
                    print(f"  step {k + 1}/{rollout_steps}")

    print(f"\nSaved: {out_path}")
    gt_state_rollout = states_raw[:total_frames].detach().cpu()
    pred_state_rollout = torch.cat(pred_states, dim=0)[:total_frames]
    state_plot_path = os.path.join(args.output_dir, f"state_rollout_ep{args.episode}.png")
    plot_state_rollout(gt_state_rollout, pred_state_rollout, state_plot_path)
    print(f"Saved: {state_plot_path}")
    print(f"Layout: [GT Front | GT Wrist] (top) / [Pred Front | Pred Wrist] (bottom)")
    print(f"Context frames 0..{H - 1} show GT in both rows.")
    print(f"Total frames: {total_frames}  |  FPS: {args.fps}")


if __name__ == "__main__":
    main()
