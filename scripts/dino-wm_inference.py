#!/usr/bin/env python3
"""
Generate one DINO world-model rollout for a single LeRobot episode.

Quickstart:
    python scripts/dino-wm_inference.py \
        --wm-checkpoint checkpoints/dino3_wm_checkpoints/best_wm_from_hpc.pth \
        --decoder-checkpoint checkpoints/dino3_decoder_checkpoints/best_decoder.pth \
        --dataset villekuosmanen/bin_pick_pack_coffee_capsules_eval \
        --episode 0 \
        --context-length 3 \
        --full-episode \
        --reset-interval 10 \
        --future-action-steps 1 \
        --video-query-batch 8 \
        --output-dir outputs/dino_replay_full_ep_reset10_k1 \
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
import torch.nn.functional as F
from einops import rearrange

DEFAULT_DATASET_STATS = "arx5_datasets_6Feb_26_stats.json"

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dino_wm.config import MODEL_CONFIG, get_decoder_image_size, get_dino_config
from dino_wm.data_utils import compute_action_deltas
from dino_wm.dino_decoder import VQVAE
from dino_wm.dino_models import VideoTransformer, normalize_acs, normalize_states, unnormalize_states
from scripts.utils import get_dino_model, preprocess_images_for_dino
from scripts.wan_wm_utils import load_lerobot_episode, load_stats


def _load_state_dict_with_meta(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "decoder_state_dict", "state_dict"):
            if key in ckpt:
                meta = {k: v for k, v in ckpt.items() if k != key}
                return ckpt[key], meta
    return ckpt, {}


def _resolve_quantize_flag(ckpt_meta: dict, checkpoint_path: str) -> bool:
    if isinstance(ckpt_meta, dict) and "quantize" in ckpt_meta:
        return bool(ckpt_meta["quantize"])
    return "_vq" in os.path.basename(checkpoint_path)


def _to_hwc_uint8_list(tensor: torch.Tensor) -> list[np.ndarray]:
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"Expected 4D video tensor, got shape={tuple(tensor.shape)}")
    if tensor.max() > 1.0:
        arr = tensor.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    else:
        arr = (tensor.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
    return [arr[i] for i in range(arr.shape[0])]


class EpisodeFrameProvider:
    def __init__(
        self,
        *,
        dataset_id: str,
        episode: int,
        max_frames: int,
        video_query_batch: int,
    ):
        dataset, start_idx, end_idx = load_lerobot_episode(dataset_id, episode)
        batch = dataset.hf_dataset.with_format(None)[start_idx:end_idx]
        episode_total = len(batch["timestamp"])
        if episode_total == 0:
            raise ValueError(f"Episode {episode} has no frames.")
        if video_query_batch <= 0:
            raise ValueError("--video-query-batch must be >= 1.")

        action_key = next((k for k in ["action", "action.pos", "action.position", "action.eef_pose"] if k in batch), None)
        state_key = next((k for k in ["observation.state", "observation.state.pos", "observation.state.eef_pose"] if k in batch), None)
        if action_key is None:
            raise KeyError("No action key found in episode batch.")
        if state_key is None:
            raise KeyError("No state key found in episode batch.")

        self.dataset = dataset
        self.episode = int(episode)
        self.total_frames = min(int(max_frames), int(episode_total))
        self.episode_total_frames = int(episode_total)
        self.video_query_batch = int(video_query_batch)
        self.timestamps = [np.float64(t) for t in batch["timestamp"][: self.total_frames]]
        actions_np = np.asarray(batch[action_key][: self.total_frames], dtype=np.float32)
        states_np = np.asarray(batch[state_key][: self.total_frames], dtype=np.float32)
        self.actions_delta = torch.from_numpy(compute_action_deltas(actions_np, states_np))
        self.states = torch.from_numpy(states_np)

        self._chunk_cache: OrderedDict[int, tuple[list[np.ndarray], list[np.ndarray], int]] = OrderedDict()
        self._max_cached_chunks = 3

    def _load_chunk_containing(self, frame_idx: int) -> None:
        if not (0 <= frame_idx < self.total_frames):
            raise IndexError(f"Frame index {frame_idx} out of range for total_frames={self.total_frames}")
        chunk_start = (frame_idx // self.video_query_batch) * self.video_query_batch
        if chunk_start in self._chunk_cache:
            self._chunk_cache.move_to_end(chunk_start)
            return

        chunk_end = min(chunk_start + self.video_query_batch, self.total_frames)
        query = {k: self.timestamps[chunk_start:chunk_end] for k in self.dataset.meta.video_keys}
        orig_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            video_chunk = self.dataset._query_videos(query, self.episode)
        finally:
            torch.set_default_dtype(orig_dtype)

        self._chunk_cache[chunk_start] = (
            _to_hwc_uint8_list(video_chunk["observation.images.front"]),
            _to_hwc_uint8_list(video_chunk["observation.images.wrist"]),
            chunk_end,
        )
        if len(self._chunk_cache) > self._max_cached_chunks:
            self._chunk_cache.popitem(last=False)
        print(f"  loaded frames {chunk_end}/{self.total_frames}")

    def get_frame_pair(self, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        self._load_chunk_containing(frame_idx)
        chunk_start = (frame_idx // self.video_query_batch) * self.video_query_batch
        front_cache, wrist_cache, _ = self._chunk_cache[chunk_start]
        local_idx = frame_idx - chunk_start
        return front_cache[local_idx], wrist_cache[local_idx]


class DinoDecoderAdapter:
    def __init__(self, decoder: VQVAE):
        self.decoder = decoder

    @torch.no_grad()
    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        pred_ims, _ = self.decoder(tokens)
        pred_ims = rearrange(pred_ims, "(b t) c h w -> b t h w c", t=tokens.shape[1])
        return pred_ims.clamp(0.0, 1.0)


@torch.no_grad()
def _encode_dino_frame_pair(
    dino_model: torch.nn.Module,
    front_hwc_uint8: np.ndarray,
    wrist_hwc_uint8: np.ndarray,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    front = (
        torch.from_numpy(front_hwc_uint8)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        / 255.0
    )
    wrist = (
        torch.from_numpy(wrist_hwc_uint8)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        / 255.0
    )
    front_prep = preprocess_images_for_dino(front, is_front_camera=True)
    wrist_prep = preprocess_images_for_dino(wrist, is_front_camera=False)
    front_emb = dino_model.forward_features(front_prep)["x_norm_patchtokens"].cpu()
    wrist_emb = dino_model.forward_features(wrist_prep)["x_norm_patchtokens"].cpu()
    return front_emb, wrist_emb


@torch.no_grad()
def generate_rollout_dino_streaming(
    *,
    transition,
    decoder_adapter,
    dino_model: torch.nn.Module,
    frame_provider: EpisodeFrameProvider,
    actions_raw: torch.Tensor,
    states_raw: torch.Tensor,
    context_length: int,
    horizon: int,
    device: str,
    stats: dict,
    render_size: tuple[int, int],
    writer,
    reset_interval: int | None = None,
    future_action_steps: int = 0,
):
    H = int(context_length)
    total_length = H + int(horizon)
    if total_length > int(frame_provider.total_frames):
        raise ValueError(f"Need {total_length} frames, but only {frame_provider.total_frames} available.")

    action_min = stats["action_min"].to(device)
    action_max = stats["action_max"].to(device)
    state_min = stats["state_min"].to(device)
    state_max = stats["state_max"].to(device)
    action_q02 = stats["action_delta_q02"].to(device) if "action_delta_q02" in stats else None
    action_q98 = stats["action_delta_q98"].to(device) if "action_delta_q98" in stats else None
    state_q02 = stats["state_q02"].to(device) if "state_q02" in stats else None
    state_q98 = stats["state_q98"].to(device) if "state_q98" in stats else None

    all_states_raw = states_raw[:, :total_length].to(device)
    all_actions_raw = actions_raw[:, :total_length].to(device)
    all_states = normalize_states(all_states_raw, state_min, state_max, q02=state_q02, q98=state_q98)
    all_actions = normalize_acs(all_actions_raw, action_min, action_max, q02=action_q02, q98=action_q98)

    front_ctx = []
    wrist_ctx = []
    context_gt_front = []
    context_gt_wrist = []
    for i in range(H):
        front_frame, wrist_frame = frame_provider.get_frame_pair(i)
        front_emb, wrist_emb = _encode_dino_frame_pair(dino_model, front_frame, wrist_frame, device)
        front_ctx.append(front_emb)
        wrist_ctx.append(wrist_emb)
        context_gt_front.append(_resize_uint8_frame(front_frame, render_size))
        context_gt_wrist.append(_resize_uint8_frame(wrist_frame, render_size))
    inputs1 = torch.cat(front_ctx, dim=0).unsqueeze(0).to(device)
    inputs2 = torch.cat(wrist_ctx, dim=0).unsqueeze(0).to(device)
    inputs_states = all_states[:, :H]
    actions = all_actions[:, :H]

    pred_states = [all_states_raw[0, :H]]
    for gt_front, gt_wrist in zip(context_gt_front, context_gt_wrist):
        writer.append_data(_make_comparison_frame(gt_front, gt_wrist, gt_front, gt_wrist))

    print(f"Running rollout for {horizon} steps (reset_interval={reset_interval}, future_action_steps={future_action_steps})...")
    for k in range(horizon):
        current_idx = H + k
        should_reset = (
            reset_interval is not None
            and reset_interval > 0
            and k > 0
            and k % reset_interval == 0
            and current_idx <= total_length
        )
        if should_reset:
            reset_start = current_idx - H
            reset_end = current_idx
            front_ctx = []
            wrist_ctx = []
            for i in range(reset_start, reset_end):
                front_frame, wrist_frame = frame_provider.get_frame_pair(i)
                front_emb, wrist_emb = _encode_dino_frame_pair(dino_model, front_frame, wrist_frame, device)
                front_ctx.append(front_emb)
                wrist_ctx.append(wrist_emb)
            inputs1 = torch.cat(front_ctx, dim=0).unsqueeze(0).to(device)
            inputs2 = torch.cat(wrist_ctx, dim=0).unsqueeze(0).to(device)
            inputs_states = all_states[:, reset_start:reset_end]
            actions = all_actions[:, reset_start:reset_end]
            print(f"  reset @ step {k}: GT context frames [{reset_start}:{reset_end}]")

        if future_action_steps > 0:
            future_end = min(current_idx + int(future_action_steps), total_length)
            future_actions = all_actions[:, current_idx:future_end]
            if future_actions.shape[1] == 0:
                future_actions = None
        else:
            future_actions = None

        next_front, next_wrist, next_state, _ = transition(inputs1, inputs2, inputs_states, actions, future_actions)
        decoded_front = _resize_decoded_frame(decoder_adapter.decode_tokens(next_front[:, [-1]]), render_size)
        decoded_wrist = _resize_decoded_frame(decoder_adapter.decode_tokens(next_wrist[:, [-1]]), render_size)
        gt_front_frame, gt_wrist_frame = frame_provider.get_frame_pair(current_idx)
        writer.append_data(
            _make_comparison_frame(
                _resize_uint8_frame(gt_front_frame, render_size),
                _resize_uint8_frame(gt_wrist_frame, render_size),
                decoded_front,
                decoded_wrist,
            )
        )

        if current_idx < all_actions.shape[1]:
            next_action = all_actions[:, current_idx:current_idx + 1]
        else:
            next_action = actions[:, -1:]

        inputs1 = torch.cat([inputs1[:, 1:], next_front[:, -1:]], dim=1)
        inputs2 = torch.cat([inputs2[:, 1:], next_wrist[:, -1:]], dim=1)
        inputs_states = torch.cat([inputs_states[:, 1:], next_state[:, -1:]], dim=1)
        actions = torch.cat([actions[:, 1:], next_action], dim=1)

        next_state_raw = unnormalize_states(next_state[:, -1], state_min, state_max, q02=state_q02, q98=state_q98)
        pred_states.append(next_state_raw.squeeze(0).unsqueeze(0))
        if (k + 1) % 10 == 0 or (k + 1) == horizon:
            print(f"  step {k + 1}/{horizon}")

    return all_states_raw.squeeze(0), torch.cat(pred_states, dim=0)[:total_length]


def _resize_uint8_frame(frame_hwc_uint8: np.ndarray, render_size: tuple[int, int]) -> np.ndarray:
    frame = torch.from_numpy(frame_hwc_uint8).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    frame = F.interpolate(frame, size=render_size, mode="bilinear", align_corners=False)
    frame = (frame.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return frame


def _resize_decoded_frame(decoded: torch.Tensor, render_size: tuple[int, int]) -> np.ndarray:
    frame = F.interpolate(
        decoded.squeeze(1).permute(0, 3, 1, 2).float().cpu(),
        size=render_size,
        mode="bilinear",
        align_corners=False,
    )
    frame = (frame.squeeze(0).permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return frame


def _make_comparison_frame(
    gt_front: np.ndarray,
    gt_wrist: np.ndarray,
    pred_front: np.ndarray,
    pred_wrist: np.ndarray,
) -> np.ndarray:
    width = gt_front.shape[1]
    gt_row = np.concatenate([gt_front, gt_wrist], axis=1)
    pred_row = np.concatenate([pred_front, pred_wrist], axis=1)
    separator = np.ones((16, 2 * width, 3), dtype=np.uint8) * 255
    return np.concatenate([gt_row, separator, pred_row], axis=0)


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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate one DINO world-model rollout from a LeRobot episode")
    parser.add_argument("--wm-checkpoint", type=str, required=True, help="Path to world model checkpoint")
    parser.add_argument("--decoder-checkpoint", type=str, required=True, help="Path to DINO decoder checkpoint")
    parser.add_argument("--dataset", type=str, required=True, help="LeRobot dataset id/path")
    parser.add_argument("--dataset-stats", type=str, default=DEFAULT_DATASET_STATS, help="Path to dataset statistics JSON file")
    parser.add_argument("--episode", type=int, default=0, help="Episode index")
    parser.add_argument("--dino-version", type=str, default="v3", choices=["v2", "v3"], help="DINO version")
    parser.add_argument("--horizon", type=int, default=10, help="Rollout horizon")
    parser.add_argument("--context-length", type=int, default=3, help="Context length H")
    parser.add_argument("--sequence-length", type=int, default=4, help="Sequence length used during training")
    parser.add_argument("--action-horizon", type=int, default=100, help="Action horizon for checkpoint compatibility")
    parser.add_argument("--future-action-steps", type=int, default=0, help="If >0, pass K future GT actions at each step")
    parser.add_argument("--full-episode", action="store_true", help="Ignore --horizon and roll to episode end")
    parser.add_argument("--video-query-batch", type=int, default=16, help="Frames per dataset video query")
    parser.add_argument("--reset-interval", type=int, default=None, help="Reset with fresh GT context every N steps")
    parser.add_argument("--output-dir", type=str, default="rollout_videos", help="Directory to save outputs")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--fps", type=int, default=20, help="Video FPS")
    parser.add_argument("--quantize", action="store_true", help="Enable VQ codebook quantization")
    parser.add_argument("--render-height", type=int, default=224, help="Rendered comparison image height")
    parser.add_argument("--render-width", type=int, default=224, help="Rendered comparison image width")
    return parser.parse_args(argv)


def validate_args(args):
    if args.render_height <= 0 or args.render_width <= 0:
        raise ValueError("--render-height and --render-width must be > 0")
    if args.context_length <= 0:
        raise ValueError("--context-length must be >= 1")
    if args.horizon < 0:
        raise ValueError("--horizon must be >= 0")
    if args.future_action_steps < 0:
        raise ValueError("--future-action-steps must be >= 0")
    if args.video_query_batch <= 0:
        raise ValueError("--video-query-batch must be >= 1")
    if args.reset_interval is not None and args.reset_interval < 0:
        raise ValueError("--reset-interval must be >= 0")


def main():
    args = parse_args()
    validate_args(args)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    else:
        device = args.device
    render_size = (int(args.render_height), int(args.render_width))

    print(f"Loading dataset stats from {args.dataset_stats}")
    stats = load_stats(args.dataset_stats, device)
    state_dim = int(stats["state_dim"])
    action_dim = int(stats["action_dim"])
    print(f"Inferred state_dim={state_dim}, action_dim={action_dim} from stats")

    dino_cfg = get_dino_config(args.dino_version)
    MODEL_CONFIG["dim"] = int(dino_cfg["dim"])
    MODEL_CONFIG["image_size"] = get_decoder_image_size(args.dino_version)

    decoder_state, decoder_meta = _load_state_dict_with_meta(args.decoder_checkpoint, device)
    quantize = bool(args.quantize) if args.quantize else _resolve_quantize_flag(decoder_meta, args.decoder_checkpoint)
    decoder = VQVAE(quantize=quantize).to(device)
    decoder.load_state_dict(decoder_state)
    decoder.eval()
    decoder_adapter = DinoDecoderAdapter(decoder)

    transition = VideoTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        num_frames=args.sequence_length - 1,
        action_horizon=int(args.action_horizon),
        backbone="dino",
        dino_version=args.dino_version,
        num_patches=None,
        **MODEL_CONFIG,
    ).to(device)
    wm_state, _ = _load_state_dict_with_meta(args.wm_checkpoint, device)
    transition.load_state_dict(wm_state)
    transition.eval()

    print(f"Loading LeRobot episode {args.episode} from {args.dataset}")
    max_frames = 10**9 if args.full_episode else args.context_length + args.horizon
    frame_provider = EpisodeFrameProvider(
        dataset_id=args.dataset,
        episode=int(args.episode),
        max_frames=max_frames,
        video_query_batch=int(args.video_query_batch),
    )

    rollout_horizon = (
        max(0, int(frame_provider.episode_total_frames) - int(args.context_length))
        if args.full_episode
        else int(args.horizon)
    )

    print(f"Loading DINO encoder {args.dino_version}")
    dino_encoder = get_dino_model(device, args.dino_version)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"rollout_ep{int(args.episode):03d}.mp4")
    with imageio.get_writer(
        output_path,
        fps=args.fps,
        codec="libx264",
        pixelformat="yuv420p",
    ) as writer:
        with torch.no_grad():
            gt_states, pred_states = generate_rollout_dino_streaming(
                transition=transition,
                decoder_adapter=decoder_adapter,
                dino_model=dino_encoder,
                frame_provider=frame_provider,
                actions_raw=frame_provider.actions_delta.unsqueeze(0),
                states_raw=frame_provider.states.unsqueeze(0),
                context_length=args.context_length,
                horizon=rollout_horizon,
                device=device,
                stats=stats,
                render_size=render_size,
                writer=writer,
                reset_interval=args.reset_interval,
                future_action_steps=args.future_action_steps,
            )
    print(f"Saved: {output_path}")

    state_plot_path = os.path.join(args.output_dir, f"state_rollout_ep{int(args.episode):03d}.png")
    plot_state_rollout(gt_states, pred_states, state_plot_path)
    print(f"Saved: {state_plot_path}")

    print(f"Done! Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
