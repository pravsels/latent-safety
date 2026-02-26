#!/usr/bin/env python3
"""
WAN World Model autoregressive rollout video.

Runs the WM autoregressively from a context window, feeding predicted
latents back as input at each step (1 GT action at a time, no future action
chunk). Produces a side-by-side GT vs predicted video for visual quality
assessment.

Layout:
    Top row:    GT front  |  GT wrist
    Bottom row: Pred front | Pred wrist
    (Context frames 0..H-1 show GT in both rows.)

Usage:
    python scripts/wan_wm_rollout.py \
        --dataset villekuosmanen/fail_bil_pick_capsules_drop_on_table \
        --episode 0 \
        --wm-checkpoint wan_wm_checkpoints/best_wm.pth \
        --dataset-stats wan_wm_checkpoints/dataset_stats.json \
        --context-length 3 \
        --horizon 50
"""
import argparse
import os
import sys

import numpy as np
import torch

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
    p = argparse.ArgumentParser(description="WAN WM autoregressive rollout video")
    p.add_argument("--dataset", required=True, help="LeRobot dataset ID or local path")
    p.add_argument("--episode", type=int, default=0, help="Episode index")
    p.add_argument("--wm-checkpoint", default="wan_wm_checkpoints/best_wm.pth",
                   help="Path to WM checkpoint .pth")
    p.add_argument("--dataset-stats", default="wan_wm_checkpoints/dataset_stats.json",
                   help="Path to normalization stats JSON")
    p.add_argument("--wan-vae-model", default="ByteDance/Video-As-Prompt-Wan2.1-14B",
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
    p.add_argument("--render-size", type=int, default=224,
                   help="Output frame size in pixels (square)")
    p.add_argument("--fps", type=int, default=20, help="Video FPS")
    p.add_argument("--output-dir", default="outputs/rollout",
                   help="Directory to save the output video")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def _resize_frame(frame_hwc: np.ndarray, size: int) -> np.ndarray:
    """Resize HWC uint8 frame to (size, size, 3) using PIL."""
    from PIL import Image
    return np.array(Image.fromarray(frame_hwc).resize((size, size), Image.LANCZOS))


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    H = args.context_length
    latent_side = WAN_CONFIG["latent_side"]
    num_patches = latent_side * latent_side
    render_size = args.render_size
    total_frames = H + args.horizon

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
    ep = get_episode_frames_and_data(dataset, start_idx, end_idx, args.episode)
    T = len(ep["front_frames"])
    print(f"  Episode length: {T} frames")

    if T < total_frames:
        raise ValueError(
            f"Episode too short ({T} frames), need context={H} + horizon={args.horizon}={total_frames}"
        )

    # Encode context frames
    print(f"Encoding {H} context frames with WAN VAE...")
    ctx_front_list = [encode_frame(vae, ep["front_frames"][i], vae_device, vae_dtype).to(device)
                      for i in range(H)]
    ctx_wrist_list = [encode_frame(vae, ep["wrist_frames"][i], vae_device, vae_dtype).to(device)
                      for i in range(H)]

    # Rolling context window: (1, H, num_patches, latent_dim)
    ctx_front = torch.cat(ctx_front_list, dim=0).unsqueeze(0)   # (1, H, P, D)
    ctx_wrist = torch.cat(ctx_wrist_list, dim=0).unsqueeze(0)

    states_raw = torch.tensor(ep["states"], dtype=torch.float32, device=device)   # (T, state_dim)
    actions_raw = torch.tensor(ep["actions"], dtype=torch.float32, device=device) # (T, action_dim)

    states_norm = normalize_states(
        states_raw.unsqueeze(0),
        stats["state_min"], stats["state_max"],
        q02=stats.get("state_q02"), q98=stats.get("state_q98"),
    )  # (1, T, state_dim)
    actions_norm = normalize_acs(
        actions_raw.unsqueeze(0),
        stats["action_min"], stats["action_max"],
        q02=stats.get("action_delta_q02"), q98=stats.get("action_delta_q98"),
    )  # (1, T, action_dim)

    ctx_states  = states_norm[:, :H]    # (1, H, state_dim)
    ctx_actions = actions_norm[:, :H]   # (1, H, action_dim)

    # GT frames for the full window (resized)
    gt_front_frames = [_resize_frame(ep["front_frames"][i], render_size) for i in range(total_frames)]
    gt_wrist_frames = [_resize_frame(ep["wrist_frames"][i], render_size) for i in range(total_frames)]

    # Predicted frames — context steps show GT (no prediction yet)
    pred_front_frames = [_resize_frame(ep["front_frames"][i], render_size) for i in range(H)]
    pred_wrist_frames = [_resize_frame(ep["wrist_frames"][i], render_size) for i in range(H)]

    chunk = args.future_action_steps
    print(f"Running autoregressive rollout for {args.horizon} steps "
          f"(future action chunk={chunk})...")
    with torch.no_grad():
        for k in range(args.horizon):
            t = H + k  # absolute step in episode

            if chunk > 0:
                end = min(t + chunk, actions_norm.shape[1])
                fut_actions = actions_norm[:, t:end]  # (1, <=chunk, action_dim)
            else:
                fut_actions = None

            pred1, pred2, pred_state, _ = wm(
                ctx_front, ctx_wrist, ctx_states, ctx_actions, fut_actions
            )
            # pred1/pred2: (1, H, num_patches, latent_dim)
            # pred_state:  (1, H, state_dim)

            # Decode the last-slot prediction (next frame after context window end)
            front_frame = decode_latent(vae, pred1[:, -1], latent_side, latent_side, vae_device, vae_dtype)
            wrist_frame = decode_latent(vae, pred2[:, -1], latent_side, latent_side, vae_device, vae_dtype)
            pred_front_frames.append(_resize_frame(front_frame, render_size))
            pred_wrist_frames.append(_resize_frame(wrist_frame, render_size))

            # Roll the context window: drop oldest, append predicted
            new_front = pred1[:, -1:].to(device)   # (1, 1, P, D)
            new_wrist = pred2[:, -1:].to(device)
            ctx_front  = torch.cat([ctx_front[:, 1:],  new_front],  dim=1)
            ctx_wrist  = torch.cat([ctx_wrist[:, 1:],  new_wrist],  dim=1)
            ctx_states = torch.cat([ctx_states[:, 1:],  pred_state[:, -1:]], dim=1)

            # Advance GT action by 1 step (use last action if at end of episode)
            act_idx = min(t, actions_norm.shape[1] - 1)
            next_act = actions_norm[:, act_idx:act_idx + 1]   # (1, 1, action_dim)
            ctx_actions = torch.cat([ctx_actions[:, 1:], next_act], dim=1)

            if (k + 1) % 10 == 0:
                print(f"  step {k + 1}/{args.horizon}")

    # Build comparison video
    sep_v = np.ones((render_size, 4, 3), dtype=np.uint8) * 180       # vertical separator
    sep_h = np.ones((4, render_size * 2 + 4, 3), dtype=np.uint8) * 180  # horizontal separator

    composite_frames = []
    for i in range(total_frames):
        gt_row   = np.concatenate([gt_front_frames[i],   sep_v, gt_wrist_frames[i]],   axis=1)
        pred_row = np.concatenate([pred_front_frames[i], sep_v, pred_wrist_frames[i]], axis=1)
        frame = np.concatenate([gt_row, sep_h, pred_row], axis=0)
        composite_frames.append(frame)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"rollout_ep{args.episode}.mp4")

    try:
        import imageio.v3 as iio
        iio.imwrite(
            out_path,
            np.stack(composite_frames),
            fps=args.fps,
            codec="libx264",
            pixelformat="yuv420p",
        )
    except Exception as e:
        # Fallback to imageio v2 API
        import imageio
        writer = imageio.get_writer(out_path, fps=args.fps, codec="libx264",
                                    ffmpeg_params=["-pix_fmt", "yuv420p"])
        for f in composite_frames:
            writer.append_data(f)
        writer.close()

    print(f"\nSaved: {out_path}")
    print(f"Layout: [GT Front | GT Wrist] (top) / [Pred Front | Pred Wrist] (bottom)")
    print(f"Context frames 0..{H - 1} show GT in both rows.")
    print(f"Total frames: {total_frames}  |  FPS: {args.fps}")


if __name__ == "__main__":
    main()
