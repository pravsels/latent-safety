#!/usr/bin/env python3
"""
WAN World Model frame prediction evaluation video.

At each evaluation step t (striding by N=future-action-steps):
  - Feed WM: GT context latents [t-H:t] + GT states + GT context actions + GT future actions [t:t+N]
  - WM predicts the frame at t+N
  - Decode predicted latent -> pixel frame
  - Compare side-by-side with GT frame at t+N

Output: one PNG per evaluation step showing predicted vs GT side by side.
Not a continuous rollout — context is always GT. Each image is an independent
single-step prediction vs GT comparison.

Layout per video frame:
    ┌─────────────────────┬─────────────────────┐
    │    GT front         │    GT wrist          │
    ├─────────────────────┼─────────────────────┤
    │  Pred front         │  Pred wrist          │
    └─────────────────────┴─────────────────────┘

Usage:
    python scripts/wan_wm_frame_eval.py \
        --dataset villekuosmanen/fail_bil_pick_capsules_drop_on_table \
        --episode 0 \
        --wm-checkpoint wan_wm_checkpoints/best_wm.pth \
        --dataset-stats wan_wm_checkpoints/dataset_stats.json \
        --future-action-steps 50 \
        --context-length 3
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
    p = argparse.ArgumentParser(description="WAN WM frame prediction evaluation video")
    p.add_argument("--dataset", required=True, help="LeRobot dataset ID or local path")
    p.add_argument("--episode", type=int, default=0, help="Episode index")
    p.add_argument("--wm-checkpoint", default="wan_wm_checkpoints/best_wm.pth")
    p.add_argument("--dataset-stats", default="wan_wm_checkpoints/dataset_stats.json")
    p.add_argument("--wan-vae-model", default="ByteDance/Video-As-Prompt-Wan2.1-14B")
    p.add_argument("--wan-vae-subfolder", default="vae")
    p.add_argument("--wan-vae-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--future-action-steps", type=int, default=50,
                   help="N: action chunk size and prediction horizon")
    p.add_argument("--context-length", type=int, default=3, help="H: GT context frames")
    p.add_argument("--sequence-length", type=int, default=4)
    p.add_argument("--action-horizon", type=int, default=100)
    p.add_argument("--render-size", type=int, default=224)
    p.add_argument("--output-dir", default="outputs/frame_eval")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def _resize_frame(frame_hwc: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image
    return np.array(Image.fromarray(frame_hwc).resize((size, size), Image.LANCZOS))


def _label_frame(frame_hwc: np.ndarray, text: str) -> np.ndarray:
    """Burn a text label into the top-left corner of a HWC uint8 frame."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.fromarray(frame_hwc)
    draw = ImageDraw.Draw(img)
    # Shadow for readability on any background
    draw.text((3, 3), text, fill=(0, 0, 0))
    draw.text((2, 2), text, fill=(255, 255, 255))
    return np.array(img)


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    H = args.context_length
    N = args.future_action_steps
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
    latent_dim = encode_frame(vae, dummy, vae_device, vae_dtype).shape[-1]
    print(f"  latent: num_patches={num_patches}, latent_dim={latent_dim}")

    print(f"Loading WM from {args.wm_checkpoint}")
    wm = load_wm(
        args.wm_checkpoint,
        state_dim=state_dim, action_dim=action_dim,
        num_patches=num_patches, latent_dim=latent_dim,
        sequence_length=args.sequence_length,
        action_horizon=args.action_horizon, device=device,
    )

    print(f"Loading episode {args.episode} from {args.dataset}")
    dataset, start_idx, end_idx = load_lerobot_episode(args.dataset, args.episode)
    ep = get_episode_frames_and_data(dataset, start_idx, end_idx, args.episode)
    T = len(ep["front_frames"])
    print(f"  Episode length: {T} frames")

    if T < H + N + 1:
        raise ValueError(f"Episode too short ({T}) for context={H} + future={N} + 1")

    # Encode all frames upfront
    print("Encoding all frames with WAN VAE...")
    front_latents = torch.cat(
        [encode_frame(vae, ep["front_frames"][i], vae_device, vae_dtype) for i in range(T)], dim=0
    ).unsqueeze(0).to(device)  # (1, T, num_patches, latent_dim)
    wrist_latents = torch.cat(
        [encode_frame(vae, ep["wrist_frames"][i], vae_device, vae_dtype) for i in range(T)], dim=0
    ).unsqueeze(0).to(device)

    states_norm = normalize_states(
        torch.tensor(ep["states"], dtype=torch.float32, device=device).unsqueeze(0),
        stats["state_min"], stats["state_max"],
        q02=stats.get("state_q02"), q98=stats.get("state_q98"),
    )
    actions_norm = normalize_acs(
        torch.tensor(ep["actions"], dtype=torch.float32, device=device).unsqueeze(0),
        stats["action_min"], stats["action_max"],
        q02=stats.get("action_delta_q02"), q98=stats.get("action_delta_q98"),
    )

    sep_v = np.ones((render_size, 4, 3), dtype=np.uint8) * 180
    sep_h = np.ones((4, render_size * 3 + 8, 3), dtype=np.uint8) * 180

    composite_frames = []
    eval_steps = []

    print(f"\nEvaluating {N}-step-ahead frame prediction (stride={N})...")
    print(f"{'Step t':>8}  {'Target t+N':>10}  frames collected: ", end="", flush=True)

    with torch.no_grad():
        t = H
        while t + N < T:
            ctx_front   = front_latents[:, t - H:t]    # (1, H, P, D)
            ctx_wrist   = wrist_latents[:, t - H:t]
            ctx_states  = states_norm[:, t - H:t]
            ctx_actions = actions_norm[:, t - H:t]
            fut_actions = actions_norm[:, t:t + N]      # (1, N, action_dim)

            pred1, pred2, _, _ = wm(ctx_front, ctx_wrist, ctx_states, ctx_actions, fut_actions)
            # pred latent for t+N is the last slot
            pred_front_frame = decode_latent(
                vae, pred1[:, -1], latent_side, latent_side, vae_device, vae_dtype
            )
            pred_wrist_frame = decode_latent(
                vae, pred2[:, -1], latent_side, latent_side, vae_device, vae_dtype
            )

            gt_front_t     = _label_frame(_resize_frame(ep["front_frames"][t],     render_size), f"GT front  t={t}")
            gt_wrist_t     = _label_frame(_resize_frame(ep["wrist_frames"][t],     render_size), f"GT wrist  t={t}")
            gt_front_tN    = _label_frame(_resize_frame(ep["front_frames"][t + N], render_size), f"GT front  t={t+N}")
            gt_wrist_tN    = _label_frame(_resize_frame(ep["wrist_frames"][t + N], render_size), f"GT wrist  t={t+N}")
            pred_front_tN  = _label_frame(_resize_frame(pred_front_frame,          render_size), f"Pred front t={t+N}")
            pred_wrist_tN  = _label_frame(_resize_frame(pred_wrist_frame,          render_size), f"Pred wrist t={t+N}")

            front_row = np.concatenate([gt_front_t,  sep_v, gt_front_tN,  sep_v, pred_front_tN], axis=1)
            wrist_row = np.concatenate([gt_wrist_t,  sep_v, gt_wrist_tN,  sep_v, pred_wrist_tN], axis=1)
            frame = np.concatenate([front_row, sep_h, wrist_row], axis=0)
            composite_frames.append(frame)
            eval_steps.append(t)

            print(f"{len(composite_frames)}", end=" ", flush=True)
            t += N

    print(f"\nTotal evaluation frames: {len(composite_frames)}")

    from PIL import Image
    os.makedirs(args.output_dir, exist_ok=True)
    out_dir = os.path.join(args.output_dir, f"ep{args.episode}_N{N}")
    os.makedirs(out_dir, exist_ok=True)

    for i, (frame, t) in enumerate(zip(composite_frames, eval_steps)):
        out_path = os.path.join(out_dir, f"step{t:04d}_pred_t+{N}.png")
        Image.fromarray(frame).save(out_path)
        print(f"  [{i+1}/{len(composite_frames)}] t={t:4d} -> t+{N}={t+N:4d}  saved: {out_path}")

    print(f"\nLayout: [GT@t | GT@t+{N} | Pred@t+{N}] — front row / wrist row")
    print(f"Saved {len(composite_frames)} PNGs to {out_dir}")


if __name__ == "__main__":
    main()
