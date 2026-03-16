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
        --future-action-steps 50 \
        --context-length 3
"""
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

DEFAULT_DATASET_STATS = "arx5_datasets_6Feb_26_stats.json"

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.wan_wm_utils import (
    load_stats, load_wan_vae, load_wm,
    load_lerobot_episode, get_episode_frames_and_data,
    encode_frame,
)
from dino_wm.config import WAN_CONFIG
from dino_wm.dino_models import normalize_states, normalize_acs, unnormalize_states


def parse_args():
    p = argparse.ArgumentParser(description="WAN WM N-step state prediction eval")
    p.add_argument("--dataset", required=True, help="LeRobot dataset ID or local path")
    p.add_argument("--episode", type=int, default=0, help="Episode index")
    p.add_argument("--wm-checkpoint", default="wan_wm_checkpoints/best_wm.pth",
                   help="Path to WM checkpoint .pth")
    p.add_argument("--dataset-stats", default=DEFAULT_DATASET_STATS,
                   help="Path to normalization stats JSON")
    p.add_argument("--wan-vae-model", default="ByteDance/Video-As-Prompt-Wan2.1-14B",
                   help="WAN VAE model ID or local path")
    p.add_argument("--wan-vae-subfolder", default="vae")
    p.add_argument("--wan-vae-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--future-action-steps", type=int, default=50,
                   help="N: predict state N steps ahead")
    p.add_argument("--context-length", type=int, default=3,
                   help="H: number of context frames fed to WM")
    p.add_argument("--sequence-length", type=int, default=4,
                   help="WM sequence_length parameter (num_frames = sequence_length - 1)")
    p.add_argument("--action-horizon", type=int, default=100,
                   help="WM action_horizon parameter")
    p.add_argument("--output-dir", default="outputs/state_eval",
                   help="Directory to save the output plot")
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
    print(f"  state_dim={state_dim}, action_dim={action_dim}")

    print(f"Loading WAN VAE from {args.wan_vae_model}")
    vae, vae_device, vae_dtype = load_wan_vae(
        args.wan_vae_model, args.wan_vae_subfolder, device, args.wan_vae_dtype
    )

    # Infer latent_dim by encoding a dummy frame
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

    if T < H + N + 1:
        raise ValueError(
            f"Episode too short ({T} frames) for context={H} + future={N} + 1 target step"
        )

    # Encode all frames upfront (avoids repeated VAE calls in eval loop)
    print("Encoding all frames with WAN VAE...")
    front_latents = torch.cat(
        [encode_frame(vae, ep["front_frames"][i], vae_device, vae_dtype) for i in range(T)],
        dim=0,
    ).unsqueeze(0).to(device)  # (1, T, num_patches, latent_dim)

    wrist_latents = torch.cat(
        [encode_frame(vae, ep["wrist_frames"][i], vae_device, vae_dtype) for i in range(T)],
        dim=0,
    ).unsqueeze(0).to(device)  # (1, T, num_patches, latent_dim)

    states_raw = torch.tensor(ep["states"], dtype=torch.float32, device=device)   # (T, state_dim)
    actions_raw = torch.tensor(ep["actions"], dtype=torch.float32, device=device) # (T, action_dim)

    states_norm = normalize_states(
        states_raw.unsqueeze(0),  # (1, T, state_dim)
        stats["state_min"], stats["state_max"],
        q02=stats.get("state_q02"), q98=stats.get("state_q98"),
    )
    actions_norm = normalize_acs(
        actions_raw.unsqueeze(0),  # (1, T, action_dim)
        stats["action_min"], stats["action_max"],
        q02=stats.get("action_delta_q02"), q98=stats.get("action_delta_q98"),
    )

    os.makedirs(args.output_dir, exist_ok=True)

    eval_steps = []
    l2_diffs = []

    col_w = 42
    header = f"{'Step':>6}  {'GT State':>{col_w}}  {'Pred State':>{col_w}}  {'L2':>8}"
    print(f"\n{header}")
    print("-" * len(header))

    with torch.no_grad():
        t = H
        while t + N < T:
            ctx_front   = front_latents[:, t - H:t]    # (1, H, num_patches, latent_dim)
            ctx_wrist   = wrist_latents[:, t - H:t]
            ctx_states  = states_norm[:, t - H:t]       # (1, H, state_dim)
            ctx_actions = actions_norm[:, t - H:t]      # (1, H, action_dim)
            fut_actions = actions_norm[:, t:t + N]      # (1, N, action_dim)

            _, _, state_preds, _ = wm(
                ctx_front, ctx_wrist, ctx_states, ctx_actions, fut_actions
            )
            # state_preds: (1, H, state_dim)
            # The last prediction corresponds to the state after the context window + N future steps.
            pred_state_norm = state_preds[:, -1]  # (1, state_dim)

            pred_state = unnormalize_states(
                pred_state_norm,
                stats["state_min"], stats["state_max"],
                q02=stats.get("state_q02"), q98=stats.get("state_q98"),
            ).squeeze(0)  # (state_dim,)

            gt_state = states_raw[t + N]  # (state_dim,)
            l2 = (pred_state - gt_state).norm().item()

            gt_str   = str(gt_state.cpu().numpy().round(3))
            pred_str = str(pred_state.cpu().numpy().round(3))
            print(f"{t:>6}  {gt_str:>{col_w}}  {pred_str:>{col_w}}  {l2:>8.4f}")

            eval_steps.append(t)
            l2_diffs.append(l2)
            t += N

    if not l2_diffs:
        print("No evaluation steps — episode may be too short.")
        return

    print(f"\nMean L2: {np.mean(l2_diffs):.4f}  |  Max L2: {np.max(l2_diffs):.4f}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(eval_steps, l2_diffs, marker="o")
    ax.set_xlabel(f"Context start step t  (predicting t+{N})")
    ax.set_ylabel("L2 state diff (unnormalized)")
    ax.set_title(
        f"WM {N}-step state prediction error\n"
        f"Dataset: {args.dataset}  |  Episode: {args.episode}"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(args.output_dir, f"state_diff_ep{args.episode}.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
