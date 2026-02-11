#!/usr/bin/env python3
"""
Evaluate the future-action-horizon world model by sweeping K from 1..max_horizon.

For each horizon K:
  - Feed H context frames + K future delta-actions to the world model
  - Compare predicted latent / state at t+K+1 against ground truth
  - Record latent MSE (front + wrist) and state MSE

Outputs two plots: latent_mse_vs_horizon.png and state_mse_vs_horizon.png.

Data source is a LeRobot HF dataset (one episode).  DINO features and
delta-actions are computed on the fly.

Example:
    python scripts/eval_horizon_sweep.py \
        --hf-repo villekuosmanen/bin_pick_pack_coffee_capsules_eval \
        --episode 0 \
        --wm-checkpoint ./checkpoints/dino3_wm_checkpoints/latest_wm.pth \
        --dataset-stats ./checkpoints/arx5_datasets_6Feb_26_stats.json \
        --out-dir outputs/horizon_sweep
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure repo root is on sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dino_wm.config import MODEL_CONFIG, get_dino_config
from dino_wm.data_utils import compute_action_deltas
from dino_wm.dino_models import VideoTransformer, normalize_acs, normalize_states
from scripts.utils import get_dino_model, preprocess_images_for_dino


# ---------------------------------------------------------------------------
# LeRobot data loading
# ---------------------------------------------------------------------------

def _get_episode_range(dataset, ep_idx: int) -> tuple[int, int]:
    """Return (start, end) frame indices for an episode."""
    if hasattr(dataset, "episode_data_index"):
        return (
            int(dataset.episode_data_index["from"][ep_idx].item()),
            int(dataset.episode_data_index["to"][ep_idx].item()),
        )
    if hasattr(dataset, "meta") and hasattr(dataset.meta, "episodes"):
        ep = dataset.meta.episodes
        info = ep.get(ep_idx) if isinstance(ep, dict) else ep[ep_idx]
        return int(info["dataset_from_index"]), int(info["dataset_to_index"])
    raise AttributeError("Cannot determine episode range from dataset")


def load_episode(
    repo_id: str,
    episode: int,
    device: str,
    dino_model: torch.nn.Module,
    batch_size: int = 32,
) -> dict[str, torch.Tensor]:
    """
    Load a single episode from a LeRobot HF dataset and return:
      front_embd  (T, num_patches, dim)   – DINO features for front camera
      wrist_embd  (T, num_patches, dim)   – DINO features for wrist camera
      actions_delta (T, action_dim)        – delta actions
      states      (T, state_dim)           – raw states
    Embeddings are kept on CPU to avoid OOM; moved to device during evaluation.
    Actions/states are on *device*.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id, episodes=[episode], video_backend="pyav")
    start_idx, end_idx = _get_episode_range(dataset, episode)
    T = end_idx - start_idx

    # --- Collect tabular data (actions, states, timestamps) ----------------
    batch = dataset.hf_dataset[start_idx:end_idx]

    # Auto-detect action & state keys
    action_candidates = [
        "action", "action.pos", "action.position",
        "action.eef_pose", "action.velocity", "action.effort",
    ]
    state_candidates = [
        "observation.state", "observation.state.pos",
        "observation.state.eef_pose", "observation.state.velocity",
        "observation.state.effort",
    ]
    action_key = next((k for k in action_candidates if k in batch), None)
    state_key = next((k for k in state_candidates if k in batch), None)
    if action_key is None:
        available = [k for k in batch.keys() if "action" in k.lower()]
        raise KeyError(f"No action key found. Available keys with 'action': {available}")
    if state_key is None:
        available = [k for k in batch.keys() if "state" in k.lower()]
        raise KeyError(f"No state key found. Available keys with 'state': {available}")
    print(f"  Using action_key={action_key!r}, state_key={state_key!r}")

    actions_np = np.array(batch[action_key], dtype=np.float32)
    states_np = np.array(batch[state_key], dtype=np.float32)
    actions_delta_np = compute_action_deltas(actions_np, states_np)

    timestamps = batch["timestamp"]
    batch_timestamps = [np.float64(t) for t in timestamps]

    # --- DINO encode in chunks (load video frames chunk-by-chunk) ----------
    front_embds: list[torch.Tensor] = []
    wrist_embds: list[torch.Tensor] = []

    dino_model.eval()
    for i in range(0, T, batch_size):
        j = min(i + batch_size, T)
        chunk_ts = batch_timestamps[i:j]

        # Query video frames for this chunk only
        query = {k: chunk_ts for k in dataset.meta.video_keys}
        orig_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            video_chunk = dataset._query_videos(query, episode)
        finally:
            torch.set_default_dtype(orig_dtype)

        front_chunk = video_chunk["observation.images.front"].to(device, dtype=torch.float32)
        wrist_chunk = video_chunk["observation.images.wrist"].to(device, dtype=torch.float32)

        with torch.no_grad():
            f_prep = preprocess_images_for_dino(front_chunk, is_front_camera=True)
            w_prep = preprocess_images_for_dino(wrist_chunk, is_front_camera=False)
            front_embds.append(
                dino_model.forward_features(f_prep)["x_norm_patchtokens"].cpu()
            )
            wrist_embds.append(
                dino_model.forward_features(w_prep)["x_norm_patchtokens"].cpu()
            )

        # Free GPU memory between chunks
        del front_chunk, wrist_chunk, f_prep, w_prep, video_chunk
        torch.cuda.empty_cache()
        print(f"  DINO encoded frames {i}–{j-1} / {T}")

    return {
        "front_embd": torch.cat(front_embds, dim=0),                     # (T, P, D) on CPU
        "wrist_embd": torch.cat(wrist_embds, dim=0),                     # (T, P, D) on CPU
        "actions_delta": torch.from_numpy(actions_delta_np).to(device),   # (T, A) on device
        "states": torch.from_numpy(states_np).to(device),                 # (T, S) on device
    }


# ---------------------------------------------------------------------------
# Checkpoint loading (same helper used by inference scripts)
# ---------------------------------------------------------------------------

def _load_state_dict_with_meta(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "decoder_state_dict", "state_dict"):
            if key in ckpt:
                meta = {k: v for k, v in ckpt.items() if k != key}
                return ckpt[key], meta
    return ckpt, {}


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_horizon(
    transition: VideoTransformer,
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
    """
    Sweep K = 1 … max_horizon and compute MSEs over sliding windows.

    Returns dict with keys:
        horizons    (max_horizon,)  – 1 … max_horizon
        latent_mse_mean / latent_mse_std
        state_mse_mean  / state_mse_std
    """
    # Embeddings may be on CPU to save GPU memory; actions/states on device.
    front_cpu = episode_data["front_embd"]   # (T, P, D) possibly CPU
    wrist_cpu = episode_data["wrist_embd"]
    acs   = episode_data["actions_delta"]    # on device
    sts   = episode_data["states"]           # on device
    T = front_cpu.shape[0]

    # Normalisation tensors
    action_min = stats["action_min"]
    action_max = stats["action_max"]
    state_min  = stats["state_min"]
    state_max  = stats["state_max"]
    action_q02 = stats.get("action_delta_q02")
    action_q98 = stats.get("action_delta_q98")
    state_q02  = stats.get("state_q02")
    state_q98  = stats.get("state_q98")

    # Pre-normalise everything once (on device)
    norm_acs = normalize_acs(acs, action_min, action_max, q02=action_q02, q98=action_q98)
    norm_sts = normalize_states(sts, state_min, state_max, q02=state_q02, q98=state_q98)

    # Context frame indices (spaced by pred_step)
    ctx_idx_dev = torch.arange(context_length, device=device, dtype=torch.long) * pred_step
    ctx_idx_cpu = ctx_idx_dev.cpu()
    t = int(ctx_idx_dev[-1].item())  # last context raw-frame index

    transition.eval()

    latent_mses: list[list[float]] = []   # [K][window]
    state_mses:  list[list[float]] = []

    for K in range(1, max_horizon + 1):
        target_idx = t + K + 1
        min_len = target_idx + 1  # episode must be at least this long

        k_latent: list[float] = []
        k_state:  list[float] = []

        # Slide the starting offset through the episode
        for offset in range(0, T - min_len + 1, stride):
            # Context embeddings — index on CPU, move to device
            in_front = front_cpu[offset + ctx_idx_cpu].unsqueeze(0).to(device)  # (1, H, P, D)
            in_wrist = wrist_cpu[offset + ctx_idx_cpu].unsqueeze(0).to(device)
            in_state = norm_sts[offset + ctx_idx_dev].unsqueeze(0)
            in_acs   = norm_acs[offset + ctx_idx_dev].unsqueeze(0)

            # Future actions: a_{t+1} .. a_{t+K}  (or None for ablation)
            if no_future_actions:
                future_actions = None
            else:
                fa_start = offset + t + 1
                fa_end   = offset + t + 1 + K
                future_actions = norm_acs[fa_start:fa_end].unsqueeze(0)  # (1, K, A)

            pred_front, pred_wrist, pred_state, _ = transition(
                in_front, in_wrist, in_state, in_acs, future_actions,
            )

            # Ground truth at target_idx — move embeddings to device for comparison
            gt_front = front_cpu[offset + target_idx].to(device)  # (P, D)
            gt_wrist = wrist_cpu[offset + target_idx].to(device)
            gt_state = norm_sts[offset + target_idx]

            # Latent MSE (average of front + wrist)
            mse_front = torch.mean((pred_front[0, -1] - gt_front) ** 2).item()
            mse_wrist = torch.mean((pred_wrist[0, -1] - gt_wrist) ** 2).item()
            k_latent.append((mse_front + mse_wrist) / 2.0)

            # State MSE
            mse_state = torch.mean((pred_state[0, -1] - gt_state) ** 2).item()
            k_state.append(mse_state)

        latent_mses.append(k_latent)
        state_mses.append(k_state)
        if K % 10 == 0 or K == 1:
            n = len(k_latent)
            print(f"  K={K:3d}  windows={n}  latent_mse={np.mean(k_latent):.6f}" if n else f"  K={K:3d}  no valid windows")

    # Aggregate
    horizons = np.arange(1, max_horizon + 1)
    latent_means = np.array([np.mean(v) if v else np.nan for v in latent_mses])
    latent_stds  = np.array([np.std(v) if v else np.nan for v in latent_mses])
    state_means  = np.array([np.mean(v) if v else np.nan for v in state_mses])
    state_stds   = np.array([np.std(v) if v else np.nan for v in state_mses])

    return {
        "horizons": horizons,
        "latent_mse_mean": latent_means,
        "latent_mse_std": latent_stds,
        "state_mse_mean": state_means,
        "state_mse_std": state_stds,
        "num_windows": [len(v) for v in latent_mses],
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _annotate_endpoints(ax, h, values, color):
    """Add text labels at K=1 and K=max with exact values."""
    v_first, v_last = values[0], values[-1]
    ax.annotate(
        f"{v_first:.4f}",
        xy=(h[0], v_first),
        xytext=(h[0] + 3, v_first),
        fontsize=10, fontweight="bold", color=color,
        arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
        va="center",
    )
    ax.annotate(
        f"{v_last:.4f}",
        xy=(h[-1], v_last),
        xytext=(h[-1] - 3, v_last),
        fontsize=10, fontweight="bold", color=color,
        ha="right", va="center",
        arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
    )


def plot_sweep(
    results: dict[str, np.ndarray],
    out_dir: Path,
    title_suffix: str = "",
    tag: str = "",
) -> None:
    """Plot latent and state MSE vs horizon. *tag* is inserted into filenames."""
    h = results["horizons"]
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""

    # --- Latent MSE ---
    fig, ax = plt.subplots(figsize=(10, 5))
    mean = results["latent_mse_mean"]
    std = results["latent_mse_std"]
    ax.plot(h, mean, color="tab:blue", linewidth=1.5)
    ax.fill_between(h, mean - std, mean + std, alpha=0.25, color="tab:blue")
    _annotate_endpoints(ax, h, mean, "tab:blue")
    ax.set_xlabel("Future Action Horizon (K)")
    ax.set_ylabel("Latent MSE (DINO feature space)")
    ax.set_title(f"Latent MSE vs Future Action Horizon{title_suffix}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"latent_mse_vs_horizon{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    # --- State MSE ---
    fig, ax = plt.subplots(figsize=(10, 5))
    mean = results["state_mse_mean"]
    std = results["state_mse_std"]
    ax.plot(h, mean, color="tab:orange", linewidth=1.5)
    ax.fill_between(h, mean - std, mean + std, alpha=0.25, color="tab:orange")
    _annotate_endpoints(ax, h, mean, "tab:orange")
    ax.set_xlabel("Future Action Horizon (K)")
    ax.set_ylabel("State MSE (normalised)")
    ax.set_title(f"State MSE vs Future Action Horizon{title_suffix}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"state_mse_vs_horizon{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Evaluate WM prediction quality across future-action horizons",
    )
    p.add_argument("--hf-repo", default=None, help="LeRobot HF dataset repo id (required unless --replot)")
    p.add_argument("--episode", type=int, default=0, help="Episode index (default: 0)")
    p.add_argument("--wm-checkpoint", default=None, help="Path to world model checkpoint (required unless --replot)")
    p.add_argument("--dataset-stats", default=None, help="Path to dataset_stats.json (required unless --replot)")
    p.add_argument("--context-length", type=int, default=3, help="Context length H (default: 3)")
    p.add_argument("--pred-step", type=int, default=5, help="Prediction step in raw frames (default: 5)")
    p.add_argument("--max-horizon", type=int, default=100, help="Max future-action horizon K (default: 100)")
    p.add_argument("--stride", type=int, default=10,
                   help="Stride (in raw frames) for sliding the context window across the episode (default: 10)")
    p.add_argument("--no-future-actions", action="store_true",
                   help="Ablation: pass zeros instead of real future actions (tests whether trajectory encoder matters)")
    p.add_argument("--dino-version", type=str, default="v3", choices=["v2", "v3"],
                   help="DINO version (default: v3)")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out-dir", type=str, default="outputs/horizon_sweep")
    p.add_argument("--dino-batch-size", type=int, default=64,
                   help="Batch size for DINO feature extraction (default: 64)")
    p.add_argument("--replot", type=str, default=None, metavar="JSON_PATH",
                   help="Skip evaluation; just re-plot from an existing results JSON file")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)

    # --- Replot mode: regenerate PNGs from existing JSON -------------------
    if args.replot:
        print(f"Re-plotting from {args.replot}")
        with open(args.replot) as f:
            data = json.load(f)
        results = {
            "horizons": np.array(data["horizons"]),
            "latent_mse_mean": np.array(data["latent_mse_mean"]),
            "latent_mse_std": np.array(data["latent_mse_std"]),
            "state_mse_mean": np.array(data["state_mse_mean"]),
            "state_mse_std": np.array(data["state_mse_std"]),
        }
        # Auto-detect mode from filename or CLI flag
        is_ablation = args.no_future_actions or "no_future_actions" in os.path.basename(args.replot)
        tag = "no_future_actions" if is_ablation else "with_future_actions"
        mode_label = "no future actions (ablation)" if is_ablation else "with future actions"
        ep = data.get("episode", "?")
        ps = data.get("pred_step", "?")
        title_suffix = f"  (ep {ep}, pred_step={ps}, {mode_label})"
        plot_sweep(results, out_dir, title_suffix=title_suffix, tag=tag)
        return 0

    # Validate required args for full eval mode
    if not args.hf_repo:
        raise SystemExit("error: --hf-repo is required (unless using --replot)")
    if not args.wm_checkpoint:
        raise SystemExit("error: --wm-checkpoint is required (unless using --replot)")
    if not args.dataset_stats:
        raise SystemExit("error: --dataset-stats is required (unless using --replot)")

    device = args.device if torch.cuda.is_available() else "cpu"

    # --- Load stats --------------------------------------------------------
    print(f"Loading dataset stats from {args.dataset_stats}")
    with open(args.dataset_stats) as f:
        stats_raw = json.load(f)

    stats: dict[str, torch.Tensor] = {}
    for key in ("action_min", "action_max", "state_min", "state_max",
                "action_delta_q02", "action_delta_q98", "state_q02", "state_q98"):
        if key in stats_raw:
            stats[key] = torch.tensor(stats_raw[key], dtype=torch.float32, device=device)

    state_dim = len(stats_raw["state_min"])
    action_dim = len(stats_raw["action_min"])
    print(f"  state_dim={state_dim}, action_dim={action_dim}")

    # --- Configure DINO / model dims --------------------------------------
    dino_cfg = get_dino_config(args.dino_version)
    MODEL_CONFIG["dim"] = dino_cfg["dim"]

    # --- Load DINO encoder -------------------------------------------------
    print("Loading DINO encoder …")
    dino_model = get_dino_model(device, version=args.dino_version)

    # --- Load episode ------------------------------------------------------
    print(f"Loading episode {args.episode} from {args.hf_repo} …")
    episode_data = load_episode(
        args.hf_repo,
        args.episode,
        device,
        dino_model,
        batch_size=args.dino_batch_size,
    )
    T = episode_data["front_embd"].shape[0]
    print(f"  Episode has {T} frames")

    # --- Load world model --------------------------------------------------
    print(f"Loading world model from {args.wm_checkpoint} …")
    # action_horizon must match the value in dino_wm_config.yaml (used at training time).
    # The model pads shorter future-action tensors internally, so we can sweep
    # K < action_horizon freely.
    from dino_wm.dino_models import FUTURE_ACTION_HORIZON_MAX

    transition = VideoTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        num_frames=args.context_length,
        action_horizon=FUTURE_ACTION_HORIZON_MAX,
        dino_version=args.dino_version,
        **MODEL_CONFIG,
    ).to(device)

    wm_sd, _ = _load_state_dict_with_meta(args.wm_checkpoint, device)
    transition.load_state_dict(wm_sd, strict=True)
    transition.eval()
    print("  World model loaded.")

    # --- Run sweep ---------------------------------------------------------
    print(f"Running horizon sweep K=1..{args.max_horizon} "
          f"(context_length={args.context_length}, pred_step={args.pred_step}, "
          f"stride={args.stride}) …")
    if args.no_future_actions:
        print("ABLATION MODE: future actions zeroed out")

    results = evaluate_horizon(
        transition,
        episode_data,
        stats,
        context_length=args.context_length,
        pred_step=args.pred_step,
        max_horizon=args.max_horizon,
        stride=args.stride,
        device=device,
        no_future_actions=args.no_future_actions,
    )

    # --- Save results ------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)

    # File tag: distinguishes normal vs ablation outputs in the same directory
    tag = "no_future_actions" if args.no_future_actions else "with_future_actions"

    # JSON for later analysis
    json_path = out_dir / f"horizon_sweep_results_{tag}.json"
    json_payload = {
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
    json_path.write_text(json.dumps(json_payload, indent=2))
    print(f"Saved: {json_path}")

    mode_label = "no future actions (ablation)" if args.no_future_actions else "with future actions"
    title_suffix = f"  (ep {args.episode}, pred_step={args.pred_step}, {mode_label})"
    plot_sweep(results, out_dir, title_suffix=title_suffix, tag=tag)

    # Quick summary
    print(f"\nSummary (K=1 → K={args.max_horizon}):")
    print(f"  Latent MSE: {results['latent_mse_mean'][0]:.6f} → {results['latent_mse_mean'][-1]:.6f}")
    print(f"  State  MSE: {results['state_mse_mean'][0]:.6f} → {results['state_mse_mean'][-1]:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
