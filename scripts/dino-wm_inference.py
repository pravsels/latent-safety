#!/usr/bin/env python3
"""
Generate world model rollouts and save as videos.

Usage:
    python scripts/dino-wm_inference.py \
            --wm-checkpoint dino_wm_checkpoints/best_wm.pth \
            --decoder-checkpoint dino_decoder_checkpoints/testing_decoder.pth \
            --hdf5-file arx5_subset_eval.h5 \
            --dataset-stats dataset_stats.json \
            --horizon 200 \
            --reset-interval 13 \
            --num-rollouts 5
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange
import imageio.v3 as iio
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt

# Add parent directory to path to import dino_wm modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from dino_wm.test_loader import SplitTrajectoryDataset
from dino_wm.dino_decoder import VQVAE
from dino_wm.dino_models import VideoTransformer, normalize_acs, normalize_states, unnormalize_states
from dino_wm.config import MODEL_CONFIG, get_dino_config, get_decoder_image_size


def _load_state_dict_with_meta(path: str, device: str):
    """
    Load checkpoint and return (state_dict, meta_dict).
    Supports raw state_dict or dict checkpoints with common keys.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "decoder_state_dict", "state_dict"):
            if key in ckpt:
                meta = {k: v for k, v in ckpt.items() if k != key}
                return ckpt[key], meta
    return ckpt, {}


def _resolve_quantize_flag(ckpt_meta: dict, checkpoint_path: str) -> bool:
    """
    Decide whether to enable VQ codebook quantization.
    Priority:
      1) checkpoint metadata key "quantize"
      2) filename heuristic: "_vq" in basename
      3) default False
    """
    if isinstance(ckpt_meta, dict) and "quantize" in ckpt_meta:
        return bool(ckpt_meta["quantize"])
    if "_vq" in os.path.basename(checkpoint_path):
        return True
    return False


def _infer_latent_shape(hdf5_file: str, front_key: str, wrist_key: str) -> tuple[int, int]:
    """Infer (num_patches, latent_dim) from first trajectory in HDF5."""
    with h5py.File(hdf5_file, "r") as hf:
        traj_ids = sorted(list(hf.keys()))
        if not traj_ids:
            raise ValueError(f"No trajectories found in {hdf5_file}")
        traj = hf[traj_ids[0]]
        if front_key not in traj:
            raise KeyError(f"Front latent key '{front_key}' not found in trajectory '{traj_ids[0]}'")
        if wrist_key not in traj:
            raise KeyError(f"Wrist latent key '{wrist_key}' not found in trajectory '{traj_ids[0]}'")
        front_shape = tuple(traj[front_key].shape)
        wrist_shape = tuple(traj[wrist_key].shape)
    if len(front_shape) != 3 or len(wrist_shape) != 3:
        raise ValueError(
            "Expected latent tensors with shape (T, num_patches, dim). "
            f"Got front={front_shape}, wrist={wrist_shape}"
        )
    if front_shape[1:] != wrist_shape[1:]:
        raise ValueError(
            "Front and wrist latent shapes must match. "
            f"Got front={front_shape[1:]}, wrist={wrist_shape[1:]}"
        )
    return int(front_shape[1]), int(front_shape[2])


class DinoDecoderAdapter:
    def __init__(self, decoder: VQVAE):
        self.decoder = decoder

    @torch.no_grad()
    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, T, N, D) -> (B, T, H, W, C) in [0, 1]
        pred_ims, _ = self.decoder(tokens)
        pred_ims = rearrange(pred_ims, "(b t) c h w -> b t h w c", t=tokens.shape[1])
        return pred_ims.clamp(0.0, 1.0)


class WanDecoderAdapter:
    def __init__(
        self,
        *,
        model: str,
        subfolder: str,
        device: str,
        dtype: str,
        latent_h: int = 0,
        latent_w: int = 0,
    ):
        from diffusers import AutoencoderKLWan

        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        self.model_dtype = dtype_map[dtype] if self.device.type == "cuda" else torch.float32
        self.latent_h = int(latent_h)
        self.latent_w = int(latent_w)
        self.vae = AutoencoderKLWan.from_pretrained(
            model, subfolder=subfolder, torch_dtype=self.model_dtype
        ).to(self.device).eval()

    def _infer_hw(self, num_patches: int) -> tuple[int, int]:
        if self.latent_h > 0 and self.latent_w > 0:
            if self.latent_h * self.latent_w != num_patches:
                raise ValueError(
                    f"wan_latent_height*wan_latent_width ({self.latent_h*self.latent_w}) "
                    f"must equal num_patches ({num_patches})."
                )
            return self.latent_h, self.latent_w
        side = int(np.sqrt(num_patches))
        if side * side != num_patches:
            raise ValueError(
                "Cannot infer WAN latent H/W from non-square num_patches. "
                "Set --wan-latent-height and --wan-latent-width explicitly."
            )
        return side, side

    @torch.no_grad()
    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, T, N, C) -> image tensor (B, T, H, W, 3) in [0, 1]
        _, _, n, _ = tokens.shape
        h, w = self._infer_hw(n)
        z = rearrange(tokens, "b t (h w) c -> b c t h w", h=h, w=w).to(
            device=self.device, dtype=self.model_dtype
        )
        y = self.vae.decode(z).sample
        y = y.clamp(-1, 1).add(1.0).mul(0.5)
        return rearrange(y, "b c t h w -> b t h w c").float()


def generate_rollout(transition, decoder_adapter, data, context_length, horizon, device, stats, render_size, reset_interval=None):
    """
    Generate a single rollout.
    
    Args:
        transition: World model (VideoTransformer)
        decoder_adapter: Decoder adapter with decode_tokens()
        data: Batch from dataset
        context_length: Number of context frames H
        horizon: Number of rollout steps
        device: Device to run on
        stats: Dictionary containing normalization stats (action_min, action_max, etc.)
        reset_interval: If set, every N steps reset with fresh GT context (Option B: full context reset)
    
    Returns:
        gt_im1, gt_im2: Ground truth images (T, H, W, C)
        pred_im1, pred_im2: Predicted images (T, H, W, C)
    """
    H = context_length
    
    # Unpack stats
    action_min = stats['action_min'].to(device)
    action_max = stats['action_max'].to(device)
    state_min = stats['state_min'].to(device)
    state_max = stats['state_max'].to(device)
    action_q02 = stats['action_delta_q02'].to(device) if "action_delta_q02" in stats else None
    action_q98 = stats['action_delta_q98'].to(device) if "action_delta_q98" in stats else None
    state_q02 = stats['state_q02'].to(device) if "state_q02" in stats else None
    state_q98 = stats['state_q98'].to(device) if "state_q98" in stats else None
    
    # Get all ground truth data upfront
    all_data1 = data['cam_zed_embd'][[0]].to(device)
    all_data2 = data['cam_rs_embd'][[0]].to(device)
    
    # Normalize states and actions
    all_states_raw = data['state'][[0]].to(device)
    all_states = normalize_states(
        all_states_raw, state_min, state_max, q02=state_q02, q98=state_q98
    )
    
    all_acs = data['action'][[0]].to(device)
    all_acs = normalize_acs(
        all_acs, action_min, action_max, q02=action_q02, q98=action_q98
    )
    
    # Initialize context with first H frames
    inputs1 = all_data1[:, :H]
    inputs2 = all_data2[:, :H]
    inputs_states = all_states[:, :H]
    acs = all_acs[:, :H]
    
    # Initialize with context images
    im1s = data['agentview_image'][[0], :H].squeeze().to(device) / 255.  # (T, H, W, C)
    im2s = data['robot0_eye_in_hand_image'][[0], :H].squeeze().to(device) / 255.
    im1s = F.interpolate(im1s.permute(0, 3, 1, 2), size=render_size, mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
    im2s = F.interpolate(im2s.permute(0, 3, 1, 2), size=render_size, mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
    
    # Track predicted states (for plotting)
    pred_states = [all_states_raw[0, :H]]

    # Autoregressive rollout
    for k in range(horizon):
        current_idx = H + k
        
        # Check if we should reset with fresh GT context (Option B: full context reset)
        # Reset happens at the start of prediction steps: k = reset_interval, 2*reset_interval, etc.
        should_reset = (reset_interval is not None and 
                       k > 0 and 
                       k % reset_interval == 0 and
                       current_idx + H <= all_data1.shape[1])
        
        if should_reset:
            # Full context reset: give it H fresh GT frames starting from current position
            # This resets error accumulation by replacing the context with ground truth
            reset_start = current_idx
            reset_end = reset_start + H
            
            # Reset inputs with fresh GT frames (internal context reset only)
            inputs1 = all_data1[:, reset_start:reset_end]
            inputs2 = all_data2[:, reset_start:reset_end]
            inputs_states = all_states[:, reset_start:reset_end]
            acs = all_acs[:, reset_start:reset_end]
            
            # Note: We don't modify visualization here - the reset is internal
            # The prediction from this reset context will be added to visualization below
        
        # Predict next frame (using either current context or reset context)
        pred1, pred2, pred_state, _ = transition(inputs1, inputs2, inputs_states, acs)
        
        # Decode predictions
        pred_im1 = decoder_adapter.decode_tokens(pred1[:, [-1]])
        pred_im2 = decoder_adapter.decode_tokens(pred2[:, [-1]])
        pred_im1 = F.interpolate(
            pred_im1.squeeze(1).permute(0, 3, 1, 2),
            size=render_size,
            mode='bilinear',
            align_corners=False,
        ).permute(0, 2, 3, 1).unsqueeze(1)
        pred_im2 = F.interpolate(
            pred_im2.squeeze(1).permute(0, 3, 1, 2),
            size=render_size,
            mode='bilinear',
            align_corners=False,
        ).permute(0, 2, 3, 1).unsqueeze(1)
        
        im1s = torch.cat([im1s, pred_im1.squeeze(0)], dim=0)
        im2s = torch.cat([im2s, pred_im2.squeeze(0)], dim=0)
        
        # Update inputs for next step (rolling window)
        # Use GT actions from dataset (no normalization needed as they are already normalized in all_acs)
        if current_idx < all_acs.shape[1]:
            acs = torch.cat([acs[[0], 1:], all_acs[0, current_idx].unsqueeze(0).unsqueeze(0)], dim=1)
        else:
            # If we run out of GT actions, repeat the last one
            acs = torch.cat([acs[[0], 1:], acs[[0], -1:]], dim=1)
        
        inputs1 = torch.cat([inputs1[[0], 1:], pred1[:, -1].unsqueeze(1)], dim=1)
        inputs2 = torch.cat([inputs2[[0], 1:], pred2[:, -1].unsqueeze(1)], dim=1)
        # pred_state is already normalized (model output), so we can use it directly
        inputs_states = torch.cat([inputs_states[[0], 1:], pred_state[:, -1].unsqueeze(1)], dim=1)
        pred_state_raw = unnormalize_states(
            pred_state[:, -1], state_min, state_max, q02=state_q02, q98=state_q98
        )
        pred_states.append(pred_state_raw.squeeze(0).unsqueeze(0))
    
    # Get ground truth for comparison
    total_length = H + horizon
    gt_im1 = data['agentview_image'][[0], :total_length].squeeze().to(device)
    gt_im2 = data['robot0_eye_in_hand_image'][[0], :total_length].squeeze().to(device)
    gt_im1 = F.interpolate(gt_im1.permute(0, 3, 1, 2).float(), size=render_size, mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
    gt_im2 = F.interpolate(gt_im2.permute(0, 3, 1, 2).float(), size=render_size, mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
    gt_im1 = gt_im1.squeeze(0) / 255.  # (T, H, W, C)
    gt_im2 = gt_im2.squeeze(0) / 255.
    
    gt_states = all_states_raw[:, :total_length].squeeze(0)
    pred_states = torch.cat(pred_states, dim=0)[:total_length]
    return gt_im1, gt_im2, im1s, im2s, gt_states, pred_states


def create_comparison_video(gt_im1, gt_im2, pred_im1, pred_im2):
    """
    Create vertical comparison video layout:
    [GT Front]  [GT Wrist]
    ───────────────────────
    [Pred Front] [Pred Wrist]
    
    Args:
        gt_im1, gt_im2: (T, H, W, C) ground truth images (front, wrist)
        pred_im1, pred_im2: (T, H, W, C) predicted images (front, wrist)
    
    Returns:
        video: (T, H_total, W_total, C) numpy array where H_total=2H+16, W_total=2W
    """
    T = gt_im1.shape[0]
    H, W = gt_im1.shape[1], gt_im1.shape[2]
    
    # Convert to numpy
    gt_im1_np = (gt_im1.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    gt_im2_np = (gt_im2.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pred_im1_np = (pred_im1.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pred_im2_np = (pred_im2.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    
    # Concatenate cameras horizontally for each section
    gt_top = np.concatenate([gt_im1_np, gt_im2_np], axis=2)  # (T, H, 2W, C) - GT Front | GT Wrist
    pred_bottom = np.concatenate([pred_im1_np, pred_im2_np], axis=2)  # (T, H, 2W, C) - Pred Front | Pred Wrist
    
    # Create horizontal separator (white line between GT and predictions)
    # Separator height = 16 pixels, width = 2W to match concatenated images
    # This ensures total height (2H + 16) is divisible by 16
    separator_height = 16
    separator_width = 2 * W  # Match the width of concatenated images
    separator = np.ones((T, separator_height, separator_width, 3), dtype=np.uint8) * 255
    
    # Stack vertically: GT on top, separator, predictions on bottom
    video = np.concatenate([gt_top, separator, pred_bottom], axis=1)  # (T, 2H+16, 2W, C)
    
    return video


def plot_state_rollout(gt_states: torch.Tensor, pred_states: torch.Tensor, output_path: str):
    """
    Plot per-dimension state rollouts and an L2 error curve.
    gt_states/pred_states: (T, D) tensors on any device.
    """
    gt = gt_states.detach().cpu().numpy()
    pred = pred_states.detach().cpu().numpy()
    T, D = gt.shape
    err = np.linalg.norm(gt - pred, axis=1)

    num_plots = D + 1
    ncols = 3
    nrows = int(np.ceil(num_plots / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 2.8 * nrows), squeeze=False)
    axes = axes.flatten()

    t = np.arange(T)
    for i in range(D):
        ax = axes[i]
        ax.plot(t, gt[:, i], label="gt", linewidth=1.2)
        ax.plot(t, pred[:, i], label="pred", linewidth=1.2, alpha=0.8)
        ax.set_title(f"state[{i}]")
        ax.grid(True, alpha=0.3)

    ax = axes[D]
    ax.plot(t, err, color="tab:red", linewidth=1.2)
    ax.set_title("L2 error")
    ax.grid(True, alpha=0.3)

    for j in range(D + 1, len(axes)):
        axes[j].axis("off")

    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate world model rollouts")
    parser.add_argument("--wm-checkpoint", type=str, required=True,
                       help="Path to world model checkpoint")
    parser.add_argument("--decoder-checkpoint", type=str, default=None,
                       help="Path to DINO decoder checkpoint (required when --backbone=dino)")
    parser.add_argument("--hdf5-file", type=str, required=True,
                       help="Path to HDF5 dataset file")
    parser.add_argument("--dataset-stats", type=str, required=True,
                       help="Path to dataset statistics JSON file")
    parser.add_argument("--backbone", type=str, default="dino", choices=["dino", "wan"],
                       help="Backbone flow to use: dino or wan (default: dino)")
    parser.add_argument("--dino-version", type=str, default="v3", choices=["v2", "v3"],
                       help="DINO version used when --backbone=dino (default: v3)")
    parser.add_argument("--front-latent-key", type=str, default="cam_zed_embd",
                       help="HDF5 key for front camera latents (default: cam_zed_embd)")
    parser.add_argument("--wrist-latent-key", type=str, default="cam_rs_embd",
                       help="HDF5 key for wrist camera latents (default: cam_rs_embd)")
    parser.add_argument("--wan-vae-model", type=str, default=None,
                       help="Diffusers WAN VAE model id/path (required when --backbone=wan)")
    parser.add_argument("--wan-vae-subfolder", type=str, default="vae",
                       help="WAN VAE subfolder (default: vae)")
    parser.add_argument("--wan-vae-dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"],
                       help="WAN VAE dtype (default: bf16)")
    parser.add_argument("--wan-latent-height", type=int, default=0,
                       help="WAN latent height for decode (0 = infer square)")
    parser.add_argument("--wan-latent-width", type=int, default=0,
                       help="WAN latent width for decode (0 = infer square)")
    parser.add_argument("--horizon", type=int, default=10,
                       help="Rollout horizon (default: 10)")
    parser.add_argument("--context-length", type=int, default=3,
                       help="Context length H (default: 3)")
    parser.add_argument("--sequence-length", type=int, default=4,
                       help="Sequence length used during training (default: 4). This determines num_frames=sequence_length-1.")
    parser.add_argument("--action-horizon", type=int, default=100,
                       help="Action horizon used by VideoTransformer checkpoint compatibility (default: 100).")
    parser.add_argument("--reset-interval", type=int, default=None,
                       help="Reset with fresh GT context every N steps (Option B: full context reset). "
                            "Useful for long rollouts to prevent error accumulation. Default: None (no resets).")
    parser.add_argument("--num-rollouts", type=int, default=5,
                       help="Number of rollouts to generate (default: 5)")
    parser.add_argument("--output-dir", type=str, default="rollout_videos",
                       help="Directory to save videos (default: rollout_videos)")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use (default: cuda:0)")
    parser.add_argument("--fps", type=int, default=20,
                       help="Video FPS (default: 20)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for trajectory selection. Use the same seed for different checkpoints to ensure same trajectories are used for comparison (default: None, random)")
    parser.add_argument("--quantize", action="store_true",
                        help="Enable VQ codebook quantization (must match training setting)")
    parser.add_argument("--render-height", type=int, default=224,
                        help="Rendered comparison image height (default: 224)")
    parser.add_argument("--render-width", type=int, default=224,
                        help="Rendered comparison image width (default: 224)")
    return parser.parse_args(argv)


def validate_args(args):
    if args.backbone == "dino" and not args.decoder_checkpoint:
        raise ValueError("--decoder-checkpoint is required when --backbone=dino")
    if args.backbone == "wan" and not args.wan_vae_model:
        raise ValueError("--wan-vae-model is required when --backbone=wan")
    if args.render_height <= 0 or args.render_width <= 0:
        raise ValueError("--render-height and --render-width must be > 0")


def main():
    args = parse_args()
    validate_args(args)
    
    # Set random seed for reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        import random
        random.seed(args.seed)
        print(f"Using random seed: {args.seed} (for reproducible trajectory selection)")
    
    device = args.device
    render_size = (int(args.render_height), int(args.render_width))
    
    # Load dataset stats
    print(f"Loading dataset stats from {args.dataset_stats}")
    with open(args.dataset_stats, 'r') as f:
        stats_data = json.load(f)
        
    # Create stats tensors
    stats = {
        'action_min': torch.tensor(stats_data['action_min']).float().to(device),
        'action_max': torch.tensor(stats_data['action_max']).float().to(device),
        'state_min': torch.tensor(stats_data['state_min']).float().to(device),
        'state_max': torch.tensor(stats_data['state_max']).float().to(device),
    }
    if "action_delta_q02" in stats_data:
        stats["action_delta_q02"] = torch.tensor(stats_data["action_delta_q02"]).float().to(device)
    if "action_delta_q98" in stats_data:
        stats["action_delta_q98"] = torch.tensor(stats_data["action_delta_q98"]).float().to(device)
    if "state_q02" in stats_data:
        stats["state_q02"] = torch.tensor(stats_data["state_q02"]).float().to(device)
    if "state_q98" in stats_data:
        stats["state_q98"] = torch.tensor(stats_data["state_q98"]).float().to(device)
    
    # Infer dimensions from stats
    state_dim = len(stats_data['state_min'])
    action_dim = len(stats_data['action_min'])
    print(f"Inferred state_dim={state_dim}, action_dim={action_dim} from stats")
    
    # Load models
    print("Loading models...")
    backbone = str(args.backbone)
    latent_num_patches = None
    if backbone == "dino":
        dino_cfg = get_dino_config(args.dino_version)
        MODEL_CONFIG['dim'] = int(dino_cfg['dim'])
        MODEL_CONFIG['image_size'] = get_decoder_image_size(args.dino_version)
        dec_state, dec_meta = _load_state_dict_with_meta(args.decoder_checkpoint, device)
        quantize = bool(args.quantize) if args.quantize else _resolve_quantize_flag(dec_meta, args.decoder_checkpoint)
        decoder = VQVAE(quantize=quantize).to(device)
        if quantize:
            print("VQ codebook quantization enabled")
        else:
            print("VQ codebook quantization disabled (standard autoencoder)")
        decoder.load_state_dict(dec_state)
        decoder.eval()
        decoder_adapter = DinoDecoderAdapter(decoder)
    else:
        latent_num_patches, latent_dim = _infer_latent_shape(
            args.hdf5_file, args.front_latent_key, args.wrist_latent_key
        )
        MODEL_CONFIG['dim'] = int(latent_dim)
        MODEL_CONFIG['image_size'] = (224, 224)
        decoder_adapter = WanDecoderAdapter(
            model=args.wan_vae_model,
            subfolder=args.wan_vae_subfolder,
            device=device,
            dtype=args.wan_vae_dtype,
            latent_h=args.wan_latent_height,
            latent_w=args.wan_latent_width,
        )
        print(
            f"WAN backbone active: latent num_patches={latent_num_patches}, dim={latent_dim}, model={args.wan_vae_model}"
        )

    transition = VideoTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        num_frames=args.sequence_length - 1,  # Must match training: sequence_length - 1
        action_horizon=int(args.action_horizon),
        backbone=backbone,
        dino_version=args.dino_version,
        num_patches=latent_num_patches,
        **MODEL_CONFIG
    ).to(device)
    wm_state, _ = _load_state_dict_with_meta(args.wm_checkpoint, device)
    transition.load_state_dict(wm_state)
    transition.eval()
    
    # Setup dataset
    print("Loading dataset...")
    
    # Use split='train' with num_test=0 to load ALL trajectories
    # Logic: ids[num_test:] -> ids[0:] -> All trajectories
    dataset = SplitTrajectoryDataset(
        args.hdf5_file,
        segment_length=args.context_length + args.horizon,
        split='train',
        num_test=0,
        front_embd_key=args.front_latent_key,
        wrist_embd_key=args.wrist_latent_key,
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Select trajectory indices deterministically if seed is set
    if args.seed is not None:
        # Use seed to deterministically select trajectory indices
        np.random.seed(args.seed)
        num_trajectories = len(dataset)
        # Select random but deterministic indices (without replacement)
        if args.num_rollouts <= num_trajectories:
            trajectory_indices = np.random.choice(num_trajectories, size=args.num_rollouts, replace=False)
        else:
            # If we need more rollouts than trajectories, allow replacement but still deterministic
            trajectory_indices = np.random.choice(num_trajectories, size=args.num_rollouts, replace=True)
        trajectory_indices = sorted(trajectory_indices.tolist())  # Sort for deterministic order
        print(f"Using seed {args.seed}: Selected trajectory indices {trajectory_indices}")
    else:
        # Random selection each time - use DataLoader with shuffle
        trajectory_indices = None
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        dataloader_iter = iter(dataloader)
    
    print(f"Generating {args.num_rollouts} rollouts...")
    
    for i in tqdm(range(args.num_rollouts), desc="Rollouts"):
        if trajectory_indices is not None:
            # Get specific trajectory by index (deterministic)
            data = dataset[trajectory_indices[i]]
            # Wrap in dict format to match what DataLoader returns
            if isinstance(data, dict):
                # Already in dict format, just ensure batch dimension
                data = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) and v.dim() > 0 else v for k, v in data.items()}
            else:
                # If dataset returns something else, wrap it
                data = {'data': data.unsqueeze(0) if isinstance(data, torch.Tensor) else data}
        else:
            # Random selection - get next from iterator
            try:
                data = next(dataloader_iter)
            except StopIteration:
                # Reset iterator if we run out
                dataloader_iter = iter(dataloader)
                data = next(dataloader_iter)
        
        with torch.no_grad():
            gt_im1, gt_im2, pred_im1, pred_im2, gt_states, pred_states = generate_rollout(
                transition,
                decoder_adapter,
                data,
                args.context_length,
                args.horizon,
                device,
                stats,
                render_size,
                args.reset_interval,
            )
            
            video = create_comparison_video(gt_im1, gt_im2, pred_im1, pred_im2)
            
            # Save video
            output_path = os.path.join(args.output_dir, f"rollout_{i:03d}.mp4")
            # imageio expects (T, H, W, C) format
            iio.imwrite(output_path, video, fps=args.fps, codec='libx264', pixelformat='yuv420p')
            print(f"Saved: {output_path}")

            # Save state rollout plot
            state_plot_path = os.path.join(args.output_dir, f"state_rollout_{i:03d}.png")
            plot_state_rollout(gt_states, pred_states, state_plot_path)
            print(f"Saved: {state_plot_path}")
    
    print(f"Done! Videos saved to {args.output_dir}")


if __name__ == "__main__":
    main()
