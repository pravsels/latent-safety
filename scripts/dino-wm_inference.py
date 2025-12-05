#!/usr/bin/env python3
"""
Generate world model rollouts and save as videos.

Usage:
    python scripts/dino-wm_inference.py \
            --wm-checkpoint dino_wm_checkpoints/best_wm.pth \
            --decoder-checkpoint dino_decoder_checkpoints/testing_decoder.pth \
            --hdf5-file arx5_subset_eval.h5 \
            --horizon 200 \
            --reset-interval 13 \
            --num-rollouts 5
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange
import imageio.v3 as iio
from tqdm import tqdm

# Add parent directory to path to import dino_wm modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from dino_wm.test_loader import SplitTrajectoryDataset
from dino_wm.dino_decoder import VQVAE
from dino_wm.dino_models import VideoTransformer


def generate_rollout(transition, decoder, data, context_length, horizon, device, reset_interval=None):
    """
    Generate a single rollout.
    
    Args:
        transition: World model (VideoTransformer)
        decoder: VQVAE decoder
        data: Batch from dataset
        context_length: Number of context frames H
        horizon: Number of rollout steps
        device: Device to run on
        reset_interval: If set, every N steps reset with fresh GT context (Option B: full context reset)
    
    Returns:
        gt_im1, gt_im2: Ground truth images (T, H, W, C)
        pred_im1, pred_im2: Predicted images (T, H, W, C)
    """
    H = context_length
    
    # Get all ground truth data upfront
    all_data1 = data['cam_zed_embd'][[0]].to(device)
    all_data2 = data['cam_rs_embd'][[0]].to(device)
    all_states = data['state'][[0]].to(device)
    all_acs = data['action'][[0]].to(device)
    
    # Initialize context with first H frames
    inputs1 = all_data1[:, :H]
    inputs2 = all_data2[:, :H]
    inputs_states = all_states[:, :H]
    acs = all_acs[:, :H]
    
    # Initialize with context images
    im1s = data['agentview_image'][[0], :H].squeeze().to(device) / 255.  # (T, H, W, C)
    im2s = data['robot0_eye_in_hand_image'][[0], :H].squeeze().to(device) / 255.
    im1s = F.interpolate(im1s.permute(0, 3, 1, 2), size=(224, 224), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
    im2s = F.interpolate(im2s.permute(0, 3, 1, 2), size=(224, 224), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
    
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
        pred_latent = torch.cat([pred1[:, [-1]], pred2[:, [-1]]], dim=0)
        pred_ims, _ = decoder(pred_latent)
        pred_ims = rearrange(pred_ims, "(b t) c h w -> b t h w c", t=1)
        pred_im1, pred_im2 = torch.split(pred_ims, [inputs1.shape[0], inputs2.shape[0]], dim=0)
        
        im1s = torch.cat([im1s, pred_im1.squeeze(0)], dim=0)
        im2s = torch.cat([im2s, pred_im2.squeeze(0)], dim=0)
        
        # Update inputs for next step (rolling window)
        # Use GT actions from dataset (no normalization)
        if current_idx < all_acs.shape[1]:
            acs = torch.cat([acs[[0], 1:], all_acs[0, current_idx].unsqueeze(0).unsqueeze(0)], dim=1)
        else:
            # If we run out of GT actions, repeat the last one
            acs = torch.cat([acs[[0], 1:], acs[[0], -1:]], dim=1)
        
        inputs1 = torch.cat([inputs1[[0], 1:], pred1[:, -1].unsqueeze(1)], dim=1)
        inputs2 = torch.cat([inputs2[[0], 1:], pred2[:, -1].unsqueeze(1)], dim=1)
        inputs_states = torch.cat([inputs_states[[0], 1:], pred_state[:, -1].unsqueeze(1)], dim=1)
    
    # Get ground truth for comparison
    total_length = H + horizon
    gt_im1 = data['agentview_image'][[0], :total_length].squeeze().to(device)
    gt_im2 = data['robot0_eye_in_hand_image'][[0], :total_length].squeeze().to(device)
    gt_im1 = F.interpolate(gt_im1.permute(0, 3, 1, 2).float(), size=(224, 224), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
    gt_im2 = F.interpolate(gt_im2.permute(0, 3, 1, 2).float(), size=(224, 224), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
    gt_im1 = gt_im1.squeeze(0) / 255.  # (T, H, W, C)
    gt_im2 = gt_im2.squeeze(0) / 255.
    
    return gt_im1, gt_im2, im1s, im2s


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


def main():
    parser = argparse.ArgumentParser(description="Generate world model rollouts")
    parser.add_argument("--wm-checkpoint", type=str, required=True,
                       help="Path to world model checkpoint")
    parser.add_argument("--decoder-checkpoint", type=str, required=True,
                       help="Path to decoder checkpoint")
    parser.add_argument("--hdf5-file", type=str, required=True,
                       help="Path to HDF5 dataset file")
    parser.add_argument("--horizon", type=int, default=10,
                       help="Rollout horizon (default: 10)")
    parser.add_argument("--context-length", type=int, default=3,
                       help="Context length H (default: 3)")
    parser.add_argument("--sequence-length", type=int, default=4,
                       help="Sequence length used during training (default: 4). This determines num_frames=sequence_length-1.")
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
    parser.add_argument("--state-dim", type=int, default=7,
                       help="State dimension (default: 7 for ARX5)")
    parser.add_argument("--action-dim", type=int, default=None,
                       help="Action dimension (default: infer from dataset)")
    
    args = parser.parse_args()
    
    device = args.device
    
    # Infer action dimension from dataset if not provided
    if args.action_dim is None:
        import h5py
        with h5py.File(args.hdf5_file, 'r') as hf:
            first_traj = list(hf.keys())[0]
            action_dim = hf[first_traj]['actions'].shape[1]
        print(f"Inferred action_dim={action_dim} from dataset")
    else:
        action_dim = args.action_dim
    
    # Load models
    print("Loading models...")
    decoder = VQVAE().to(device)
    decoder.load_state_dict(torch.load(args.decoder_checkpoint, map_location=device, weights_only=False))
    decoder.eval()
    
    transition = VideoTransformer(
        image_size=(224, 224),
        dim=384,
        ac_dim=10,  # Action embedding dimension
        state_dim=args.state_dim,
        action_dim=action_dim,
        depth=6,
        heads=16,
        mlp_dim=2048,
        num_frames=args.sequence_length - 1,  # Must match training: sequence_length - 1
        dropout=0.1
    ).to(device)
    transition.load_state_dict(torch.load(args.wm_checkpoint, map_location=device, weights_only=False))
    transition.eval()
    
    # Setup dataset
    print("Loading dataset...")
    dataset = SplitTrajectoryDataset(
        args.hdf5_file,
        segment_length=args.context_length + args.horizon,
        split='test',
        num_test=100  # Use test split
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating {args.num_rollouts} rollouts...")
    for i in tqdm(range(args.num_rollouts), desc="Rollouts"):
        data = next(iter(dataloader))
        
        with torch.no_grad():
            gt_im1, gt_im2, pred_im1, pred_im2 = generate_rollout(
                transition, decoder, data, args.context_length, args.horizon, device, args.reset_interval
            )
            
            video = create_comparison_video(gt_im1, gt_im2, pred_im1, pred_im2)
            
            # Save video
            output_path = os.path.join(args.output_dir, f"rollout_{i:03d}.mp4")
            # imageio expects (T, H, W, C) format
            iio.imwrite(output_path, video, fps=args.fps, codec='libx264', pixelformat='yuv420p')
            print(f"Saved: {output_path}")
    
    print(f"Done! Videos saved to {args.output_dir}")


if __name__ == "__main__":
    main()

