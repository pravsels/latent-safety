#!/usr/bin/env python3
"""
Test the trained DINO decoder by visualizing reconstructions.

Usage:
  # Show dataset info only
  python dino_wm/test_dino_decoder.py --input arx5_subset_eval.h5 --info

  # Run evaluation
  python dino_wm/test_dino_decoder.py \
    --input arx5_subset_eval.h5 \
    --output decoder_eval_results/ \
    --num_images 15 \
    --checkpoint dino_decoder_checkpoints/testing_decoder.pth
"""

import argparse
from collections import defaultdict
import h5py
import os
import random
import torch
import matplotlib.pyplot as plt
from einops import rearrange
import torch.nn.functional as F
import numpy as np

from dino_decoder import VQVAE
from test_loader import SplitTrajectoryDataset
from dino_wm.config import MODEL_CONFIG


# Default configuration
DEFAULT_CHECKPOINT = "checkpoints/testing_decoder.pth"
DEFAULT_HDF5_FILE = "test_v2.h5"
DEFAULT_OUTPUT_DIR = "decoder_results"
DEFAULT_NUM_IMAGES = 5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_dataset_info(hdf5_file):
    """Get detailed info about trajectories and frames in the dataset."""
    info = {"trajectories": {}, "total_frames": 0}
    with h5py.File(hdf5_file, "r") as hf:
        for traj_id in hf.keys():
            num_frames = len(hf[traj_id]["actions"])
            info["trajectories"][traj_id] = num_frames
            info["total_frames"] += num_frames
    return info


def print_dataset_info(hdf5_file):
    """Print detailed dataset statistics."""
    info = get_dataset_info(hdf5_file)
    num_traj = len(info["trajectories"])
    frame_counts = list(info["trajectories"].values())
    
    print(f"\n{'='*60}")
    print(f"DATASET INFO: {hdf5_file}")
    print(f"{'='*60}")
    print(f"Total trajectories: {num_traj}")
    print(f"Total frames:       {info['total_frames']}")
    print(f"\nFrames per trajectory:")
    print(f"  Min:    {min(frame_counts)}")
    print(f"  Max:    {max(frame_counts)}")
    print(f"  Mean:   {np.mean(frame_counts):.1f}")
    print(f"  Median: {np.median(frame_counts):.1f}")
    print(f"\nTrajectory breakdown:")
    for traj_id, frames in info["trajectories"].items():
        print(f"  {traj_id}: {frames} frames")
    print(f"{'='*60}\n")
    return info


def parse_args():
    parser = argparse.ArgumentParser(description="Test DINO decoder with visualizations")
    parser.add_argument("--input", "-i", type=str, default=DEFAULT_HDF5_FILE,
                        help=f"Input HDF5 file (default: {DEFAULT_HDF5_FILE})")
    parser.add_argument("--output", "-o", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory for results (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--num_images", "-n", type=int, default=DEFAULT_NUM_IMAGES,
                        help=f"Number of images to generate (default: {DEFAULT_NUM_IMAGES})")
    parser.add_argument("--checkpoint", "-c", type=str, default=DEFAULT_CHECKPOINT,
                        help=f"Model checkpoint path (default: {DEFAULT_CHECKPOINT})")
    parser.add_argument("--device", "-d", type=str, default=DEVICE,
                        help=f"Device to use (default: {DEVICE})")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (default: None)")
    parser.add_argument("--info", action="store_true",
                        help="Only show dataset info, don't run evaluation")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Info-only mode
    if args.info:
        print_dataset_info(args.input)
        return
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    print(f"Output directory: {args.output}")
    
    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    decoder = VQVAE().to(device)
    decoder.load_state_dict(torch.load(args.checkpoint, map_location=device))
    decoder.eval()
    
    # Load test data - use ALL trajectories in the file
    print(f"Loading data from: {args.input}")
    with h5py.File(args.input, "r") as hf:
        num_traj = len(hf.keys())
    
    # Use all trajectories in the eval file (num_test = total trajectories)
    test_dataset = SplitTrajectoryDataset(args.input, segment_length=1, split="test", num_test=num_traj)
    
    # Print dataset summary
    print(f"\nDataset summary:")
    print(f"  Trajectories: {num_traj}")
    print(f"  Total frames: {len(test_dataset)}")
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"  Random seed: {args.seed}")
    
    # Random sampling - picks random frames from across all trajectories
    indices = random.sample(range(len(test_dataset)), min(args.num_images, len(test_dataset)))
    
    print(f"\nGenerating {len(indices)} images (randomly sampled across all frames)...")
    
    total_loss1 = 0.0
    total_loss2 = 0.0
    sampled_trajectories = defaultdict(int)  # Track which trajectories were sampled
    
    for img_idx, dataset_idx in enumerate(indices):
        # Get data from dataset by index
        data = test_dataset[dataset_idx]
        # Get trajectory info from slice_indices
        traj_id, frame_idx = test_dataset.slice_indices[dataset_idx]
        sampled_trajectories[traj_id] += 1
        
        with torch.no_grad():
            # Load embeddings and images (add batch dimension)
            inputs1 = data["cam_zed_embd"].unsqueeze(0).to(device)
            inputs2 = data["cam_rs_embd"].unsqueeze(0).to(device)
            gt1 = data["agentview_image"].unsqueeze(0).to(device) / 255.0
            gt2 = data["robot0_eye_in_hand_image"].unsqueeze(0).to(device) / 255.0
            
            # Resize GT to MODEL_CONFIG['image_size']
            img_size = MODEL_CONFIG['image_size']
            B, T, H, W, C = gt1.shape
            gt1 = gt1.permute(0, 1, 4, 2, 3).reshape(B*T, C, H, W)
            gt2 = gt2.permute(0, 1, 4, 2, 3).reshape(B*T, C, H, W)
            gt1 = F.interpolate(gt1, size=img_size, mode="bilinear", align_corners=False)
            gt2 = F.interpolate(gt2, size=img_size, mode="bilinear", align_corners=False)
            gt1 = gt1.view(B, T, C, img_size[0], img_size[1]).permute(0, 1, 3, 4, 2).squeeze(1)
            gt2 = gt2.view(B, T, C, img_size[0], img_size[1]).permute(0, 1, 3, 4, 2).squeeze(1)
            
            # Run decoder
            inputs = torch.cat([inputs1, inputs2], dim=0)
            pred, _ = decoder(inputs)
            pred = rearrange(pred, "(b t) c h w -> b t c h w", t=1)
            pred1, pred2 = torch.split(pred, [B, B], dim=0)
            pred1 = pred1.squeeze(1).permute(0, 2, 3, 1)
            pred2 = pred2.squeeze(1).permute(0, 2, 3, 1)
            
            # Calculate loss
            loss1 = torch.nn.MSELoss()(pred1, gt1)
            loss2 = torch.nn.MSELoss()(pred2, gt2)
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            
            # Convert to numpy
            gt1_np = gt1.cpu().numpy()[0]
            gt2_np = gt2.cpu().numpy()[0]
            pred1_np = np.clip(pred1.cpu().numpy()[0], 0, 1)
            pred2_np = np.clip(pred2.cpu().numpy()[0], 0, 1)
            
            # Create visualization for this sample
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            axes[0].imshow(gt1_np)
            axes[0].set_title(f"Front GT")
            axes[0].axis('off')
            
            axes[1].imshow(pred1_np)
            axes[1].set_title(f"Front Pred (MSE: {loss1.item():.4f})")
            axes[1].axis('off')
            
            axes[2].imshow(gt2_np)
            axes[2].set_title(f"Wrist GT")
            axes[2].axis('off')
            
            axes[3].imshow(pred2_np)
            axes[3].set_title(f"Wrist Pred (MSE: {loss2.item():.4f})")
            axes[3].axis('off')
            
            plt.tight_layout()
            
            # Save individual image
            output_path = os.path.join(args.output, f"sample_{img_idx+1:03d}.png")
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"  [{img_idx+1}/{len(indices)}] traj={traj_id} frame={frame_idx} | {output_path} | Front: {loss1.item():.4f} Wrist: {loss2.item():.4f}")
    
    # Print summary
    num_generated = len(indices)
    avg_loss1 = total_loss1 / num_generated
    avg_loss2 = total_loss2 / num_generated
    
    print(f"\n{'='*60}")
    print(f"SUMMARY ({num_generated} images)")
    print(f"{'='*60}")
    print(f"Average Front MSE: {avg_loss1:.6f}")
    print(f"Average Wrist MSE: {avg_loss2:.6f}")
    print(f"Average Total MSE: {avg_loss1 + avg_loss2:.6f}")
    print(f"\nSampled from {len(sampled_trajectories)} trajectories:")
    for traj, count in sorted(sampled_trajectories.items()):
        print(f"  {traj}: {count} frames")
    print(f"\nResults saved to: {args.output}/")


if __name__ == "__main__":
    main()