#!/usr/bin/env python3
"""
Test the trained DINO decoder by visualizing reconstructions.

Usage:
  python dino_wm/test_dino_decoder.py
"""

import h5py
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from einops import rearrange
import torch.nn.functional as F
import numpy as np

from test_loader import SplitTrajectoryDataset
from dino_decoder import VQVAE


# Configuration
CHECKPOINT = "checkpoints/testing_decoder.pth"
HDF5_FILE = "test_v2.h5"
BATCH_SIZE = 1
TEST_FRAC = 0.2
OUTPUT_FILE = "decoder_test_results.png"
DEVICE = "cuda:0"

def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading checkpoint: {CHECKPOINT}")
    decoder = VQVAE().to(device)
    decoder.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    decoder.eval()
    
    # Load test data
    with h5py.File(HDF5_FILE, "r") as hf:
        num_traj = len(hf.keys())
    num_test = max(1, int(round(TEST_FRAC * num_traj)))
    
    test_dataset = SplitTrajectoryDataset(HDF5_FILE, segment_length=1, split="test", num_test=num_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Loaded {num_traj} trajectories, using {num_test} for testing")
    
    # Get one batch
    data = next(iter(test_loader))
    
    with torch.no_grad():
        # Load embeddings and images
        inputs1 = data["cam_zed_embd"].to(device)
        inputs2 = data["cam_rs_embd"].to(device)
        gt1 = data["agentview_image"].to(device) / 255.0
        gt2 = data["robot0_eye_in_hand_image"].to(device) / 255.0
        
        # Resize GT to 224x224
        B, T, H, W, C = gt1.shape
        gt1 = gt1.permute(0, 1, 4, 2, 3).reshape(B*T, C, H, W)
        gt2 = gt2.permute(0, 1, 4, 2, 3).reshape(B*T, C, H, W)
        gt1 = F.interpolate(gt1, size=(224, 224), mode="bilinear", align_corners=False)
        gt2 = F.interpolate(gt2, size=(224, 224), mode="bilinear", align_corners=False)
        gt1 = gt1.view(B, T, C, 224, 224).permute(0, 1, 3, 4, 2).squeeze(1)
        gt2 = gt2.view(B, T, C, 224, 224).permute(0, 1, 3, 4, 2).squeeze(1)
        
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
        total_loss = loss1 + loss2
        
        print(f"Front MSE: {loss1.item():.6f}")
        print(f"Wrist MSE: {loss2.item():.6f}")
        print(f"Total MSE: {total_loss.item():.6f}")
        
        # Convert to numpy
        gt1_np = gt1.cpu().numpy()
        gt2_np = gt2.cpu().numpy()
        pred1_np = np.clip(pred1.cpu().numpy(), 0, 1)
        pred2_np = np.clip(pred2.cpu().numpy(), 0, 1)
        
        # Create visualization
        fig, axes = plt.subplots(BATCH_SIZE, 4, figsize=(16, 4 * BATCH_SIZE))
        if BATCH_SIZE == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(BATCH_SIZE):
            axes[i, 0].imshow(gt1_np[i])
            axes[i, 0].set_title(f"Front GT {i}")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(pred1_np[i])
            axes[i, 1].set_title(f"Front Pred {i}")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(gt2_np[i])
            axes[i, 2].set_title(f"Wrist GT {i}")
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(pred2_np[i])
            axes[i, 3].set_title(f"Wrist Pred {i}")
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to {OUTPUT_FILE}")
        plt.show()

if __name__ == "__main__":
    main()