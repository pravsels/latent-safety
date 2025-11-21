#!/usr/bin/env python3
"""
Train the DINO decoder (VQVAE) to reconstruct both cameras from DINO patch embeddings.

Quickstart:

  python dino_wm/train_dino_decoder.py --hdf5-file test_v2.h5 --batch-size 2
"""

import argparse
import h5py
import os
import torch
import wandb
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from einops import rearrange
import matplotlib.pyplot as plt
import torch.nn.functional as F

from test_loader import SplitTrajectoryDataset
from dino_decoder import VQVAE

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hdf5-file",
        "--hdf5",
        dest="hdf5_file",
        type=str,
        default="test_v2.h5",
        help="Path to consolidated DINO WM HDF5 file (default: test_v2.h5)",
    )
    parser.add_argument(
        "--num-test-trajectories",
        type=int,
        default=None,
        help="Explicit number of trajectories to use for test split (overrides --test-frac).",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.2,
        help="Fraction of trajectories to use for test split (used if --num-test-trajectories is not set).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training and evaluation (default: 64).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device to use (default: cuda:0).",
    )
    parser.add_argument(
        "--train-iters",
        type=int,
        default=5000,
        help="Number of training iterations (default: 5000).",
    )
    args = parser.parse_args()

    wandb.init(project="dino-WM", name="Decoder", entity="pravsels", mode="offline")

    hdf5_file = args.hdf5_file
    H = 1
    BS = args.batch_size

    # Determine train/test split via percentage, with minimum of 1 trajectory
    # in the smaller split when possible.
    with h5py.File(hdf5_file, "r") as hf:
        num_traj = len(hf.keys())

    if num_traj == 0:
        raise ValueError(f"No trajectories found in HDF5 file: {hdf5_file}")

    if args.num_test_trajectories is not None:
        num_test = max(1, min(args.num_test_trajectories, num_traj))
    else:
        raw_num = int(round(args.test_frac * num_traj))
        num_test = max(1, raw_num)

    # Ensure at least 1 train trajectory when more than 1 total trajectory exists.
    if num_traj - num_test < 1 and num_traj > 1:
        num_test = num_traj - 1

    print(
        f"Found {num_traj} trajectories in {hdf5_file}. "
        f"Using {num_traj - num_test} for train and {num_test} for test."
    )

    expert_data = SplitTrajectoryDataset(
        hdf5_file, H, split="train", num_test=num_test
    )
    expert_data_eval = SplitTrajectoryDataset(
        hdf5_file, H, split="test", num_test=num_test
    )

    expert_loader = iter(DataLoader(expert_data, batch_size=BS, shuffle=True))
    expert_loader_eval = iter(DataLoader(expert_data_eval, batch_size=BS, shuffle=True))
    device = args.device
    
    decoder = VQVAE().to(device)
    print('decoder with parameters', count_parameters(decoder))
    
    optimizer = AdamW([
        {'params': decoder.parameters(), 'lr': 3e-4}
    ])

    best_eval = float('inf')
    iters = []
    train_losses = []
    eval_losses = []
    train_iter = args.train_iters
    for i in range(train_iter):
        if i % len(expert_loader) == 0:
            expert_loader = iter(DataLoader(expert_data, batch_size=BS, shuffle=True))
        if i % len(expert_loader_eval) == 0:
            expert_loader_eval = iter(DataLoader(expert_data_eval, batch_size=BS, shuffle=True))
        data = next(expert_loader)

        inputs1 = data["cam_zed_embd"].to(device)
        inputs2 = data["cam_rs_embd"].to(device)
        # Ground truth images start as (B, T, H_img, W_img, C); we resize to 224x224 to match decoder output.
        output1 = data["agentview_image"].to(device) / 255.0  # (B, T, H, W, C)
        output2 = data["robot0_eye_in_hand_image"].to(device) / 255.0

        B, T, H_img, W_img, C = output1.shape
        # Flatten batch & time and go to BCHW for interpolate: (B, T, H, W, C) -> (B*T, C, H, W)
        output1_btchw = output1.permute(0, 1, 4, 2, 3).contiguous().view(
            B * T, C, H_img, W_img
        )
        output2_btchw = output2.permute(0, 1, 4, 2, 3).contiguous().view(
            B * T, C, H_img, W_img
        )
        # Resize spatial dims to 224x224 so loss compares at decoder resolution
        output1_btchw = F.interpolate(
            output1_btchw, size=(224, 224), mode="bilinear", align_corners=False
        )
        output2_btchw = F.interpolate(
            output2_btchw, size=(224, 224), mode="bilinear", align_corners=False
        )
        # Back to (B, T, H, W, C) after resize
        output1 = (
            output1_btchw.view(B, T, C, 224, 224).permute(0, 1, 3, 4, 2).contiguous()
        )
        output2 = (
            output2_btchw.view(B, T, C, 224, 224).permute(0, 1, 3, 4, 2).contiguous()
        )


        inputs = torch.cat([inputs1, inputs2], dim=0)

        pred, _ = decoder(inputs)
        # Decoder returns (B*T, C, H_dec, W_dec); restore (B, T, C, H, W) with T=1
        pred = rearrange(pred, "(b t) c h w -> b t c h w", t=1)
        
        pred1, pred2 = torch.split(pred, [inputs1.shape[0], inputs2.shape[0]], dim=0)
        # Drop only the time dim, keep batch dim: (B, 1, C, H, W) -> (B, C, H, W)
        pred1 = pred1.squeeze(1).permute(0, 2, 3, 1)  # (B, H, W, C)
        pred2 = pred2.squeeze(1).permute(0, 2, 3, 1)
        # output1, output2: (B, T, H, W, C) -> drop time dim only
        output1_bhwc = output1.squeeze(1)
        output2_bhwc = output2.squeeze(1)

        loss = nn.MSELoss()(pred1, output1_bhwc)
        loss += nn.MSELoss()(pred2, output2_bhwc)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        wandb.log({'train_loss': loss.item()})
        print(f"\rIter {i}, Train Loss: {loss.item():.4f}", end='', flush=True)
        
        if i % 100 == 0:
            train_losses.append(loss.item())
            iters.append(i)
            eval_data = next(expert_loader_eval)
            decoder.eval()
            with torch.no_grad():
                inputs1 = eval_data["cam_zed_embd"].to(device)
                inputs2 = eval_data["cam_rs_embd"].to(device)
                # Same resizing as above for eval images
                output1 = eval_data["agentview_image"].to(device) / 255.0
                output2 = eval_data["robot0_eye_in_hand_image"].to(device) / 255.0

                B_eval, T_eval, H_img_e, W_img_e, C_e = output1.shape
                output1_btchw_e = output1.permute(0, 1, 4, 2, 3).contiguous().view(
                    B_eval * T_eval, C_e, H_img_e, W_img_e
                )
                output2_btchw_e = output2.permute(0, 1, 4, 2, 3).contiguous().view(
                    B_eval * T_eval, C_e, H_img_e, W_img_e
                )
                output1_btchw_e = F.interpolate(
                    output1_btchw_e,
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False,
                )
                output2_btchw_e = F.interpolate(
                    output2_btchw_e,
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False,
                )
                output1 = (
                    output1_btchw_e.view(B_eval, T_eval, C_e, 224, 224)
                    .permute(0, 1, 3, 4, 2)
                    .contiguous()
                )
                output2 = (
                    output2_btchw_e.view(B_eval, T_eval, C_e, 224, 224)
                    .permute(0, 1, 3, 4, 2)
                    .contiguous()
                )


                inputs = torch.cat([inputs1, inputs2], dim=0)
                pred, _ = decoder(inputs)
                pred = rearrange(pred, "(b t) c h w -> b t c h w", t=1)
                pred1, pred2 = torch.split(
                    pred, [inputs1.shape[0], inputs2.shape[0]], dim=0
                )
                pred1 = pred1.squeeze(1).permute(0, 2, 3, 1)
                pred2 = pred2.squeeze(1).permute(0, 2, 3, 1)

                output1_bhwc = output1.squeeze(1)
                output2_bhwc = output2.squeeze(1)
                
                loss = nn.MSELoss()(pred1, output1_bhwc)
                loss += nn.MSELoss()(pred2, output2_bhwc)

            print()
            print(f"\rIter {i}, Eval Loss: {loss.item():.4f}")
            if loss < best_eval:
                best_eval = loss
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(decoder.state_dict(), 'checkpoints/testing_decoder.pth')
            decoder.train()
            
            out_log = (output1_bhwc[0].detach().cpu().numpy())
            pred_log = (pred1[0].detach().detach().cpu().numpy())
            out_log2 = (output2_bhwc[0].detach().cpu().numpy())
            pred_log2 = (pred2[0].detach().detach().cpu().numpy())

            wandb.log({'eval_loss': loss.item(), 'ground_truth_front': wandb.Image(out_log), 'pred_front': wandb.Image(pred_log), 'ground_truth_wrist': wandb.Image(out_log2), 'pred_wrist': wandb.Image(pred_log2)})
            eval_losses.append(loss.item())


    plt.plot(iters, train_losses, label='train')
    plt.plot(iters, eval_losses, label='eval')
    plt.legend()
    plt.savefig('training curve.png')


if __name__ == "__main__":
    main()