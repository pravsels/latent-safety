#!/usr/bin/env python3
"""
Train the DINO decoder (VQVAE) to reconstruct both cameras from DINO patch embeddings.

Quickstart (run from repo root):

  # Standard autoencoder (no quantization), resume from latest checkpoint in checkpoint-dir
  python -u dino_wm/train_dino_decoder.py \
    --hdf5-file ${data_dir}/arx5_subset_train.h5 \
    --batch-size 256 \
    --checkpoint-dir ${data_dir}/dino_decoder_checkpoints \
    --wandb-mode offline \
    --auto-resume

  # With VQ codebook quantization enabled (writes *_vq checkpoints/plots)
  python -u dino_wm/train_dino_decoder.py \
    --hdf5-file ${data_dir}/arx5_subset_train.h5 \
    --batch-size 256 \
    --checkpoint-dir ${data_dir}/dino_decoder_checkpoints \
    --wandb-mode offline \
    --quantize \
    --auto-resume

Notes:
  - Non-quantized outputs: testing_decoder.pth / latest_decoder.pth / best_decoder.pth / training_curve.png
  - Quantized outputs:     testing_decoder_vq.pth / latest_decoder_vq.pth / best_decoder_vq.pth / training_curve_vq.png
  - Use --eval-every and --save-every to control evaluation and checkpoint frequency.
"""

import os
import sys
# Ensure repo root is on sys.path regardless of current working directory.
# This makes `import dino_wm.*` work when running via `python dino_wm/train_dino_decoder.py`.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import h5py
import torch
import wandb
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from einops import rearrange
import matplotlib.pyplot as plt
import torch.nn.functional as F

from dino_wm.test_loader import SplitTrajectoryDataset
from dino_wm.dino_decoder import VQVAE
from dino_wm.config import MODEL_CONFIG, DECODER_CONFIG, TRAIN_CONFIG

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
    parser.add_argument(
        "--eval-every",
        type=int,
        default=100,
        help="Run evaluation every N iterations (default: 100).",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Save a latest checkpoint every N iterations for preemption-safe resume (default: 100).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="dino_decoder_checkpoints",
        help="Directory to save checkpoints (default: dino_decoder_checkpoints).",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="offline",
        choices=["online", "offline", "disabled"],
        help="Wandb logging mode (default: offline).",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="dino-WM",
        help="Wandb project name (default: dino-WM).",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="pravsels",
        help="Wandb entity/team name (default: pravsels).",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default="Decoder",
        help="Wandb run name (default: Decoder).",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="If set, automatically resume from the latest checkpoint in --checkpoint-dir (if present).",
    )
    parser.add_argument(
        "--start-iter",
        type=int,
        default=None,
        help="Iteration to start training from. If omitted and resuming from a checkpoint that stores 'iter', it will resume from there.",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Enable VQ codebook quantization (default: False).",
    )
    args = parser.parse_args()

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        config=vars(args)
    )

    hdf5_file = args.hdf5_file
    H = 1
    BS = args.batch_size
    run_suffix = "_vq" if args.quantize else ""

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
    
    decoder = VQVAE(quantize=args.quantize).to(device)
    if args.quantize:
        print("VQ codebook quantization enabled")
    else:
        print("VQ codebook quantization disabled (standard autoencoder)")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Paths:
    # - testing_decoder*.pth remains a plain state_dict for backward compatibility with older scripts.
    # - latest_decoder*.pth is a richer dict checkpoint for preemption-safe resume (model+optimizer+iter).
    latest_state_dict_path = os.path.join(args.checkpoint_dir, f"testing_decoder{run_suffix}.pth")
    latest_ckpt_path = os.path.join(args.checkpoint_dir, f"latest_decoder{run_suffix}.pth")
    best_ckpt_path = os.path.join(args.checkpoint_dir, f"best_decoder{run_suffix}.pth")

    # Resolve resume path.
    resume_path = args.resume_checkpoint
    if resume_path is None and args.auto_resume and os.path.exists(latest_ckpt_path):
        resume_path = latest_ckpt_path

    optimizer = AdamW([
        {'params': decoder.parameters(), 'lr': 3e-4}
    ])

    # best_eval is tracked and persisted via best checkpoint when available
    best_eval = float("inf")
    if os.path.exists(best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        if isinstance(best_ckpt, dict) and "best_eval" in best_ckpt:
            best_eval = best_ckpt["best_eval"]
            print(f"Loaded previous best eval: {best_eval:.4f}")

    # Resume (supports legacy state_dict-only checkpoints and newer dict checkpoints).
    start_iter_from_ckpt = None
    if resume_path is not None:
        print(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            decoder.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except Exception as e:
                    print(f"Warning: failed to load optimizer state ({e}); continuing with fresh optimizer.")
            if "iter" in ckpt:
                start_iter_from_ckpt = int(ckpt["iter"]) + 1
            if "best_eval" in ckpt and best_eval == float("inf"):
                best_eval = ckpt["best_eval"]
        else:
            # Legacy checkpoints may just be a raw state_dict
            decoder.load_state_dict(ckpt)

    print('decoder with parameters', count_parameters(decoder))
    
    iters = []
    train_losses = []
    eval_losses = []
    train_iter = args.train_iters
    start_iter = args.start_iter
    if start_iter is None:
        start_iter = start_iter_from_ckpt or 0
    if start_iter_from_ckpt is not None:
        # start_iter_from_ckpt is stored as (ckpt_iter + 1)
        print(f"Auto-resume: continuing from iter {start_iter} (loaded iter={start_iter - 1} from checkpoint).")
    for i in range(start_iter, train_iter):
        # Refresh iterators when they exhaust (avoids relying on len() of iterator).
        try:
            data = next(expert_loader)
        except StopIteration:
            expert_loader = iter(DataLoader(expert_data, batch_size=BS, shuffle=True))
            data = next(expert_loader)

        inputs1 = data["cam_zed_embd"].to(device)
        inputs2 = data["cam_rs_embd"].to(device)
        # Ground truth images start as (B, T, H_img, W_img, C); we resize to DECODER_CONFIG['decoder_image_size'] to match decoder's native output.
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
        # Resize spatial dims to DECODER_CONFIG['decoder_image_size'] so loss compares at decoder resolution
        img_size = DECODER_CONFIG['decoder_image_size']
        output1_btchw = F.interpolate(
            output1_btchw, size=img_size, mode="bilinear", align_corners=False
        )
        output2_btchw = F.interpolate(
            output2_btchw, size=img_size, mode="bilinear", align_corners=False
        )
        # Back to (B, T, H, W, C) after resize
        output1 = (
            output1_btchw.view(B, T, C, img_size[0], img_size[1]).permute(0, 1, 3, 4, 2).contiguous()
        )
        output2 = (
            output2_btchw.view(B, T, C, img_size[0], img_size[1]).permute(0, 1, 3, 4, 2).contiguous()
        )


        inputs = torch.cat([inputs1, inputs2], dim=0)

        pred, diff = decoder(inputs)
        # Decoder returns (B*T, C, H_dec, W_dec); restore (B, T, C, H, W) with T=1
        pred = rearrange(pred, "(b t) c h w -> b t c h w", t=1)
        
        pred1, pred2 = torch.split(pred, [inputs1.shape[0], inputs2.shape[0]], dim=0)
        # Drop only the time dim, keep batch dim: (B, 1, C, H, W) -> (B, C, H, W)
        pred1 = pred1.squeeze(1).permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        pred2 = pred2.squeeze(1).permute(0, 2, 3, 1)
        # output1, output2: (B, T, H, W, C) -> (B, H, W, C)
        output1_bhwc = output1.squeeze(1)
        output2_bhwc = output2.squeeze(1)

        recon_loss = nn.MSELoss()(pred1, output1_bhwc)
        recon_loss += nn.MSELoss()(pred2, output2_bhwc)
        # VQ commitment / codebook loss (scalar)
        vq_loss = diff.mean()
        loss = recon_loss + 0.25 * vq_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        wandb.log({'train_loss': loss.item()})
        print(f"\rIter {i}, Train Loss: {loss.item():.4f}", end='', flush=True)

        # Periodic "latest" checkpoint for HPC preemption / manual restarts.
        if args.save_every and (i % args.save_every == 0):
            torch.save(
                {
                    "model_state_dict": decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "iter": i,
                    "best_eval": best_eval.item() if hasattr(best_eval, "item") else best_eval,
                    "quantize": bool(args.quantize),
                },
                latest_ckpt_path,
            )
        
        if args.eval_every and (i % args.eval_every == 0):
            train_losses.append(loss.item())
            iters.append(i)
            try:
                eval_data = next(expert_loader_eval)
            except StopIteration:
                expert_loader_eval = iter(DataLoader(expert_data_eval, batch_size=BS, shuffle=True))
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
                img_size = DECODER_CONFIG['decoder_image_size']
                output1_btchw_e = F.interpolate(
                    output1_btchw_e,
                    size=img_size,
                    mode="bilinear",
                    align_corners=False,
                )
                output2_btchw_e = F.interpolate(
                    output2_btchw_e,
                    size=img_size,
                    mode="bilinear",
                    align_corners=False,
                )
                output1 = (
                    output1_btchw_e.view(B_eval, T_eval, C_e, img_size[0], img_size[1])
                    .permute(0, 1, 3, 4, 2)
                    .contiguous()
                )
                output2 = (
                    output2_btchw_e.view(B_eval, T_eval, C_e, img_size[0], img_size[1])
                    .permute(0, 1, 3, 4, 2)
                    .contiguous()
                )


                inputs = torch.cat([inputs1, inputs2], dim=0)
                pred, diff = decoder(inputs)
                pred = rearrange(pred, "(b t) c h w -> b t c h w", t=1)
                pred1, pred2 = torch.split(
                    pred, [inputs1.shape[0], inputs2.shape[0]], dim=0
                )
                pred1 = pred1.squeeze(1).permute(0, 2, 3, 1)
                pred2 = pred2.squeeze(1).permute(0, 2, 3, 1)

                output1_bhwc = output1.squeeze(1)
                output2_bhwc = output2.squeeze(1)
                
                recon_loss = nn.MSELoss()(pred1, output1_bhwc)
                recon_loss += nn.MSELoss()(pred2, output2_bhwc)
                vq_loss = diff.mean()
                loss = recon_loss + 0.25 * vq_loss

            print()
            print(f"\rIter {i}, Eval Loss: {loss.item():.4f}")
            if loss < best_eval:
                best_eval = loss
                # Save backward-compatible latest weights (state_dict)
                torch.save(decoder.state_dict(), latest_state_dict_path)
                # Save rich "latest" checkpoint for restart/resume
                torch.save(
                    {
                        "model_state_dict": decoder.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "iter": i,
                        "best_eval": best_eval.item() if hasattr(best_eval, "item") else best_eval,
                        "quantize": bool(args.quantize),
                    },
                    latest_ckpt_path,
                )
                # Save best checkpoint with metadata to persist best_eval across sessions
                torch.save({
                    'model_state_dict': decoder.state_dict(),
                    'best_eval': best_eval.item() if hasattr(best_eval, 'item') else best_eval
                }, best_ckpt_path)
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
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    plt.savefig(os.path.join(args.checkpoint_dir, f'training_curve{run_suffix}.png'))

    best_eval_val = best_eval.item() if hasattr(best_eval, 'item') else best_eval
    print(f"\nTraining complete. Best eval loss: {best_eval_val:.4f}")


if __name__ == "__main__":
    main()