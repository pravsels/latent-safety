#!/usr/bin/env python3
"""
Train the DINO World Model (VideoTransformer) on trajectory data.

Quickstart:

  python dino_wm/train_dino_wm.py --hdf5-file arx5_subset_train.h5

With custom parameters:

  python dino_wm/train_dino_wm.py \
    --hdf5-file arx5_subset_train.h5 \
    --resume-checkpoint dino_wm_checkpoints/wm_iter5000.pth \
    --start-iter 5000 --batch-size 128
"""

import argparse
import os
import h5py
import json
import numpy as np
import torch
import random
import wandb
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import matplotlib.pyplot as plt
from tqdm import tqdm

from test_loader import SplitTrajectoryDataset
from dino_decoder import VQVAE
from dino_models import VideoTransformer, normalize_acs, normalize_states, unnormalize_states
from dino_wm.config import MODEL_CONFIG, TRAIN_CONFIG, DECODER_CONFIG


def main():
    parser = argparse.ArgumentParser(
        description="Train DINO World Model on trajectory data"
    )
    parser.add_argument(
        "--hdf5-file",
        "--hdf5",
        dest="hdf5_file",
        type=str,
        default="arx5_subset_train.h5",
        help="Path to HDF5 file (default: arx5_subset_train.h5). Will be split into train/eval.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training (default: 16).",
    )
    parser.add_argument(
        "--train-iters",
        type=int,
        default=100000,
        help="Number of training iterations (default: 100000).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device to use (default: cuda:0).",
    )
    parser.add_argument(
        "--decoder-checkpoint",
        type=str,
        default="dino_decoder_checkpoints/testing_decoder.pth",
        help="Path to decoder checkpoint (default: dino_decoder_checkpoints/testing_decoder.pth).",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=4,
        help="Sequence length for training (default: 4).",
    )
    parser.add_argument(
        "--eval-horizon",
        type=int,
        default=16,
        help="Evaluation rollout horizon (default: 16).",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=3,
        help="Context length for autoregressive evaluation (default: 3).",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1000,
        help="Evaluation interval in iterations (default: 1000).",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.1,
        help="Fraction of trajectories to use for evaluation (default: 0.1).",
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
        default="WM",
        help="Wandb run name (default: WM).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="dino_wm_checkpoints",
        help="Directory to save checkpoints (default: dino_wm_checkpoints).",
    )
    parser.add_argument(
        "--dataset-stats",
        type=str,
        default="dataset_stats.json",
        help="Path to dataset statistics JSON file.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )
    parser.add_argument(
        "--start-iter",
        type=int,
        default=0,
        help="Iteration to start training from (default: 0).",
    )
    args = parser.parse_args()

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        config=vars(args)
    )

    # Set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    use_amp = True
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    BS = args.batch_size
    BL = args.sequence_length
    EVAL_H = args.eval_horizon
    H = args.context_length
    device = args.device
    
    # LOAD STATS
    stats_path = args.dataset_stats
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"Stats file '{stats_path}' not found! Please run scripts/compute_stats_json.py to generate it."
        )

    print(f"Loading dataset stats from {stats_path}")
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    # Check for required keys
    required_keys = ["action_min", "action_max", "state_min", "state_max"]
    missing_keys = [k for k in required_keys if k not in stats]
    
    if missing_keys:
        raise ValueError(f"Stats file missing required keys: {missing_keys}")

    # Create tensors on device
    action_min = torch.tensor(stats['action_min']).float().to(device)
    action_max = torch.tensor(stats['action_max']).float().to(device)
    state_min = torch.tensor(stats['state_min']).float().to(device)
    state_max = torch.tensor(stats['state_max']).float().to(device)
    
    # Infer dimensions from stats
    state_dim = len(stats['state_min'])
    action_dim = len(stats['action_min'])
    
    print(f"Loaded state normalization stats from {stats_path}")
    print(f"Inferred state_dim={state_dim}, action_dim={action_dim} from dataset stats")

    # Dataset setup
    hdf5_file = args.hdf5_file
    
    # Count trajectories and compute split
    with h5py.File(hdf5_file, "r") as hf:
        num_traj = len(hf.keys())
    
    num_test = max(1, int(args.test_frac * num_traj))
    if num_traj - num_test < 1 and num_traj > 1:
        num_test = num_traj - 1
    
    expert_data = SplitTrajectoryDataset(hdf5_file, BL, split='train', num_test=num_test)
    expert_data_eval = SplitTrajectoryDataset(hdf5_file, BL, split='test', num_test=num_test)
    expert_data_imagine = SplitTrajectoryDataset(hdf5_file, 32, split='test', num_test=num_test)
    
    print(f"Dataset: {hdf5_file}")
    print(f"  Train: {num_traj - num_test} trajectories")
    print(f"  Eval:  {num_test} trajectories")

    expert_loader = iter(DataLoader(expert_data, batch_size=BS, shuffle=True))
    expert_loader_eval = iter(DataLoader(expert_data_eval, batch_size=BS, shuffle=True))
    expert_loader_imagine = iter(DataLoader(expert_data_imagine, batch_size=1, shuffle=True))

    # Load decoder
    decoder = VQVAE().to(device)
    decoder.load_state_dict(torch.load(args.decoder_checkpoint, map_location=device))
    decoder.eval()
    print(f"Loaded decoder from {args.decoder_checkpoint}")

    # Initialize world model
    transition = VideoTransformer(
        state_dim=state_dim,    # Inferred from dataset stats
        action_dim=action_dim,  # Inferred from dataset stats
        num_frames=BL-1,        # context window size (input sequence length)
        **MODEL_CONFIG
    ).to(device)
    
    if args.resume_checkpoint is not None:
        print(f"Resuming from checkpoint: {args.resume_checkpoint}")
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        # Handle both old (state_dict) and new (dict with model_state_dict) formats
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            transition.load_state_dict(ckpt['model_state_dict'])
        else:
            transition.load_state_dict(ckpt)
    
    transition.train()
    
    # Optimizer
    optimizer = AdamW([
        {'params': transition.transformer.parameters(), 'lr': 5e-5},
        {'params': transition.state_head.parameters(), 'lr': 5e-5}, 
        {'params': transition.front_head.parameters(), 'lr': 5e-5}, 
        {'params': transition.wrist_head.parameters(), 'lr': 5e-5}, 
        {'params': transition.action_encoder.parameters(), 'lr': 5e-4},
        {'params': transition.state_encoder.parameters(), 'lr': 5e-4},
        {'params': [transition.pos_embedding], 'lr': 5e-4},
        {'params': [transition.temp_embedding], 'lr': 5e-4}
    ])

    # Load best_eval from existing best checkpoint to persist across sessions
    best_eval = float('inf')
    best_ckpt_path = os.path.join(args.checkpoint_dir, 'best_wm.pth')
    if os.path.exists(best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        if isinstance(best_ckpt, dict) and 'best_eval' in best_ckpt:
            best_eval = best_ckpt['best_eval']
            print(f"Loaded previous best eval: {best_eval:.4f}")
    
    iters = []
    train_iter = args.train_iters
    start_iter = args.start_iter

    for i in tqdm(range(start_iter, train_iter), desc="Training", unit="iter"):
        if i > 0 and i % len(expert_loader) == 0:
            expert_loader = iter(DataLoader(expert_data, batch_size=BS, shuffle=True))
        if i > 0 and i % len(expert_loader_eval) == 0:
            expert_loader_eval = iter(DataLoader(expert_data_eval, batch_size=BS, shuffle=True))
        if i > 0 and i % len(expert_loader_imagine) == 0:
            expert_loader_imagine = iter(DataLoader(expert_data_imagine, batch_size=1, shuffle=True))

        data = next(expert_loader)

        gt_front_embd = data['cam_zed_embd'].to(device)
        
        # Teacher Forcing Setup:
        # Input:  Frames [0, 1, ..., N-1]
        # Target: Frames [1, 2, ..., N]
        # The model predicts t+1 given history up to t.
        input_front_embd = gt_front_embd[:, :-1]
        target_front_embd = gt_front_embd[:, 1:]

        gt_wrist_embd = data['cam_rs_embd'].to(device)
        input_wrist_embd = gt_wrist_embd[:, :-1]
        target_wrist_embd = gt_wrist_embd[:, 1:]

        gt_state = data['state'].to(device)
        norm_gt_state = normalize_states(gt_state, state_min, state_max)
        input_state = norm_gt_state[:, :-1]
        target_state = norm_gt_state[:, 1:]

        gt_acs = data['action'].to(device)
        norm_gt_acs = normalize_acs(gt_acs, action_min, action_max)
        input_acs = norm_gt_acs[:, :-1]

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            pred_front, pred_wrist, pred_state, _ = transition(input_front_embd, input_wrist_embd, input_state, input_acs)
            loss_front_tf = nn.MSELoss()(pred_front, target_front_embd)
            loss_wrist_tf = nn.MSELoss()(pred_wrist, target_wrist_embd)
            loss_state_tf = nn.MSELoss()(pred_state, target_state)
            loss_tf = loss_front_tf + loss_wrist_tf + loss_state_tf

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            # Detach predictions for AR step
            detach_pred_front = pred_front
            detach_pred_wrist = pred_wrist
            detach_pred_state = pred_state.detach()
            
            # Create hybrid AR inputs: [GT_0, Pred_1]
            input_front_ar = torch.cat([gt_front_embd[:, [0]], detach_pred_front[:, [0]]], dim=1)
            input_wrist_ar = torch.cat([gt_wrist_embd[:, [0]], detach_pred_wrist[:, [0]]], dim=1)
            input_state_ar = torch.cat([norm_gt_state[:,[0]], detach_pred_state[:, [0]]], dim=1)
            input_acs_ar = norm_gt_acs[:, [0,1]]

            # AR Forward pass
            pred_front_ar, pred_wrist_ar, pred_state_ar, _ = transition(input_front_ar, input_wrist_ar, input_state_ar, input_acs_ar)
            
            # Targets for AR step: Frame 2 (index 2 in GT)
            target_front_ar = gt_front_embd[:, 2]
            target_wrist_ar = gt_wrist_embd[:, 2]
            target_state_ar = norm_gt_state[:, 2]
            
            # Calculate AR losses on the second step of prediction (corresponding to Frame 2)
            loss_front_ar = nn.MSELoss()(pred_front_ar[:,1], target_front_ar)
            loss_wrist_ar = nn.MSELoss()(pred_wrist_ar[:,1], target_wrist_ar)
            loss_state_ar = nn.MSELoss()(pred_state_ar[:,1], target_state_ar)
            loss_ar = loss_front_ar + loss_wrist_ar + loss_state_ar       

        loss = loss_tf + loss_ar*0.5

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss = loss.item()
        print(f"\rIter {i}, TF Loss: {loss_tf:.4f}, AR loss:{loss_ar:.4f}, front Loss: {loss_front_tf.item():.4f}, wrist Loss: {loss_wrist_tf.item():.4f}, state Loss: {loss_state_tf.item():.4f}", end='', flush=True)
        wandb.log({'train_loss': loss_tf, "train_loss_ar": loss_ar})
        
        # Evaluation
        if (i) % args.eval_interval == 0:
            iters.append(i)
            eval_data = next(expert_loader_imagine)
            transition.eval()
            with torch.no_grad():
                gt_front_embd_eval = eval_data['cam_zed_embd'].to(device)
                input_front_embd_eval = gt_front_embd_eval[[0], :H].to(device)

                gt_wrist_embd_eval = eval_data['cam_rs_embd'].to(device)
                input_wrist_embd_eval = gt_wrist_embd_eval[[0], :H].to(device)
                
                all_acs = eval_data['action'][[0]].to(device)
                all_acs = normalize_acs(all_acs, action_min, action_max)
                
                acs = eval_data['action'][[0],:H].to(device)
                acs = normalize_acs(acs, action_min, action_max)

                gt_states_eval = eval_data['state'][[0],:H].to(device)
                input_states_eval = normalize_states(gt_states_eval, state_min, state_max)
                # Resize images to DECODER_CONFIG['decoder_image_size'] to match decoder output
                im1s = eval_data['agentview_image'][[0], :H].squeeze().to(device)/255.  # (T, H, W, C)
                im2s = eval_data['robot0_eye_in_hand_image'][[0], :H].squeeze().to(device)/255.
                im1s = F.interpolate(im1s.permute(0, 3, 1, 2), size=DECODER_CONFIG['decoder_image_size'], mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                im2s = F.interpolate(im2s.permute(0, 3, 1, 2), size=DECODER_CONFIG['decoder_image_size'], mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                for k in range(EVAL_H-H):
                    pred_front, pred_wrist, pred_state, _ = transition(input_front_embd_eval, input_wrist_embd_eval, input_states_eval, acs)

                    pred_latent = torch.cat([pred_front[:,[-1]], pred_wrist[:,[-1]]], dim=0)
                    pred_ims, _ = decoder(pred_latent)

                    pred_ims = rearrange(pred_ims, "(b t) c h w -> b t h w c", t=1)
                    pred_im1, pred_im2 = torch.split(pred_ims, [input_front_embd_eval.shape[0], input_wrist_embd_eval.shape[0]], dim=0)

                    im1s = torch.cat([im1s, pred_im1.squeeze(0)], dim=0)
                    im2s = torch.cat([im2s, pred_im2.squeeze(0)], dim=0)
                    
                    # getting next inputs
                    acs = torch.cat([acs[[0], 1:], all_acs[0,H+k].unsqueeze(0).unsqueeze(0)], dim=1)
                    input_front_embd_eval = torch.cat([input_front_embd_eval[[0], 1:], pred_front[:, -1].unsqueeze(1)], dim=1)
                    input_wrist_embd_eval = torch.cat([input_wrist_embd_eval[[0], 1:], pred_wrist[:, -1].unsqueeze(1)], dim=1)
                    input_states_eval = torch.cat([input_states_eval[[0], 1:], pred_state[:,-1].unsqueeze(1)], dim=1)

                gt_im1 = eval_data['agentview_image'][[0], :EVAL_H].squeeze().to(device)  # (T, H, W, C)
                gt_im2 = eval_data['robot0_eye_in_hand_image'][[0], :EVAL_H].squeeze().to(device)
                # Resize to DECODER_CONFIG['decoder_image_size'] to match decoder output
                gt_im1 = F.interpolate(gt_im1.permute(0, 3, 1, 2).float(), size=DECODER_CONFIG['decoder_image_size'], mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                gt_im2 = F.interpolate(gt_im2.permute(0, 3, 1, 2).float(), size=DECODER_CONFIG['decoder_image_size'], mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

                gt_imgs = torch.cat([gt_im1, gt_im2], dim=-2)/255.
                pred_imgs = torch.cat([im1s, im2s], dim=-2)
                vid = torch.cat([gt_imgs, pred_imgs], dim=-3)
                vid = vid.detach().cpu().numpy()
                vid = (vid * 255).clip(0, 255).astype(np.uint8)
                vid = rearrange(vid, "t h w c -> t c h w")
                wandb.log({"video": wandb.Video(vid, fps=20, format='mp4')})
                
                # done logging video

                eval_data = next(expert_loader_eval)
                gt_front_embd_eval = eval_data['cam_zed_embd'].to(device)
                gt_wrist_embd_eval = eval_data['cam_rs_embd'].to(device)

                input_front_embd_eval = gt_front_embd_eval[:, :-1]
                target_front_embd_eval = gt_front_embd_eval[:, 1:]

                input_wrist_embd_eval = gt_wrist_embd_eval[:, :-1]
                target_wrist_embd_eval = gt_wrist_embd_eval[:, 1:]

                gt_state_eval = eval_data['state'].to(device)
                norm_eval_states = normalize_states(gt_state_eval, state_min, state_max)
                input_state_eval = norm_eval_states[:, :-1]
                target_state_eval = norm_eval_states[:, 1:]

                data_acs = eval_data['action'].to(device)
                norm_acs = normalize_acs(data_acs, action_min, action_max)
                acs = norm_acs[:, :-1]
                pred_front, pred_wrist, pred_state, _ = transition(input_front_embd_eval, input_wrist_embd_eval, input_state_eval, acs)

                pred_latent = torch.cat([pred_front[:,[H-1]], pred_wrist[:,[H-1]]], dim=0)
                pred_ims, _ = decoder(pred_latent)
                pred_im1, pred_im2 = torch.split(pred_ims, [input_front_embd_eval.shape[0], input_wrist_embd_eval.shape[0]], dim=0)
                pred_im1 = pred_im1[0].permute(1,2,0).detach().cpu().numpy()
                pred_im2 = pred_im2[0].permute(1,2,0).detach().cpu().numpy()
                im1 = eval_data['agentview_image'][0, H].numpy()
                im2 = eval_data['robot0_eye_in_hand_image'][0, H].numpy()
                loss_front = nn.MSELoss()(pred_front, target_front_embd_eval)
                loss_wrist = nn.MSELoss()(pred_wrist, target_wrist_embd_eval)
                loss_state = nn.MSELoss()(pred_state, target_state_eval)
                loss = loss_front + loss_wrist + loss_state
            print()
            print(f"\rIter {i}, Eval Loss: {loss.item():.4f}, front Loss: {loss_front.item():.4f}, wrist Loss: {loss_wrist.item():.4f}, state Loss: {loss_state.item():.4f}")

            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save(transition.state_dict(), os.path.join(args.checkpoint_dir, f'wm_iter{i}.pth'))

            if loss < best_eval:
                best_eval = loss
                torch.save({
                    'model_state_dict': transition.state_dict(),
                    'best_eval': best_eval.item() if hasattr(best_eval, 'item') else best_eval
                }, os.path.join(args.checkpoint_dir, 'best_wm.pth'))
            
            transition.train()
            wandb.log({'eval_loss': loss.item(), 'front_loss': loss_front.item(), 'wrist_loss': loss_wrist.item(), 'state_loss': loss_state.item(), 'pred_front': wandb.Image(pred_im1), 'pred_wrist': wandb.Image(pred_im2), 'front': wandb.Image(im1), 'wrist': wandb.Image(im2)})

    plt.legend()
    plt.savefig(os.path.join(args.checkpoint_dir, 'training_curve.png'))

    best_eval_val = best_eval.item() if hasattr(best_eval, 'item') else best_eval
    print(f"\nTraining complete. Best eval loss: {best_eval_val:.4f}")


if __name__ == "__main__":
    main()
