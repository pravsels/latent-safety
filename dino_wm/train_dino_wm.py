#!/usr/bin/env python3
"""
Train the DINO World Model (VideoTransformer) on trajectory data.

Quickstart:

  python dino_wm/train_dino_wm.py --hdf5-file arx5_subset_train.h5

With custom parameters:

  python dino_wm/train_dino_wm.py --hdf5-file arx5_subset_train.h5 --batch-size 128 
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
from torchvision import transforms
from torch.optim import AdamW
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import matplotlib.pyplot as plt
from tqdm import tqdm

from test_loader import SplitTrajectoryDataset
from dino_decoder import VQVAE
from dino_models import VideoTransformer, normalize_acs, normalize_states, unnormalize_states

dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')


transform = transforms.Compose([           
                                transforms.Resize(256),                    
                                transforms.CenterCrop(224),               
                                transforms.ToTensor(),                    
                                transforms.Normalize(                      
                                mean=[0.485, 0.456, 0.406],                
                                std=[0.229, 0.224, 0.225]              
                                )])


DINO_transform = transforms.Compose([           
                            transforms.Resize(224),
                            
                            transforms.ToTensor(),])
norm_transform = transforms.Normalize(                      
                                mean=[0.485, 0.456, 0.406],                
                                std=[0.229, 0.224, 0.225]              
                                )


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
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

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
    decoder.load_state_dict(torch.load(args.decoder_checkpoint))
    decoder.eval()
    print(f"Loaded decoder from {args.decoder_checkpoint}")

    # Initialize world model
    transition = VideoTransformer(
        image_size=(224, 224),
        dim=384,  # DINO feature dimension
        action_embed_dim=10,  # Action embedding dimension
        state_embed_dim=10,  # State embedding dimension
        state_dim=state_dim,  # Inferred from dataset stats
        action_dim=action_dim,  # Inferred from dataset stats
        depth=6,
        heads=16,
        mlp_dim=2048,
        num_frames=BL-1,
        dropout=0.1
    ).to(device)
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

    best_eval = float('inf')
    iters = []
    train_iter = args.train_iters

    for i in tqdm(range(train_iter), desc="Training", unit="iter"):
        if i % len(expert_loader) == 0:
            expert_loader = iter(DataLoader(expert_data, batch_size=BS, shuffle=True))
        if i % len(expert_loader_eval) == 0:
            expert_loader_eval = iter(DataLoader(expert_data_eval, batch_size=BS, shuffle=True))
        if i % len(expert_loader_imagine) == 0:
            expert_loader_imagine = iter(DataLoader(expert_data_imagine, batch_size=1, shuffle=True))

        data = next(expert_loader)

        data1 = data['cam_zed_embd'].to(device)
        inputs1 = data1[:, :-1]
        output1 = data1[:, 1:]

        data2 = data['cam_rs_embd'].to(device)
        inputs2 = data2[:, :-1]
        output2 = data2[:, 1:]

        data_state = data['state'].to(device)
        norm_states = normalize_states(data_state, state_min, state_max)
        inputs_states = norm_states[:, :-1]
        output_state = norm_states[:, 1:]

        data_acs = data['action'].to(device)
        norm_acs = normalize_acs(data_acs, action_min, action_max)
        acs = norm_acs[:, :-1]

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            pred1, pred2, pred_state, _ = transition(inputs1, inputs2, inputs_states, acs)
            im1_loss_tf = nn.MSELoss()(pred1, output1)
            im2_loss_tf = nn.MSELoss()(pred2, output2)
            state_loss_tf = nn.MSELoss()(pred_state, output_state)
            loss_tf = im1_loss_tf + im2_loss_tf + state_loss_tf

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            detach_pred1 = pred1
            detach_pred2 = pred2
            detach_pred_state = pred_state.detach()
            inputs1_ar = torch.cat([data1[:, [0]], detach_pred1[:, [0]]], dim=1)
            inputs2_ar = torch.cat([data2[:, [0]], detach_pred2[:, [0]]], dim=1)
            states_ar = torch.cat([norm_states[:,[0]], detach_pred_state[:, [0]]], dim=1)
            acs_ar = norm_acs[:, [0,1]]

            pred1_ar, pred2_ar, pred_state_ar, _ = transition(inputs1_ar, inputs2_ar, states_ar, acs_ar)
            output1_ar = data1[:, 2]
            output2_ar = data2[:, 2]
            output_state_ar = norm_states[:, 2]
            im1_loss_ar = nn.MSELoss()(pred1_ar[:,1], output1_ar)
            im2_loss_ar = nn.MSELoss()(pred2_ar[:,1], output2_ar)
            state_loss_ar = nn.MSELoss()(pred_state_ar[:,1], output_state_ar)
            loss_ar = im1_loss_ar + im2_loss_ar + state_loss_ar       

        loss = loss_tf + loss_ar*0.5

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss = loss.item()
        print(f"\rIter {i}, TF Loss: {loss_tf:.4f}, AR loss:{loss_ar:.4f}, front Loss: {im1_loss_tf.item():.4f}, wrist Loss: {im2_loss_tf.item():.4f}, state Loss: {state_loss_tf.item():.4f}", end='', flush=True)
        wandb.log({'train_loss': loss_tf, "train_loss_ar": loss_ar})
        
        # Evaluation
        if (i) % args.eval_interval == 0:
            iters.append(i)
            eval_data = next(expert_loader_imagine)
            transition.eval()
            with torch.no_grad():
                eval_data1 = eval_data['cam_zed_embd'].to(device)
                inputs1 = eval_data1[[0], :H].to(device)

                eval_data2 = eval_data['cam_rs_embd'].to(device)
                inputs2 = eval_data2[[0], :H].to(device)
                
                all_acs = eval_data['action'][[0]].to(device)
                all_acs = normalize_acs(all_acs, action_min, action_max)
                
                acs = eval_data['action'][[0],:H].to(device)
                acs = normalize_acs(acs, action_min, action_max)

                eval_states = eval_data['state'][[0],:H].to(device)
                inputs_states = normalize_states(eval_states, state_min, state_max)
                # Resize images to 224x224 to match decoder output
                im1s = eval_data['agentview_image'][[0], :H].squeeze().to(device)/255.  # (T, H, W, C)
                im2s = eval_data['robot0_eye_in_hand_image'][[0], :H].squeeze().to(device)/255.
                im1s = F.interpolate(im1s.permute(0, 3, 1, 2), size=(224, 224), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                im2s = F.interpolate(im2s.permute(0, 3, 1, 2), size=(224, 224), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                for k in range(EVAL_H-H):
                    pred1, pred2, pred_state, _ = transition(inputs1, inputs2, inputs_states, acs)

                    pred_latent = torch.cat([pred1[:,[-1]], pred2[:,[-1]]], dim=0)
                    pred_ims, _ = decoder(pred_latent)

                    pred_ims = rearrange(pred_ims, "(b t) c h w -> b t h w c", t=1)
                    pred_im1, pred_im2 = torch.split(pred_ims, [inputs1.shape[0], inputs2.shape[0]], dim=0)

                    im1s = torch.cat([im1s, pred_im1.squeeze(0)], dim=0)
                    im2s = torch.cat([im2s, pred_im2.squeeze(0)], dim=0)
                    
                    # getting next inputs
                    acs = torch.cat([acs[[0], 1:], all_acs[0,H+k].unsqueeze(0).unsqueeze(0)], dim=1)
                    inputs1 = torch.cat([inputs1[[0], 1:], pred1[:, -1].unsqueeze(1)], dim=1)
                    inputs2 = torch.cat([inputs2[[0], 1:], pred2[:, -1].unsqueeze(1)], dim=1)
                    inputs_states = torch.cat([inputs_states[[0], 1:], pred_state[:,-1].unsqueeze(1)], dim=1)

                gt_im1 = eval_data['agentview_image'][[0], :EVAL_H].squeeze().to(device)  # (T, H, W, C)
                gt_im2 = eval_data['robot0_eye_in_hand_image'][[0], :EVAL_H].squeeze().to(device)
                # Resize to 224x224 to match decoder output
                gt_im1 = F.interpolate(gt_im1.permute(0, 3, 1, 2).float(), size=(224, 224), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                gt_im2 = F.interpolate(gt_im2.permute(0, 3, 1, 2).float(), size=(224, 224), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

                gt_imgs = torch.cat([gt_im1, gt_im2], dim=-2)/255.
                pred_imgs = torch.cat([im1s, im2s], dim=-2)
                vid = torch.cat([gt_imgs, pred_imgs], dim=-3)
                vid = vid.detach().cpu().numpy()
                vid = (vid * 255).clip(0, 255).astype(np.uint8)
                vid = rearrange(vid, "t h w c -> t c h w")
                wandb.log({"video": wandb.Video(vid, fps=20, format='mp4')})
                
                # done logging video

                eval_data = next(expert_loader_eval)
                data1 = eval_data['cam_zed_embd'].to(device)
                data2 = eval_data['cam_rs_embd'].to(device)

                inputs1 = data1[:, :-1]
                output1 = data1[:, 1:]

                inputs2 = data2[:, :-1]
                output2 = data2[:, 1:]

                data_state = eval_data['state'].to(device)
                norm_eval_states = normalize_states(data_state, state_min, state_max)
                states = norm_eval_states[:, :-1]
                output_state = norm_eval_states[:, 1:]

                data_acs = eval_data['action'].to(device)
                data_acs = normalize_acs(data_acs, action_min, action_max)
                acs = data_acs[:, :-1]
                pred1, pred2, pred_state, _ = transition(inputs1, inputs2, states, acs)

                pred_latent = torch.cat([pred1[:,[H-1]], pred2[:,[H-1]]], dim=0)
                pred_ims, _ = decoder(pred_latent)
                pred_im1, pred_im2 = torch.split(pred_ims, [inputs1.shape[0], inputs2.shape[0]], dim=0)
                pred_im1 = pred_im1[0].permute(1,2,0).detach().cpu().numpy()
                pred_im2 = pred_im2[0].permute(1,2,0).detach().cpu().numpy()
                im1 = eval_data['agentview_image'][0, H].numpy()
                im2 = eval_data['robot0_eye_in_hand_image'][0, H].numpy()
                im1_loss = nn.MSELoss()(pred1, output1)
                im2_loss = nn.MSELoss()(pred2, output2)
                state_loss = nn.MSELoss()(pred_state, output_state)
                loss = im1_loss + im2_loss + state_loss
            print()
            print(f"\rIter {i}, Eval Loss: {loss.item():.4f}, front Loss: {im1_loss.item():.4f}, wrist Loss: {im2_loss.item():.4f}, state Loss: {state_loss.item():.4f}")

            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save(transition.state_dict(), os.path.join(args.checkpoint_dir, f'wm_iter{i}.pth'))

            if loss < best_eval:
                best_eval = loss
                torch.save(transition.state_dict(), os.path.join(args.checkpoint_dir, 'best_wm.pth'))
            
            transition.train()
            wandb.log({'eval_loss': loss.item(), 'front_loss': im1_loss.item(), 'wrist_loss': im2_loss.item(), 'state_loss': state_loss.item(), 'pred_front': wandb.Image(pred_im1), 'pred_wrist': wandb.Image(pred_im2), 'front': wandb.Image(im1), 'wrist': wandb.Image(im2)})

    plt.legend()
    plt.savefig(os.path.join(args.checkpoint_dir, 'training_curve.png'))


if __name__ == "__main__":
    main()
