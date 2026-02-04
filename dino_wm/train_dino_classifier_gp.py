#!/usr/bin/env python3
"""
Train the failure classifier head with gradient penalty regularization on top of a frozen DINO World Model.

This variant uses WGAN-GP style gradient penalty to enforce Lipschitz constraints on the classifier,
which can improve stability and generalization.

Quickstart:

  python dino_wm/train_dino_classifier_gp.py --hdf5-file /data/labeled/train.h5

Resume training:

  python dino_wm/train_dino_classifier_gp.py \
    --hdf5-file /data/labeled/train.h5 \
    --resume-checkpoint dino_wm_checkpoints/classifier_gp.pth \
    --start-iter 5000
"""

import argparse
import os
import h5py
import json
import numpy as np
import torch
import random
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader
from einops import rearrange
from tqdm import tqdm

from dino_decoder import VQVAE
from test_loader import SplitTrajectoryDataset
from dino_models import VideoTransformer, normalize_acs, normalize_states
from dino_wm.config import (
    MODEL_CONFIG,
    TRAIN_CONFIG,
    DECODER_CONFIG,
    get_dino_config,
    get_decoder_image_size,
)


def fail_loss(pred, fail_data):
    """
    Basic failure classification loss without gradient penalty.
    
    Args:
        pred: Predicted failure scores
        fail_data: Ground truth labels (0=safe, 1=unsafe, 2=weak unsafe)
    
    Returns:
        Loss tensor (always a tensor, even if zero)
    """
    safe_data = torch.where(fail_data == 0.)
    unsafe_data = torch.where(fail_data == 1.)
    unsafe_data_weak = torch.where(fail_data == 2.)
    
    pos = pred[safe_data]
    neg = pred[unsafe_data]
    neg_weak = pred[unsafe_data_weak]

    gamma = 0.75
    # Initialize as tensor to ensure we always return a tensor
    lx_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    
    if pos.size(0) > 0:
        lx_loss = lx_loss + (1/pos.size(0))*torch.sum(torch.relu(gamma - pos))  # penalizes safe for being negative
    if neg.size(0) > 0:
        lx_loss = lx_loss + (1/neg.size(0))*torch.sum(torch.relu(gamma + neg))  # penalizes unsafe for being positive
    if neg_weak.size(0) > 0:
        lx_loss = lx_loss + (1/neg_weak.size(0))*torch.sum(torch.relu(neg_weak))  # penalizes weak unsafe for being positive

    return lx_loss


def fail_loss_gp(transition, feat, fail_data, gp_weight=10.0, relu_weight=100.0, target_gradient_norm=2.1):
    """
    Failure classification loss with gradient penalty regularization.
    
    The gradient penalty enforces a Lipschitz constraint on the classifier,
    similar to WGAN-GP, which can improve training stability and generalization.
    
    Implements an Ordinal Hierarchical Margin for safety-critical classification:
      - Label 0 (Safe):        target score > +gamma  (penalized once)
      - Label 2 (Weak Unsafe): target score < 0       (penalized once)
      - Label 1 (Hard Unsafe): target score < -gamma  (penalized twice - creates buffer zone)
    
    Args:
        transition: The VideoTransformer model
        feat: Latent features from the model
        fail_data: Ground truth failure labels (0=safe, 1=hard unsafe, 2=weak unsafe)
        gp_weight: Weight for gradient penalty term (default: 10.0)
        relu_weight: Weight for ReLU margin loss term (default: 100.0)
        target_gradient_norm: Target gradient norm for GP (default: 2.1)
    """
    safe_data = torch.where(fail_data == 0.)
    hard_unsafe_data = torch.where(fail_data == 1.)
    all_unsafe_data = torch.where(fail_data != 0)  # Both label 1 and 2

    pred = transition.failure_pred(feat)
    pos = pred[safe_data]
    neg = pred[hard_unsafe_data]
    all_unsafe = pred[all_unsafe_data]

    safe_dataset = feat[safe_data]
    unsafe_dataset = feat[hard_unsafe_data]
    N = max(safe_dataset.shape[0], unsafe_dataset.shape[0])
    if min(safe_dataset.shape[0], unsafe_dataset.shape[0]) != 0:
        if N > safe_dataset.shape[0]:
            repeat_times = (N + safe_dataset.shape[0] - 1) // safe_dataset.shape[0]  # Ceiling division
            safe_repeated = safe_dataset.repeat((repeat_times,) + (1,) * (safe_dataset.dim() - 1))  # Repeat along batch dim
            indices = torch.randperm(safe_repeated.shape[0], device=safe_dataset.device)[:N]
            pos_data =  safe_repeated[indices]
        else:
            pos_data = safe_dataset
        if N > unsafe_dataset.shape[0]:
            repeat_times = (N + unsafe_dataset.shape[0] - 1) // unsafe_dataset.shape[0]  # Ceiling division
            unsafe_repeated = unsafe_dataset.repeat((repeat_times,) + (1,) * (unsafe_dataset.dim() - 1))  # Repeat along batch dim
            indices = torch.randperm(unsafe_repeated.shape[0], device=unsafe_dataset.device)[:N]
            neg_data =  unsafe_repeated[indices]
        else:
            neg_data = unsafe_dataset

        # gradient penalty
        alpha = torch.rand(pos_data.shape[0], 1, device=pos_data.device)
        alpha = alpha.view(-1, 1, 1)  # Shape: (N, 1, 1)
        interpolates = alpha * pos_data + (1 - alpha) * neg_data
        interpolates.requires_grad_(True)
        disc_interpolates = transition.failure_pred(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(pos_data.shape[0], -1)
        gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)
        gp_loss = ((gradients_norm - target_gradient_norm) ** 2).mean()
    else:
        gp_loss = torch.tensor(0.0, device=feat.device)
    
    gamma = 0.75
    # Handle empty tensor case to avoid NaN
    if all_unsafe.size(0) > 0 and pos.size(0) > 0:
        zero_sum_loss = all_unsafe.mean() + (-pos.mean())
    elif all_unsafe.size(0) > 0:
        zero_sum_loss = all_unsafe.mean()
    elif pos.size(0) > 0:
        zero_sum_loss = -pos.mean()
    else:
        zero_sum_loss = torch.tensor(0.0, device=feat.device)
    
    # Hierarchical margin loss:
    # - Safe (label 0): push above +gamma
    # - Hard unsafe (label 1): push below -gamma (via neg) AND below 0 (via all_unsafe)
    # - Weak unsafe (label 2): push below 0 (via all_unsafe only)
    relu_loss = torch.tensor(0.0, device=feat.device)
    if pos.size(0) > 0:
        relu_loss = relu_loss + (1/pos.size(0))*torch.sum(torch.relu(gamma - pos))
    if neg.size(0) > 0:
        relu_loss = relu_loss + (1/neg.size(0))*torch.sum(torch.relu(gamma + neg))
    if all_unsafe.size(0) > 0:
        relu_loss = relu_loss + (1/all_unsafe.size(0))*torch.sum(torch.relu(all_unsafe))

    lx_loss = zero_sum_loss + gp_weight * gp_loss + relu_weight * relu_loss
    
    return lx_loss


def _compute_confusion(pred_scores, labels, threshold: float = 0.0):
    pred_pos = pred_scores > threshold
    gt_pos = labels > 0
    tp = torch.sum(pred_pos & gt_pos).float()
    fn = torch.sum((~pred_pos) & gt_pos).float()
    fp = torch.sum(pred_pos & (~gt_pos)).float()
    tn = torch.sum((~pred_pos) & (~gt_pos)).float()
    return tp, fn, fp, tn


def _precision_recall_f1(tp, fn, fp):
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser(
        description="Train failure classifier with gradient penalty on top of frozen DINO World Model"
    )
    parser.add_argument(
        "--hdf5-file",
        "--hdf5",
        dest="hdf5_file",
        type=str,
        required=True,
        help="Path to HDF5 file. Will be split into train/test using --test-frac.",
    )
    parser.add_argument(
        "--action-key",
        type=str,
        default="actions_delta",
        help="Action dataset key to load (default: actions_delta).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=TRAIN_CONFIG['batch_size'],
        help=f"Batch size for training (default: {TRAIN_CONFIG['batch_size']}).",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=TRAIN_CONFIG['sequence_length'],
        help=f"Sequence length for training (default: {TRAIN_CONFIG['sequence_length']}).",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=TRAIN_CONFIG['context_length'],
        help=f"Context length for autoregressive evaluation (default: {TRAIN_CONFIG['context_length']}).",
    )
    parser.add_argument(
        "--eval-horizon",
        type=int,
        default=16,
        help="Evaluation rollout horizon (default: 16).",
    )
    parser.add_argument(
        "--train-iters",
        type=int,
        default=10000,
        help="Number of training iterations (default: 10000).",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=500,
        help="Evaluation interval in iterations (default: 500).",
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
        default=None,
        help="Path to decoder checkpoint (defaults to a DINO-version-specific path).",
    )
    parser.add_argument(
        "--wm-checkpoint",
        type=str,
        default=None,
        help="Path to world model checkpoint to load (defaults to a DINO-version-specific path).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints (defaults to a DINO-version-specific path).",
    )
    parser.add_argument(
        "--dino-version",
        type=str,
        default="v3",
        choices=["v2", "v3"],
        help="DINO version used to generate embeddings (default: v3).",
    )
    parser.add_argument(
        "--dataset-stats",
        type=str,
        default="dataset_stats.json",
        help="Path to dataset statistics JSON file (default: dataset_stats.json).",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.1,
        help="Fraction of trajectories to use for evaluation (default: 0.1).",
    )
    parser.add_argument(
        "--num-test-trajectories",
        type=int,
        default=None,
        help="Explicit number of test trajectories (overrides --test-frac).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for failure head (default: 1e-4).",
    )
    parser.add_argument(
        "--gp-weight",
        type=float,
        default=10.0,
        help="Weight for gradient penalty term (default: 10.0).",
    )
    parser.add_argument(
        "--relu-weight",
        type=float,
        default=100.0,
        help="Weight for ReLU margin loss term (default: 100.0).",
    )
    parser.add_argument(
        "--target-gradient-norm",
        type=float,
        default=2.1,
        help="Target gradient norm for gradient penalty (default: 2.1).",
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
        default="Classifier-GP",
        help="Wandb run name (default: Classifier-GP).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0).",
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

    if args.decoder_checkpoint is None:
        args.decoder_checkpoint = (
            "dino3_decoder_checkpoints/best_decoder.pth"
            if args.dino_version == "v3"
            else "dino2_decoder_checkpoints/testing_decoder.pth"
        )
    if args.wm_checkpoint is None:
        args.wm_checkpoint = (
            "dino3_wm_checkpoints/best_wm.pth"
            if args.dino_version == "v3"
            else "dino2_wm_checkpoints/best_wm.pth"
        )
    if args.checkpoint_dir is None:
        args.checkpoint_dir = (
            "dino3_wm_checkpoints"
            if args.dino_version == "v3"
            else "dino2_wm_checkpoints"
        )

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        config=vars(args)
    )

    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    BS = args.batch_size
    BL = args.sequence_length
    EVAL_H = args.eval_horizon
    H = args.context_length
    device = args.device

    # Load state normalization stats
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
    action_q02 = torch.tensor(stats['action_delta_q02']).float().to(device) if "action_delta_q02" in stats else None
    action_q98 = torch.tensor(stats['action_delta_q98']).float().to(device) if "action_delta_q98" in stats else None
    state_q02 = torch.tensor(stats['state_q02']).float().to(device) if "state_q02" in stats else None
    state_q98 = torch.tensor(stats['state_q98']).float().to(device) if "state_q98" in stats else None
    
    # Infer dimensions from stats
    state_dim = len(stats['state_min'])
    action_dim = len(stats['action_min'])
    
    print(f"Loaded state normalization stats from {stats_path}")
    print(f"Inferred state_dim={state_dim}, action_dim={action_dim} from dataset stats")

    # Dataset setup
    hdf5_file = args.hdf5_file
    
    with h5py.File(hdf5_file, "r") as hf:
        num_traj = len(hf.keys())
    
    if args.num_test_trajectories is not None:
        num_test = max(1, min(args.num_test_trajectories, num_traj))
    else:
        num_test = max(1, int(args.test_frac * num_traj))
    
    if num_traj - num_test < 1 and num_traj > 1:
        num_test = num_traj - 1
    
    expert_data = SplitTrajectoryDataset(
        hdf5_file, BL, split='train', num_test=num_test, action_key=args.action_key
    )
    expert_data_eval = SplitTrajectoryDataset(
        hdf5_file, BL, split='test', num_test=num_test, action_key=args.action_key
    )
    expert_data_imagine = SplitTrajectoryDataset(
        hdf5_file, 32, split='test', num_test=num_test, action_key=args.action_key
    )
    
    print(f"Dataset: {hdf5_file}")
    print(f"  Train: {num_traj - num_test} trajectories")
    print(f"  Eval:  {num_test} trajectories")

    expert_loader = iter(DataLoader(expert_data, batch_size=BS, shuffle=True))
    expert_loader_eval = iter(DataLoader(expert_data_eval, batch_size=BS, shuffle=True))
    expert_loader_imagine = iter(DataLoader(expert_data_imagine, batch_size=1, shuffle=True))
   
    # Configure model dimensions and image sizes based on selected DINO version
    dino_cfg = get_dino_config(args.dino_version)
    decoder_img_size = get_decoder_image_size(args.dino_version)
    MODEL_CONFIG['dim'] = dino_cfg['dim']
    MODEL_CONFIG['image_size'] = decoder_img_size
    DECODER_CONFIG['decoder_image_size'] = decoder_img_size

    # Load decoder
    decoder = VQVAE().to(device)
    decoder_ckpt = torch.load(args.decoder_checkpoint, map_location=device)
    if isinstance(decoder_ckpt, dict) and 'model_state_dict' in decoder_ckpt:
        decoder.load_state_dict(decoder_ckpt['model_state_dict'])
    else:
        decoder.load_state_dict(decoder_ckpt)
    decoder.eval()
    print(f"Loaded decoder from {args.decoder_checkpoint}")

    # Initialize world model and load checkpoint
    transition = VideoTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        num_frames=BL-1,
        dino_version=args.dino_version,
        **MODEL_CONFIG
    ).to(device)
    
    if args.resume_checkpoint is not None:
        print(f"Resuming from checkpoint: {args.resume_checkpoint}")
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        # Resume only the failure head (world model remains frozen).
        if isinstance(ckpt, dict) and 'failure_head_state_dict' in ckpt:
            transition.failure_head.load_state_dict(ckpt['failure_head_state_dict'])
        else:
            transition.failure_head.load_state_dict(ckpt)
    else:
        print(f"Loading world model from {args.wm_checkpoint}")
        wm_ckpt = torch.load(args.wm_checkpoint, map_location=device)
        if isinstance(wm_ckpt, dict) and 'model_state_dict' in wm_ckpt:
            transition.load_state_dict(wm_ckpt['model_state_dict'])
        else:
            transition.load_state_dict(wm_ckpt)

    # Freeze all parameters except failure head
    for name, param in transition.named_parameters():
        param.requires_grad = name.startswith("failure_head")

    # Optimizer for failure head only
    optimizer = AdamW([
        {'params': transition.failure_head.parameters(), 'lr': args.learning_rate}, 
    ])

    # Load best_eval from existing best checkpoint to persist across sessions
    best_eval = float('inf')
    best_ckpt_path = os.path.join(args.checkpoint_dir, 'best_classifier_gp.pth')
    if os.path.exists(best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        if isinstance(best_ckpt, dict):
            if 'failure_head_state_dict' in best_ckpt:
                transition.failure_head.load_state_dict(best_ckpt['failure_head_state_dict'])
            if 'best_eval' in best_ckpt:
                best_eval = best_ckpt['best_eval']
                print(f"Loaded previous best eval: {best_eval:.4f}")
    
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

        data1 = data['cam_zed_embd'].to(device)
        data2 = data['cam_rs_embd'].to(device)
        inputs1 = data1[:, :-1]
        inputs2 = data2[:, :-1]

        data_state = data['state'].to(device)
        norm_states = normalize_states(
            data_state, state_min, state_max, q02=state_q02, q98=state_q98
        )
        states = norm_states[:, :-1]

        data_acs = data['action'].to(device)
        norm_acs = normalize_acs(
            data_acs, action_min, action_max, q02=action_q02, q98=action_q98
        )
        acs = norm_acs[:, :-1]
        
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            # Get latent features for gradient penalty computation
            latent = transition.forward_features(inputs1, inputs2, states, acs)

            failure_loss = fail_loss_gp(
                transition, latent, data['failure'][:, 1:].to(device),
                gp_weight=args.gp_weight,
                relu_weight=args.relu_weight,
                target_gradient_norm=args.target_gradient_norm
            )
            loss = failure_loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss = loss.item()
        wandb.log({'train_loss': train_loss})
        print(f"\rIter {i}, Train Loss: {train_loss:.4f}", end='', flush=True)
        
        if (i) % args.eval_interval == 0:
            eval_data = next(expert_loader_imagine)
            transition.eval()
            with torch.no_grad():
                eval_data1 = eval_data['cam_zed_embd'].to(device)
                eval_data2 = eval_data['cam_rs_embd'].to(device)

                inputs1 = eval_data1[[0], :H]
                inputs2 = eval_data2[[0], :H]
                all_acs = eval_data['action'][[0]].to(device)
                all_acs = normalize_acs(
                    all_acs, action_min, action_max, q02=action_q02, q98=action_q98
                )
                acs = eval_data['action'][[0],:H].to(device)
                acs = normalize_acs(
                    acs, action_min, action_max, q02=action_q02, q98=action_q98
                )
                eval_states = eval_data['state'][[0],:H].to(device)
                states = normalize_states(
                    eval_states, state_min, state_max, q02=state_q02, q98=state_q98
                )
                decoder_h, decoder_w = DECODER_CONFIG['decoder_image_size']
                im1s_raw = eval_data['agentview_image'][[0], :H].squeeze().to(device)/255.
                im2s_raw = eval_data['robot0_eye_in_hand_image'][[0], :H].squeeze().to(device)/255.
                im1s = torch.nn.functional.interpolate(
                    im1s_raw.permute(0,3,1,2), size=(decoder_h, decoder_w),
                    mode='bilinear', align_corners=False
                ).permute(0,2,3,1)
                im2s = torch.nn.functional.interpolate(
                    im2s_raw.permute(0,3,1,2), size=(decoder_h, decoder_w),
                    mode='bilinear', align_corners=False
                ).permute(0,2,3,1)
                for k in range(EVAL_H-H):
                    pred1, pred2, pred_state, pred_fail = transition(inputs1, inputs2, states, acs)
                    pred_latent = torch.cat([pred1[:,[-1]], pred2[:,[-1]]], dim=0)
                    pred_ims, _ = decoder(pred_latent)

                    pred_ims = rearrange(pred_ims, "(b t) c h w -> b t c h w", t=1)
                    pred_im1, pred_im2 = torch.split(pred_ims, [inputs1.shape[0], inputs2.shape[0]], dim=0)

                    pred_im1 = pred_im1[0].permute(0,2,3,1).detach()
                    pred_im2 = pred_im2[0].permute(0,2,3,1).detach()
                    pred_fail = pred_fail[:,-1]

                    if pred_fail < 0:
                        pred_im1[:,:,:,0] *= 2
                        pred_im2[:,:,:,0] *= 2
                    
                    im1s = torch.cat([im1s, pred_im1], dim=0)
                    im2s = torch.cat([im2s, pred_im2], dim=0)
                    
                    # getting next inputs
                    acs = torch.cat([acs[[0], 1:], all_acs[0,H+k].unsqueeze(0).unsqueeze(0)], dim=1)
                    inputs1 = torch.cat([inputs1[[0], 1:], pred1[:, -1].unsqueeze(1)], dim=1)
                    inputs2 = torch.cat([inputs2[[0], 1:], pred2[:, -1].unsqueeze(1)], dim=1)
                    states = torch.cat([states[[0], 1:], pred_state[:,-1].unsqueeze(1)], dim=1)
                
                gt_im1_raw = eval_data['agentview_image'][[0], :EVAL_H].squeeze().to(device)
                gt_im2_raw = eval_data['robot0_eye_in_hand_image'][[0], :EVAL_H].squeeze().to(device)
                gt_im1 = torch.nn.functional.interpolate(
                    gt_im1_raw.permute(0,3,1,2), size=(decoder_h, decoder_w),
                    mode='bilinear', align_corners=False
                ).permute(0,2,3,1)
                gt_im2 = torch.nn.functional.interpolate(
                    gt_im2_raw.permute(0,3,1,2), size=(decoder_h, decoder_w),
                    mode='bilinear', align_corners=False
                ).permute(0,2,3,1)
                gt_fail = eval_data['failure'][[0], :EVAL_H].squeeze().to(device)
                
                for j in range(EVAL_H):
                    if gt_fail[j] > 0:
                        gt_im1[j,:,:,0] *= 2
                        gt_im2[j,:,:,0] *= 2
                
                gt_imgs = torch.cat([gt_im1, gt_im2], dim=-3)/255.
                pred_imgs = torch.cat([im1s, im2s], dim=-3)

                vid = torch.cat([gt_imgs, pred_imgs], dim=-2)
                vid = vid[H:]

                vid = rearrange(vid, "t h w c -> t c h w")
                vid = vid.detach().cpu().numpy()
                vid = (vid * 255).clip(0, 255).astype(np.uint8)

                wandb.log({"video": wandb.Video(vid, fps=20)})

                # Compute eval loss on held-out batch
                eval_data = next(expert_loader_eval)

                data1 = eval_data['cam_zed_embd'].to(device)
                data2 = eval_data['cam_rs_embd'].to(device)

                inputs1 = data1[:, :-1]
                inputs2 = data2[:, :-1]

                data_state = eval_data['state'].to(device)
                norm_eval_states = normalize_states(
                    data_state, state_min, state_max, q02=state_q02, q98=state_q98
                )
                states = norm_eval_states[:, :-1]

                data_acs = eval_data['action'].to(device)
                norm_acs = normalize_acs(
                    data_acs, action_min, action_max, q02=action_q02, q98=action_q98
                )
                acs = norm_acs[:, :-1]

                pred1, pred2, pred_state, pred_fail = transition(inputs1, inputs2, states, acs)
                
                failure_loss = fail_loss(pred_fail, eval_data['failure'][:, 1:].to(device))
                loss = failure_loss
            print(f"\rIter {i}, Eval Loss: {loss.item():.4f},")

            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save(
                transition.failure_head.state_dict(),
                os.path.join(args.checkpoint_dir, 'classifier_gp.pth')
            )

            if loss < best_eval:
                best_eval = loss
                print(f"New best at iter {i}, saving model.")
                torch.save({
                    'failure_head_state_dict': transition.failure_head.state_dict(),
                    'best_eval': best_eval.item() if hasattr(best_eval, 'item') else best_eval
                }, os.path.join(args.checkpoint_dir, 'best_classifier_gp.pth'))

            
            transition.train()
            # --- eval metrics ---
            with torch.no_grad():
                eval_scores = pred_fail.detach().reshape(-1)
                eval_labels = eval_data['failure'][:, 1:].to(device).detach().reshape(-1)
                tp, fn, fp, tn = _compute_confusion(eval_scores, eval_labels, threshold=0.0)
                precision, recall, f1 = _precision_recall_f1(tp, fn, fp)

                thresholds = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
                sweep = {}
                for thr in thresholds:
                    ttp, tfn, tfp, _ = _compute_confusion(eval_scores, eval_labels, threshold=thr)
                    p, r, f = _precision_recall_f1(ttp, tfn, tfp)
                    sweep[f"eval/precision@{thr}"] = p.item()
                    sweep[f"eval/recall@{thr}"] = r.item()
                    sweep[f"eval/f1@{thr}"] = f.item()

            wandb.log({
                'eval_loss': loss.item(),
                'eval/tp': tp.item(),
                'eval/fn': fn.item(),
                'eval/fp': fp.item(),
                'eval/tn': tn.item(),
                'eval/precision': precision.item(),
                'eval/recall': recall.item(),
                'eval/f1': f1.item(),
                **sweep,
            })

    best_eval_val = best_eval.item() if hasattr(best_eval, 'item') else best_eval
    print(f"\nTraining complete. Best eval loss: {best_eval_val:.4f}")


if __name__ == "__main__":
    main()
