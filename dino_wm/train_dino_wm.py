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
import sys
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

# Ensure repo root is on sys.path regardless of current working directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from test_loader import SplitTrajectoryDataset
from dino_decoder import VQVAE
from dino_models import VideoTransformer, normalize_acs, normalize_states, unnormalize_states
from dino_wm.config import MODEL_CONFIG, TRAIN_CONFIG, DECODER_CONFIG, get_dino_config, get_decoder_image_size


def _global_grad_norm(parameters, norm_type: float = 2.0) -> float:
    """Compute global grad norm over a set of parameters."""
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    if norm_type == float("inf"):
        return max(g.abs().max().item() for g in grads)
    total = 0.0
    for g in grads:
        total += g.norm(norm_type).item() ** norm_type
    return total ** (1.0 / norm_type)


def _global_weight_norm(parameters, norm_type: float = 2.0) -> float:
    params = [p.detach() for p in parameters]
    if not params:
        return 0.0
    if norm_type == float("inf"):
        return max(p.abs().max().item() for p in params)
    total = 0.0
    for p in params:
        total += p.norm(norm_type).item() ** norm_type
    return total ** (1.0 / norm_type)


def _load_yaml_config(path: str) -> dict:
    """
    Load YAML into a plain dict.
    Uses ruamel.yaml (repo dependency via setup.py) and supports env var expansion.
    """
    import pathlib
    import ruamel.yaml as ryaml

    p = os.path.expandvars(os.path.expanduser(path))
    if not os.path.exists(p):
        return {}
    cfg = ryaml.YAML(typ="safe", pure=True).load(pathlib.Path(p).read_text()) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping (YAML dict). Got: {type(cfg)}")
    return cfg


def _compute_lr_factor(
    step: int,
    total_steps: int,
    warmup_steps: int,
    min_lr_factor: float,
    schedule: str,
) -> float:
    """Compute LR multiplier factor for a given global step."""
    if schedule == "constant":
        return 1.0

    # warmup
    warmup_steps = int(max(0, warmup_steps))
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)

    # cosine over the remaining steps
    denom = max(1, int(total_steps) - warmup_steps)
    t = min(1.0, max(0.0, float(step - warmup_steps) / float(denom)))
    cosine = 0.5 * (1.0 + float(torch.cos(torch.tensor(t * 3.141592653589793)).item()))
    return float(min_lr_factor) + (1.0 - float(min_lr_factor)) * cosine


def main():
    # Parse config path first so we can apply YAML values as argparse defaults.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("configs", "wm_config.yaml"),
        help="Path to YAML config file (default: configs/wm_config.yaml). CLI flags override it.",
    )
    pre_args, remaining_argv = pre_parser.parse_known_args()
    cfg = _load_yaml_config(pre_args.config)

    parser = argparse.ArgumentParser(
        description="Train DINO World Model on trajectory data",
        parents=[pre_parser]
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
        "--dino-version",
        type=str,
        default="v3",
        choices=["v2", "v3"],
        help="DINO version to use (default: v3).",
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
        "--eval-samples",
        type=int,
        default=8,
        help="Number of evaluation samples to rollout for failure analysis (default: 8).",
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
    parser.add_argument(
        "--auto-resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Auto-resume from latest checkpoint in --checkpoint-dir (default: False).",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1000,
        help="Save a latest checkpoint every N iterations (default: 1000).",
    )
    # --- learning-rate scheduling ---
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="cosine",
        choices=["cosine", "constant"],
        help="Learning-rate schedule (default: cosine).",
    )
    parser.add_argument(
        "--lr-min-factor",
        type=float,
        default=0.1,
        help="Minimum LR factor for cosine schedule (default: 0.1).",
    )
    parser.add_argument(
        "--lr-warmup-iters",
        type=int,
        default=1000,
        help="Number of warmup iterations (default: 1000).",
    )

    # Apply YAML config as defaults (CLI overrides because we parse after this).
    known_dests = {a.dest for a in parser._actions}
    for k, v in (cfg or {}).items():
        if k in known_dests:
            parser.set_defaults(**{k: v})

    args = parser.parse_args(remaining_argv)

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

    # Configure model dimensions and image sizes based on selected DINO version
    dino_cfg = get_dino_config(args.dino_version)
    decoder_img_size = get_decoder_image_size(args.dino_version)

    MODEL_CONFIG['dim'] = dino_cfg['dim']
    MODEL_CONFIG['image_size'] = decoder_img_size
    DECODER_CONFIG['decoder_image_size'] = decoder_img_size

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
        dino_version=args.dino_version,
        **MODEL_CONFIG
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
    base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    # Load best_eval from existing best checkpoint to persist across sessions
    best_eval = float('inf')
    best_ckpt_path = os.path.join(args.checkpoint_dir, 'best_wm.pth')
    latest_ckpt_path = os.path.join(args.checkpoint_dir, 'latest_wm.pth')
    if os.path.exists(best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        if isinstance(best_ckpt, dict) and 'best_eval' in best_ckpt:
            best_eval = best_ckpt['best_eval']
            print(f"Loaded previous best eval: {best_eval:.4f}")

    # Resume logic
    resume_path = args.resume_checkpoint
    if resume_path is None and args.auto_resume and os.path.exists(latest_ckpt_path):
        resume_path = latest_ckpt_path

    start_iter = args.start_iter
    if resume_path is not None:
        print(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            transition.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt:
                try:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                except Exception as e:
                    print(f"Warning: failed to load optimizer state ({e}); continuing with fresh optimizer.")
            if 'iter' in ckpt:
                start_iter = int(ckpt['iter']) + 1
            if 'best_eval' in ckpt and best_eval == float('inf'):
                best_eval = ckpt['best_eval']
        else:
            transition.load_state_dict(ckpt)

    transition.train()

    iters = []
    train_iter = args.train_iters

    def _make_ckpt_dict(iter_idx: int) -> dict:
        return {
            'model_state_dict': transition.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iter': int(iter_idx),
            'best_eval': float(best_eval),
            'seed': int(args.seed),
        }

    for i in tqdm(range(start_iter, train_iter), desc="Training", unit="iter"):
        # --- LR update ---
        lr_factor = _compute_lr_factor(
            step=i,
            total_steps=train_iter,
            warmup_steps=int(args.lr_warmup_iters),
            min_lr_factor=float(args.lr_min_factor),
            schedule=str(args.lr_schedule),
        )
        for pg, base_lr in zip(optimizer.param_groups, base_lrs):
            pg['lr'] = base_lr * lr_factor

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
            input_state_ar = torch.cat([norm_gt_state[:, [0]], detach_pred_state[:, [0]]], dim=1)
            input_acs_ar = norm_gt_acs[:, [0, 1]]

            # AR Forward pass
            pred_front_ar, pred_wrist_ar, pred_state_ar, _ = transition(input_front_ar, input_wrist_ar, input_state_ar, input_acs_ar)

            # Targets for AR step: Frame 2 (index 2 in GT)
            target_front_ar = gt_front_embd[:, 2]
            target_wrist_ar = gt_wrist_embd[:, 2]
            target_state_ar = norm_gt_state[:, 2]

            # Calculate AR losses on the second step of prediction (corresponding to Frame 2)
            loss_front_ar = nn.MSELoss()(pred_front_ar[:, 1], target_front_ar)
            loss_wrist_ar = nn.MSELoss()(pred_wrist_ar[:, 1], target_wrist_ar)
            loss_state_ar = nn.MSELoss()(pred_state_ar[:, 1], target_state_ar)
            loss_ar = loss_front_ar + loss_wrist_ar + loss_state_ar

        loss = loss_tf + loss_ar * 0.5

        scaler.scale(loss).backward()

        # Norms and Step
        scaler.unscale_(optimizer)
        grad_norm = _global_grad_norm(transition.parameters())
        scaler.step(optimizer)
        scaler.update()
        weight_norm = _global_weight_norm(transition.parameters())

        train_loss = loss.item()
        print(
            f"\rIter {i} | lr {optimizer.param_groups[0]['lr']:.2e} | TF {loss_tf:.4f} | AR {loss_ar:.4f} | grad {grad_norm:.2f} | weight {weight_norm:.2f}",
            end='',
            flush=True
        )
        wandb.log({
            'train_loss': loss_tf,
            'train_loss_ar': loss_ar,
            'grad_norm': grad_norm,
            'weight_norm': weight_norm,
            'lr': optimizer.param_groups[0]['lr'],
            'lr_factor': lr_factor,
        })

        # Periodic "latest" checkpoint
        if args.save_every and (i % args.save_every == 0):
            torch.save(_make_ckpt_dict(i), latest_ckpt_path)

        # Evaluation
        if (i) % args.eval_interval == 0:
            iters.append(i)
            transition.eval()

            eval_specimens = []

            # Metrics to average
            avg_metrics = {
                'eval_loss': 0.0,
                'front_loss': 0.0,
                'wrist_loss': 0.0,
                'state_loss': 0.0,
            }

            num_samples = args.eval_samples

            print(f"\nRunning evaluation on {num_samples} samples...")

            for s_idx in range(num_samples):
                with torch.no_grad():
                    # Get sample for rollout (long sequence)
                    eval_data = next(expert_loader_imagine)
                    gt_front_embd_eval = eval_data['cam_zed_embd'].to(device)
                    input_front_embd_eval = gt_front_embd_eval[[0], :H].to(device)

                    gt_wrist_embd_eval = eval_data['cam_rs_embd'].to(device)
                    input_wrist_embd_eval = gt_wrist_embd_eval[[0], :H].to(device)

                    all_acs = eval_data['action'][[0]].to(device)
                    all_acs = normalize_acs(all_acs, action_min, action_max)

                    acs = eval_data['action'][[0], :H].to(device)
                    acs = normalize_acs(acs, action_min, action_max)

                    gt_states_eval = eval_data['state'][[0], :H].to(device)
                    input_states_eval = normalize_states(gt_states_eval, state_min, state_max)

                    # Original images for comparison and video
                    # Resize to DECODER_CONFIG['decoder_image_size']
                    im1s = eval_data['agentview_image'][[0], :H].squeeze().to(device) / 255.
                    im2s = eval_data['robot0_eye_in_hand_image'][[0], :H].squeeze().to(device) / 255.
                    im1s = F.interpolate(im1s.permute(0, 3, 1, 2), size=DECODER_CONFIG['decoder_image_size'], mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                    im2s = F.interpolate(im2s.permute(0, 3, 1, 2), size=DECODER_CONFIG['decoder_image_size'], mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

                    # Rollout
                    rollout_latent_mse = 0
                    for k in range(EVAL_H - H):
                        pred_front, pred_wrist, pred_state, _ = transition(input_front_embd_eval, input_wrist_embd_eval, input_states_eval, acs)

                        # Track rollout error (latent MSE)
                        target_front = gt_front_embd_eval[[0], H + k]
                        target_wrist = gt_wrist_embd_eval[[0], H + k]
                        rollout_latent_mse += nn.MSELoss()(pred_front[:, [-1]], target_front.unsqueeze(1)).item()
                        rollout_latent_mse += nn.MSELoss()(pred_wrist[:, [-1]], target_wrist.unsqueeze(1)).item()

                        pred_latent = torch.cat([pred_front[:, [-1]], pred_wrist[:, [-1]]], dim=0)
                        pred_ims, _ = decoder(pred_latent)
                        pred_ims = rearrange(pred_ims, "(b t) c h w -> b t h w c", t=1)
                        pred_im1, pred_im2 = torch.split(pred_ims, [input_front_embd_eval.shape[0], input_wrist_embd_eval.shape[0]], dim=0)

                        im1s = torch.cat([im1s, pred_im1.squeeze(0)], dim=0)
                        im2s = torch.cat([im2s, pred_im2.squeeze(0)], dim=0)

                        # Get next inputs
                        acs = torch.cat([acs[[0], 1:], all_acs[0, H + k].unsqueeze(0).unsqueeze(0)], dim=1)
                        input_front_embd_eval = torch.cat([input_front_embd_eval[[0], 1:], pred_front[:, -1].unsqueeze(1)], dim=1)
                        input_wrist_embd_eval = torch.cat([input_wrist_embd_eval[[0], 1:], pred_wrist[:, -1].unsqueeze(1)], dim=1)
                        input_states_eval = torch.cat([input_states_eval[[0], 1:], pred_state[:, -1].unsqueeze(1)], dim=1)

                    rollout_latent_mse /= (EVAL_H - H)

                    # Video prep
                    gt_im1 = eval_data['agentview_image'][[0], :EVAL_H].squeeze().to(device)
                    gt_im2 = eval_data['robot0_eye_in_hand_image'][[0], :EVAL_H].squeeze().to(device)
                    gt_im1 = F.interpolate(gt_im1.permute(0, 3, 1, 2).float(), size=DECODER_CONFIG['decoder_image_size'], mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                    gt_im2 = F.interpolate(gt_im2.permute(0, 3, 1, 2).float(), size=DECODER_CONFIG['decoder_image_size'], mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

                    gt_imgs = torch.cat([gt_im1, gt_im2], dim=-2) / 255.
                    pred_imgs = torch.cat([im1s, im2s], dim=-2)
                    vid = torch.cat([gt_imgs, pred_imgs], dim=-3)
                    vid = vid.detach().cpu().numpy()
                    vid = (vid * 255).clip(0, 255).astype(np.uint8)
                    vid = rearrange(vid, "t h w c -> t c h w")

                    # Store specimen
                    eval_specimens.append({
                        'traj_id': eval_data['traj_id'][0],
                        'start_idx': eval_data['start_idx'][0].item(),
                        'loss': rollout_latent_mse,
                        'video': wandb.Video(vid, fps=20, format='mp4'),
                        'last_im1': im1s[-1].detach().cpu().numpy(),
                        'last_im2': im2s[-1].detach().cpu().numpy(),
                        'gt_last_im1': (gt_im1[-1] / 255.).detach().cpu().numpy(),
                        'gt_last_im2': (gt_im2[-1] / 255.).detach().cpu().numpy(),
                    })

                    # Teacher Forcing metrics for this sample (using original seq length BL)
                    # We need a new sample from expert_loader_eval for standard metrics
                    eval_data_tf = next(expert_loader_eval)
                    tf_front = eval_data_tf['cam_zed_embd'].to(device)
                    tf_wrist = eval_data_tf['cam_rs_embd'].to(device)
                    tf_state = normalize_states(eval_data_tf['state'].to(device), state_min, state_max)
                    tf_acs = normalize_acs(eval_data_tf['action'].to(device), action_min, action_max)

                    p_front, p_wrist, p_state, _ = transition(tf_front[:, :-1], tf_wrist[:, :-1], tf_state[:, :-1], tf_acs[:, :-1])
                    l_front = nn.MSELoss()(p_front, tf_front[:, 1:]).item()
                    l_wrist = nn.MSELoss()(p_wrist, tf_wrist[:, 1:]).item()
                    l_state = nn.MSELoss()(p_state, tf_state[:, 1:]).item()

                    avg_metrics['eval_loss'] += (l_front + l_wrist + l_state)
                    avg_metrics['front_loss'] += l_front
                    avg_metrics['wrist_loss'] += l_wrist
                    avg_metrics['state_loss'] += l_state

            # Finalize metrics
            for k in avg_metrics:
                avg_metrics[k] /= num_samples

            # Sort and Log Table
            eval_specimens.sort(key=lambda x: x['loss'], reverse=True)
            columns = ["traj_id", "start_idx", "rollout_mse", "video"]
            table = wandb.Table(columns=columns)
            for spec in eval_specimens:
                table.add_data(spec['traj_id'], spec['start_idx'], spec['loss'], spec['video'])

            # Print summary
            print(f"\rIter {i}, Eval Loss: {avg_metrics['eval_loss']:.4f}, front: {avg_metrics['front_loss']:.4f}, wrist: {avg_metrics['wrist_loss']:.4f}, state: {avg_metrics['state_loss']:.4f}")

            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save(_make_ckpt_dict(i), os.path.join(args.checkpoint_dir, f'wm_iter{i}.pth'))

            if avg_metrics['eval_loss'] < best_eval:
                best_eval = avg_metrics['eval_loss']
                torch.save(_make_ckpt_dict(i), os.path.join(args.checkpoint_dir, 'best_wm.pth'))

            transition.train()

            # Log all to WandB
            log_dict = {
                'eval_loss': avg_metrics['eval_loss'],
                'front_loss': avg_metrics['front_loss'],
                'wrist_loss': avg_metrics['wrist_loss'],
                'state_loss': avg_metrics['state_loss'],
                'eval_failures_table': table,
                'video': eval_specimens[0]['video'], # Worst case video
            }
            # Add some sample images from the worst case
            worst_case = eval_specimens[0]
            log_dict.update({
                'pred_front': wandb.Image(worst_case['last_im1']),
                'pred_wrist': wandb.Image(worst_case['last_im2']),
                'front': wandb.Image(worst_case['gt_last_im1']),
                'wrist': wandb.Image(worst_case['gt_last_im2']),
            })
            wandb.log(log_dict)

    plt.legend()
    plt.savefig(os.path.join(args.checkpoint_dir, 'training_curve.png'))

    best_eval_val = best_eval.item() if hasattr(best_eval, 'item') else best_eval
    print(f"\nTraining complete. Best eval loss: {best_eval_val:.4f}")


if __name__ == "__main__":
    main()
