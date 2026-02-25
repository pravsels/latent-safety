#!/usr/bin/env python3
"""
Train the failure classifier head on top of a frozen DINO World Model.

Quickstart (training from scratch):

  python dino_wm/train_dino_classifier.py --hdf5-file /data/labeled/train.h5

Auto-resume (will automatically continue from last checkpoint if it exists):

  python dino_wm/train_dino_classifier.py --hdf5-file /data/labeled/train.h5

  The script automatically detects existing checkpoints in --checkpoint-dir and:
  - Loads the failure head weights from best_classifier.pth
  - Restores the iteration count
  - Preserves the best eval score

Resume from explicit checkpoint:

  python dino_wm/train_dino_classifier.py \
    --hdf5-file /data/labeled/train.h5 \
    --resume-checkpoint dino_wm_checkpoints/classifier.pth \
    --start-iter 5000
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
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from einops import rearrange
from tqdm import tqdm

from dino_decoder import VQVAE
from test_loader import SplitTrajectoryDataset
from dino_models import (
    FUTURE_ACTION_HORIZON_MAX,
    VideoTransformer,
    normalize_acs,
    normalize_states,
)
from checkpoint_utils import filter_state_dict_by_shape
from train_wm_common import (
    build_train_loader,
    init_distributed_from_env,
    load_yaml_config as _load_yaml_config,
    sample_future_action_window,
)
from dino_wm.config import (
    MODEL_CONFIG,
    TRAIN_CONFIG,
    DECODER_CONFIG,
    get_dino_config,
    get_decoder_image_size,
)


def fail_loss(pred, fail_data):
    """
    Failure classification loss using margin-based hinge loss.
    
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


def _resolve_data_path(path: str | None, *, prefer_data_root: bool = False) -> str | None:
    """
    Resolve possibly-relative paths with optional scratch-root fallback.

    If LATENT_SAFETY_DATA_ROOT is set (e.g., by SLURM script), relative paths can be
    resolved under that directory instead of repo cwd.
    """
    if path is None:
        return None
    expanded = os.path.expanduser(os.path.expandvars(path))
    if os.path.isabs(expanded):
        return expanded

    data_root = os.environ.get("LATENT_SAFETY_DATA_ROOT")
    if data_root:
        candidate = os.path.join(data_root, expanded)
        if prefer_data_root or not os.path.exists(expanded):
            return candidate
    return expanded


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("configs", "dino_classifier_config.yaml"),
        help="Path to YAML config file (default: configs/dino_classifier_config.yaml). CLI flags override it.",
    )
    pre_args, remaining_argv = pre_parser.parse_known_args(argv)
    cfg = _load_yaml_config(pre_args.config)

    parser = argparse.ArgumentParser(
        description="Train failure classifier on top of frozen DINO World Model",
        parents=[pre_parser],
    )
    parser.add_argument(
        "--hdf5-file",
        "--hdf5",
        dest="hdf5_file",
        type=str,
        default=None,
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
        default=5e-5,
        help="Learning rate for failure head (default: 5e-5).",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=FUTURE_ACTION_HORIZON_MAX,
        help=(
            "Maximum future-action horizon expected by trajectory encoder "
            f"(default: {FUTURE_ACTION_HORIZON_MAX})."
        ),
    )
    parser.add_argument(
        "--future-action-steps-train",
        type=int,
        default=50,
        help="Max number of future actions to sample for training (default: 50).",
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
        default="Classifier",
        help="Wandb run name (default: Classifier).",
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
    known_dests = {a.dest for a in parser._actions}
    for k, v in (cfg or {}).items():
        if k in known_dests:
            parser.set_defaults(**{k: v})

    return parser.parse_args(remaining_argv)


def main(argv=None):
    args = parse_args(argv)
    rank, world_size, local_rank, is_distributed = init_distributed_from_env()
    is_rank0 = rank == 0

    if args.hdf5_file is None:
        raise ValueError(
            "Missing --hdf5-file. Provide it via CLI or set hdf5_file in the YAML config."
        )

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

    # Resolve config paths. In SLURM, LATENT_SAFETY_DATA_ROOT can redirect relative
    # data/checkpoint paths to scratch storage without hardcoding absolute paths in YAML.
    args.hdf5_file = _resolve_data_path(args.hdf5_file)
    args.dataset_stats = _resolve_data_path(args.dataset_stats)
    args.decoder_checkpoint = _resolve_data_path(args.decoder_checkpoint)
    args.wm_checkpoint = _resolve_data_path(args.wm_checkpoint)
    args.resume_checkpoint = _resolve_data_path(args.resume_checkpoint)
    args.checkpoint_dir = _resolve_data_path(args.checkpoint_dir, prefer_data_root=True)

    if is_distributed and str(args.device).startswith("cuda"):
        device = f"cuda:{local_rank}"
    else:
        device = args.device

    if is_rank0:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            config=vars(args)
        )

    use_amp = str(device).startswith("cuda") and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Set seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    BS = args.batch_size
    BL = args.sequence_length
    EVAL_H = args.eval_horizon
    H = args.context_length
    action_horizon = int(args.action_horizon)
    future_action_steps_train = int(args.future_action_steps_train)

    if action_horizon < 1:
        raise ValueError(f"--action-horizon must be >= 1 (got {action_horizon}).")
    if future_action_steps_train < 1:
        raise ValueError(
            f"--future-action-steps-train must be >= 1 (got {future_action_steps_train})."
        )
    max_future_len = min(action_horizon, future_action_steps_train)

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
    
    # Classifier predicts over BL-1 context frames, and additionally conditions on a
    # variable-size chunk of future actions after the context window.
    train_segment_len = BL + max_future_len
    eval_segment_len = max(32, EVAL_H + max_future_len)

    expert_data = SplitTrajectoryDataset(
        hdf5_file, train_segment_len, split='train', num_test=num_test, action_key=args.action_key
    )
    expert_data_eval = SplitTrajectoryDataset(
        hdf5_file, train_segment_len, split='test', num_test=num_test, action_key=args.action_key
    )
    expert_data_imagine = SplitTrajectoryDataset(
        hdf5_file, eval_segment_len, split='test', num_test=num_test, action_key=args.action_key
    )
    
    print(f"Dataset: {hdf5_file}")
    print(f"  Train: {num_traj - num_test} trajectories")
    print(f"  Eval:  {num_test} trajectories")

    train_loader, train_sampler = build_train_loader(
        expert_data,
        BS,
        is_distributed=is_distributed,
        rank=rank,
        world_size=world_size,
    )
    expert_loader = iter(train_loader)
    if is_rank0:
        expert_loader_eval = iter(DataLoader(expert_data_eval, batch_size=BS, shuffle=True))
        expert_loader_imagine = iter(DataLoader(expert_data_imagine, batch_size=1, shuffle=True))
    else:
        expert_loader_eval = None
        expert_loader_imagine = None
   
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
    if is_rank0:
        print(f"Loaded decoder from {args.decoder_checkpoint}")

    # Initialize world model and load checkpoint
    transition = VideoTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        num_frames=BL-1,
        action_horizon=action_horizon,
        dino_version=args.dino_version,
        **MODEL_CONFIG
    ).to(device)
    if is_distributed:
        transition = DistributedDataParallel(transition, device_ids=[local_rank], find_unused_parameters=True)
    transition_module = transition.module if is_distributed else transition
    
    # Always load the world model backbone first
    if is_rank0:
        print(f"Loading world model from {args.wm_checkpoint}")
    wm_ckpt = torch.load(args.wm_checkpoint, map_location=device)
    if isinstance(wm_ckpt, dict) and 'model_state_dict' in wm_ckpt:
        wm_state = wm_ckpt['model_state_dict']
    else:
        wm_state = wm_ckpt
    filtered, missing, unexpected, mismatched = filter_state_dict_by_shape(
        transition_module.state_dict(),
        wm_state,
    )
    transition_module.load_state_dict(filtered, strict=False)
    if is_rank0 and missing:
        print(f"Warning: missing {len(missing)} keys from world-model checkpoint.")
    if is_rank0 and unexpected:
        print(f"Warning: world-model checkpoint has {len(unexpected)} unexpected keys.")
    if is_rank0 and mismatched:
        print(f"Warning: skipped {len(mismatched)} mismatched world-model keys.")

    # Freeze all parameters except failure head
    for name, param in transition_module.named_parameters():
        param.requires_grad = name.startswith("failure_head")

    # Optimizer for failure head only
    optimizer = AdamW([
        {'params': transition_module.failure_head.parameters(), 'lr': args.learning_rate},
    ])

    # Load checkpoint for resuming training
    best_eval = float('inf')
    start_iter = args.start_iter
    best_ckpt_path = os.path.join(args.checkpoint_dir, 'best_classifier.pth')
    
    # Determine which checkpoint to load for failure head:
    # 1. If --resume-checkpoint is explicitly passed, use that
    # 2. Else if best_classifier.pth exists, auto-load from there
    # 3. Else train from scratch
    
    if args.resume_checkpoint is not None:
        # Explicit checkpoint specified
        if is_rank0:
            print(f"\n{'='*60}")
            print(f"RESUMING TRAINING - Using explicit checkpoint")
            print(f"{'='*60}")
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        if isinstance(ckpt, dict):
            if 'failure_head_state_dict' in ckpt:
                transition_module.failure_head.load_state_dict(ckpt['failure_head_state_dict'])
                if is_rank0:
                    print(f"  Loaded failure head weights from: {args.resume_checkpoint}")
            else:
                transition_module.failure_head.load_state_dict(ckpt)
                if is_rank0:
                    print(f"  Loaded failure head weights from: {args.resume_checkpoint}")
            if 'best_eval' in ckpt:
                best_eval = ckpt['best_eval']
                if is_rank0:
                    print(f"  Previous best eval loss: {best_eval:.4f}")
            if 'iteration' in ckpt and args.start_iter == 0:
                start_iter = ckpt['iteration'] + 1
                if is_rank0:
                    print(f"  Resuming from iteration: {start_iter}")
            elif args.start_iter > 0:
                if is_rank0:
                    print(f"  Using explicit start iteration: {args.start_iter}")
        else:
            transition_module.failure_head.load_state_dict(ckpt)
            if is_rank0:
                print(f"  Loaded failure head weights from: {args.resume_checkpoint}")
            if args.start_iter > 0:
                if is_rank0:
                    print(f"  Using explicit start iteration: {args.start_iter}")
        if is_rank0:
            print(f"{'='*60}\n")
        
    elif os.path.exists(best_ckpt_path):
        # Auto-load from best checkpoint
        if is_rank0:
            print(f"\n{'='*60}")
            print(f"RESUMING TRAINING - Found existing checkpoint")
            print(f"{'='*60}")
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        if isinstance(best_ckpt, dict):
            if 'failure_head_state_dict' in best_ckpt:
                transition_module.failure_head.load_state_dict(best_ckpt['failure_head_state_dict'])
                if is_rank0:
                    print(f"  Loaded failure head weights from: {best_ckpt_path}")
            if 'best_eval' in best_ckpt:
                best_eval = best_ckpt['best_eval']
                if is_rank0:
                    print(f"  Previous best eval loss: {best_eval:.4f}")
            if 'iteration' in best_ckpt and args.start_iter == 0:
                start_iter = best_ckpt['iteration'] + 1
                if is_rank0:
                    print(f"  Resuming from iteration: {start_iter}")
            elif args.start_iter > 0:
                if is_rank0:
                    print(f"  Using explicit start iteration: {args.start_iter}")
        if is_rank0:
            print(f"{'='*60}\n")
        
    else:
        # Training from scratch
        if is_rank0:
            print(f"\n{'='*60}")
            print(f"TRAINING FROM SCRATCH - No existing checkpoint found")
            print(f"  Checkpoint dir: {args.checkpoint_dir}")
            print(f"  Starting from iteration: {start_iter}")
            print(f"{'='*60}\n")
    
    train_iter = args.train_iters

    for i in tqdm(range(start_iter, train_iter), desc="Training", unit="iter", disable=not is_rank0):
        if i > 0 and i % len(train_loader) == 0:
            if train_sampler is not None:
                train_sampler.set_epoch(i)
            train_loader, train_sampler = build_train_loader(
                expert_data,
                BS,
                is_distributed=is_distributed,
                rank=rank,
                world_size=world_size,
            )
            expert_loader = iter(train_loader)
        if is_rank0 and i > 0 and i % len(expert_loader_eval) == 0:
            expert_loader_eval = iter(DataLoader(expert_data_eval, batch_size=BS, shuffle=True))
        if is_rank0 and i > 0 and i % len(expert_loader_imagine) == 0:
            expert_loader_imagine = iter(DataLoader(expert_data_imagine, batch_size=1, shuffle=True))

        data = next(expert_loader)

        data1 = data['cam_zed_embd'].to(device)
        data2 = data['cam_rs_embd'].to(device)
        inputs1 = data1[:, :BL-1]
        inputs2 = data2[:, :BL-1]

        data_state = data['state'].to(device)
        norm_states = normalize_states(
            data_state, state_min, state_max, q02=state_q02, q98=state_q98
        )
        states = norm_states[:, :BL-1]

        data_acs = data['action'].to(device)
        norm_acs = normalize_acs(
            data_acs, action_min, action_max, q02=action_q02, q98=action_q98
        )
        acs = norm_acs[:, :BL-1]
        future_len = sample_future_action_window(
            action_horizon=action_horizon,
            future_action_steps_train=future_action_steps_train,
        )
        t = BL - 2
        future_actions = norm_acs[:, t + 1 : t + 1 + future_len]
        
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            pred1, pred2, pred_state, pred_fail = transition(
                inputs1, inputs2, states, acs, future_actions
            )
            # Context indices are [0 .. BL-2] (e.g., BL=4 -> {BL-4, BL-3, BL-2} = {0,1,2});
            # with future_len actions starting at BL-1, target is one step after: (BL-2)+future_len+1.
            target_idx = BL - 1 + future_len 
            pred_fail_target = pred_fail[:, -1].squeeze(-1)
            target_fail = data['failure'][:, target_idx].to(device)
            failure_loss = fail_loss(pred_fail_target, target_fail)
            loss = failure_loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss = loss.item()
        if is_rank0:
            wandb.log({'train_loss': train_loss})
            print(f"\rIter {i}, Train Loss: {train_loss:.4f}", end='', flush=True)
        
        if (i) % args.eval_interval == 0:
            if is_distributed:
                torch.distributed.barrier()
            if is_rank0:
                eval_data = next(expert_loader_imagine)
                transition_module.eval()
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
                    
                    # Get decoder output size from config
                    decoder_h, decoder_w = DECODER_CONFIG['decoder_image_size']
                    
                    # Load and resize ground truth to match decoder output
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
                        t = (H - 1) + k
                        future_actions = all_acs[:, t + 1 : t + 1 + max_future_len]
                        pred1, pred2, pred_state, pred_fail = transition(
                            inputs1, inputs2, states, acs, future_actions
                        )
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
                    
                    gt_im1_raw = eval_data['agentview_image'][[0], :EVAL_H].squeeze().to(device).float()
                    gt_im2_raw = eval_data['robot0_eye_in_hand_image'][[0], :EVAL_H].squeeze().to(device).float()
                    
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

                    wandb.log({"video": wandb.Video(vid, fps=20, format="mp4")})

                    # Compute eval loss on held-out batch
                    eval_data = next(expert_loader_eval)

                    data1 = eval_data['cam_zed_embd'].to(device)
                    data2 = eval_data['cam_rs_embd'].to(device)

                    inputs1 = data1[:, :BL-1]
                    inputs2 = data2[:, :BL-1]

                    data_state = eval_data['state'].to(device)
                    norm_eval_states = normalize_states(
                        data_state, state_min, state_max, q02=state_q02, q98=state_q98
                    )
                    states = norm_eval_states[:, :BL-1]

                    data_acs = eval_data['action'].to(device)
                    norm_acs = normalize_acs(
                        data_acs, action_min, action_max, q02=action_q02, q98=action_q98
                    )
                    acs = norm_acs[:, :BL-1]
                    future_len = sample_future_action_window(
                        action_horizon=action_horizon,
                        future_action_steps_train=future_action_steps_train,
                    )
                    t = BL - 2
                    future_actions = norm_acs[:, t + 1 : t + 1 + future_len]

                    pred1, pred2, pred_state, pred_fail = transition(
                        inputs1, inputs2, states, acs, future_actions
                    )
                    target_idx = BL - 1 + future_len
                    pred_fail_target = pred_fail[:, -1].squeeze(-1)
                    target_fail = eval_data['failure'][:, target_idx].to(device)
                    failure_loss = fail_loss(pred_fail_target, target_fail)
                    loss = failure_loss
                    print(f"\rIter {i}, Eval Loss: {loss.item():.4f},")

                    os.makedirs(args.checkpoint_dir, exist_ok=True)
                    # Save latest checkpoint with iteration for resuming
                    torch.save({
                        'failure_head_state_dict': transition_module.failure_head.state_dict(),
                        'iteration': i,
                    }, os.path.join(args.checkpoint_dir, 'classifier.pth'))

                    if loss < best_eval:
                        best_eval = loss
                        print(f"New best at iter {i}, saving model.")
                        torch.save({
                            'failure_head_state_dict': transition_module.failure_head.state_dict(),
                            'best_eval': best_eval.item() if hasattr(best_eval, 'item') else best_eval,
                            'iteration': i,
                        }, os.path.join(args.checkpoint_dir, 'best_classifier.pth'))

                    transition_module.train()
                    # --- eval metrics ---
                    with torch.no_grad():
                        eval_scores = pred_fail_target.detach().reshape(-1)
                        eval_labels = target_fail.detach().reshape(-1)
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
            if is_distributed:
                torch.distributed.barrier()

    best_eval_val = best_eval.item() if hasattr(best_eval, 'item') else best_eval
    if is_rank0:
        print(f"\nTraining complete. Best eval loss: {best_eval_val:.4f}")
    if is_distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
