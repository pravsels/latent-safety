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
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch import nn
from torch.nn.parallel import DistributedDataParallel
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


def init_distributed_from_env() -> tuple[int, int, int, bool]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1
    if is_distributed:
        torch.distributed.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
        )
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, is_distributed


def build_train_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    *,
    is_distributed: bool,
    rank: int,
    world_size: int,
) -> tuple[DataLoader, DistributedSampler | None]:
    if is_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        sampler = None
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, sampler


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


def _build_action_tokens_from_raw(
    norm_actions_raw: torch.Tensor,
    step_starts: torch.Tensor,
    pred_step: int,
    mode: str,
) -> torch.Tensor:
    """
    Build one action token per model step from raw per-frame actions.

    Args:
        norm_actions_raw: (B, T, A) normalized actions at raw frame rate.
        step_starts: (L,) raw indices t for each model step start.
        pred_step: size of each model step in raw frames (>=1).
        mode:
          - "start": use a_t
          - "last":  use a_{t+pred_step-1}
          - "mean":  mean over a_t..a_{t+pred_step-1} (clipped at sequence end)
          - "sum":   sum  over a_t..a_{t+pred_step-1} (clipped at sequence end)

    Returns:
        (B, L, A) action tokens aligned to each step start.
    """
    if pred_step < 1:
        raise ValueError(f"pred_step must be >= 1 (got {pred_step})")
    if norm_actions_raw.ndim != 3:
        raise ValueError(f"norm_actions_raw must have shape (B,T,A); got {tuple(norm_actions_raw.shape)}")
    if step_starts.ndim != 1:
        raise ValueError(f"step_starts must have shape (L,); got {tuple(step_starts.shape)}")

    B, T, A = norm_actions_raw.shape
    step_starts = step_starts.to(device=norm_actions_raw.device, dtype=torch.long)

    if mode == "start" or pred_step == 1:
        # (B, T, A): dim=1 is the time axis, so we select timesteps given by step_starts.
        return norm_actions_raw.index_select(1, step_starts)

    if mode == "last":
        idx_last = torch.clamp(step_starts + (pred_step - 1), max=T - 1)
        return norm_actions_raw.index_select(1, idx_last)

    if mode in ("mean", "sum"):
        toks = []
        for t in step_starts.tolist():
            if t >= T:
                window = norm_actions_raw[:, (T - 1):T]  # (B,1,A)
            else:
                t_end = min(T, t + pred_step)
                window = norm_actions_raw[:, t:t_end]  # (B,K,A)
            toks.append(window.mean(dim=1) if mode == "mean" else window.sum(dim=1))
        return torch.stack(toks, dim=1)

    raise ValueError(f"Unknown action aggregation mode '{mode}'")


def compute_action_horizon_indices(
    *,
    context_length: int,
    pred_step: int,
    action_horizon: int,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, slice, int, int]:
    if context_length < 1:
        raise ValueError(f"context_length must be >= 1 (got {context_length})")
    if pred_step < 1:
        raise ValueError(f"pred_step must be >= 1 (got {pred_step})")
    if action_horizon < 1:
        raise ValueError(f"action_horizon must be >= 1 (got {action_horizon})")
    ctx_idx = torch.arange(context_length, device=device, dtype=torch.long) * pred_step
    t = int(ctx_idx[-1].item())
    future_slice = slice(t, t + action_horizon)
    target_idx = t + action_horizon
    segment_length = target_idx + 1
    return ctx_idx, future_slice, target_idx, segment_length


def compute_action_horizon_ar_indices(
    *,
    context_length: int,
    pred_step: int,
    action_horizon: int,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, slice, int, slice, int, int]:
    ctx_idx, future_slice, target_idx, _ = compute_action_horizon_indices(
        context_length=context_length,
        pred_step=pred_step,
        action_horizon=action_horizon,
        device=device,
    )
    ar_future_slice = slice(future_slice.start + 1, future_slice.stop + 1)
    ar_target_idx = target_idx + 1
    segment_length = ar_target_idx + 1
    return ctx_idx, future_slice, target_idx, ar_future_slice, ar_target_idx, segment_length


def compute_eval_t_plus_k_indices(
    *,
    context_length: int,
    pred_step: int,
    action_horizon: int,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, int, int]:
    ctx_idx, _, target_idx, segment_length = compute_action_horizon_indices(
        context_length=context_length,
        pred_step=pred_step,
        action_horizon=action_horizon,
        device=device,
    )
    return ctx_idx, target_idx, segment_length


def sample_future_action_window(
    *,
    action_horizon: int,
    future_action_max_steps: int,
    future_action_small_max_steps: int,
    future_action_small_prob: float,
    rng: random.Random | None = None,
) -> int:
    if action_horizon < 1:
        raise ValueError(f"action_horizon must be >= 1 (got {action_horizon})")
    if future_action_max_steps < 1:
        raise ValueError(f"future_action_max_steps must be >= 1 (got {future_action_max_steps})")
    if future_action_small_max_steps < 1:
        raise ValueError(
            f"future_action_small_max_steps must be >= 1 (got {future_action_small_max_steps})"
        )
    if not (0.0 <= future_action_small_prob <= 1.0):
        raise ValueError(
            f"future_action_small_prob must be in [0, 1] (got {future_action_small_prob})"
        )
    max_len = min(action_horizon, future_action_max_steps)
    rng = rng or random
    small_cap = min(future_action_small_max_steps, max_len)
    if max_len <= small_cap:
        return int(rng.randint(1, max_len))
    if rng.random() < future_action_small_prob:
        return int(rng.randint(1, small_cap))
    return int(rng.randint(small_cap + 1, max_len))


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    # Parse config path first so we can apply YAML values as argparse defaults.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("configs", "wm_config.yaml"),
        help="Path to YAML config file (default: configs/wm_config.yaml). CLI flags override it.",
    )
    pre_args, remaining_argv = pre_parser.parse_known_args(argv)
    cfg = _load_yaml_config(pre_args.config)

    parser = argparse.ArgumentParser(
        description="Train DINO World Model on trajectory data",
        parents=[pre_parser],
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
        "--pred-step",
        type=int,
        default=1,
        help="Prediction step in raw dataset frames. pred_step=5 means train/eval on t->t+5 (default: 1).",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=100,
        help="Number of future raw-frame actions to condition on (default: 100).",
    )
    parser.add_argument(
        "--future-action-max-steps",
        type=int,
        default=50,
        help="Max number of future actions to sample for conditioning (default: 50).",
    )
    parser.add_argument(
        "--future-action-small-max-steps",
        type=int,
        default=20,
        help="Upper bound for preferred small windows (default: 20).",
    )
    parser.add_argument(
        "--future-action-small-prob",
        type=float,
        default=0.8,
        help="Probability of sampling from small windows (default: 0.8).",
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

    return parser.parse_args(remaining_argv)


def main():
    args = parse_args()

    rank, world_size, local_rank, is_distributed = init_distributed_from_env()
    is_rank0 = rank == 0

    # Initialize wandb
    if is_rank0:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            config=vars(args),
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
    pred_step = int(args.pred_step)
    action_agg = "start"
    action_horizon = int(args.action_horizon)
    future_action_max_steps = int(args.future_action_max_steps)
    future_action_small_max_steps = int(args.future_action_small_max_steps)
    future_action_small_prob = float(args.future_action_small_prob)
    if is_distributed and str(args.device).startswith("cuda"):
        device = f"cuda:{local_rank}"
    else:
        device = args.device

    if BL < 2:
        raise ValueError(f"--sequence-length must be >= 2 (got {BL}).")
    if H < 1:
        raise ValueError(f"--context-length must be >= 1 (got {H}).")
    if EVAL_H < 1:
        raise ValueError(f"--eval-horizon must be >= 1 (got {EVAL_H}).")
    if EVAL_H < H:
        raise ValueError(f"--eval-horizon must be >= --context-length (got eval_horizon={EVAL_H}, context_length={H}).")
    
    if action_horizon < 1:
        raise ValueError(f"--action-horizon must be >= 1 (got {action_horizon}).")
    if future_action_max_steps < 1:
        raise ValueError(
            f"--future-action-max-steps must be >= 1 (got {future_action_max_steps})."
        )
    if future_action_small_max_steps < 1:
        raise ValueError(
            f"--future-action-small-max-steps must be >= 1 (got {future_action_small_max_steps})."
        )
    if not (0.0 <= future_action_small_prob <= 1.0):
        raise ValueError(
            f"--future-action-small-prob must be in [0, 1] (got {future_action_small_prob})."
        )
    if int(args.eval_samples) < 1:
        raise ValueError(f"--eval-samples must be >= 1 (got {args.eval_samples}).")
    if pred_step < 1:
        raise ValueError(f"--pred-step must be >= 1 (got {pred_step}).")
    
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
    
    # We load contiguous raw-frame segments, then later index into them using pred_step and action_horizon.
    _, _, _, _, _, train_raw_len = compute_action_horizon_ar_indices(
        context_length=H,
        pred_step=pred_step,
        action_horizon=action_horizon,
        device="cpu",
    )
    imagine_raw_len = train_raw_len

    expert_data = SplitTrajectoryDataset(hdf5_file, train_raw_len, split='train', num_test=num_test)
    expert_data_eval = SplitTrajectoryDataset(hdf5_file, train_raw_len, split='test', num_test=num_test)
    expert_data_imagine = SplitTrajectoryDataset(hdf5_file, imagine_raw_len, split='test', num_test=num_test)
    
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
        num_frames=H,           # context window size
        action_horizon=action_horizon,
        dino_version=args.dino_version,
        **MODEL_CONFIG
    ).to(device)

    if is_distributed:
        transition = DistributedDataParallel(transition, device_ids=[local_rank])
    transition_module = transition.module if is_distributed else transition

    transition.train()

    # Optimizer
    optimizer = AdamW([
        {'params': transition_module.transformer.parameters(), 'lr': 5e-5},
        {'params': transition_module.state_head.parameters(), 'lr': 5e-5},
        {'params': transition_module.front_head.parameters(), 'lr': 5e-5},
        {'params': transition_module.wrist_head.parameters(), 'lr': 5e-5},
        {'params': transition_module.action_encoder.parameters(), 'lr': 5e-4},
        {'params': transition_module.state_encoder.parameters(), 'lr': 5e-4},
        {'params': [transition_module.pos_embedding], 'lr': 5e-4},
        {'params': [transition_module.temp_embedding], 'lr': 5e-4}
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
            transition_module.load_state_dict(ckpt['model_state_dict'])
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
            transition_module.load_state_dict(ckpt)

    transition.train()

    iters = []
    train_iter = args.train_iters

    def _make_ckpt_dict(iter_idx: int) -> dict:
        return {
            'model_state_dict': transition_module.state_dict(),
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
        if i > 0 and i % len(expert_loader_eval) == 0:
            expert_loader_eval = iter(DataLoader(expert_data_eval, batch_size=BS, shuffle=True))
        if i > 0 and i % len(expert_loader_imagine) == 0:
            expert_loader_imagine = iter(DataLoader(expert_data_imagine, batch_size=1, shuffle=True))

        data = next(expert_loader)

        ctx_idx, _, target_idx, _, ar_target_idx, _ = compute_action_horizon_ar_indices(
            context_length=H,
            pred_step=pred_step,
            action_horizon=action_horizon,
            device=device,
        )
        t = int(ctx_idx[-1].item())
        future_len = sample_future_action_window(
            action_horizon=action_horizon,
            future_action_max_steps=future_action_max_steps,
            future_action_small_max_steps=future_action_small_max_steps,
            future_action_small_prob=future_action_small_prob,
        )
        future_slice = slice(t, t + future_len)
        ar_future_slice = slice(t + 1, t + 1 + future_len)

        gt_front_raw = data['cam_zed_embd'].to(device)
        input_front_embd = gt_front_raw.index_select(1, ctx_idx)
        target_front_embd = gt_front_raw[:, target_idx]

        gt_wrist_raw = data['cam_rs_embd'].to(device)
        input_wrist_embd = gt_wrist_raw.index_select(1, ctx_idx)
        target_wrist_embd = gt_wrist_raw[:, target_idx]

        gt_state_raw = data['state'].to(device)
        norm_gt_state_raw = normalize_states(gt_state_raw, state_min, state_max)
        input_state = norm_gt_state_raw.index_select(1, ctx_idx)
        target_state = norm_gt_state_raw[:, target_idx]

        gt_acs_raw = data['action'].to(device)
        norm_gt_acs_raw = normalize_acs(gt_acs_raw, action_min, action_max)
        # Build action tokens aligned to each context step start in ctx_idx.
        input_acs = _build_action_tokens_from_raw(norm_gt_acs_raw, ctx_idx, pred_step=pred_step, mode=action_agg)
        future_actions = norm_gt_acs_raw[:, future_slice]

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            pred_front, pred_wrist, pred_state, _ = transition(
                input_front_embd,
                input_wrist_embd,
                input_state,
                input_acs,
                future_actions,
            )
            loss_front_tf = nn.MSELoss()(pred_front[:, -1], target_front_embd)
            loss_wrist_tf = nn.MSELoss()(pred_wrist[:, -1], target_wrist_embd)
            loss_state_tf = nn.MSELoss()(pred_state[:, -1], target_state)
            loss_tf = loss_front_tf + loss_wrist_tf + loss_state_tf

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            input_front_ar = torch.cat(
                [input_front_embd[:, 1:], pred_front[:, -1].unsqueeze(1)],
                dim=1,
            )
            input_wrist_ar = torch.cat(
                [input_wrist_embd[:, 1:], pred_wrist[:, -1].unsqueeze(1)],
                dim=1,
            )
            input_state_ar = torch.cat(
                [input_state[:, 1:], pred_state[:, -1].unsqueeze(1)],
                dim=1,
            )
            ctx_idx_ar = torch.cat([ctx_idx[1:], torch.tensor([target_idx], device=device)])
            input_acs_ar = _build_action_tokens_from_raw(
                norm_gt_acs_raw,
                ctx_idx_ar,
                pred_step=pred_step,
                mode=action_agg,
            )
            future_actions_ar = norm_gt_acs_raw[:, ar_future_slice]
            pred_front_ar, pred_wrist_ar, pred_state_ar, _ = transition(
                input_front_ar,
                input_wrist_ar,
                input_state_ar,
                input_acs_ar,
                future_actions_ar,
            )
            target_front_ar = gt_front_raw[:, ar_target_idx]
            target_wrist_ar = gt_wrist_raw[:, ar_target_idx]
            target_state_ar = norm_gt_state_raw[:, ar_target_idx]
            loss_front_ar = nn.MSELoss()(pred_front_ar[:, -1], target_front_ar)
            loss_wrist_ar = nn.MSELoss()(pred_wrist_ar[:, -1], target_wrist_ar)
            loss_state_ar = nn.MSELoss()(pred_state_ar[:, -1], target_state_ar)
            loss_ar = loss_front_ar + loss_wrist_ar + loss_state_ar

        loss = loss_tf + loss_ar * 0.5

        scaler.scale(loss).backward()

        # Norms and Step
        scaler.unscale_(optimizer)
        grad_norm = _global_grad_norm(transition_module.parameters())
        scaler.step(optimizer)
        scaler.update()
        weight_norm = _global_weight_norm(transition_module.parameters())

        train_loss = loss.item()
        print(
            f"\rIter {i} | lr {optimizer.param_groups[0]['lr']:.2e} | TF {loss_tf:.4f} | AR {loss_ar:.4f} | grad {grad_norm:.2f} | weight {weight_norm:.2f}",
            end='',
            flush=True
        )
        if is_rank0:
            wandb.log({
                'train_loss': loss_tf,
                'train_loss_ar': loss_ar,
                'grad_norm': grad_norm,
                'weight_norm': weight_norm,
                'lr': optimizer.param_groups[0]['lr'],
                'lr_factor': lr_factor,
            })

        # Periodic "latest" checkpoint
        if is_rank0 and args.save_every and (i % args.save_every == 0):
            torch.save(_make_ckpt_dict(i), latest_ckpt_path)

        # Evaluation
        if is_rank0 and (i) % args.eval_interval == 0:
            iters.append(i)
            transition.eval()

            # Metrics to average
            avg_metrics = {
                'eval_loss': 0.0,
                'front_loss': 0.0,
                'wrist_loss': 0.0,
                'state_loss': 0.0,
            }

            num_samples = args.eval_samples

            print(f"\nRunning evaluation on {num_samples} samples...")

            sample_images = None
            for s_idx in range(num_samples):
                with torch.no_grad():
                    # Get sample for single-step t+K evaluation
                    eval_data = next(expert_loader_imagine)
                    gt_front_embd_eval = eval_data['cam_zed_embd'].to(device)
                    ctx_idx, target_idx, _ = compute_eval_t_plus_k_indices(
                        context_length=H,
                        pred_step=pred_step,
                        action_horizon=action_horizon,
                        device=device,
                    )
                    t = int(ctx_idx[-1].item())
                    future_len = sample_future_action_window(
                        action_horizon=action_horizon,
                        future_action_max_steps=future_action_max_steps,
                        future_action_small_max_steps=future_action_small_max_steps,
                        future_action_small_prob=future_action_small_prob,
                    )
                    input_front_embd_eval = gt_front_embd_eval.index_select(1, ctx_idx)

                    gt_wrist_embd_eval = eval_data['cam_rs_embd'].to(device)
                    input_wrist_embd_eval = gt_wrist_embd_eval.index_select(1, ctx_idx)

                    all_acs = eval_data['action'][[0]].to(device)
                    all_acs = normalize_acs(all_acs, action_min, action_max)

                    # Action tokens aligned to each context step start.
                    acs = _build_action_tokens_from_raw(all_acs, ctx_idx, pred_step=pred_step, mode=action_agg)

                    gt_states_eval = eval_data['state'][[0]].to(device)
                    input_states_eval = normalize_states(gt_states_eval, state_min, state_max).index_select(1, ctx_idx)
                    future_actions = all_acs[:, t:t + future_len]
                    pred_front, pred_wrist, pred_state, _ = transition(
                        input_front_embd_eval,
                        input_wrist_embd_eval,
                        input_states_eval,
                        acs,
                        future_actions,
                    )

                    target_front = gt_front_embd_eval[[0], target_idx]
                    target_wrist = gt_wrist_embd_eval[[0], target_idx]
                    target_state = normalize_states(gt_states_eval[[0], target_idx], state_min, state_max)

                    l_front = nn.MSELoss()(pred_front[:, -1], target_front).item()
                    l_wrist = nn.MSELoss()(pred_wrist[:, -1], target_wrist).item()
                    l_state = nn.MSELoss()(pred_state[:, -1], target_state).item()

                    avg_metrics['eval_loss'] += (l_front + l_wrist + l_state)
                    avg_metrics['front_loss'] += l_front
                    avg_metrics['wrist_loss'] += l_wrist
                    avg_metrics['state_loss'] += l_state

                    if sample_images is None:
                        pred_latent = torch.cat([pred_front[:, [-1]], pred_wrist[:, [-1]]], dim=0)
                        pred_ims, _ = decoder(pred_latent)
                        pred_ims = rearrange(pred_ims, "(b t) c h w -> b t h w c", t=1)
                        pred_im1, pred_im2 = torch.split(pred_ims, [1, 1], dim=0)
                        gt_im1 = eval_data['agentview_image'][[0]].to(device)[:, target_idx] / 255.
                        gt_im2 = eval_data['robot0_eye_in_hand_image'][[0]].to(device)[:, target_idx] / 255.
                        gt_im1 = F.interpolate(
                            gt_im1.permute(0, 3, 1, 2),
                            size=DECODER_CONFIG['decoder_image_size'],
                            mode='bilinear',
                            align_corners=False,
                        ).permute(0, 2, 3, 1)
                        gt_im2 = F.interpolate(
                            gt_im2.permute(0, 3, 1, 2),
                            size=DECODER_CONFIG['decoder_image_size'],
                            mode='bilinear',
                            align_corners=False,
                        ).permute(0, 2, 3, 1)
                        sample_images = {
                            'pred_front': pred_im1.squeeze(0).squeeze(0).detach().cpu().numpy(),
                            'pred_wrist': pred_im2.squeeze(0).squeeze(0).detach().cpu().numpy(),
                            'front': gt_im1.squeeze(0).detach().cpu().numpy(),
                            'wrist': gt_im2.squeeze(0).detach().cpu().numpy(),
                        }

            # Finalize metrics
            for k in avg_metrics:
                avg_metrics[k] /= num_samples

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
            }
            if sample_images is not None:
                log_dict.update({
                    'pred_front': wandb.Image(sample_images['pred_front']),
                    'pred_wrist': wandb.Image(sample_images['pred_wrist']),
                    'front': wandb.Image(sample_images['front']),
                    'wrist': wandb.Image(sample_images['wrist']),
                })
            wandb.log(log_dict)

    if is_rank0:
        plt.legend()
        plt.savefig(os.path.join(args.checkpoint_dir, 'training_curve.png'))

    best_eval_val = best_eval.item() if hasattr(best_eval, 'item') else best_eval
    print(f"\nTraining complete. Best eval loss: {best_eval_val:.4f}")


if __name__ == "__main__":
    main()
