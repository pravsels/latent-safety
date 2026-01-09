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
import random
import numpy as np
import torch
import wandb
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from einops import rearrange
import matplotlib.pyplot as plt
import torch.nn.functional as F

from dino_wm.test_loader import SplitTrajectoryDataset
from dino_wm.dino_decoder import VQVAE
from dino_wm.config import MODEL_CONFIG, DECODER_CONFIG, TRAIN_CONFIG, get_dino_config

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def _load_yaml_config(path: str) -> dict:
    """
    Load YAML into a plain dict.
    Uses ruamel.yaml (repo dependency via setup.py) and supports env var expansion.
    """
    import pathlib
    import ruamel.yaml as ryaml

    p = os.path.expandvars(os.path.expanduser(path))
    cfg = ryaml.YAML(typ="safe", pure=True).load(pathlib.Path(p).read_text()) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping (YAML dict). Got: {type(cfg)}")
    return cfg

def _global_grad_norm(parameters, norm_type: float = 2.0) -> float:
    """Compute global grad norm over a set of parameters (like clip_grad_norm_ but without clipping)."""
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

def _psnr_from_mse(mse: torch.Tensor, max_val: float = 1.0, eps: float = 1e-10) -> torch.Tensor:
    """Compute PSNR in dB from MSE. mse can be scalar or per-sample tensor."""
    mse = torch.clamp(mse, min=eps)
    return 10.0 * torch.log10((max_val ** 2) / mse)

def _gaussian_kernel_2d(kernel_size: int = 11, sigma: float = 1.5, device=None, dtype=None) -> torch.Tensor:
    """Returns 2D Gaussian kernel of shape (1, 1, K, K)."""
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = torch.outer(g, g)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d.view(1, 1, kernel_size, kernel_size)

def _ssim(img1_bchw: torch.Tensor, img2_bchw: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """
    Compute SSIM for images in [0, 1]. Returns a scalar tensor (mean over batch).
    Uses a standard Gaussian window approach.
    """
    assert img1_bchw.shape == img2_bchw.shape, "SSIM inputs must have same shape"
    # Ensure float32 for numerical stability
    x = img1_bchw.float()
    y = img2_bchw.float()
    B, C, H, W = x.shape

    kernel_size = 11
    sigma = 1.5
    kernel = _gaussian_kernel_2d(kernel_size, sigma, device=x.device, dtype=x.dtype)
    # Apply per-channel via groups convolution
    kernel = kernel.expand(C, 1, kernel_size, kernel_size)
    padding = kernel_size // 2

    mu_x = F.conv2d(x, kernel, padding=padding, groups=C)
    mu_y = F.conv2d(y, kernel, padding=padding, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, kernel, padding=padding, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=padding, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=padding, groups=C) - mu_xy

    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    # Mean over channels and spatial dims, then batch
    return ssim_map.mean(dim=(1, 2, 3)).mean()

def _load_dino_model(device: str):
    """
    Load the DINO model used for embeddings (matches dino_wm.config.get_dino_config()).
    Resolved paths are anchored at repo root to avoid cwd sensitivity.
    """
    import warnings

    dino_cfg = get_dino_config()
    hub_repo = dino_cfg["hub_repo"]
    weights_path = dino_cfg.get("weights_path", None)

    # Resolve relative paths to be robust to current working directory.
    if isinstance(hub_repo, str) and hub_repo.startswith("."):
        hub_repo = os.path.abspath(os.path.join(_REPO_ROOT, hub_repo))
    if isinstance(hub_repo, str) and hub_repo.startswith(".."):
        hub_repo = os.path.abspath(os.path.join(_REPO_ROOT, hub_repo))
    if isinstance(weights_path, str) and (weights_path.startswith(".") or weights_path.startswith("..")):
        weights_path = os.path.abspath(os.path.join(_REPO_ROOT, weights_path))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="xFormers is not available")
        if "hub_source" in dino_cfg:
            model = torch.hub.load(
                hub_repo,
                dino_cfg["model_name"],
                source=dino_cfg["hub_source"],
                weights=weights_path,
            ).to(device)
        else:
            model = torch.hub.load(hub_repo, dino_cfg["model_name"]).to(device)
    model.eval()
    return model

def _preprocess_images_for_dino(images_bchw_01: torch.Tensor, *, is_front_camera: bool) -> torch.Tensor:
    """
    Match the preprocessing used when creating cam_{zed,rs}_embd in this repo:
      - front: gaussian blur + crop + resize + imagenet normalize
      - wrist: resize + imagenet normalize
    Input expected in [0, 1], shape (B, 3, H, W)
    Output: normalized tensor sized MODEL_CONFIG['image_size']
    """
    try:
        from torchvision.transforms import functional as TF
    except Exception as e:
        raise ImportError(
            "DINO preprocessing requires torchvision. Install torchvision or disable DINO-based losses/metrics."
        ) from e

    out = images_bchw_01
    if is_front_camera:
        out = TF.gaussian_blur(out, kernel_size=(5, 5), sigma=(0.1, 0.1))
        out = TF.crop(out, top=30, left=46, height=180, width=180)

    out = TF.resize(out, MODEL_CONFIG["image_size"], antialias=True)
    out = TF.normalize(out, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return out

def _dino_patchtokens(model, preprocessed_bchw: torch.Tensor) -> torch.Tensor:
    """Return DINO patch tokens: (B, num_patches, dim)."""
    feats = model.forward_features(preprocessed_bchw)
    return feats["x_norm_patchtokens"]

def _assert_token_shapes_match(
    predicted_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    *,
    context: str,
) -> None:
    """
    Ensure DINO patch token shapes match (B, N, D). If they don't, it's almost always
    because the dataset embeddings were generated with a different DINO variant than
    the one currently configured (v2: 256 patches, v3: 196 patches).
    """
    if predicted_tokens.ndim != 3 or target_tokens.ndim != 3:
        raise ValueError(
            f"{context}: expected tokens with shape (B, N, D), got "
            f"pred={tuple(predicted_tokens.shape)} target={tuple(target_tokens.shape)}"
        )
    if predicted_tokens.shape[0] != target_tokens.shape[0]:
        raise ValueError(
            f"{context}: batch size mismatch: pred B={predicted_tokens.shape[0]} target B={target_tokens.shape[0]}"
        )
    if predicted_tokens.shape[1] != target_tokens.shape[1]:
        raise ValueError(
            f"{context}: num_patches mismatch: pred N={predicted_tokens.shape[1]} target N={target_tokens.shape[1]}. "
            f"This usually means your HDF5 embeddings were generated with a different DINO version than "
            f"`dino_wm/config.py:DINO_VERSION`."
        )
    if predicted_tokens.shape[2] != target_tokens.shape[2]:
        raise ValueError(
            f"{context}: feature-dim mismatch: pred D={predicted_tokens.shape[2]} target D={target_tokens.shape[2]}. "
            f"This usually means your HDF5 embeddings were generated with a different DINO backbone/config."
        )

def _dino_token_distance(pred_tokens: torch.Tensor, target_tokens: torch.Tensor, metric: str) -> torch.Tensor:
    """
    Compute distance between DINO patch tokens.
    pred_tokens/target_tokens: (B, N, D)
    Returns scalar tensor.
    """
    _assert_token_shapes_match(pred_tokens, target_tokens, context="dino_token_distance")
    if metric == "l2":
        return (pred_tokens - target_tokens).pow(2).mean()
    if metric == "cosine":
        # 1 - cosine similarity (mean over tokens and batch)
        pred_n = F.normalize(pred_tokens, dim=-1)
        tgt_n = F.normalize(target_tokens, dim=-1)
        return (1.0 - (pred_n * tgt_n).sum(dim=-1)).mean()
    raise ValueError(f"Unknown DINO token distance metric: {metric}")

def _vgg16_perceptual_loss(pred_bchw_01: torch.Tensor, gt_bchw_01: torch.Tensor) -> torch.Tensor:
    """
    Lightweight VGG16 perceptual loss (L1 on intermediate features).
    NOTE: Requires torchvision; pretrained weights may need to be available/cached.
    """
    try:
        from torchvision.models import vgg16, VGG16_Weights
    except Exception as e:
        raise ImportError(
            "VGG perceptual loss requires torchvision>=0.13. Install torchvision or use --perceptual-kind none/dino."
        ) from e

    # Load pretrained VGG16 once per call site (caller should cache via closure if needed).
    vgg = vgg16(weights=VGG16_Weights.DEFAULT).features.to(pred_bchw_01.device).eval()
    for p in vgg.parameters():
        p.requires_grad_(False)

    # VGG expects ImageNet-normalized inputs at ~224 resolution.
    pred = F.interpolate(pred_bchw_01, size=MODEL_CONFIG["image_size"], mode="bilinear", align_corners=False)
    gt = F.interpolate(gt_bchw_01, size=MODEL_CONFIG["image_size"], mode="bilinear", align_corners=False)
    pred = _preprocess_images_for_dino(pred, is_front_camera=False)  # just resize+normalize path
    gt = _preprocess_images_for_dino(gt, is_front_camera=False)

    # Common perceptual layers: relu1_2, relu2_2, relu3_3
    layer_ids = {3, 8, 15}
    loss = pred.new_tensor(0.0)
    x = pred
    y = gt
    for idx, layer in enumerate(vgg):
        x = layer(x)
        y = layer(y)
        if idx in layer_ids:
            loss = loss + (x - y).abs().mean()
    return loss

def _compute_lr(
    step: int,
    total_steps: int,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
    schedule: str,
) -> float:
    """Compute learning rate for a given global step."""
    if schedule == "constant":
        return float(base_lr)

    # warmup + cosine decay
    warmup_steps = int(max(0, warmup_steps))
    if warmup_steps > 0 and step < warmup_steps:
        return float(base_lr) * float(step + 1) / float(warmup_steps)

    # cosine over the remaining steps (or all steps if warmup=0)
    denom = max(1, int(total_steps) - warmup_steps)
    t = min(1.0, max(0.0, float(step - warmup_steps) / float(denom)))
    cosine = 0.5 * (1.0 + float(torch.cos(torch.tensor(t * 3.141592653589793)).item()))
    return float(min_lr) + (float(base_lr) - float(min_lr)) * cosine

def main():
    # Parse config path first so we can apply YAML values as argparse defaults.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("configs", "dino_decoder_config.yaml"),
        help="Path to YAML config file (default: configs/dino_decoder_config.yaml). CLI flags override it.",
    )
    pre_args, remaining_argv = pre_parser.parse_known_args()
    cfg = _load_yaml_config(pre_args.config) if pre_args.config else {}

    parser = argparse.ArgumentParser(parents=[pre_parser])
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
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader num_workers (default: 0). Increase for faster HDF5 reading on CPU-heavy nodes.",
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DataLoader pin_memory (default: True). Use --no-pin-memory to disable.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for train/test trajectory split and DataLoader shuffling (default: 0).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Number of timesteps per sample for decoder training (default: 1).",
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
        "--save-iter-checkpoints-every",
        type=int,
        default=1000,
        help="Save numbered iteration checkpoints decoder_iterXXXX*.pth every N iterations (default: 1000). Set to 0 to disable.",
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
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Auto-resume from latest checkpoint in --checkpoint-dir (default: False). Use --no-auto-resume to disable.",
    )
    parser.add_argument(
        "--start-iter",
        type=int,
        default=None,
        help="Iteration to start training from. If omitted and resuming from a checkpoint that stores 'iter', it will resume from there.",
    )
    parser.add_argument(
        "--quantize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable VQ codebook quantization (default: False). Use --no-quantize to disable.",
    )
    # --- learning-rate scheduling ---
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Base learning rate (default: 3e-4).",
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=1e-5,
        help="Minimum learning rate for cosine schedule (default: 1e-5). Ignored for constant schedule.",
    )
    parser.add_argument(
        "--lr-warmup-iters",
        type=int,
        default=100,
        help="Number of warmup iterations (default: 100).",
    )
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="cosine",
        choices=["cosine", "constant", "plateau"],
        help="Learning-rate schedule (default: cosine).",
    )
    parser.add_argument(
        "--plateau-patience",
        type=int,
        default=10,
        help="ReduceLROnPlateau: number of eval steps with no improvement before reducing LR (default: 10).",
    )
    parser.add_argument(
        "--plateau-factor",
        type=float,
        default=0.5,
        help="ReduceLROnPlateau: multiplicative factor of LR reduction (default: 0.5).",
    )
    parser.add_argument(
        "--plateau-threshold",
        type=float,
        default=1e-4,
        help="ReduceLROnPlateau: threshold for measuring new optimum (default: 1e-4).",
    )
    parser.add_argument(
        "--plateau-cooldown",
        type=int,
        default=0,
        help="ReduceLROnPlateau: cooldown eval steps after LR reduction (default: 0).",
    )
    parser.add_argument(
        "--plateau-min-lr",
        type=float,
        default=None,
        help="ReduceLROnPlateau: minimum LR. If omitted, uses --lr-min.",
    )

    # --- Two-stage loss schedule (optional refinements after pixel MSE plateaus) ---
    parser.add_argument(
        "--perceptual-kind",
        type=str,
        default="vgg16",
        choices=["none", "vgg16", "dino"],
        help="Optional perceptual loss type (default: vgg16).",
    )
    parser.add_argument(
        "--perceptual-weight",
        type=float,
        default=0.05,
        help="Weight for perceptual loss term (default: 0.05). Set to 0.0 to disable.",
    )
    parser.add_argument(
        "--dino-cycle-weight",
        type=float,
        default=0.1,
        help="Weight for DINO cycle-consistency loss term (default: 0.1). Set to 0.0 to disable.",
    )
    parser.add_argument(
        "--dino-cycle-metric",
        type=str,
        default="cosine",
        choices=["l2", "cosine"],
        help="Distance metric for DINO cycle loss (default: cosine).",
    )
    parser.add_argument(
        "--dino-cycle-train-every",
        type=int,
        default=1,
        help="Compute DINO-cycle loss every N train iterations when enabled (default: 1).",
    )
    # Apply YAML config as defaults (CLI overrides because we parse after this).
    known_dests = {a.dest for a in parser._actions}
    for k, v in (cfg or {}).items():
        if k in known_dests:
            parser.set_defaults(**{k: v})

    args = parser.parse_args(remaining_argv)

    # Reproducibility: controls trajectory split ordering and DataLoader shuffles.
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        config=vars(args)
    )

    hdf5_file = args.hdf5_file
    H = int(args.horizon)
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
        hdf5_file, H, split="train", num_test=num_test, seed=int(args.seed)
    )
    expert_data_eval = SplitTrajectoryDataset(
        hdf5_file, H, split="test", num_test=num_test, seed=int(args.seed)
    )

    # DataLoaders: eval loader should NOT shuffle for stable metrics.
    persistent = bool(args.num_workers) and int(args.num_workers) > 0
    train_loader = DataLoader(
        expert_data,
        batch_size=BS,
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        persistent_workers=persistent,
    )
    eval_loader = DataLoader(
        expert_data_eval,
        batch_size=BS,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        persistent_workers=persistent,
    )
    expert_loader = iter(train_loader)
    expert_loader_eval = iter(eval_loader)
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
    iter_ckpt_template = os.path.join(args.checkpoint_dir, f"decoder_iter{{iter}}{run_suffix}.pth")

    # Resolve resume path.
    resume_path = args.resume_checkpoint
    if resume_path is None and args.auto_resume and os.path.exists(latest_ckpt_path):
        resume_path = latest_ckpt_path

    optimizer = AdamW([
        {'params': decoder.parameters(), 'lr': float(args.lr)}
    ])

    # Optional plateau scheduler (steps on eval_loss)
    plateau_scheduler = None
    if args.lr_schedule == "plateau":
        plateau_min_lr = float(args.lr_min) if args.plateau_min_lr is None else float(args.plateau_min_lr)
        plateau_scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(args.plateau_factor),
            patience=int(args.plateau_patience),
            threshold=float(args.plateau_threshold),
            cooldown=int(args.plateau_cooldown),
            min_lr=plateau_min_lr,
        )

    # Optional heavy models (truly lazy-loaded)
    dino_model = None
    vgg_model = None

    def _ensure_dino_model():
        nonlocal dino_model
        if dino_model is None:
            print("Loading DINO model for perceptual/cycle losses...")
            dino_model = _load_dino_model(device)
        return dino_model

    def _ensure_vgg_model():
        nonlocal vgg_model
        if vgg_model is None:
            print("Loading VGG16 for perceptual loss...")
            try:
                from torchvision.models import vgg16, VGG16_Weights
            except Exception as e:
                raise ImportError(
                    "VGG perceptual loss requires torchvision. Install torchvision or use --perceptual-kind none/dino."
                ) from e
            vgg_model = vgg16(weights=VGG16_Weights.DEFAULT).features.to(device).eval()
            for p in vgg_model.parameters():
                p.requires_grad_(False)
        return vgg_model

    # --- Two-stage loss schedule (no new knobs) ---
    # Stage 1: pixel-space reconstruction (MSE) only.
    # Stage 2: enable perceptual + DINO-cycle terms once eval_loss plateaus.
    # Plateau definition: no improvement beyond `--plateau-threshold` for `--plateau-patience` eval events.
    extra_losses_enabled = False
    plateau_patience_evals = int(getattr(args, "plateau_patience", 10))
    plateau_threshold = float(getattr(args, "plateau_threshold", 1e-4))
    eval_no_improve = 0
    best_eval_for_plateau = float("inf")

    # best_eval is tracked and persisted via best checkpoint when available
    best_eval = float("inf")
    if os.path.exists(best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        if isinstance(best_ckpt, dict) and "best_eval" in best_ckpt:
            best_eval = float(best_ckpt["best_eval"])
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
            # Restore two-stage schedule state if present (so resume doesn't "forget" Stage 2).
            if "extra_losses_enabled" in ckpt:
                extra_losses_enabled = bool(ckpt["extra_losses_enabled"])
            if "best_eval_for_plateau" in ckpt:
                best_eval_for_plateau = float(ckpt["best_eval_for_plateau"])
            if "eval_no_improve" in ckpt:
                eval_no_improve = int(ckpt["eval_no_improve"])
            if "best_eval" in ckpt and best_eval == float("inf"):
                best_eval = float(ckpt["best_eval"])
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

    def _make_ckpt_dict(iter_idx: int) -> dict:
        """Single source of truth for resume-safe checkpoints."""
        return {
            "model_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iter": int(iter_idx),
            "best_eval": float(best_eval),
            "quantize": bool(args.quantize),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "lr_schedule": str(args.lr_schedule),
            # Two-stage schedule state (resume-safe)
            "extra_losses_enabled": bool(extra_losses_enabled),
            "best_eval_for_plateau": float(best_eval_for_plateau),
            "eval_no_improve": int(eval_no_improve),
            "seed": int(args.seed),
            "decoder_image_size": tuple(DECODER_CONFIG["decoder_image_size"]),
        }

    for i in range(start_iter, train_iter):
        # --- LR update ---
        # - cosine/constant: updated per-iteration from global step (requires known train_iters)
        # - plateau: updated on eval_loss inside the eval block (does not require known horizon)
        if args.lr_schedule in ("cosine", "constant"):
            lr = _compute_lr(
                step=i,
                total_steps=train_iter,
                base_lr=float(args.lr),
                min_lr=float(args.lr_min),
                warmup_steps=int(args.lr_warmup_iters),
                schedule=str(args.lr_schedule),
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr
        else:
            lr = float(optimizer.param_groups[0]["lr"])

        # Refresh iterators when they exhaust (avoids relying on len() of iterator).
        try:
            data = next(expert_loader)
        except StopIteration:
            expert_loader = iter(train_loader)
            data = next(expert_loader)

        inputs1 = data["cam_zed_embd"].to(device)
        inputs2 = data["cam_rs_embd"].to(device)
        # Ground truth images start as (B, T, H_img, W_img, C); we resize to DECODER_CONFIG['decoder_image_size'] to match decoder's native output.
        output1 = data["agentview_image"].to(device) / 255.0  # (B, T, H, W, C)
        output2 = data["robot0_eye_in_hand_image"].to(device) / 255.0
        B, T, H_img, W_img, C = output1.shape
        if T != H:
            raise ValueError(f"Dataset returned T={T} frames but --horizon={H}. Set horizon to match dataset.")
        if inputs1.ndim == 4 and inputs1.shape[1] != H:
            raise ValueError(f"cam_zed_embd has T={inputs1.shape[1]} but --horizon={H}.")
        if inputs2.ndim == 4 and inputs2.shape[1] != H:
            raise ValueError(f"cam_rs_embd has T={inputs2.shape[1]} but --horizon={H}.")
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
        # Robustness: if decoder output resolution doesn't match configured GT resize, align here.
        img_size = DECODER_CONFIG["decoder_image_size"]
        if tuple(pred.shape[-2:]) != tuple(img_size):
            pred = F.interpolate(pred, size=img_size, mode="bilinear", align_corners=False)
        # Decoder returns (B*T, C, H_dec, W_dec); restore (B, T, C, H, W)
        pred = rearrange(pred, "(b t) c h w -> b t c h w", t=H)
        
        pred1, pred2 = torch.split(pred, [inputs1.shape[0], inputs2.shape[0]], dim=0)
        # (B, T, C, H, W) -> (B, T, H, W, C) to match outputs
        pred1_bthwc = pred1.permute(0, 1, 3, 4, 2).contiguous()
        pred2_bthwc = pred2.permute(0, 1, 3, 4, 2).contiguous()

        recon_loss = nn.MSELoss()(pred1_bthwc, output1)
        recon_loss += nn.MSELoss()(pred2_bthwc, output2)
        # VQ commitment / codebook loss (scalar)
        vq_loss = diff.mean()
        loss = recon_loss + 0.25 * vq_loss

        # --- Stage 2: optional perceptual loss ---
        perceptual_loss_val = None
        if (
            extra_losses_enabled
            and args.perceptual_weight
            and args.perceptual_weight > 0.0
            and args.perceptual_kind != "none"
        ):
            # Flatten time into batch for perceptual losses: (B, T, H, W, C) -> (B*T, C, H, W)
            pred1_bchw_01 = pred1_bthwc.permute(0, 1, 4, 2, 3).contiguous().view(-1, C, img_size[0], img_size[1]).clamp(0.0, 1.0)
            gt1_bchw_01 = output1.permute(0, 1, 4, 2, 3).contiguous().view(-1, C, img_size[0], img_size[1]).clamp(0.0, 1.0)
            pred2_bchw_01 = pred2_bthwc.permute(0, 1, 4, 2, 3).contiguous().view(-1, C, img_size[0], img_size[1]).clamp(0.0, 1.0)
            gt2_bchw_01 = output2.permute(0, 1, 4, 2, 3).contiguous().view(-1, C, img_size[0], img_size[1]).clamp(0.0, 1.0)

            if args.perceptual_kind == "vgg16":
                # Use cached VGG model; compute feature L1 at a few layers.
                # VGG expects ImageNet-normalized inputs.
                vgg_model = _ensure_vgg_model()
                def _vgg_feats(x):
                    # Resize to 224 and normalize like ImageNet
                    x = F.interpolate(x, size=MODEL_CONFIG["image_size"], mode="bilinear", align_corners=False)
                    x = _preprocess_images_for_dino(x, is_front_camera=False)
                    out = []
                    layer_ids = {3, 8, 15}
                    h = x
                    for idx, layer in enumerate(vgg_model):
                        h = layer(h)
                        if idx in layer_ids:
                            out.append(h)
                    return out

                pf = _vgg_feats(pred1_bchw_01)
                gf = _vgg_feats(gt1_bchw_01)
                pw = _vgg_feats(pred2_bchw_01)
                gw = _vgg_feats(gt2_bchw_01)
                perceptual_loss_val = sum((a - b).abs().mean() for a, b in zip(pf, gf)) + sum((a - b).abs().mean() for a, b in zip(pw, gw))

            elif args.perceptual_kind == "dino":
                # DINO feature distance between pred and GT pixels (semantic/perceptual).
                dino_model = _ensure_dino_model()
                pre_pf = _preprocess_images_for_dino(pred1_bchw_01, is_front_camera=True)
                pre_gf = _preprocess_images_for_dino(gt1_bchw_01, is_front_camera=True)
                pre_pw = _preprocess_images_for_dino(pred2_bchw_01, is_front_camera=False)
                pre_gw = _preprocess_images_for_dino(gt2_bchw_01, is_front_camera=False)
                tok_pf = _dino_patchtokens(dino_model, pre_pf)
                tok_gf = _dino_patchtokens(dino_model, pre_gf)
                tok_pw = _dino_patchtokens(dino_model, pre_pw)
                tok_gw = _dino_patchtokens(dino_model, pre_gw)
                perceptual_loss_val = _dino_token_distance(tok_pf, tok_gf, metric=args.dino_cycle_metric) + _dino_token_distance(tok_pw, tok_gw, metric=args.dino_cycle_metric)
            else:
                raise ValueError(f"Unknown perceptual kind: {args.perceptual_kind}")

            loss = loss + float(args.perceptual_weight) * perceptual_loss_val

        # --- Stage 2: optional DINO cycle-consistency loss (pred pixels -> DINO tokens vs input tokens) ---
        dino_cycle_loss_val = None
        if extra_losses_enabled and args.dino_cycle_weight and args.dino_cycle_weight > 0.0:
            every = max(1, int(args.dino_cycle_train_every))
            if (i % every) == 0:
                dino_model = _ensure_dino_model()
                # Use the first timestep for cycle-consistency against stored DINO embeddings.
                pred1_bchw_01 = pred1_bthwc[:, 0].permute(0, 3, 1, 2).contiguous().clamp(0.0, 1.0)
                pred2_bchw_01 = pred2_bthwc[:, 0].permute(0, 3, 1, 2).contiguous().clamp(0.0, 1.0)

                # Inputs can be (B, T, N, D) or (B, N, D)
                tgt_front = inputs1[:, 0] if inputs1.ndim == 4 else inputs1
                tgt_wrist = inputs2[:, 0] if inputs2.ndim == 4 else inputs2

                pre_front = _preprocess_images_for_dino(pred1_bchw_01, is_front_camera=True)
                pre_wrist = _preprocess_images_for_dino(pred2_bchw_01, is_front_camera=False)
                tok_front = _dino_patchtokens(dino_model, pre_front)
                tok_wrist = _dino_patchtokens(dino_model, pre_wrist)

                dino_cycle_loss_val = _dino_token_distance(tok_front, tgt_front, metric=args.dino_cycle_metric) + _dino_token_distance(tok_wrist, tgt_wrist, metric=args.dino_cycle_metric)
                loss = loss + float(args.dino_cycle_weight) * dino_cycle_loss_val
        optimizer.zero_grad()
        loss.backward()
        grad_norm = _global_grad_norm(decoder.parameters())
        optimizer.step()
        weight_norm = _global_weight_norm(decoder.parameters())
        log_dict = {
            'train_loss': loss.item(),
            'grad_norm': grad_norm,
            'weight_norm': weight_norm,
            'lr': lr,
            'extra_losses_enabled': int(extra_losses_enabled),
        }
        if args.perceptual_weight and args.perceptual_weight > 0.0:
            log_dict["perceptual_weight_effective"] = float(args.perceptual_weight) if extra_losses_enabled else 0.0
        if args.dino_cycle_weight and args.dino_cycle_weight > 0.0:
            log_dict["dino_cycle_weight_effective"] = float(args.dino_cycle_weight) if extra_losses_enabled else 0.0
        if perceptual_loss_val is not None:
            log_dict["train_perceptual_loss"] = float(perceptual_loss_val.detach().item()) if hasattr(perceptual_loss_val, "detach") else float(perceptual_loss_val)
        if dino_cycle_loss_val is not None:
            log_dict["train_dino_cycle_loss"] = float(dino_cycle_loss_val.detach().item()) if hasattr(dino_cycle_loss_val, "detach") else float(dino_cycle_loss_val)
        wandb.log(log_dict)
        print(
            f"\rIter {i} | lr {lr:.2e} | train_loss {loss.item():.4f} | grad_norm {grad_norm:.2f} | weight_norm {weight_norm:.2f}",
            end="",
            flush=True,
        )

        # Periodic "latest" checkpoint for HPC preemption / manual restarts.
        if args.save_every and (i % args.save_every == 0):
            torch.save(_make_ckpt_dict(i), latest_ckpt_path)

        # Periodic numbered checkpoints for experiment tracking / rollback (like wm_iterXXXX.pth).
        if args.save_iter_checkpoints_every and (i % args.save_iter_checkpoints_every == 0):
            iter_ckpt_path = iter_ckpt_template.format(iter=i)
            torch.save(_make_ckpt_dict(i), iter_ckpt_path)
        
        if args.eval_every and (i % args.eval_every == 0):
            train_losses.append(loss.item())
            iters.append(i)
            try:
                eval_data = next(expert_loader_eval)
            except StopIteration:
                expert_loader_eval = iter(eval_loader)
                eval_data = next(expert_loader_eval)
            decoder.eval()
            with torch.no_grad():
                inputs1 = eval_data["cam_zed_embd"].to(device)
                inputs2 = eval_data["cam_rs_embd"].to(device)
                # Same resizing as above for eval images
                output1 = eval_data["agentview_image"].to(device) / 255.0
                output2 = eval_data["robot0_eye_in_hand_image"].to(device) / 255.0

                B_eval, T_eval, H_img_e, W_img_e, C_e = output1.shape
                if T_eval != H:
                    raise ValueError(f"Eval dataset returned T={T_eval} frames but --horizon={H}. Set horizon to match dataset.")
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
                img_size = DECODER_CONFIG["decoder_image_size"]
                if tuple(pred.shape[-2:]) != tuple(img_size):
                    pred = F.interpolate(pred, size=img_size, mode="bilinear", align_corners=False)
                pred = rearrange(pred, "(b t) c h w -> b t c h w", t=H)
                pred1, pred2 = torch.split(
                    pred, [inputs1.shape[0], inputs2.shape[0]], dim=0
                )
                pred1_bthwc = pred1.permute(0, 1, 3, 4, 2).contiguous()
                pred2_bthwc = pred2.permute(0, 1, 3, 4, 2).contiguous()
                
                recon_loss = nn.MSELoss()(pred1_bthwc, output1)
                recon_loss += nn.MSELoss()(pred2_bthwc, output2)
                vq_loss = diff.mean()
                loss = recon_loss + 0.25 * vq_loss

                # --- Phase 1 eval metrics: PSNR + SSIM ---
                # Convert to BCHW for metrics
                pred1_bchw = pred1_bthwc.permute(0, 1, 4, 2, 3).contiguous().view(-1, C_e, img_size[0], img_size[1]).clamp(0.0, 1.0)
                gt1_bchw = output1.permute(0, 1, 4, 2, 3).contiguous().view(-1, C_e, img_size[0], img_size[1])
                pred2_bchw = pred2_bthwc.permute(0, 1, 4, 2, 3).contiguous().view(-1, C_e, img_size[0], img_size[1]).clamp(0.0, 1.0)
                gt2_bchw = output2.permute(0, 1, 4, 2, 3).contiguous().view(-1, C_e, img_size[0], img_size[1])

                mse1 = (pred1_bchw - gt1_bchw).pow(2).mean(dim=(1, 2, 3))  # per-sample
                mse2 = (pred2_bchw - gt2_bchw).pow(2).mean(dim=(1, 2, 3))
                psnr1 = _psnr_from_mse(mse1).mean()
                psnr2 = _psnr_from_mse(mse2).mean()
                ssim1 = _ssim(pred1_bchw, gt1_bchw)
                ssim2 = _ssim(pred2_bchw, gt2_bchw)
                eval_psnr = (psnr1 + psnr2) / 2.0
                eval_ssim = (ssim1 + ssim2) / 2.0

                # Optional: DINO cycle metric on eval (pred pixels -> DINO tokens vs input tokens)
                eval_dino_cycle_loss = None
                if extra_losses_enabled and args.dino_cycle_weight and args.dino_cycle_weight > 0.0:
                    dino_model = _ensure_dino_model()
                    pred1_bchw_01 = pred1_bchw
                    pred2_bchw_01 = pred2_bchw

                    # Targets: if embeddings include time (B, T, N, D), compare per-timestep by flattening to (B*T, N, D).
                    if inputs1.ndim == 4:
                        tgt_front = inputs1.reshape(-1, inputs1.shape[-2], inputs1.shape[-1])
                    else:
                        tgt_front = inputs1
                    if inputs2.ndim == 4:
                        tgt_wrist = inputs2.reshape(-1, inputs2.shape[-2], inputs2.shape[-1])
                    else:
                        tgt_wrist = inputs2

                    pre_front = _preprocess_images_for_dino(pred1_bchw_01, is_front_camera=True)
                    pre_wrist = _preprocess_images_for_dino(pred2_bchw_01, is_front_camera=False)
                    tok_front = _dino_patchtokens(dino_model, pre_front)
                    tok_wrist = _dino_patchtokens(dino_model, pre_wrist)
                    eval_dino_cycle_loss = _dino_token_distance(tok_front, tgt_front, metric=args.dino_cycle_metric) + _dino_token_distance(tok_wrist, tgt_wrist, metric=args.dino_cycle_metric)

            print()
            eval_msg = (
                f"\rIter {i} | eval_loss {loss.item():.4f} | psnr {eval_psnr.item():.2f} | ssim {eval_ssim.item():.3f} "
                f"(front psnr {psnr1.item():.2f} ssim {ssim1.item():.3f}, wrist psnr {psnr2.item():.2f} ssim {ssim2.item():.3f})"
            )
            if eval_dino_cycle_loss is not None:
                eval_msg += f" | dino_cycle {eval_dino_cycle_loss.item():.4f}"
            print(eval_msg)
            if plateau_scheduler is not None:
                plateau_scheduler.step(loss.item())

            # Plateau detector for enabling extra losses (based on eval_loss only).
            cur_eval = float(loss.item())
            if cur_eval < (best_eval_for_plateau - plateau_threshold):
                best_eval_for_plateau = cur_eval
                eval_no_improve = 0
            else:
                eval_no_improve += 1

            if (
                (not extra_losses_enabled)
                and (
                    (args.perceptual_weight and args.perceptual_weight > 0.0)
                    or (args.dino_cycle_weight and args.dino_cycle_weight > 0.0)
                )
                and eval_no_improve >= plateau_patience_evals
            ):
                extra_losses_enabled = True
                print(
                    f"Switching to Stage 2 losses after plateau: no improvement for {eval_no_improve} evals "
                    f"(patience={plateau_patience_evals}, threshold={plateau_threshold}). "
                    f"perceptual_weight={float(args.perceptual_weight):.4f}, dino_cycle_weight={float(args.dino_cycle_weight):.4f}"
                )
                wandb.log(
                    {
                        "extra_losses_enabled": 1,
                        "extra_losses_enabled_at_iter": i,
                        "plateau_patience_evals": plateau_patience_evals,
                        "plateau_threshold": plateau_threshold,
                    }
                )
            if float(loss.item()) < best_eval:
                best_eval = float(loss.item())
                # Save backward-compatible latest weights (state_dict)
                torch.save(decoder.state_dict(), latest_state_dict_path)
                # Save best checkpoint with metadata to persist best_eval across sessions
                torch.save(_make_ckpt_dict(i), best_ckpt_path)
            decoder.train()
            
            out_log = (output1[0, 0].detach().cpu().numpy())
            pred_log = (pred1_bthwc[0, 0].detach().cpu().numpy())
            out_log2 = (output2[0, 0].detach().cpu().numpy())
            pred_log2 = (pred2_bthwc[0, 0].detach().cpu().numpy())
            pred_log = np.clip(pred_log, 0.0, 1.0)
            pred_log2 = np.clip(pred_log2, 0.0, 1.0)

            # --- Phase 1 visuals: diff maps (abs error) ---
            diff_front = np.abs(out_log - pred_log)
            diff_wrist = np.abs(out_log2 - pred_log2)

            eval_log = {
                'eval_loss': loss.item(),
                'eval_psnr': eval_psnr.item(),
                'eval_ssim': eval_ssim.item(),
                'eval_psnr_front': psnr1.item(),
                'eval_psnr_wrist': psnr2.item(),
                'eval_ssim_front': ssim1.item(),
                'eval_ssim_wrist': ssim2.item(),
                'ground_truth_front': wandb.Image(out_log),
                'pred_front': wandb.Image(pred_log),
                'diff_front': wandb.Image(diff_front),
                'ground_truth_wrist': wandb.Image(out_log2),
                'pred_wrist': wandb.Image(pred_log2),
                'diff_wrist': wandb.Image(diff_wrist),
            }
            if eval_dino_cycle_loss is not None:
                eval_log["eval_dino_cycle_loss"] = float(eval_dino_cycle_loss.detach().item())
            wandb.log(eval_log)
            eval_losses.append(float(loss.item()))


    plt.plot(iters, train_losses, label='train')
    plt.plot(iters, eval_losses, label='eval')
    plt.legend()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    plt.savefig(os.path.join(args.checkpoint_dir, f'training_curve{run_suffix}.png'))

    print(f"\nTraining complete. Best eval loss: {float(best_eval):.4f}")


if __name__ == "__main__":
    main()