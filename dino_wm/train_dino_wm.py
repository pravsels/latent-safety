#!/usr/bin/env python3
"""
Train the DINO World Model (VideoTransformer) on trajectory data.

Quickstart:

  python dino_wm/train_dino_wm.py --config configs/dino_wm_config.yaml

With common overrides:

  python dino_wm/train_dino_wm.py \
    --config configs/dino_wm_config.yaml \
    --hdf5-file arx5_datasets_6Feb_26.h5 \
    --dataset-stats arx5_datasets_6Feb_26_stats.json \
    --checkpoint-dir dino_wm_checkpoints \
    --auto-resume
"""

import argparse
import os
import sys
import numpy as np
import torch
import random
import wandb
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
from einops import rearrange
import matplotlib.pyplot as plt

# Ensure repo root is on sys.path regardless of current working directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from test_loader import SplitTrajectoryDataset
from dino_decoder import VQVAE
from dino_models import VideoTransformer, normalize_acs, normalize_states
from dino_wm.checkpoint_utils import filter_state_dict_by_shape
from dino_wm.config import MODEL_CONFIG, DECODER_CONFIG, get_dino_config, get_decoder_image_size
from dino_wm.train_wm_common import (
    build_wm_optimizer,
    build_train_loader,
    build_split_datasets,
    freeze_failure_head_for_wm_training,
    init_distributed_from_env,
    load_stats_tensors,
    load_yaml_config as _load_yaml_config,
    run_train_eval_loop,
    resolve_wm_checkpoint as _resolve_wm_checkpoint,
)


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    # Parse config path first so we can apply YAML values as argparse defaults.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("configs", "dino_wm_config.yaml"),
        help="Path to YAML config file (default: configs/dino_wm_config.yaml). CLI flags override it.",
    )
    pre_args, remaining_argv = pre_parser.parse_known_args(argv)
    cfg = _load_yaml_config(pre_args.config)
    if isinstance(cfg, dict) and isinstance(cfg.get("dino_assets"), dict):
        dino_assets = cfg["dino_assets"]
        if "decoder_checkpoint_dir" in dino_assets:
            cfg.setdefault(
                "decoder_checkpoint",
                os.path.join(dino_assets["decoder_checkpoint_dir"], "best_decoder.pth"),
            )
        if "wm_checkpoint_dir" in dino_assets:
            # Do not implicitly set resume_checkpoint here.
            # Resume selection is handled in main() so it can:
            # - prefer best_wm.pth when it exists
            # - fall back to latest_wm.pth / highest wm_iter*.pth
            # - avoid crashing when best_wm.pth hasn't been created yet
            cfg.setdefault("checkpoint_dir", dino_assets["wm_checkpoint_dir"])

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
        "--action-key",
        type=str,
        default="actions_delta",
        help="Action dataset key to load (default: actions_delta).",
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
        "--future-action-steps-train",
        type=int,
        default=50,
        help="Max number of future actions to sample for training (default: 50).",
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


def main(argv=None):
    args = parse_args(argv)

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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    use_amp = str(args.device).startswith("cuda") and torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    BS = args.batch_size
    BL = args.sequence_length
    EVAL_H = args.eval_horizon
    H = args.context_length
    pred_step = int(args.pred_step)
    action_horizon = int(args.action_horizon)
    future_action_steps_train = int(args.future_action_steps_train)
    if is_distributed and str(args.device).startswith("cuda"):
        device = f"cuda:{local_rank}"
    else:
        device = args.device

    if BL < 2:
        raise ValueError(f"--sequence-length must be >= 2 (got {BL}).")
    if H < 1:
        raise ValueError(f"--context-length must be >= 1 (got {H}).")
    # The training target is one step beyond the context window.
    # Keep sequence_length explicit to prevent silent config drift.
    if BL != H + 1:
        raise ValueError(
            f"--sequence-length must equal --context-length + 1 for current WM training "
            f"(got sequence_length={BL}, context_length={H})."
        )
    if EVAL_H < 1:
        raise ValueError(f"--eval-horizon must be >= 1 (got {EVAL_H}).")
    if EVAL_H < H:
        raise ValueError(f"--eval-horizon must be >= --context-length (got eval_horizon={EVAL_H}, context_length={H}).")
    
    if action_horizon < 1:
        raise ValueError(f"--action-horizon must be >= 1 (got {action_horizon}).")
    if future_action_steps_train < 1:
        raise ValueError(
            f"--future-action-steps-train must be >= 1 (got {future_action_steps_train})."
        )
    if int(args.eval_samples) < 1:
        raise ValueError(f"--eval-samples must be >= 1 (got {args.eval_samples}).")
    if pred_step < 1:
        raise ValueError(f"--pred-step must be >= 1 (got {pred_step}).")
    if is_rank0:
        print("Backbone flow: dino | front_latent_key=cam_zed_embd | wrist_latent_key=cam_rs_embd")
    
    # LOAD STATS
    stats_path = args.dataset_stats
    if is_rank0:
        print(f"Loading dataset stats from {stats_path}")
    _, stats_tensors, state_dim, action_dim = load_stats_tensors(stats_path, device)
    action_min = stats_tensors["action_min"]
    action_max = stats_tensors["action_max"]
    state_min = stats_tensors["state_min"]
    state_max = stats_tensors["state_max"]
    action_q02 = stats_tensors["action_q02"]
    action_q98 = stats_tensors["action_q98"]
    state_q02 = stats_tensors["state_q02"]
    state_q98 = stats_tensors["state_q98"]
    
    if is_rank0:
        print(f"Loaded state normalization stats from {stats_path}")
        print(f"Inferred state_dim={state_dim}, action_dim={action_dim} from dataset stats")

    # Dataset setup
    hdf5_file = args.hdf5_file
    dataset_info = build_split_datasets(
        SplitTrajectoryDataset,
        hdf5_file=hdf5_file,
        test_frac=float(args.test_frac),
        context_length=H,
        pred_step=pred_step,
        action_horizon=action_horizon,
        action_key=args.action_key,
        front_latent_key="cam_zed_embd",
        wrist_latent_key="cam_rs_embd",
    )
    # Primary training split used for gradient updates.
    expert_data = dataset_info["expert_data"]
    # Held-out split for numeric eval loss tracking.
    expert_data_eval = dataset_info["expert_data_eval"]
    # Held-out split sampled for qualitative "imagine" visualizations during eval.
    expert_data_imagine = dataset_info["expert_data_imagine"]
    num_traj = dataset_info["num_traj"]
    num_test = dataset_info["num_test"]
    
    if is_rank0:
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

    # Configure model dimensions and image sizes based on selected DINO version.
    dino_cfg = get_dino_config(args.dino_version)
    decoder_img_size = get_decoder_image_size(args.dino_version)
    MODEL_CONFIG['dim'] = dino_cfg['dim']
    MODEL_CONFIG['image_size'] = decoder_img_size
    latent_num_patches = int(dino_cfg['num_patches'])
    DECODER_CONFIG['decoder_image_size'] = decoder_img_size

    decoder = VQVAE().to(device)
    decoder_ckpt = torch.load(args.decoder_checkpoint, map_location=device)

    if isinstance(decoder_ckpt, dict) and "model_state_dict" in decoder_ckpt:
        decoder_state = decoder_ckpt["model_state_dict"]
    else:
        decoder_state = decoder_ckpt

    filtered, missing, unexpected, mismatched = filter_state_dict_by_shape(
        decoder.state_dict(),
        decoder_state,
    )

    decoder.load_state_dict(filtered, strict=False)
    if is_rank0:
        if missing:
            print(f"Warning: decoder missing {len(missing)} keys from checkpoint.")
        if unexpected:
            print(f"Warning: decoder has {len(unexpected)} unexpected keys in checkpoint.")
        if mismatched:
            print(f"Warning: decoder skipped {len(mismatched)} mismatched keys.")
    decoder.eval()
    if is_rank0:
        print(f"Loaded DINO decoder from {args.decoder_checkpoint}")

    # Initialize world model
    transition = VideoTransformer(
        state_dim=state_dim,    # Inferred from dataset stats
        action_dim=action_dim,  # Inferred from dataset stats
        num_frames=H,           # context window size
        action_horizon=action_horizon,
        backbone="dino",
        dino_version=args.dino_version,
        num_patches=latent_num_patches,
        **MODEL_CONFIG
    ).to(device)

    if is_distributed:
        transition = DistributedDataParallel(transition, device_ids=[local_rank])
    transition_module = transition.module if is_distributed else transition

    # failure_head is trained separately by classifier scripts.
    freeze_failure_head_for_wm_training(transition_module)

    transition.train()

    # Optimizer
    optimizer = build_wm_optimizer(transition_module)
    base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    # Load best_eval from existing best checkpoint to persist across sessions
    best_eval = float('inf')
    best_ckpt_path = os.path.join(args.checkpoint_dir, 'best_wm.pth')
    latest_ckpt_path = os.path.join(args.checkpoint_dir, 'latest_wm.pth')
    if os.path.exists(best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        if isinstance(best_ckpt, dict) and 'best_eval' in best_ckpt:
            best_eval = best_ckpt['best_eval']
            if is_rank0:
                print(f"Loaded previous best eval: {best_eval:.4f}")

    # Resume logic
    resume_path = args.resume_checkpoint
    if resume_path is not None and not os.path.exists(resume_path):
        # Common pattern: configs point at best_wm.pth, but first run hasn't produced it yet.
        # Instead of crashing, fall back to the best available checkpoint in the dir.
        if os.path.basename(resume_path) == "best_wm.pth":
            fallback = _resolve_wm_checkpoint(args.checkpoint_dir)
            if fallback is not None:
                if is_rank0:
                    print(
                        f"Warning: requested '{resume_path}' not found; "
                        f"falling back to '{fallback}'."
                    )
                resume_path = fallback
            else:
                if is_rank0:
                    print(
                        f"Warning: requested '{resume_path}' not found and no other checkpoints "
                        f"exist in '{args.checkpoint_dir}'. Starting from scratch."
                    )
                resume_path = None
        elif args.auto_resume:
            fallback = _resolve_wm_checkpoint(args.checkpoint_dir)
            if fallback is not None:
                if is_rank0:
                    print(
                        f"Warning: requested '{resume_path}' not found; "
                        f"auto-resume enabled, falling back to '{fallback}'."
                    )
                resume_path = fallback
            else:
                if is_rank0:
                    print(
                        f"Warning: requested '{resume_path}' not found and no checkpoints exist in "
                        f"'{args.checkpoint_dir}'. Starting from scratch."
                    )
                resume_path = None
        else:
            raise FileNotFoundError(
                f"Resume checkpoint '{resume_path}' not found. "
                f"Either point --resume-checkpoint to an existing file, or enable --auto-resume."
            )

    if resume_path is None and args.auto_resume:
        # Auto-resume default: prefer best if it exists, else fall back to latest/highest-iter.
        resume_path = _resolve_wm_checkpoint(args.checkpoint_dir)

    start_iter = args.start_iter
    if resume_path is not None:
        if is_rank0:
            print(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            filtered, missing, unexpected, mismatched = filter_state_dict_by_shape(
                transition_module.state_dict(),
                ckpt['model_state_dict'],
            )
            transition_module.load_state_dict(filtered, strict=False)
            if is_rank0:
                if missing:
                    print(f"Warning: missing {len(missing)} keys from checkpoint.")
                if unexpected:
                    print(f"Warning: {len(unexpected)} unexpected keys in checkpoint.")
                if mismatched:
                    print(f"Warning: {len(mismatched)} keys had shape mismatches and were skipped.")
            if 'optimizer_state_dict' in ckpt:
                if missing or mismatched:
                    if is_rank0:
                        print("Warning: skipping optimizer state due to partial model load.")
                else:
                    try:
                        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    except Exception as e:
                        if is_rank0:
                            print(f"Warning: failed to load optimizer state ({e}); continuing with fresh optimizer.")
            if 'iter' in ckpt:
                start_iter = int(ckpt['iter']) + 1
            if 'best_eval' in ckpt and best_eval == float('inf'):
                best_eval = ckpt['best_eval']
        else:
            filtered, missing, unexpected, mismatched = filter_state_dict_by_shape(
                transition_module.state_dict(),
                ckpt,
            )
            transition_module.load_state_dict(filtered, strict=False)
            if is_rank0:
                if missing:
                    print(f"Warning: missing {len(missing)} keys from checkpoint.")
                if unexpected:
                    print(f"Warning: {len(unexpected)} unexpected keys in checkpoint.")
                if mismatched:
                    print(f"Warning: {len(mismatched)} keys had shape mismatches and were skipped.")
                if missing or mismatched:
                    print("Warning: skipping optimizer state due to partial model load.")

    transition.train()

    train_iter = args.train_iters
    def _render_eval_images_fn(*, pred_front, pred_wrist, eval_data, target_idx, device):
        pred_front_last = pred_front[:, [-1]]
        pred_wrist_last = pred_wrist[:, [-1]]
        pred_latent = torch.cat([pred_front_last, pred_wrist_last], dim=0)
        pred_ims, _ = decoder(pred_latent)
        pred_ims = rearrange(pred_ims, "(b t) c h w -> b t h w c", t=1)
        pred_im1, pred_im2 = torch.split(pred_ims, [1, 1], dim=0)
        pred_im1 = pred_im1.squeeze(0).squeeze(0)
        pred_im2 = pred_im2.squeeze(0).squeeze(0)
        target_size = DECODER_CONFIG['decoder_image_size']

        gt_im1 = eval_data['agentview_image'][[0]].to(device)[:, target_idx] / 255.
        gt_im2 = eval_data['robot0_eye_in_hand_image'][[0]].to(device)[:, target_idx] / 255.
        gt_im1 = F.interpolate(
            gt_im1.permute(0, 3, 1, 2), size=target_size, mode='bilinear', align_corners=False
        ).permute(0, 2, 3, 1)
        gt_im2 = F.interpolate(
            gt_im2.permute(0, 3, 1, 2), size=target_size, mode='bilinear', align_corners=False
        ).permute(0, 2, 3, 1)
        return {
            'pred_front': pred_im1.detach().cpu().numpy(),
            'pred_wrist': pred_im2.detach().cpu().numpy(),
            'front': gt_im1.squeeze(0).detach().cpu().numpy(),
            'wrist': gt_im2.squeeze(0).detach().cpu().numpy(),
        }

    best_eval = run_train_eval_loop(
        args=args,
        start_iter=start_iter,
        train_iter=train_iter,
        transition=transition,
        transition_module=transition_module,
        optimizer=optimizer,
        base_lrs=base_lrs,
        scaler=scaler,
        use_amp=use_amp,
        train_loader=train_loader,
        train_sampler=train_sampler,
        expert_data=expert_data,
        expert_data_eval=expert_data_eval,
        expert_data_imagine=expert_data_imagine,
        H=H,
        pred_step=pred_step,
        action_horizon=action_horizon,
        future_action_steps_train=future_action_steps_train,
        device=device,
        is_rank0=is_rank0,
        is_distributed=is_distributed,
        rank=rank,
        world_size=world_size,
        action_min=action_min,
        action_max=action_max,
        state_min=state_min,
        state_max=state_max,
        action_q02=action_q02,
        action_q98=action_q98,
        state_q02=state_q02,
        state_q98=state_q98,
        latest_ckpt_path=latest_ckpt_path,
        checkpoint_dir=args.checkpoint_dir,
        best_eval=best_eval,
        seed=args.seed,
        normalize_acs_fn=normalize_acs,
        normalize_states_fn=normalize_states,
        render_eval_images_fn=_render_eval_images_fn,
    )

    if is_rank0:
        plt.legend()
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        plt.savefig(os.path.join(args.checkpoint_dir, 'training_curve.png'))

    best_eval_val = best_eval.item() if hasattr(best_eval, 'item') else best_eval
    if is_rank0:
        print(f"\nTraining complete. Best eval loss: {best_eval_val:.4f}")

    if is_distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
