#!/usr/bin/env python3
"""
Train the WAN-backbone World Model on trajectory data.

Quickstart:

  python dino_wm/train_wan_wm.py --config configs/wan_wm_config.yaml

With common overrides:

  python dino_wm/train_wan_wm.py \
    --config configs/wan_wm_config.yaml \
    --hdf5-file arx5_datasets_6Feb_26_wan.h5 \
    --dataset-stats arx5_datasets_6Feb_26_stats.json \
    --checkpoint-dir wan_wm_checkpoints \
    --auto-resume
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from einops import rearrange
from torch.nn.parallel import DistributedDataParallel

# Ensure repo root import works when executed as a script.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dino_wm.dino_models import VideoTransformer, normalize_acs, normalize_states
from dino_wm.checkpoint_utils import filter_state_dict_by_shape
from dino_wm.config import DECODER_CONFIG, MODEL_CONFIG
from dino_wm.train_wm_common import (
    build_wm_optimizer,
    build_split_datasets,
    build_train_loader,
    freeze_failure_head_for_wm_training,
    init_distributed_from_env,
    load_stats_tensors,
    load_yaml_config,
    resolve_wm_checkpoint,
    run_train_eval_loop,
)
from dino_wm.test_loader import SplitTrajectoryDataset


def infer_latent_shape(hdf5_file: str, front_key: str, wrist_key: str) -> tuple[int, int]:
    """Infer (num_patches, latent_dim) from HDF5 latent datasets."""
    import h5py

    with h5py.File(hdf5_file, "r") as hf:
        traj_ids = sorted(list(hf.keys()))
        if not traj_ids:
            raise ValueError(f"No trajectories found in {hdf5_file}")
        traj = hf[traj_ids[0]]
        if front_key not in traj:
            raise KeyError(f"Front latent key '{front_key}' not found in trajectory '{traj_ids[0]}'")
        if wrist_key not in traj:
            raise KeyError(f"Wrist latent key '{wrist_key}' not found in trajectory '{traj_ids[0]}'")
        front_shape = tuple(traj[front_key].shape)
        wrist_shape = tuple(traj[wrist_key].shape)

    if len(front_shape) != 3 or len(wrist_shape) != 3:
        raise ValueError(
            "Expected latent tensors with shape (T, num_patches, dim). "
            f"Got front={front_shape}, wrist={wrist_shape}"
        )
    if front_shape[1:] != wrist_shape[1:]:
        raise ValueError(
            "Front and wrist latent shapes must match for shared transformer heads. "
            f"Got front={front_shape[1:]}, wrist={wrist_shape[1:]}"
        )
    return int(front_shape[1]), int(front_shape[2])


class WanDecoderAdapter:
    """Thin wrapper around Diffusers AutoencoderKLWan decode API."""

    def __init__(
        self,
        *,
        model: str,
        subfolder: str,
        device: str,
        dtype: str,
        latent_h: int = 0,
        latent_w: int = 0,
    ):
        from diffusers import AutoencoderKLWan

        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        self.model_dtype = dtype_map[dtype] if self.device.type == "cuda" else torch.float32
        self.latent_h = int(latent_h)
        self.latent_w = int(latent_w)
        self.vae = AutoencoderKLWan.from_pretrained(
            model, subfolder=subfolder, torch_dtype=self.model_dtype
        ).to(self.device).eval()

    def _infer_hw(self, num_patches: int) -> tuple[int, int]:
        if self.latent_h > 0 and self.latent_w > 0:
            if self.latent_h * self.latent_w != num_patches:
                raise ValueError(
                    f"wan_latent_height*wan_latent_width ({self.latent_h*self.latent_w}) "
                    f"must equal num_patches ({num_patches})."
                )
            return self.latent_h, self.latent_w

        side = int(np.sqrt(num_patches))
        if side * side != num_patches:
            raise ValueError(
                "Cannot infer WAN latent H/W from non-square num_patches. "
                "Set --wan-latent-height and --wan-latent-width explicitly."
            )
        return side, side

    @torch.no_grad()
    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, T, N, C) -> image tensor (B, T, H, W, 3) in [0, 1]
        _, _, n, _ = tokens.shape
        h, w = self._infer_hw(n)
        z = rearrange(tokens, "b t (h w) c -> b c t h w", h=h, w=w).to(
            device=self.device, dtype=self.model_dtype
        )
        y = self.vae.decode(z).sample
        y = y.clamp(-1, 1).add(1.0).mul(0.5)
        return rearrange(y, "b c t h w -> b t h w c").float()


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("configs", "wan_wm_config.yaml"),
        help="Path to YAML config file (default: configs/wan_wm_config.yaml). CLI flags override it.",
    )
    pre_args, remaining_argv = pre_parser.parse_known_args(argv)
    cfg = load_yaml_config(pre_args.config)

    parser = argparse.ArgumentParser(
        description="Train WAN-backbone World Model on trajectory data",
        parents=[pre_parser],
    )
    parser.add_argument("--hdf5-file", "--hdf5", dest="hdf5_file", type=str, default="arx5_subset_train.h5")
    parser.add_argument("--dataset-stats", type=str, default="dataset_stats.json")
    parser.add_argument("--checkpoint-dir", type=str, default="wan_wm_checkpoints")
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--start-iter", type=int, default=0)
    parser.add_argument(
        "--auto-resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Auto-resume from latest checkpoint in --checkpoint-dir.",
    )

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-iters", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=3)
    parser.add_argument("--pred-step", type=int, default=1)
    parser.add_argument("--action-horizon", type=int, default=100)
    parser.add_argument("--future-action-steps-train", type=int, default=50)
    parser.add_argument("--eval-horizon", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--eval-samples", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--action-key", type=str, default="actions_delta")

    parser.add_argument("--front-latent-key", type=str, default="wan_front_embd")
    parser.add_argument("--wrist-latent-key", type=str, default="wan_wrist_embd")
    parser.add_argument("--wan-vae-model", type=str, default=None)
    parser.add_argument("--wan-vae-subfolder", type=str, default="vae")
    parser.add_argument("--wan-vae-dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--wan-latent-height", type=int, default=0)
    parser.add_argument("--wan-latent-width", type=int, default=0)

    parser.add_argument("--wandb-mode", type=str, choices=["online", "offline", "disabled"], default="offline")
    parser.add_argument("--wandb-project", type=str, default="wan-WM")
    parser.add_argument("--wandb-entity", type=str, default="pravsels")
    parser.add_argument("--wandb-name", type=str, default="wan-WM")
    parser.add_argument("--lr-schedule", type=str, choices=["cosine", "constant"], default="cosine")
    parser.add_argument("--lr-min-factor", type=float, default=0.1)
    parser.add_argument("--lr-warmup-iters", type=int, default=1000)

    known_dests = {a.dest for a in parser._actions}
    for k, v in (cfg or {}).items():
        if k in known_dests:
            parser.set_defaults(**{k: v})

    return parser.parse_args(remaining_argv)


def main(argv=None):
    args = parse_args(argv)

    rank, world_size, local_rank, is_distributed = init_distributed_from_env()
    is_rank0 = rank == 0

    if is_rank0:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            config=vars(args),
        )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    use_amp = str(args.device).startswith("cuda") and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    bs = int(args.batch_size)
    bl = int(args.sequence_length)
    h = int(args.context_length)
    eval_h = int(args.eval_horizon)
    pred_step = int(args.pred_step)
    action_horizon = int(args.action_horizon)
    future_action_steps_train = int(args.future_action_steps_train)

    if is_distributed and str(args.device).startswith("cuda"):
        device = f"cuda:{local_rank}"
    else:
        device = args.device

    if bl < 2:
        raise ValueError(f"--sequence-length must be >= 2 (got {bl}).")
    if h < 1:
        raise ValueError(f"--context-length must be >= 1 (got {h}).")
    # The training target is one step beyond the context window.
    # Keep sequence_length explicit to prevent silent config drift.
    if bl != h + 1:
        raise ValueError(
            f"--sequence-length must equal --context-length + 1 for current WM training "
            f"(got sequence_length={bl}, context_length={h})."
        )
    if eval_h < 1:
        raise ValueError(f"--eval-horizon must be >= 1 (got {eval_h}).")
    if eval_h < h:
        raise ValueError(
            f"--eval-horizon must be >= --context-length (got eval_horizon={args.eval_horizon}, context_length={h})."
        )
    if pred_step < 1:
        raise ValueError(f"--pred-step must be >= 1 (got {pred_step}).")
    if action_horizon < 1:
        raise ValueError(f"--action-horizon must be >= 1 (got {action_horizon}).")
    if future_action_steps_train < 1:
        raise ValueError(
            f"--future-action-steps-train must be >= 1 (got {future_action_steps_train})."
        )
    if int(args.eval_samples) < 1:
        raise ValueError(f"--eval-samples must be >= 1 (got {args.eval_samples}).")

    _, stats_tensors, state_dim, action_dim = load_stats_tensors(args.dataset_stats, device)
    action_min = stats_tensors["action_min"]
    action_max = stats_tensors["action_max"]
    state_min = stats_tensors["state_min"]
    state_max = stats_tensors["state_max"]
    action_q02 = stats_tensors["action_q02"]
    action_q98 = stats_tensors["action_q98"]
    state_q02 = stats_tensors["state_q02"]
    state_q98 = stats_tensors["state_q98"]

    dataset_info = build_split_datasets(
        SplitTrajectoryDataset,
        hdf5_file=args.hdf5_file,
        test_frac=float(args.test_frac),
        context_length=h,
        pred_step=pred_step,
        action_horizon=action_horizon,
        action_key=args.action_key,
        front_latent_key=args.front_latent_key,
        wrist_latent_key=args.wrist_latent_key,
    )
    expert_data = dataset_info["expert_data"]
    expert_data_eval = dataset_info["expert_data_eval"]
    expert_data_imagine = dataset_info["expert_data_imagine"]

    train_loader, train_sampler = build_train_loader(
        expert_data,
        bs,
        is_distributed=is_distributed,
        rank=rank,
        world_size=world_size,
    )

    latent_num_patches, latent_dim = infer_latent_shape(
        args.hdf5_file,
        args.front_latent_key,
        args.wrist_latent_key,
    )
    MODEL_CONFIG["dim"] = int(latent_dim)
    MODEL_CONFIG["image_size"] = (224, 224)
    DECODER_CONFIG["decoder_image_size"] = MODEL_CONFIG["image_size"]

    wan_decoder = None
    if args.wan_vae_model:
        wan_decoder = WanDecoderAdapter(
            model=args.wan_vae_model,
            subfolder=args.wan_vae_subfolder,
            device=device,
            dtype=args.wan_vae_dtype,
            latent_h=args.wan_latent_height,
            latent_w=args.wan_latent_width,
        )
        print(f"Loaded WAN VAE decoder from {args.wan_vae_model}/{args.wan_vae_subfolder}")
    else:
        print("WAN backbone selected without --wan-vae-model; eval image decoding will be skipped.")

    transition = VideoTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        num_frames=h,
        action_horizon=action_horizon,
        backbone="wan",
        dino_version="v3",
        num_patches=int(latent_num_patches),
        **MODEL_CONFIG,
    ).to(device)

    if is_distributed:
        transition = DistributedDataParallel(transition, device_ids=[local_rank])
    transition_module = transition.module if is_distributed else transition

    # failure_head is trained separately by classifier scripts.
    freeze_failure_head_for_wm_training(transition_module)

    transition.train()

    optimizer = build_wm_optimizer(transition_module)
    base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    best_eval = float("inf")
    best_ckpt_path = os.path.join(args.checkpoint_dir, "best_wm.pth")
    latest_ckpt_path = os.path.join(args.checkpoint_dir, "latest_wm.pth")
    if os.path.exists(best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        if isinstance(best_ckpt, dict) and "best_eval" in best_ckpt:
            best_eval = best_ckpt["best_eval"]

    resume_path = args.resume_checkpoint
    if resume_path is not None and not os.path.exists(resume_path):
        if os.path.basename(resume_path) == "best_wm.pth" or args.auto_resume:
            fallback = resolve_wm_checkpoint(args.checkpoint_dir)
            resume_path = fallback
            if resume_path is not None:
                print(f"Warning: requested checkpoint missing; falling back to '{resume_path}'.")
            else:
                print("Warning: no checkpoint found for resume; starting from scratch.")
        else:
            raise FileNotFoundError(
                f"Resume checkpoint '{resume_path}' not found. Use --auto-resume to allow fallback."
            )
    if resume_path is None and args.auto_resume:
        resume_path = resolve_wm_checkpoint(args.checkpoint_dir)

    start_iter = int(args.start_iter)
    if resume_path is not None:
        print(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            filtered, missing, unexpected, mismatched = filter_state_dict_by_shape(
                transition_module.state_dict(),
                ckpt["model_state_dict"],
            )
            transition_module.load_state_dict(filtered, strict=False)
            if missing:
                print(f"Warning: missing {len(missing)} keys from checkpoint.")
            if unexpected:
                print(f"Warning: {len(unexpected)} unexpected keys in checkpoint.")
            if mismatched:
                print(f"Warning: {len(mismatched)} keys had shape mismatches and were skipped.")
            if "optimizer_state_dict" in ckpt and not (missing or mismatched):
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except Exception as e:
                    print(f"Warning: failed to load optimizer state ({e}); continuing with fresh optimizer.")
            if "iter" in ckpt:
                start_iter = int(ckpt["iter"]) + 1
            if "best_eval" in ckpt and best_eval == float("inf"):
                best_eval = ckpt["best_eval"]
        else:
            filtered, _, _, _ = filter_state_dict_by_shape(transition_module.state_dict(), ckpt)
            transition_module.load_state_dict(filtered, strict=False)

    transition.train()

    def _render_eval_images_fn(*, pred_front, pred_wrist, eval_data, target_idx, device):
        if wan_decoder is None:
            return None
        pred_im1 = wan_decoder.decode_tokens(pred_front[:, [-1]]).squeeze(0).squeeze(0)
        pred_im2 = wan_decoder.decode_tokens(pred_wrist[:, [-1]]).squeeze(0).squeeze(0)
        target_size = (int(pred_im1.shape[0]), int(pred_im1.shape[1]))

        gt_im1 = eval_data["agentview_image"][[0]].to(device)[:, target_idx] / 255.0
        gt_im2 = eval_data["robot0_eye_in_hand_image"][[0]].to(device)[:, target_idx] / 255.0
        gt_im1 = F.interpolate(
            gt_im1.permute(0, 3, 1, 2), size=target_size, mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        gt_im2 = F.interpolate(
            gt_im2.permute(0, 3, 1, 2), size=target_size, mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        return {
            "pred_front": pred_im1.detach().cpu().numpy(),
            "pred_wrist": pred_im2.detach().cpu().numpy(),
            "front": gt_im1.squeeze(0).detach().cpu().numpy(),
            "wrist": gt_im2.squeeze(0).detach().cpu().numpy(),
        }

    best_eval = run_train_eval_loop(
        args=args,
        start_iter=start_iter,
        train_iter=int(args.train_iters),
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
        H=h,
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
    plt.savefig(os.path.join(args.checkpoint_dir, "training_curve.png"))
    print(f"\nTraining complete. Best eval loss: {float(best_eval):.4f}")


if __name__ == "__main__":
    main()
