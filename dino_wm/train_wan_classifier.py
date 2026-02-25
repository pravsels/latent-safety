#!/usr/bin/env python3
"""
Train the failure classifier head on top of a frozen WAN World Model.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from einops import rearrange
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from checkpoint_utils import filter_state_dict_by_shape
from dino_models import FUTURE_ACTION_HORIZON_MAX, VideoTransformer, normalize_acs, normalize_states
from test_loader import SplitTrajectoryDataset
from train_wm_common import (
    build_train_loader,
    init_distributed_from_env,
    load_yaml_config as _load_yaml_config,
    sample_future_action_window,
)
from dino_wm.config import MODEL_CONFIG, WAN_CONFIG


def fail_loss(pred, fail_data):
    safe_data = torch.where(fail_data == 0.0)
    unsafe_data = torch.where(fail_data == 1.0)
    unsafe_data_weak = torch.where(fail_data == 2.0)

    pos = pred[safe_data]
    neg = pred[unsafe_data]
    neg_weak = pred[unsafe_data_weak]

    gamma = 0.75
    lx_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    if pos.size(0) > 0:
        lx_loss = lx_loss + (1.0 / pos.size(0)) * torch.sum(torch.relu(gamma - pos))
    if neg.size(0) > 0:
        lx_loss = lx_loss + (1.0 / neg.size(0)) * torch.sum(torch.relu(gamma + neg))
    if neg_weak.size(0) > 0:
        lx_loss = lx_loss + (1.0 / neg_weak.size(0)) * torch.sum(torch.relu(neg_weak))

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


def infer_latent_shape(hdf5_file: str, front_key: str, wrist_key: str) -> tuple[int, int]:
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

        side = WAN_CONFIG["latent_side"]
        if side * side == num_patches:
            return side, side

        raise ValueError(
            f"num_patches={num_patches} doesn't match expected {side}x{side}={side*side}. "
            "Set --wan-latent-height and --wan-latent-width explicitly."
        )

    @torch.no_grad()
    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
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
        default=os.path.join("configs", "wan_classifier_config.yaml"),
        help="Path to YAML config file (default: configs/wan_classifier_config.yaml). CLI flags override it.",
    )
    pre_args, remaining_argv = pre_parser.parse_known_args(argv)
    cfg = _load_yaml_config(pre_args.config)

    parser = argparse.ArgumentParser(
        description="Train failure classifier on top of frozen WAN World Model",
        parents=[pre_parser],
    )
    parser.add_argument("--hdf5-file", "--hdf5", dest="hdf5_file", type=str, default=None)
    parser.add_argument("--dataset-stats", type=str, default="dataset_stats.json")
    parser.add_argument("--action-key", type=str, default="actions_delta")
    parser.add_argument("--front-latent-key", type=str, default="wan_front_embd")
    parser.add_argument("--wrist-latent-key", type=str, default="wan_wrist_embd")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=3)
    parser.add_argument("--eval-horizon", type=int, default=16)
    parser.add_argument("--train-iters", type=int, default=10000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--wm-checkpoint", type=str, default="wan_wm_checkpoints/best_wm.pth")
    parser.add_argument("--checkpoint-dir", type=str, default="wan_classifier_checkpoints")
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--num-test-trajectories", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--action-horizon", type=int, default=FUTURE_ACTION_HORIZON_MAX)
    parser.add_argument("--future-action-steps-train", type=int, default=50)
    parser.add_argument("--wan-vae-model", type=str, default="ByteDance/Video-As-Prompt-Wan2.1-14B")
    parser.add_argument("--wan-vae-subfolder", type=str, default="vae")
    parser.add_argument("--wan-vae-dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--wan-latent-height", type=int, default=0)
    parser.add_argument("--wan-latent-width", type=int, default=0)
    parser.add_argument("--wandb-mode", type=str, default="offline", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-project", type=str, default="wan-classifier")
    parser.add_argument("--wandb-entity", type=str, default="pravsels")
    parser.add_argument("--wandb-name", type=str, default="WAN-Classifier")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--start-iter", type=int, default=0)

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
        raise ValueError("Missing --hdf5-file. Provide it via CLI or set hdf5_file in config.")

    args.hdf5_file = _resolve_data_path(args.hdf5_file)
    args.dataset_stats = _resolve_data_path(args.dataset_stats)
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
            config=vars(args),
        )

    use_amp = str(device).startswith("cuda") and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    bs = int(args.batch_size)
    bl = int(args.sequence_length)
    h = int(args.context_length)
    eval_h = int(args.eval_horizon)
    action_horizon = int(args.action_horizon)
    future_action_steps_train = int(args.future_action_steps_train)

    if action_horizon < 1:
        raise ValueError(f"--action-horizon must be >= 1 (got {action_horizon}).")
    if future_action_steps_train < 1:
        raise ValueError(
            f"--future-action-steps-train must be >= 1 (got {future_action_steps_train})."
        )
    max_future_len = min(action_horizon, future_action_steps_train)

    if not os.path.exists(args.dataset_stats):
        raise FileNotFoundError(
            f"Stats file '{args.dataset_stats}' not found! Generate it first."
        )
    if is_rank0:
        print(f"Loading dataset stats from {args.dataset_stats}")
    with open(args.dataset_stats, "r") as f:
        stats = json.load(f)

    required = ["action_min", "action_max", "state_min", "state_max"]
    missing = [k for k in required if k not in stats]
    if missing:
        raise ValueError(f"Stats file missing required keys: {missing}")

    action_min = torch.tensor(stats["action_min"]).float().to(device)
    action_max = torch.tensor(stats["action_max"]).float().to(device)
    state_min = torch.tensor(stats["state_min"]).float().to(device)
    state_max = torch.tensor(stats["state_max"]).float().to(device)
    action_q02 = (
        torch.tensor(stats["action_delta_q02"]).float().to(device) if "action_delta_q02" in stats else None
    )
    action_q98 = (
        torch.tensor(stats["action_delta_q98"]).float().to(device) if "action_delta_q98" in stats else None
    )
    state_q02 = torch.tensor(stats["state_q02"]).float().to(device) if "state_q02" in stats else None
    state_q98 = torch.tensor(stats["state_q98"]).float().to(device) if "state_q98" in stats else None
    state_dim = len(stats["state_min"])
    action_dim = len(stats["action_min"])
    if is_rank0:
        print(f"Inferred state_dim={state_dim}, action_dim={action_dim}")

    with h5py.File(args.hdf5_file, "r") as hf:
        num_traj = len(hf.keys())
    if args.num_test_trajectories is not None:
        num_test = max(1, min(int(args.num_test_trajectories), num_traj))
    else:
        num_test = max(1, int(float(args.test_frac) * num_traj))
    if num_traj - num_test < 1 and num_traj > 1:
        num_test = num_traj - 1

    train_segment_len = bl + max_future_len
    eval_segment_len = max(32, eval_h + max_future_len)
    expert_data = SplitTrajectoryDataset(
        args.hdf5_file,
        train_segment_len,
        split="train",
        num_test=num_test,
        action_key=args.action_key,
        front_embd_key=args.front_latent_key,
        wrist_embd_key=args.wrist_latent_key,
    )
    expert_data_eval = SplitTrajectoryDataset(
        args.hdf5_file,
        train_segment_len,
        split="test",
        num_test=num_test,
        action_key=args.action_key,
        front_embd_key=args.front_latent_key,
        wrist_embd_key=args.wrist_latent_key,
    )
    expert_data_imagine = SplitTrajectoryDataset(
        args.hdf5_file,
        eval_segment_len,
        split="test",
        num_test=num_test,
        action_key=args.action_key,
        front_embd_key=args.front_latent_key,
        wrist_embd_key=args.wrist_latent_key,
    )
    if is_rank0:
        print(f"Dataset: {args.hdf5_file}")
        print(f"  Train: {num_traj - num_test} trajectories")
        print(f"  Eval:  {num_test} trajectories")

    train_loader, train_sampler = build_train_loader(
        expert_data,
        bs,
        is_distributed=is_distributed,
        rank=rank,
        world_size=world_size,
    )
    expert_loader = iter(train_loader)
    if is_rank0:
        expert_loader_eval = iter(DataLoader(expert_data_eval, batch_size=bs, shuffle=True))
        expert_loader_imagine = iter(DataLoader(expert_data_imagine, batch_size=1, shuffle=True))
    else:
        expert_loader_eval = None
        expert_loader_imagine = None

    latent_num_patches, latent_dim = infer_latent_shape(
        args.hdf5_file, args.front_latent_key, args.wrist_latent_key
    )
    MODEL_CONFIG["dim"] = int(latent_dim)
    MODEL_CONFIG["image_size"] = (224, 224)

    transition = VideoTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        num_frames=bl - 1,
        action_horizon=action_horizon,
        backbone="wan",
        dino_version="v3",
        num_patches=int(latent_num_patches),
        **MODEL_CONFIG,
    ).to(device)
    if is_distributed:
        transition = DistributedDataParallel(transition, device_ids=[local_rank], find_unused_parameters=True)
    transition_module = transition.module if is_distributed else transition

    if is_rank0:
        print(f"Loading world model from {args.wm_checkpoint}")
    wm_ckpt = torch.load(args.wm_checkpoint, map_location=device)
    wm_state = wm_ckpt["model_state_dict"] if isinstance(wm_ckpt, dict) and "model_state_dict" in wm_ckpt else wm_ckpt
    filtered, missing_keys, unexpected, mismatched = filter_state_dict_by_shape(transition_module.state_dict(), wm_state)
    transition_module.load_state_dict(filtered, strict=False)
    if is_rank0 and missing_keys:
        print(f"Warning: missing {len(missing_keys)} keys from world-model checkpoint.")
    if is_rank0 and unexpected:
        print(f"Warning: world-model checkpoint has {len(unexpected)} unexpected keys.")
    if is_rank0 and mismatched:
        print(f"Warning: skipped {len(mismatched)} mismatched world-model keys.")

    for name, p in transition_module.named_parameters():
        p.requires_grad = name.startswith("failure_head")
    optimizer = AdamW([{"params": transition_module.failure_head.parameters(), "lr": args.learning_rate}])

    wan_decoder = WanDecoderAdapter(
        model=args.wan_vae_model,
        subfolder=args.wan_vae_subfolder,
        device=device,
        dtype=args.wan_vae_dtype,
        latent_h=args.wan_latent_height,
        latent_w=args.wan_latent_width,
    )

    best_eval = float("inf")
    start_iter = int(args.start_iter)
    best_ckpt_path = os.path.join(args.checkpoint_dir, "best_classifier.pth")
    if args.resume_checkpoint is not None:
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        if isinstance(ckpt, dict):
            if "failure_head_state_dict" in ckpt:
                transition_module.failure_head.load_state_dict(ckpt["failure_head_state_dict"])
            else:
                transition_module.failure_head.load_state_dict(ckpt)
            if "best_eval" in ckpt:
                best_eval = ckpt["best_eval"]
            if "iteration" in ckpt and args.start_iter == 0:
                start_iter = int(ckpt["iteration"]) + 1
        else:
            transition_module.failure_head.load_state_dict(ckpt)
    elif os.path.exists(best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        if isinstance(best_ckpt, dict) and "failure_head_state_dict" in best_ckpt:
            transition_module.failure_head.load_state_dict(best_ckpt["failure_head_state_dict"])
            if "best_eval" in best_ckpt:
                best_eval = best_ckpt["best_eval"]
            if "iteration" in best_ckpt and args.start_iter == 0:
                start_iter = int(best_ckpt["iteration"]) + 1

    train_iter = int(args.train_iters)
    transition.train()
    for i in tqdm(range(start_iter, train_iter), desc="Training", unit="iter", disable=not is_rank0):
        if i > 0 and i % len(train_loader) == 0:
            if train_sampler is not None:
                train_sampler.set_epoch(i)
            train_loader, train_sampler = build_train_loader(
                expert_data,
                bs,
                is_distributed=is_distributed,
                rank=rank,
                world_size=world_size,
            )
            expert_loader = iter(train_loader)
        if is_rank0 and i > 0 and i % len(expert_loader_eval) == 0:
            expert_loader_eval = iter(DataLoader(expert_data_eval, batch_size=bs, shuffle=True))
        if is_rank0 and i > 0 and i % len(expert_loader_imagine) == 0:
            expert_loader_imagine = iter(DataLoader(expert_data_imagine, batch_size=1, shuffle=True))

        data = next(expert_loader)
        data1 = data[args.front_latent_key].to(device)
        data2 = data[args.wrist_latent_key].to(device)
        inputs1 = data1[:, : bl - 1]
        inputs2 = data2[:, : bl - 1]

        norm_states = normalize_states(
            data["state"].to(device), state_min, state_max, q02=state_q02, q98=state_q98
        )
        states = norm_states[:, : bl - 1]

        norm_acs = normalize_acs(
            data["action"].to(device), action_min, action_max, q02=action_q02, q98=action_q98
        )
        acs = norm_acs[:, : bl - 1]
        future_len = sample_future_action_window(
            action_horizon=action_horizon,
            future_action_steps_train=future_action_steps_train,
        )
        t = bl - 2
        future_actions = norm_acs[:, t + 1 : t + 1 + future_len]

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            _, _, _, pred_fail = transition(inputs1, inputs2, states, acs, future_actions)
            target_idx = bl - 1 + future_len
            pred_fail_target = pred_fail[:, -1].squeeze(-1)
            target_fail = data["failure"][:, target_idx].to(device)
            loss = fail_loss(pred_fail_target, target_fail)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if is_rank0:
            wandb.log({"train_loss": loss.item()})
            print(f"\rIter {i}, Train Loss: {loss.item():.4f}", end="", flush=True)

        if i % int(args.eval_interval) == 0:
            if is_distributed:
                torch.distributed.barrier()
            if is_rank0:
                eval_data = next(expert_loader_imagine)
                transition_module.eval()
                with torch.no_grad():
                    eval_front = eval_data[args.front_latent_key].to(device)
                    eval_wrist = eval_data[args.wrist_latent_key].to(device)
                    inputs1 = eval_front[[0], :h]
                    inputs2 = eval_wrist[[0], :h]
                    all_acs = normalize_acs(
                        eval_data["action"][[0]].to(device),
                        action_min,
                        action_max,
                        q02=action_q02,
                        q98=action_q98,
                    )
                    acs = all_acs[:, :h]
                    states = normalize_states(
                        eval_data["state"][[0], :h].to(device),
                        state_min,
                        state_max,
                        q02=state_q02,
                        q98=state_q98,
                    )

                    decoder_h, decoder_w = MODEL_CONFIG["image_size"]
                    im1s_raw = eval_data["agentview_image"][[0], :h].squeeze().to(device) / 255.0
                    im2s_raw = eval_data["robot0_eye_in_hand_image"][[0], :h].squeeze().to(device) / 255.0
                    im1s = F.interpolate(
                        im1s_raw.permute(0, 3, 1, 2), size=(decoder_h, decoder_w), mode="bilinear", align_corners=False
                    ).permute(0, 2, 3, 1)
                    im2s = F.interpolate(
                        im2s_raw.permute(0, 3, 1, 2), size=(decoder_h, decoder_w), mode="bilinear", align_corners=False
                    ).permute(0, 2, 3, 1)

                    for k in range(eval_h - h):
                        t = (h - 1) + k
                        future_actions = all_acs[:, t + 1 : t + 1 + max_future_len]
                        pred1, pred2, pred_state, pred_fail = transition(inputs1, inputs2, states, acs, future_actions)

                        pred_im1 = wan_decoder.decode_tokens(pred1[:, [-1]]).squeeze(0).squeeze(0)
                        pred_im2 = wan_decoder.decode_tokens(pred2[:, [-1]]).squeeze(0).squeeze(0)
                        pred_fail_last = pred_fail[:, -1].squeeze(-1)
                        if pred_fail_last.item() < 0:
                            pred_im1[:, :, 0] *= 2
                            pred_im2[:, :, 0] *= 2

                        im1s = torch.cat([im1s, pred_im1.unsqueeze(0)], dim=0)
                        im2s = torch.cat([im2s, pred_im2.unsqueeze(0)], dim=0)
                        acs = torch.cat([acs[:, 1:], all_acs[:, h + k : h + k + 1]], dim=1)
                        inputs1 = torch.cat([inputs1[:, 1:], pred1[:, -1].unsqueeze(1)], dim=1)
                        inputs2 = torch.cat([inputs2[:, 1:], pred2[:, -1].unsqueeze(1)], dim=1)
                        states = torch.cat([states[:, 1:], pred_state[:, -1].unsqueeze(1)], dim=1)

                    gt_im1 = F.interpolate(
                        eval_data["agentview_image"][[0], :eval_h].squeeze().to(device).float().permute(0, 3, 1, 2),
                        size=(decoder_h, decoder_w),
                        mode="bilinear",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)
                    gt_im2 = F.interpolate(
                        eval_data["robot0_eye_in_hand_image"][[0], :eval_h].squeeze().to(device).float().permute(0, 3, 1, 2),
                        size=(decoder_h, decoder_w),
                        mode="bilinear",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)
                    gt_fail = eval_data["failure"][[0], :eval_h].squeeze().to(device)
                    for j in range(eval_h):
                        if gt_fail[j] > 0:
                            gt_im1[j, :, :, 0] *= 2
                            gt_im2[j, :, :, 0] *= 2

                    vid = torch.cat([torch.cat([gt_im1, gt_im2], dim=-3) / 255.0, torch.cat([im1s, im2s], dim=-3)], dim=-2)
                    vid = rearrange(vid[h:], "t h w c -> t c h w").detach().cpu().numpy()
                    vid = (vid * 255).clip(0, 255).astype(np.uint8)
                    wandb.log({"video": wandb.Video(vid, fps=20, format="mp4")})

                    heldout = next(expert_loader_eval)
                    held_front = heldout[args.front_latent_key].to(device)
                    held_wrist = heldout[args.wrist_latent_key].to(device)
                    inputs1 = held_front[:, : bl - 1]
                    inputs2 = held_wrist[:, : bl - 1]
                    states = normalize_states(
                        heldout["state"].to(device), state_min, state_max, q02=state_q02, q98=state_q98
                    )[:, : bl - 1]
                    norm_acs = normalize_acs(
                        heldout["action"].to(device), action_min, action_max, q02=action_q02, q98=action_q98
                    )
                    acs = norm_acs[:, : bl - 1]
                    future_len = sample_future_action_window(
                        action_horizon=action_horizon,
                        future_action_steps_train=future_action_steps_train,
                    )
                    t = bl - 2
                    future_actions = norm_acs[:, t + 1 : t + 1 + future_len]
                    _, _, _, pred_fail = transition(inputs1, inputs2, states, acs, future_actions)
                    target_idx = bl - 1 + future_len
                    pred_fail_target = pred_fail[:, -1].squeeze(-1)
                    target_fail = heldout["failure"][:, target_idx].to(device)
                    eval_loss = fail_loss(pred_fail_target, target_fail)
                    print(f"\rIter {i}, Eval Loss: {eval_loss.item():.4f},")

                    os.makedirs(args.checkpoint_dir, exist_ok=True)
                    torch.save(
                        {
                            "failure_head_state_dict": transition_module.failure_head.state_dict(),
                            "iteration": i,
                        },
                        os.path.join(args.checkpoint_dir, "classifier.pth"),
                    )
                    if eval_loss < best_eval:
                        best_eval = eval_loss
                        print(f"New best at iter {i}, saving model.")
                        torch.save(
                            {
                                "failure_head_state_dict": transition_module.failure_head.state_dict(),
                                "best_eval": best_eval.item() if hasattr(best_eval, "item") else best_eval,
                                "iteration": i,
                            },
                            os.path.join(args.checkpoint_dir, "best_classifier.pth"),
                        )

                    transition_module.train()
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
                    wandb.log(
                        {
                            "eval_loss": eval_loss.item(),
                            "eval/tp": tp.item(),
                            "eval/fn": fn.item(),
                            "eval/fp": fp.item(),
                            "eval/tn": tn.item(),
                            "eval/precision": precision.item(),
                            "eval/recall": recall.item(),
                            "eval/f1": f1.item(),
                            **sweep,
                        }
                    )
            if is_distributed:
                torch.distributed.barrier()

    best_eval_val = best_eval.item() if hasattr(best_eval, "item") else best_eval
    if is_rank0:
        print(f"\nTraining complete. Best eval loss: {best_eval_val:.4f}")
    if is_distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
