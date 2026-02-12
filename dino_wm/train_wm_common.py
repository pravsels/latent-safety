"""
Shared utilities for world-model training scripts.
"""

from __future__ import annotations

import os
import random
import json

import h5py
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


def resolve_wm_checkpoint(checkpoint_dir: str) -> str | None:
    """
    Pick the most appropriate WM checkpoint from a directory.

    Preference order:
    1) best_wm.pth
    2) latest_wm.pth
    3) highest wm_iter{N}.pth
    """
    try:
        best_path = os.path.join(checkpoint_dir, "best_wm.pth")
        if os.path.exists(best_path):
            return best_path

        latest_path = os.path.join(checkpoint_dir, "latest_wm.pth")
        if os.path.exists(latest_path):
            return latest_path

        if not os.path.isdir(checkpoint_dir):
            return None

        best_iter = None
        best_iter_path = None
        for name in os.listdir(checkpoint_dir):
            if not (name.startswith("wm_iter") and name.endswith(".pth")):
                continue
            mid = name[len("wm_iter") : -len(".pth")]
            if not mid.isdigit():
                continue
            it = int(mid)
            if best_iter is None or it > best_iter:
                best_iter = it
                best_iter_path = os.path.join(checkpoint_dir, name)
        return best_iter_path
    except Exception:
        return None


def global_grad_norm(parameters, norm_type: float = 2.0) -> float:
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    if norm_type == float("inf"):
        return max(g.abs().max().item() for g in grads)
    total = 0.0
    for g in grads:
        total += g.norm(norm_type).item() ** norm_type
    return total ** (1.0 / norm_type)


def global_weight_norm(parameters, norm_type: float = 2.0) -> float:
    params = [p.detach() for p in parameters]
    if not params:
        return 0.0
    if norm_type == float("inf"):
        return max(p.abs().max().item() for p in params)
    total = 0.0
    for p in params:
        total += p.norm(norm_type).item() ** norm_type
    return total ** (1.0 / norm_type)


def load_yaml_config(path: str) -> dict:
    """
    Load YAML into a plain dict.
    Uses ruamel.yaml and supports env var expansion.
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
    # rank is between 0 and WORLD_SIZE-1
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    # local_rank is between 0 and NUM_GPUS_ON_THIS_NODE-1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1
    if is_distributed:
        # When SLURM uses --gpus-per-task=1, each task only sees 1 GPU via
        # cgroup isolation (torch.cuda.device_count() == 1). LOCAL_RANK may
        # still be >0 (from SLURM_LOCALID), so clamp to visible devices.
        if torch.cuda.is_available():
            local_rank = local_rank % torch.cuda.device_count()
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


def build_wm_optimizer(transition_module: nn.Module) -> AdamW:
    """Build the standard optimizer used by DINO/WAN world-model trainers."""
    return AdamW(
        [
            # VideoTransformer and prediction heads
            {"params": transition_module.transformer.parameters(), "lr": 5e-5},
            {"params": transition_module.state_head.parameters(), "lr": 5e-5},
            {"params": transition_module.front_head.parameters(), "lr": 5e-5},
            {"params": transition_module.wrist_head.parameters(), "lr": 5e-5},
            # Action/state encoders and trajectory summary encoder
            {"params": transition_module.action_encoder.parameters(), "lr": 5e-4},
            {"params": transition_module.state_encoder.parameters(), "lr": 5e-4},
            {"params": transition_module.trajectory_encoder.parameters(), "lr": 5e-4},
            # Positional and temporal embeddings
            {"params": [transition_module.pos_embedding], "lr": 5e-4},
            {"params": [transition_module.temp_embedding], "lr": 5e-4},
        ]
    )


def freeze_failure_head_for_wm_training(transition_module: nn.Module) -> None:
    """Freeze failure head params during WM training (classifier trains it separately)."""
    for p in transition_module.failure_head.parameters():
        p.requires_grad = False


def compute_lr_factor(
    step: int,
    total_steps: int,
    warmup_steps: int,
    min_lr_factor: float,
    schedule: str,
) -> float:
    if schedule == "constant":
        return 1.0

    warmup_steps = int(max(0, warmup_steps))
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)

    denom = max(1, int(total_steps) - warmup_steps)
    t = min(1.0, max(0.0, float(step - warmup_steps) / float(denom)))
    cosine = 0.5 * (1.0 + float(torch.cos(torch.tensor(t * 3.141592653589793)).item()))
    return float(min_lr_factor) + (1.0 - float(min_lr_factor)) * cosine


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
    last_ctx_idx = int(ctx_idx[-1].item())
    future_slice = slice(last_ctx_idx + 1, last_ctx_idx + 1 + action_horizon)
    target_idx = last_ctx_idx + action_horizon + 1
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


def sample_future_action_window(
    *,
    action_horizon: int,
    future_action_steps_train: int,
    rng: random.Random | None = None,
) -> int:
    if action_horizon < 1:
        raise ValueError(f"action_horizon must be >= 1 (got {action_horizon})")
    if future_action_steps_train < 1:
        raise ValueError(
            f"future_action_steps_train must be >= 1 (got {future_action_steps_train})"
        )
    max_len = min(action_horizon, future_action_steps_train)
    rng = rng or random
    return int(rng.randint(1, max_len))


def load_stats_tensors(stats_path: str, device: str):
    """
    Load dataset stats json and convert to tensors on device.
    Returns (stats_raw, stats_tensors, state_dim, action_dim).
    """
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"Stats file '{stats_path}' not found! Please run scripts/compute_stats_json.py to generate it."
        )
    with open(stats_path, "r") as f:
        stats = json.load(f)

    required_keys = ["action_min", "action_max", "state_min", "state_max"]
    missing_keys = [k for k in required_keys if k not in stats]
    if missing_keys:
        raise ValueError(f"Stats file missing required keys: {missing_keys}")

    stats_tensors = {
        "action_min": torch.tensor(stats["action_min"]).float().to(device),
        "action_max": torch.tensor(stats["action_max"]).float().to(device),
        "state_min": torch.tensor(stats["state_min"]).float().to(device),
        "state_max": torch.tensor(stats["state_max"]).float().to(device),
        "action_q02": torch.tensor(stats["action_delta_q02"]).float().to(device) if "action_delta_q02" in stats else None,
        "action_q98": torch.tensor(stats["action_delta_q98"]).float().to(device) if "action_delta_q98" in stats else None,
        "state_q02": torch.tensor(stats["state_q02"]).float().to(device) if "state_q02" in stats else None,
        "state_q98": torch.tensor(stats["state_q98"]).float().to(device) if "state_q98" in stats else None,
    }

    state_dim = len(stats["state_min"])
    action_dim = len(stats["action_min"])
    return stats, stats_tensors, state_dim, action_dim


def build_split_datasets(
    dataset_cls,
    *,
    hdf5_file: str,
    test_frac: float,
    context_length: int,
    pred_step: int,
    action_horizon: int,
    action_key: str,
    front_latent_key: str,
    wrist_latent_key: str,
):
    """
    Build train/test/imagine split datasets and return metadata.
    """
    with h5py.File(hdf5_file, "r") as hf:
        num_traj = len(hf.keys())

    num_test = max(1, int(test_frac * num_traj))
    if num_traj - num_test < 1 and num_traj > 1:
        num_test = num_traj - 1

    _, _, _, _, _, train_raw_len = compute_action_horizon_ar_indices(
        context_length=context_length,
        pred_step=pred_step,
        action_horizon=action_horizon,
        device="cpu",
    )

    kwargs = dict(
        hdf5_file=hdf5_file,
        segment_length=train_raw_len,
        num_test=num_test,
        action_key=action_key,
        front_embd_key=front_latent_key,
        wrist_embd_key=wrist_latent_key,
    )
    expert_data = dataset_cls(split="train", **kwargs)
    expert_data_eval = dataset_cls(split="test", **kwargs)
    expert_data_imagine = dataset_cls(split="test", **kwargs)
    return {
        "expert_data": expert_data,
        "expert_data_eval": expert_data_eval,
        "expert_data_imagine": expert_data_imagine,
        "num_traj": num_traj,
        "num_test": num_test,
        "train_raw_len": train_raw_len,
    }


def run_train_eval_loop(
    *,
    args,
    start_iter: int,
    train_iter: int,
    transition,
    transition_module,
    optimizer,
    base_lrs,
    scaler,
    use_amp: bool,
    train_loader,
    train_sampler,
    expert_data,
    expert_data_eval,
    expert_data_imagine,
    H: int,
    pred_step: int,
    action_horizon: int,
    future_action_steps_train: int,
    device: str,
    is_rank0: bool,
    is_distributed: bool,
    rank: int,
    world_size: int,
    action_min: torch.Tensor,
    action_max: torch.Tensor,
    state_min: torch.Tensor,
    state_max: torch.Tensor,
    action_q02,
    action_q98,
    state_q02,
    state_q98,
    latest_ckpt_path: str,
    checkpoint_dir: str,
    best_eval: float,
    seed: int,
    normalize_acs_fn,
    normalize_states_fn,
    render_eval_images_fn=None,
):
    import wandb

    expert_loader = iter(train_loader)
    expert_loader_eval = iter(DataLoader(expert_data_eval, batch_size=args.batch_size, shuffle=True))
    expert_loader_imagine = iter(DataLoader(expert_data_imagine, batch_size=1, shuffle=True))

    def _make_ckpt_dict(iter_idx: int, best_eval_value: float) -> dict:
        return {
            "model_state_dict": transition_module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iter": int(iter_idx),
            "best_eval": float(best_eval_value),
            "seed": int(seed),
        }

    current_best_eval = float(best_eval)
    # Ensure checkpoint path exists before periodic latest checkpoint saves.
    os.makedirs(checkpoint_dir, exist_ok=True)
    for i in tqdm(range(start_iter, train_iter), desc="Training", unit="iter", disable=not is_rank0):
        lr_factor = compute_lr_factor(
            step=i,
            total_steps=train_iter,
            warmup_steps=int(args.lr_warmup_iters),
            min_lr_factor=float(args.lr_min_factor),
            schedule=str(args.lr_schedule),
        )
        for pg, base_lr in zip(optimizer.param_groups, base_lrs):
            pg["lr"] = base_lr * lr_factor

        if i > 0 and i % len(train_loader) == 0:
            if train_sampler is not None:
                train_sampler.set_epoch(i)
            train_loader, train_sampler = build_train_loader(
                expert_data,
                args.batch_size,
                is_distributed=is_distributed,
                rank=rank,
                world_size=world_size,
            )
            expert_loader = iter(train_loader)
        if i > 0 and i % len(expert_loader_eval) == 0:
            expert_loader_eval = iter(DataLoader(expert_data_eval, batch_size=args.batch_size, shuffle=True))
        if i > 0 and i % len(expert_loader_imagine) == 0:
            expert_loader_imagine = iter(DataLoader(expert_data_imagine, batch_size=1, shuffle=True))

        data = next(expert_loader)
        ctx_idx = torch.arange(H, device=device, dtype=torch.long) * pred_step
        t = int(ctx_idx[-1].item())
        future_len = sample_future_action_window(
            action_horizon=action_horizon,
            future_action_steps_train=future_action_steps_train,
        )
        future_slice = slice(t + 1, t + 1 + future_len)
        target_idx = t + future_len + 1
        ar_future_slice = slice(t + 2, t + 2 + future_len)
        ar_target_idx = t + future_len + 2
        if is_rank0 and i == start_iter:
            print(f"Context idx (pred_step={pred_step}): {ctx_idx.tolist()}")
            print(f"t (last context idx): {t}")
            print(f"future_len: {future_len}")
            print(f"Target idx: {t}+{future_len}+1 -> {target_idx}; AR target idx: {t}+{future_len}+2 -> {ar_target_idx}")

        gt_front_raw = data["cam_zed_embd"].to(device)
        input_front_embd = gt_front_raw.index_select(1, ctx_idx)
        target_front_embd = gt_front_raw[:, target_idx]

        gt_wrist_raw = data["cam_rs_embd"].to(device)
        input_wrist_embd = gt_wrist_raw.index_select(1, ctx_idx)
        target_wrist_embd = gt_wrist_raw[:, target_idx]

        gt_state_raw = data["state"].to(device)
        norm_gt_state_raw = normalize_states_fn(gt_state_raw, state_min, state_max, q02=state_q02, q98=state_q98)
        input_state = norm_gt_state_raw.index_select(1, ctx_idx)
        target_state = norm_gt_state_raw[:, target_idx]

        gt_acs_raw = data["action"].to(device)
        norm_gt_acs_raw = normalize_acs_fn(gt_acs_raw, action_min, action_max, q02=action_q02, q98=action_q98)
        input_acs = norm_gt_acs_raw.index_select(1, ctx_idx)
        future_actions = norm_gt_acs_raw[:, future_slice]

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            pred_front, pred_wrist, pred_state, _ = transition(
                input_front_embd, input_wrist_embd, input_state, input_acs, future_actions
            )
            loss_front_tf = nn.MSELoss()(pred_front[:, -1], target_front_embd)
            loss_wrist_tf = nn.MSELoss()(pred_wrist[:, -1], target_wrist_embd)
            loss_state_tf = nn.MSELoss()(pred_state[:, -1], target_state)
            loss_tf = loss_front_tf + loss_wrist_tf + loss_state_tf

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            input_front_ar = torch.cat([input_front_embd[:, 1:], pred_front[:, -1].unsqueeze(1)], dim=1)
            input_wrist_ar = torch.cat([input_wrist_embd[:, 1:], pred_wrist[:, -1].unsqueeze(1)], dim=1)
            input_state_ar = torch.cat([input_state[:, 1:], pred_state[:, -1].unsqueeze(1)], dim=1)
            ctx_idx_ar = torch.cat([ctx_idx[1:], torch.tensor([target_idx], device=device)])
            input_acs_ar = norm_gt_acs_raw.index_select(1, ctx_idx_ar)
            future_actions_ar = norm_gt_acs_raw[:, ar_future_slice]
            pred_front_ar, pred_wrist_ar, pred_state_ar, _ = transition(
                input_front_ar, input_wrist_ar, input_state_ar, input_acs_ar, future_actions_ar
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
        scaler.unscale_(optimizer)
        grad_norm = global_grad_norm(transition_module.parameters())
        scaler.step(optimizer)
        scaler.update()
        weight_norm = global_weight_norm(transition_module.parameters())

        if is_rank0:
            print(
                f"\rIter {i} | lr {optimizer.param_groups[0]['lr']:.2e} | TF {loss_tf:.4f} | AR {loss_ar:.4f} | grad {grad_norm:.2f} | weight {weight_norm:.2f}",
                end="",
                flush=True,
            )
        if is_rank0:
            wandb.log(
                {
                    "train_loss": loss_tf,
                    "train_loss_ar": loss_ar,
                    "grad_norm": grad_norm,
                    "weight_norm": weight_norm,
                    "lr": optimizer.param_groups[0]["lr"],
                    "lr_factor": lr_factor,
                }
            )

        if is_rank0 and args.save_every and (i % args.save_every == 0):
            torch.save(_make_ckpt_dict(i, current_best_eval), latest_ckpt_path)

        if i % args.eval_interval == 0:
            # Keep all DDP ranks in lock-step while rank0 performs eval/checkpoint I/O.
            if is_distributed:
                torch.distributed.barrier()

            if is_rank0:
                transition.eval()
                avg_metrics = {"eval_loss": 0.0, "front_loss": 0.0, "wrist_loss": 0.0, "state_loss": 0.0}
                num_samples = args.eval_samples
                sample_images = None
                print(f"\nRunning evaluation on {num_samples} samples...")

                for _ in range(num_samples):
                    eval_data = next(expert_loader_imagine)
                    gt_front_embd_eval = eval_data["cam_zed_embd"].to(device)
                    ctx_idx = torch.arange(H, device=device, dtype=torch.long) * pred_step
                    t = int(ctx_idx[-1].item())
                    future_len = sample_future_action_window(
                        action_horizon=action_horizon,
                        future_action_steps_train=future_action_steps_train,
                    )
                    target_idx = t + future_len + 1
                    input_front_embd_eval = gt_front_embd_eval.index_select(1, ctx_idx)
                    gt_wrist_embd_eval = eval_data["cam_rs_embd"].to(device)
                    input_wrist_embd_eval = gt_wrist_embd_eval.index_select(1, ctx_idx)
                    all_acs = eval_data["action"][[0]].to(device)
                    all_acs = normalize_acs_fn(all_acs, action_min, action_max, q02=action_q02, q98=action_q98)
                    acs = all_acs.index_select(1, ctx_idx)
                    gt_states_eval = eval_data["state"][[0]].to(device)
                    input_states_eval = normalize_states_fn(
                        gt_states_eval, state_min, state_max, q02=state_q02, q98=state_q98
                    ).index_select(1, ctx_idx)
                    future_actions = all_acs[:, t + 1 : t + 1 + future_len]
                    pred_front, pred_wrist, pred_state, _ = transition(
                        input_front_embd_eval, input_wrist_embd_eval, input_states_eval, acs, future_actions
                    )

                    target_front = gt_front_embd_eval[[0], target_idx]
                    target_wrist = gt_wrist_embd_eval[[0], target_idx]
                    target_state = normalize_states_fn(
                        gt_states_eval[[0], target_idx], state_min, state_max, q02=state_q02, q98=state_q98
                    )
                    l_front = nn.MSELoss()(pred_front[:, -1], target_front).item()
                    l_wrist = nn.MSELoss()(pred_wrist[:, -1], target_wrist).item()
                    l_state = nn.MSELoss()(pred_state[:, -1], target_state).item()
                    avg_metrics["eval_loss"] += (l_front + l_wrist + l_state)
                    avg_metrics["front_loss"] += l_front
                    avg_metrics["wrist_loss"] += l_wrist
                    avg_metrics["state_loss"] += l_state

                    if sample_images is None and render_eval_images_fn is not None:
                        sample_images = render_eval_images_fn(
                            pred_front=pred_front,
                            pred_wrist=pred_wrist,
                            eval_data=eval_data,
                            target_idx=target_idx,
                            device=device,
                        )

                for k in avg_metrics:
                    avg_metrics[k] /= num_samples

                print(
                    f"\rIter {i}, Eval Loss: {avg_metrics['eval_loss']:.4f}, front: {avg_metrics['front_loss']:.4f}, wrist: {avg_metrics['wrist_loss']:.4f}, state: {avg_metrics['state_loss']:.4f}"
                )
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(_make_ckpt_dict(i, current_best_eval), os.path.join(checkpoint_dir, f"wm_iter{i}.pth"))
                if avg_metrics["eval_loss"] < current_best_eval:
                    current_best_eval = avg_metrics["eval_loss"]
                    torch.save(_make_ckpt_dict(i, current_best_eval), os.path.join(checkpoint_dir, "best_wm.pth"))

                transition.train()
                log_dict = {
                    "eval_loss": avg_metrics["eval_loss"],
                    "front_loss": avg_metrics["front_loss"],
                    "wrist_loss": avg_metrics["wrist_loss"],
                    "state_loss": avg_metrics["state_loss"],
                }
                if sample_images is not None:
                    log_dict.update(
                        {
                            "pred_front": wandb.Image(sample_images["pred_front"]),
                            "pred_wrist": wandb.Image(sample_images["pred_wrist"]),
                            "front": wandb.Image(sample_images["front"]),
                            "wrist": wandb.Image(sample_images["wrist"]),
                        }
                    )
                wandb.log(log_dict)

            if is_distributed:
                torch.distributed.barrier()

    return current_best_eval
