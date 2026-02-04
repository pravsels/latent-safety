#!/usr/bin/env python3
"""
Benchmark realtime feasibility of the latent safety score pipeline (end-to-end).

This benchmark intentionally DOES NOT use precomputed embeddings from the .h5.
It reads raw camera frames and runs DINOv2 + WorldModel(+failure_head) online.

Pipeline per step:
  1) Read front + wrist frames from HDF5 (camera_1, camera_0)
  2) Preprocess each view (matches scripts/lerobot_to_hdf5.py)
  3) Run ONE batched DINO forward_features (B=2) -> patch tokens for both views
  4) Maintain rolling window of last num_frames embeddings + states + actions
  5) Run ONE WM forward (B=1) to produce failure score

Outputs timing breakdown (ms) and implied Hz.

Example:
  python scripts/benchmark_latent_safety_realtime.py \
    --hdf5-file cubes_push_eval_10.h5 \
    --trajectory trajectory_0 \
    --start-frame 0 \
    --steps 200 \
    --dataset-stats dataset_stats.json \
    --wm-checkpoint dino_wm_checkpoints/best_wm.pth \
    --failure-checkpoint dino_wm_checkpoints/best_classifier.pth \
    --device cuda:0 \
    --amp
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch

# Repo root for imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from dino_wm.config import MODEL_CONFIG  # noqa: E402
from dino_wm.dino_models import VideoTransformer, normalize_acs, normalize_states  # noqa: E402
from scripts.utils import preprocess_images_for_dino  # noqa: E402


def load_state_dict_maybe_wrapped(path: str, device: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError(f"Unrecognized checkpoint format at {path}")


def percentile(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    arr = np.asarray(xs, dtype=np.float64)
    return float(np.percentile(arr, p))


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark realtime latent safety score (DINO + WM + failure head)")
    parser.add_argument("--hdf5-file", type=str, required=True)
    parser.add_argument(
        "--trajectory",
        type=str,
        default=None,
        help="Single trajectory group name (e.g. trajectory_0). If omitted, uses first N trajectories.",
    )
    parser.add_argument(
        "--trajectories",
        type=str,
        default=None,
        help="Comma-separated list of trajectory group names (overrides --trajectory/--max-trajectories).",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=1,
        help="If --trajectory/--trajectories not provided, benchmark the first N trajectories (default: 1).",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200, help="How many steps to benchmark (after warmup window fills)")
    parser.add_argument("--warmup-iters", type=int, default=50, help="Warmup iterations (timings not counted)")
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=4,
        help="Training sequence length (default 4) -> context_len = sequence_length - 1",
    )
    parser.add_argument("--dataset-stats", type=str, required=True)
    parser.add_argument("--wm-checkpoint", type=str, required=True)
    parser.add_argument("--failure-checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--amp", action="store_true", help="Enable fp16 autocast (recommended on CUDA)")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable autocast")
    parser.set_defaults(amp=True)
    args = parser.parse_args()

    device = args.device
    if "cuda" in device and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    # Rolling context window length used by the WM per inference step
    context_len = args.sequence_length - 1

    # Load stats
    with open(args.dataset_stats, "r") as f:
        stats_data = json.load(f)
    stats = {
        "action_min": torch.tensor(stats_data["action_min"]).float().to(device),
        "action_max": torch.tensor(stats_data["action_max"]).float().to(device),
        "state_min": torch.tensor(stats_data["state_min"]).float().to(device),
        "state_max": torch.tensor(stats_data["state_max"]).float().to(device),
    }
    if "action_delta_q02" in stats_data:
        stats["action_delta_q02"] = torch.tensor(stats_data["action_delta_q02"]).float().to(device)
    if "action_delta_q98" in stats_data:
        stats["action_delta_q98"] = torch.tensor(stats_data["action_delta_q98"]).float().to(device)
    if "state_q02" in stats_data:
        stats["state_q02"] = torch.tensor(stats_data["state_q02"]).float().to(device)
    if "state_q98" in stats_data:
        stats["state_q98"] = torch.tensor(stats_data["state_q98"]).float().to(device)
    state_dim = len(stats_data["state_min"])
    action_dim = len(stats_data["action_min"])

    # Load WM (includes DINO inside)
    transition = VideoTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        num_frames=context_len,
        **MODEL_CONFIG,
        device=device,
    ).to(device)

    wm_sd = load_state_dict_maybe_wrapped(args.wm_checkpoint, device)
    transition.load_state_dict(wm_sd, strict=True)

    cls_sd = load_state_dict_maybe_wrapped(args.failure_checkpoint, device)
    failure_only = {k: v for k, v in cls_sd.items() if k.startswith("failure_head.")}
    transition.load_state_dict(failure_only, strict=False)
    transition.eval()

    # Open dataset
    with h5py.File(args.hdf5_file, "r") as hf:
        traj_keys = sorted([k for k in hf.keys() if k.startswith("trajectory_")])
        if not traj_keys:
            raise ValueError(f"No trajectory_* groups found in {args.hdf5_file}")

        if args.trajectories:
            traj_names = [t.strip() for t in args.trajectories.split(",") if t.strip()]
        elif args.trajectory:
            traj_names = [args.trajectory]
        else:
            n = max(1, int(args.max_trajectories))
            traj_names = traj_keys[:n]

        # Validate and benchmark each trajectory
        per_traj = []

        def now_ms() -> float:
            import time
            return time.perf_counter() * 1000.0

        use_cuda_timing = "cuda" in device
        if use_cuda_timing:
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev2 = torch.cuda.Event(enable_timing=True)
            ev3 = torch.cuda.Event(enable_timing=True)
            ev4 = torch.cuda.Event(enable_timing=True)

        for traj_name in traj_names:
            if traj_name not in hf:
                raise ValueError(
                    f"Trajectory {traj_name} not found. Available: {traj_keys[:5]}{'...' if len(traj_keys) > 5 else ''}"
                )

            grp = hf[traj_name]
            required = ["camera_0", "camera_1", "actions", "states"]
            missing = [k for k in required if k not in grp]
            if missing:
                raise ValueError(f"Trajectory {traj_name} missing keys: {missing}")

            wrist = grp["camera_0"]
            front = grp["camera_1"]
            actions = grp["actions"]
            states = grp["states"]

            T = min(len(wrist), len(front), len(actions), len(states))
            start = int(args.start_frame)

            needed = start + args.warmup_iters + args.steps + context_len
            if needed >= T:
                raise ValueError(f"Not enough frames in {traj_name}: have {T}, need at least {needed+1}")

            # Rolling buffers (torch tensors on device)
            zed_buf = None
            rs_buf = None
            st_buf = None
            ac_buf = None

            # Timings (ms)
            t_dino: List[float] = []
            t_wm: List[float] = []
            t_total: List[float] = []

            total_iters = args.warmup_iters + args.steps + context_len
            for it in range(total_iters):
                t_idx = start + it

                wrist_u8 = wrist[t_idx]
                front_u8 = front[t_idx]

                wrist_t = (
                    torch.from_numpy(wrist_u8)
                    .to(device=device, dtype=torch.float32)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    / 255.0
                )
                front_t = (
                    torch.from_numpy(front_u8)
                    .to(device=device, dtype=torch.float32)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    / 255.0
                )

                wrist_prep = preprocess_images_for_dino(wrist_t, is_front_camera=False)
                front_prep = preprocess_images_for_dino(front_t, is_front_camera=True)
                dino_in = torch.cat([wrist_prep, front_prep], dim=0)

                ac_np = np.asarray(actions[t_idx], dtype=np.float32)
                st_np = np.asarray(states[t_idx], dtype=np.float32)
                ac_t = torch.from_numpy(ac_np).to(device=device, dtype=torch.float32)
                st_t = torch.from_numpy(st_np).to(device=device, dtype=torch.float32)

                if use_cuda_timing:
                    torch.cuda.synchronize()
                    ev0.record()
                else:
                    t0 = now_ms()

                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(args.amp and use_cuda_timing)):
                    feats = transition.dino.forward_features(dino_in)["x_norm_patchtokens"]

                if use_cuda_timing:
                    ev1.record()
                else:
                    t1 = now_ms()

                rs_emb = feats[0]
                zed_emb = feats[1]

                if zed_buf is None:
                    zed_buf = zed_emb.unsqueeze(0)
                    rs_buf = rs_emb.unsqueeze(0)
                    st_buf = st_t.unsqueeze(0)
                    ac_buf = ac_t.unsqueeze(0)
                else:
                    zed_buf = torch.cat([zed_buf, zed_emb.unsqueeze(0)], dim=0)
                    rs_buf = torch.cat([rs_buf, rs_emb.unsqueeze(0)], dim=0)
                    st_buf = torch.cat([st_buf, st_t.unsqueeze(0)], dim=0)
                    ac_buf = torch.cat([ac_buf, ac_t.unsqueeze(0)], dim=0)

                if zed_buf.shape[0] > context_len:
                    zed_buf = zed_buf[-context_len:]
                    rs_buf = rs_buf[-context_len:]
                    st_buf = st_buf[-context_len:]
                    ac_buf = ac_buf[-context_len:]

                if zed_buf.shape[0] < context_len:
                    if use_cuda_timing:
                        torch.cuda.synchronize()
                    continue

                zed_in = zed_buf.unsqueeze(0)
                rs_in = rs_buf.unsqueeze(0)
                st_in = st_buf.unsqueeze(0)
                ac_in = ac_buf.unsqueeze(0)

                st_in = normalize_states(
                    st_in,
                    stats["state_min"],
                    stats["state_max"],
                    q02=stats.get("state_q02"),
                    q98=stats.get("state_q98"),
                )
                ac_in = normalize_acs(
                    ac_in,
                    stats["action_min"],
                    stats["action_max"],
                    q02=stats.get("action_delta_q02"),
                    q98=stats.get("action_delta_q98"),
                )

                if use_cuda_timing:
                    ev2.record()
                else:
                    t2 = now_ms()

                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(args.amp and use_cuda_timing)):
                    _, _, _, pred_fail = transition(zed_in, rs_in, st_in, ac_in)
                    _ = pred_fail[:, -1, 0]

                if use_cuda_timing:
                    ev3.record()
                    ev4.record()
                    torch.cuda.synchronize()
                    dino_ms = ev0.elapsed_time(ev1)
                    wm_ms = ev2.elapsed_time(ev3)
                    total_ms = ev0.elapsed_time(ev4)
                else:
                    t3 = now_ms()
                    dino_ms = t1 - t0
                    wm_ms = t3 - t2
                    total_ms = t3 - t0

                if it >= args.warmup_iters:
                    t_dino.append(float(dino_ms))
                    t_wm.append(float(wm_ms))
                    t_total.append(float(total_ms))

            per_traj.append(
                {
                    "trajectory": traj_name,
                    "counted_steps": len(t_total),
                    "dino_ms": t_dino,
                    "wm_ms": t_wm,
                    "total_ms": t_total,
                }
            )

    # Report
    def summarize(name: str, xs: List[float]) -> str:
        return (
            f"{name}: mean={np.mean(xs):.2f}ms  "
            f"p50={percentile(xs, 50):.2f}  p90={percentile(xs, 90):.2f}  p99={percentile(xs, 99):.2f}"
        )

    print(f"Device: {device}  AMP: {args.amp}  context_len: {context_len}")
    print(f"warmup-iters={args.warmup_iters}  steps={args.steps}  start-frame={args.start_frame}")

    overall_means = []
    for r in per_traj:
        traj_name = r["trajectory"]
        t_dino = r["dino_ms"]
        t_wm = r["wm_ms"]
        t_total = r["total_ms"]

        print(f"\nTrajectory: {traj_name}")
        print(f"Counted steps: {len(t_total)}")
        print(summarize("DINO(batched 2 cams)", t_dino))
        print(summarize("WM+failure", t_wm))
        print(summarize("Total", t_total))
        hz = 1000.0 / float(np.mean(t_total)) if t_total else float("nan")
        print(f"Implied throughput: {hz:.2f} Hz")
        overall_means.append(float(np.mean(t_total)) if t_total else float("nan"))

    if overall_means:
        mean_over_traj = float(np.nanmean(np.asarray(overall_means, dtype=np.float64)))
        hz_over_traj = 1000.0 / mean_over_traj if mean_over_traj > 0 else float("nan")
        print("\n=== Mean over trajectories ===")
        print(f"Total mean (avg of per-traj means): {mean_over_traj:.2f}ms")
        print(f"Implied throughput: {hz_over_traj:.2f} Hz")


if __name__ == "__main__":
    main()


