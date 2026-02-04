#!/usr/bin/env python3
"""
Phase-1 evaluation: Teacher-forced failure head on recorded trajectories (GT roll-through).

What this does:
- Loads a DINO World Model (WM) checkpoint
- Loads ONLY the trained `failure_head.*` weights from a classifier checkpoint
- Runs teacher-forced sliding windows over each trajectory (no imagination / no decoder)
- Overlays a colored indicator + numeric score on the ground-truth video

Score alignment (recommended):
- If the model uses `num_frames = sequence_length - 1` (typically 3),
  then for each window starting at s with frames [s, s+1, s+2] we use
  `pred_fail[:, -1]` as the score for frame (s+3).
- Therefore scores appear starting at frame index == num_frames (e.g. frame 3, the 4th frame).

Example:
  python scripts/dino_wm_failure_tf_eval.py \
    --hdf5-file my_recording.h5 \
    --dataset-stats dataset_stats.json \
    --wm-checkpoint dino_wm_checkpoints/best_wm.pth \
    --failure-checkpoint dino_wm_checkpoints/best_classifier.pth \
    --output-dir failure_head_teacher_forced_videos \
    --indicator-size 64
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import cv2
import imageio

# Add repo root to path so `dino_wm.*` imports work when running from anywhere
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from dino_wm.dino_models import VideoTransformer, normalize_acs, normalize_states  # noqa: E402
from dino_wm.config import MODEL_CONFIG  # noqa: E402


def load_state_dict_maybe_wrapped(path: str, device: str) -> Dict[str, torch.Tensor]:
    """
    Supports both:
    - raw state_dict
    - dict with `model_state_dict` (our newer "best_*.pth" format)
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict):
        # Could still be a state_dict (OrderedDict-like)
        return ckpt
    raise ValueError(f"Unrecognized checkpoint format at {path}")


def list_trajectory_keys(h5: h5py.File) -> List[str]:
    keys = [k for k in h5.keys() if k.startswith("trajectory_")]
    # Sort by numeric suffix when possible
    def key_fn(name: str) -> Tuple[int, str]:
        try:
            return (int(name.split("_", 1)[1]), name)
        except Exception:
            return (10**18, name)
    return sorted(keys, key=key_fn)


def score_to_color(score: float, red_thresh: float, green_thresh: float) -> Tuple[int, int, int]:
    """
    Returns BGR tuple for OpenCV drawing.
    - red   if score <= red_thresh
    - green if score >= green_thresh
    - yellow otherwise (weak/uncertain band)
    """
    if score <= red_thresh:
        return (0, 0, 255)      # red (BGR)
    if score >= green_thresh:
        return (0, 200, 0)      # green (BGR)
    return (0, 255, 255)        # yellow (BGR)


def compute_failure_scores_teacher_forced(
    transition: VideoTransformer,
    cam_zed_embd: np.ndarray,  # (T, P, D)
    cam_rs_embd: np.ndarray,   # (T, P, D)
    states: np.ndarray,        # (T, S)
    actions: np.ndarray,       # (T, A)
    *,
    stats: Dict[str, torch.Tensor],
    device: str,
    num_frames: int,
    batch_windows: int,
) -> List[Optional[float]]:
    """
    Compute per-frame failure scores using teacher-forced sliding windows.

    Alignment:
      window s uses frames [s .. s+num_frames-1] as inputs
      score is assigned to frame index (s + num_frames)

    Returns a list length T with None for frames that do not have an assigned score.
    """
    T = int(cam_zed_embd.shape[0])
    scores: List[Optional[float]] = [None] * T

    if T < (num_frames + 1):
        return scores

    # Pre-pack min/max on device
    action_min = stats["action_min"].to(device)
    action_max = stats["action_max"].to(device)
    state_min = stats["state_min"].to(device)
    state_max = stats["state_max"].to(device)
    action_q02 = stats["action_delta_q02"].to(device) if "action_delta_q02" in stats else None
    action_q98 = stats["action_delta_q98"].to(device) if "action_delta_q98" in stats else None
    state_q02 = stats["state_q02"].to(device) if "state_q02" in stats else None
    state_q98 = stats["state_q98"].to(device) if "state_q98" in stats else None

    # Sliding windows start indices
    starts = np.arange(0, T - (num_frames + 1) + 1, dtype=np.int64)  # inclusive

    transition.eval()
    with torch.no_grad():
        for i in range(0, len(starts), batch_windows):
            s_batch = starts[i : i + batch_windows]

            # Build batch arrays: (B, num_frames, ...)
            zed = np.stack([cam_zed_embd[s : s + num_frames] for s in s_batch], axis=0)
            rs = np.stack([cam_rs_embd[s : s + num_frames] for s in s_batch], axis=0)
            st = np.stack([states[s : s + num_frames] for s in s_batch], axis=0)
            ac = np.stack([actions[s : s + num_frames] for s in s_batch], axis=0)

            zed_t = torch.from_numpy(zed).to(device=device, dtype=torch.float32)
            rs_t = torch.from_numpy(rs).to(device=device, dtype=torch.float32)
            st_t = torch.from_numpy(st).to(device=device, dtype=torch.float32)
            ac_t = torch.from_numpy(ac).to(device=device, dtype=torch.float32)

            st_t = normalize_states(
                st_t, state_min, state_max, q02=state_q02, q98=state_q98
            )
            ac_t = normalize_acs(
                ac_t, action_min, action_max, q02=action_q02, q98=action_q98
            )

            _, _, _, pred_fail = transition(zed_t, rs_t, st_t, ac_t)  # (B, num_frames, 1)
            step_scores = pred_fail[:, -1, 0].detach().cpu().numpy().astype(np.float64)

            for s, sc in zip(s_batch.tolist(), step_scores.tolist()):
                tgt_frame = s + num_frames
                if 0 <= tgt_frame < T:
                    scores[tgt_frame] = float(sc)

    return scores


def write_trajectory_video_with_overlay(
    *,
    out_path: str,
    wrist_rgb: np.ndarray,  # (T, H, W, 3) uint8
    front_rgb: np.ndarray,  # (T, H, W, 3) uint8
    scores: List[Optional[float]],
    fps: int,
    red_thresh: float,
    green_thresh: float,
    warmup_frames: int,
    frame_stride: int,
    indicator_size: int,
    indicator_margin: int,
    indicator_border: int,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    T = min(len(scores), wrist_rgb.shape[0], front_rgb.shape[0])

    # Video writer expects RGB frames
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p")
    try:
        for t in range(0, T, frame_stride):
            # Build side-by-side GT view (front | wrist)
            front = front_rgb[t]
            wrist = wrist_rgb[t]
            frame_rgb = np.concatenate([front, wrist], axis=1)  # (H, 2W, 3) RGB

            # Convert to BGR for OpenCV drawing
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Determine bar height based on indicator size so the square "pops" (no clipping)
            h, w = frame_bgr.shape[:2]
            m = int(indicator_margin)
            s = int(indicator_size)
            b = int(indicator_border)
            bar_h = max(40, (2 * m) + s + 4)
            cv2.rectangle(frame_bgr, (0, 0), (w, bar_h), (0, 0, 0), -1)

            # Text + indicator
            if t < warmup_frames or scores[t] is None:
                label = f"t={t:04d}  latent safety score=warmup"
                color = (200, 200, 200)
                indicator = (128, 128, 128)
            else:
                sc = float(scores[t])
                color = (255, 255, 255)
                indicator = score_to_color(sc, red_thresh=red_thresh, green_thresh=green_thresh)
                label = f"t={t:04d}  latent safety score={sc:+.3f}"

            # Indicator box (left) - make it pop
            x1, y1 = m, m
            x2, y2 = m + s, m + s
            # Ensure box stays inside the top bar
            y2 = min(y2, bar_h - m)
            x2 = min(x2, w - m)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), indicator, -1)
            if b > 0:
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 255, 255), b)

            # Main label
            text_x = x2 + max(10, m)
            text_y = int(min(bar_h - 12, max(24, round(0.65 * bar_h))))
            cv2.putText(frame_bgr, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

            # Back to RGB for writing
            out_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            writer.append_data(out_rgb)
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher-forced failure head overlay on GT trajectories")
    parser.add_argument("--hdf5-file", type=str, required=True, help="Path to consolidated HDF5 file")
    parser.add_argument("--dataset-stats", type=str, required=True, help="Path to dataset_stats.json")
    parser.add_argument("--wm-checkpoint", type=str, required=True, help="Path to world model checkpoint (best_wm.pth)")
    parser.add_argument("--failure-checkpoint", type=str, required=True, help="Path to classifier checkpoint (best_classifier.pth or classifier.pth)")
    parser.add_argument("--output-dir", type=str, default="rollout_videos_failure_tf", help="Directory to save videos")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sequence-length", type=int, default=4, help="Training sequence length (default: 4) -> num_frames=sequence_length-1")
    parser.add_argument("--fps", type=int, default=30, help="Output video fps (visualization only)")
    parser.add_argument("--frame-stride", type=int, default=1, help="Write every Nth frame to video (default: 1)")
    parser.add_argument("--indicator-size", type=int, default=64, help="Size (px) of the colored indicator square (default: 64)")
    parser.add_argument("--indicator-margin", type=int, default=8, help="Margin (px) from top-left for indicator square (default: 8)")
    parser.add_argument("--indicator-border", type=int, default=3, help="Border thickness (px) around indicator square (default: 3)")
    parser.add_argument("--max-trajectories", type=int, default=None, help="Optional limit on number of trajectories to process")
    parser.add_argument("--batch-windows", type=int, default=256, help="How many sliding windows to score per model forward pass")
    parser.add_argument("--green-thresh", type=float, default=0.75)
    parser.add_argument("--red-thresh", type=float, default=-0.75)
    parser.add_argument("--save-scores-json", action="store_true", help="Also save per-frame scores as JSON next to each video")
    args = parser.parse_args()

    device = args.device
    num_frames = args.sequence_length - 1
    warmup_frames = num_frames  # scores start at frame index == num_frames (e.g. 3 -> 4th frame)

    # Load stats
    if not os.path.exists(args.dataset_stats):
        raise FileNotFoundError(f"Stats file not found: {args.dataset_stats}")
    with open(args.dataset_stats, "r") as f:
        stats_data = json.load(f)
    stats = {
        "action_min": torch.tensor(stats_data["action_min"]).float(),
        "action_max": torch.tensor(stats_data["action_max"]).float(),
        "state_min": torch.tensor(stats_data["state_min"]).float(),
        "state_max": torch.tensor(stats_data["state_max"]).float(),
    }
    if "action_delta_q02" in stats_data:
        stats["action_delta_q02"] = torch.tensor(stats_data["action_delta_q02"]).float()
    if "action_delta_q98" in stats_data:
        stats["action_delta_q98"] = torch.tensor(stats_data["action_delta_q98"]).float()
    if "state_q02" in stats_data:
        stats["state_q02"] = torch.tensor(stats_data["state_q02"]).float()
    if "state_q98" in stats_data:
        stats["state_q98"] = torch.tensor(stats_data["state_q98"]).float()

    # Infer dims
    state_dim = len(stats_data["state_min"])
    action_dim = len(stats_data["action_min"])

    # Load model
    transition = VideoTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        num_frames=num_frames,
        **MODEL_CONFIG,
    ).to(device)

    wm_sd = load_state_dict_maybe_wrapped(args.wm_checkpoint, device)
    transition.load_state_dict(wm_sd, strict=True)

    cls_sd = load_state_dict_maybe_wrapped(args.failure_checkpoint, device)
    failure_only = {k: v for k, v in cls_sd.items() if k.startswith("failure_head.")}
    transition.load_state_dict(failure_only, strict=False)
    transition.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    with h5py.File(args.hdf5_file, "r") as hf:
        traj_keys = list_trajectory_keys(hf)
        if args.max_trajectories is not None:
            traj_keys = traj_keys[: max(0, int(args.max_trajectories))]

        if not traj_keys:
            raise ValueError(f"No trajectory_* groups found in {args.hdf5_file}")

        for traj_name in traj_keys:
            grp = hf[traj_name]

            required = ["camera_0", "camera_1", "cam_rs_embd", "cam_zed_embd", "actions", "states"]
            missing = [k for k in required if k not in grp]
            if missing:
                print(f"Skipping {traj_name} (missing keys: {missing})")
                continue

            # Load arrays
            wrist_rgb = grp["camera_0"][:]  # wrist
            front_rgb = grp["camera_1"][:]  # front
            cam_rs_embd = grp["cam_rs_embd"][:]
            cam_zed_embd = grp["cam_zed_embd"][:]
            actions = grp["actions"][:]
            states = grp["states"][:]

            # Basic length checks / trimming
            T = min(
                wrist_rgb.shape[0],
                front_rgb.shape[0],
                cam_rs_embd.shape[0],
                cam_zed_embd.shape[0],
                actions.shape[0],
                states.shape[0],
            )
            wrist_rgb = wrist_rgb[:T]
            front_rgb = front_rgb[:T]
            cam_rs_embd = cam_rs_embd[:T]
            cam_zed_embd = cam_zed_embd[:T]
            actions = actions[:T]
            states = states[:T]

            scores = compute_failure_scores_teacher_forced(
                transition=transition,
                cam_zed_embd=cam_zed_embd,
                cam_rs_embd=cam_rs_embd,
                states=states,
                actions=actions,
                stats=stats,
                device=device,
                num_frames=num_frames,
                batch_windows=args.batch_windows,
            )

            out_mp4 = os.path.join(args.output_dir, f"{traj_name}_tf_fail.mp4")
            write_trajectory_video_with_overlay(
                out_path=out_mp4,
                wrist_rgb=wrist_rgb,
                front_rgb=front_rgb,
                scores=scores,
                fps=args.fps,
                red_thresh=args.red_thresh,
                green_thresh=args.green_thresh,
                warmup_frames=warmup_frames,
                frame_stride=max(1, int(args.frame_stride)),
                indicator_size=max(10, int(args.indicator_size)),
                indicator_margin=max(0, int(args.indicator_margin)),
                indicator_border=max(0, int(args.indicator_border)),
            )
            print(f"Saved: {out_mp4}")

            if args.save_scores_json:
                out_json = os.path.join(args.output_dir, f"{traj_name}_tf_fail_scores.json")
                payload = {
                    "trajectory": traj_name,
                    "sequence_length": args.sequence_length,
                    "num_frames": num_frames,
                    "warmup_frames": warmup_frames,
                    "green_thresh": args.green_thresh,
                    "red_thresh": args.red_thresh,
                    "scores": scores,
                }
                with open(out_json, "w") as f:
                    json.dump(payload, f, indent=2)
                print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()


