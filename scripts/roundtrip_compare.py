#!/usr/bin/env python3
"""
Round-trip video frames through a DINO encoder + VQVAE decoder,
then render a side-by-side comparison video + PSNR stats.

Supports LeRobot v3 datasets (episode/camera selection) and local video files.

Examples:
  # List episodes and cameras in a dataset
  python scripts/roundtrip_compare.py \
    --hf_repo villekuosmanen/eval_pickasinglecoffeecapsulefromthecardboardtrayanddro_455ac6c3 \
    --list

  # Run DINO roundtrip on episode 0, first camera
  python scripts/roundtrip_compare.py \
    --hf_repo villekuosmanen/eval_pickasinglecoffeecapsulefromthecardboardtrayanddro_455ac6c3 \
    --episode 0 --camera observation.images.front \
    --decoder_checkpoint ./dino3_decoder_checkpoints/best_decoder.pth \
    --out_dir ./outputs/roundtrip_dino

  # Run on a local video file
  python scripts/roundtrip_compare.py \
    --video ./input.mp4 \
    --decoder_checkpoint ./dino3_decoder_checkpoints/best_decoder.pth

  # Use DINOv2 instead of default v3
  python scripts/roundtrip_compare.py \
    --video ./input.mp4 \
    --decoder_checkpoint ./dino2_decoder_checkpoints/best_decoder.pth \
    --dino_version v2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Add parent directory so imports work when run as a script
_parent = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from dino_wm.config import get_decoder_image_size
from dino_wm.dino_decoder import VQVAE
from scripts.utils import get_dino_model, preprocess_images_for_dino


# -- frame sources ------------------------------------------------------------

def read_video_frames(
    path: Path, max_frames: int,
) -> tuple[list[np.ndarray], float | None]:
    """Read frames from a local video file. Returns (frames, fps)."""
    import imageio.v2 as imageio

    try:
        reader = imageio.get_reader(str(path), format="ffmpeg")
    except Exception:
        reader = imageio.get_reader(str(path))

    fps: float | None = None
    meta = reader.get_meta_data() if hasattr(reader, "get_meta_data") else {}
    if isinstance(meta, dict):
        try:
            fps = float(meta["fps"])
        except (KeyError, TypeError, ValueError):
            pass

    frames: list[np.ndarray] = []
    try:
        for i, frame in enumerate(reader):
            if 0 < max_frames <= i:
                break
            if frame.ndim == 2:
                frame = np.repeat(frame[..., None], 3, axis=-1)
            frame = frame[..., :3].astype(np.uint8)
            frames.append(frame)
    finally:
        reader.close()

    if not frames:
        raise RuntimeError(f"No frames read from {path}")
    return frames, fps


def lerobot_list(repo_id: str) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id)
    print(f"Dataset: {repo_id}")
    print(f"  Episodes: {meta.total_episodes}")
    print(f"  Frames:   {meta.total_frames}")
    print(f"  FPS:      {meta.fps}")
    print(f"  Cameras:  {meta.camera_keys}")
    if meta.total_episodes > 0:
        print(f"  Avg frames/episode: {meta.total_frames / meta.total_episodes:.0f}")
    print(f"\nUse --episode N --camera KEY to select.")


def lerobot_load_frames(
    repo_id: str, episode: int, camera: str | None, max_frames: int,
    video_backend: str = "pyav",
) -> tuple[list[np.ndarray], float]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id, episodes=[episode], video_backend=video_backend)
    cam = camera or dataset.meta.camera_keys[0]
    if cam not in dataset.meta.camera_keys:
        raise ValueError(
            f"Camera '{cam}' not found. Available: {dataset.meta.camera_keys}"
        )

    n = min(len(dataset), max_frames) if max_frames > 0 else len(dataset)
    frames: list[np.ndarray] = []
    for idx in range(n):
        t = dataset[idx][cam]  # (C, H, W) float32 [0, 1]
        frame = (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        frames.append(frame)

    if not frames:
        raise RuntimeError(f"No frames for episode {episode} camera {cam}")
    return frames, float(dataset.meta.fps)


# -- tensor helpers -----------------------------------------------------------

def frames_to_tensor(
    frames: list[np.ndarray], device: torch.device, target_size: tuple[int, int],
) -> torch.Tensor:
    """List of HWC uint8 -> (B, 3, H, W) float32 in [0, 1], resized to *target_size*."""
    t = torch.from_numpy(np.stack(frames)).to(device=device, dtype=torch.float32)
    t = t.permute(0, 3, 1, 2).div(255.0)  # BCHW [0, 1]
    t = F.interpolate(t, size=target_size, mode="bilinear", align_corners=False)
    return t


def tensor_to_frames(video: torch.Tensor) -> list[np.ndarray]:
    """(B, 3, H, W) float [0, 1] -> list of HWC uint8."""
    v = video.detach().cpu().clamp(0, 1).mul(255).round().to(torch.uint8)
    v = v.permute(0, 2, 3, 1).contiguous()
    return [v[i].numpy() for i in range(v.shape[0])]


# -- DINO roundtrip -----------------------------------------------------------

@torch.no_grad()
def dino_roundtrip(
    frames_tensor: torch.Tensor,
    dino_model: torch.nn.Module,
    decoder: VQVAE,
    is_front_camera: bool,
) -> torch.Tensor:
    """
    Encode frames with DINO, decode with VQVAE.

    Args:
        frames_tensor: (B, 3, H, W) float32 in [0, 1]
        dino_model: DINO encoder (v2 or v3)
        decoder: VQVAE decoder
        is_front_camera: applies front-camera preprocessing (blur + crop)

    Returns:
        (B, 3, H', W') float32 in [0, 1] at decoder output resolution
    """
    preprocessed = preprocess_images_for_dino(
        frames_tensor, is_front_camera=is_front_camera,
    )

    features = dino_model.forward_features(preprocessed)["x_norm_patchtokens"]
    # features: (B, num_patches, dim)

    # Decoder expects (B, T, num_patches, dim) with T=1
    features = features.unsqueeze(1)
    decoded, _ = decoder(features)  # (B*T, 3, H', W')
    return decoded.clamp(0, 1)


# -- metrics ------------------------------------------------------------------

def compute_psnr(
    orig: list[np.ndarray], decoded: list[np.ndarray],
) -> list[dict]:
    stats: list[dict] = []
    for i in range(min(len(orig), len(decoded))):
        mse = float(
            np.mean(
                (orig[i].astype(np.float32) - decoded[i].astype(np.float32)) ** 2
            )
        )
        psnr = float("inf") if mse == 0 else 10.0 * math.log10(255.0**2 / mse)
        stats.append({"frame": i, "mse": mse, "psnr": psnr})
    return stats


def write_side_by_side(
    a: list[np.ndarray], b: list[np.ndarray], out: Path, fps: float,
) -> None:
    import imageio.v2 as imageio

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out), fps=fps, codec="libx264", quality=8, format="ffmpeg",
    )
    try:
        for i in range(min(len(a), len(b))):
            fa, fb = a[i], b[i]
            h = min(fa.shape[0], fb.shape[0])
            w = min(fa.shape[1], fb.shape[1])
            writer.append_data(np.concatenate([fa[:h, :w], fb[:h, :w]], axis=1))
    finally:
        writer.close()


# -- checkpoint loading -------------------------------------------------------

def _load_decoder_checkpoint(
    path: str, device: torch.device,
) -> tuple[dict, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "decoder_state_dict", "state_dict"):
            if key in ckpt:
                meta = {k: v for k, v in ckpt.items() if k != key}
                return ckpt[key], meta
    return ckpt, {}


def _resolve_quantize(
    user_flag: bool | None, meta: dict, checkpoint_path: str,
) -> bool:
    if user_flag is True:
        return True
    if isinstance(meta, dict) and "quantize" in meta:
        return bool(meta["quantize"])
    if "_vq" in os.path.basename(checkpoint_path):
        return True
    return False


def _infer_dino_version(meta: dict | None) -> str | None:
    if not isinstance(meta, dict):
        return None
    if "dino_version" in meta and meta["dino_version"]:
        return str(meta["dino_version"])
    if "decoder_image_size" in meta and meta["decoder_image_size"]:
        size = meta["decoder_image_size"]
        if isinstance(size, (list, tuple)) and len(size) > 0:
            side = int(size[0])
        else:
            try:
                side = int(size)
            except (TypeError, ValueError):
                return None
        if side >= 256:
            return "v2"
        if side == 224:
            return "v3"
    return None


# -- main ---------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="DINO encoder → VQVAE decoder round-trip comparison",
    )

    # Input source (mutually exclusive)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--video", help="Local video path")
    src.add_argument("--hf_repo", help="LeRobot dataset repo id")

    # Dataset options
    p.add_argument("--episode", type=int, default=0, help="Episode index (default: 0)")
    p.add_argument("--camera", default=None, help="Camera key (default: first available)")
    p.add_argument("--list", action="store_true", help="List episodes/cameras and exit")

    # Model
    p.add_argument(
        "--decoder_checkpoint", default=None,
        help="Path to VQVAE decoder checkpoint (required unless --list)",
    )
    p.add_argument(
        "--dino_version", default=None, choices=["v2", "v3"],
        help="DINO version (default: auto-detect from checkpoint, fallback v3)",
    )
    p.add_argument(
        "--quantize", action="store_true", default=None,
        help="Enable VQ codebook quantization (auto-detected from checkpoint if omitted)",
    )

    # Output / runtime
    p.add_argument("--out_dir", default="./outputs/roundtrip_dino")
    p.add_argument("--max_frames", type=int, default=0, help="Max frames to process (0 = all)")
    p.add_argument("--fps", type=float, default=0.0, help="Override fps (0 = auto)")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--camera_type", default="wrist", choices=["front", "wrist"],
        help="Camera type for DINO preprocessing (default: wrist)",
    )
    p.add_argument(
        "--batch_size", type=int, default=32,
        help="Frames per batch for encode/decode (default: 32)",
    )
    p.add_argument(
        "--video_backend", default="pyav", choices=["pyav", "torchcodec"],
        help="LeRobot video decode backend (default: pyav)",
    )

    args = p.parse_args()

    # -- list mode -------------------------------------------------------------
    if args.list:
        if not args.hf_repo:
            p.error("--list requires --hf_repo")
        lerobot_list(args.hf_repo)
        return 0

    # -- validation ------------------------------------------------------------
    if not args.video and not args.hf_repo:
        p.error("one of --video or --hf_repo is required")
    if not args.decoder_checkpoint:
        p.error("--decoder_checkpoint is required (unless using --list)")

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu",
    )
    is_front = args.camera_type == "front"

    # -- load decoder ----------------------------------------------------------
    print("Loading decoder checkpoint …")
    state_dict, ckpt_meta = _load_decoder_checkpoint(args.decoder_checkpoint, device)
    quantize = _resolve_quantize(args.quantize, ckpt_meta, args.decoder_checkpoint)

    dino_version = args.dino_version or _infer_dino_version(ckpt_meta)
    dec_size = get_decoder_image_size(dino_version)

    decoder = VQVAE(quantize=quantize).to(device)
    decoder.load_state_dict(state_dict)
    decoder.eval()
    print(f"  quantize={quantize}  dino_version={dino_version or 'default'}")
    print(f"  decoder output size={dec_size}")

    # -- load DINO encoder -----------------------------------------------------
    print("Loading DINO model …")
    dino_model = get_dino_model(str(device), version=dino_version)

    # -- load frames -----------------------------------------------------------
    if args.video:
        orig_frames, detected_fps = read_video_frames(Path(args.video), args.max_frames)
        fps = args.fps if args.fps > 0 else (detected_fps or 16.0)
    else:
        orig_frames, detected_fps = lerobot_load_frames(
            args.hf_repo, args.episode, args.camera, args.max_frames,
            video_backend=args.video_backend,
        )
        fps = args.fps if args.fps > 0 else detected_fps

    print(f"Loaded {len(orig_frames)} frames @ {fps:.1f} fps")

    # Tensor at decoder resolution for encoding + PSNR
    orig_tensor = frames_to_tensor(orig_frames, device, target_size=dec_size)
    orig_at_dec_size = tensor_to_frames(orig_tensor)

    # -- roundtrip in batches --------------------------------------------------
    bs = args.batch_size
    decoded_batches: list[torch.Tensor] = []
    for start in range(0, orig_tensor.shape[0], bs):
        end = min(start + bs, orig_tensor.shape[0])
        print(f"  Encoding/decoding frames {start}–{end - 1} …")
        dec = dino_roundtrip(orig_tensor[start:end], dino_model, decoder, is_front)
        decoded_batches.append(dec)

    decoded_tensor = torch.cat(decoded_batches, dim=0)

    # Upscale decoded frames to native resolution for the side-by-side video
    native_h, native_w = orig_frames[0].shape[:2]
    decoded_native = tensor_to_frames(
        F.interpolate(decoded_tensor, size=(native_h, native_w), mode="bilinear", align_corners=False)
    )
    # Decoded at decoder resolution for PSNR
    decoded_at_dec_size = tensor_to_frames(decoded_tensor)

    # -- write outputs ---------------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build a descriptive suffix for output filenames
    if args.video:
        tag = Path(args.video).stem
    else:
        cam_short = args.camera.rsplit(".", 1)[-1] if args.camera else "cam0"
        tag = f"{cam_short}_ep{args.episode}"

    sbs_path = out_dir / f"side_by_side_{tag}.mp4"
    write_side_by_side(orig_frames, decoded_native, sbs_path, fps)

    stats = compute_psnr(orig_at_dec_size, decoded_at_dec_size)
    finite = [s["psnr"] for s in stats if math.isfinite(s["psnr"])]
    avg_psnr = float(np.mean(finite)) if finite else float("nan")

    metrics_path = out_dir / f"metrics_{tag}.json"
    metrics_path.write_text(
        json.dumps(
            {
                "input": args.video or f"{args.hf_repo} ep{args.episode}",
                "decoder_checkpoint": args.decoder_checkpoint,
                "dino_version": dino_version or "default",
                "quantize": quantize,
                "decoder_output_size": list(dec_size),
                "fps": fps,
                "frames": len(stats),
                "avg_psnr": avg_psnr,
                "per_frame": stats,
            },
            indent=2,
        )
    )

    print(f"\nWrote: {sbs_path}")
    print(f"Wrote: {metrics_path}")
    print(f"Avg PSNR: {avg_psnr:.3f} dB  ({len(stats)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
