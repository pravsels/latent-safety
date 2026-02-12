#!/usr/bin/env python3
"""
Decode WAN VAE latents from an HDF5 file back to video.

Usage:

  python scripts/decode_wan_hdf5_to_video.py \
    --input-hdf5 arx5_datasets_single_wan.h5 \
    --model ByteDance/Video-As-Prompt-Wan2.1-14B \
    --output-dir outputs/wan_decoded

Produces one .mp4 per trajectory with front and wrist views side by side,
plus the raw camera frames for comparison.
"""

import argparse
import math
import os
import sys

import h5py
import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _load_wan_vae(model_id: str, subfolder: str, device: str, dtype: str):
    from diffusers import AutoencoderKLWan

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    model_dtype = dtype_map[dtype] if dev.type == "cuda" else torch.float32
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder=subfolder, torch_dtype=model_dtype,
    ).to(dev).eval()
    return vae, dev, model_dtype


SPATIAL_DOWNSAMPLE = 8  # Wan VAE spatial compression factor


def _infer_latent_hw_from_camera(
    grp: h5py.Group, num_patches: int,
    latent_h: int = 0, latent_w: int = 0,
    crop_multiple: int = 8,
) -> tuple[int, int]:
    """Derive latent grid (h, w) from the raw camera resolution in the HDF5 group."""
    if latent_h > 0 and latent_w > 0:
        assert latent_h * latent_w == num_patches, (
            f"latent_h*latent_w ({latent_h * latent_w}) != num_patches ({num_patches})"
        )
        return latent_h, latent_w

    # Read camera shape to get the actual image H, W
    for cam_key in ("camera_0", "camera_1"):
        if cam_key in grp:
            cam_shape = grp[cam_key].shape  # (T, H, W, C)
            img_h, img_w = int(cam_shape[1]), int(cam_shape[2])
            # Reproduce the center-crop-to-multiple logic from add_wan_embeds_to_hdf5.py
            crop_h = (img_h // crop_multiple) * crop_multiple
            crop_w = (img_w // crop_multiple) * crop_multiple
            lh = crop_h // SPATIAL_DOWNSAMPLE
            lw = crop_w // SPATIAL_DOWNSAMPLE
            if lh * lw == num_patches:
                return lh, lw
            # If it doesn't match, keep trying the other camera
            continue

    raise ValueError(
        f"Cannot infer latent H/W: no camera dataset found or num_patches={num_patches} "
        f"doesn't match camera resolution. Pass --latent-height and --latent-width explicitly."
    )


@torch.no_grad()
def decode_latents(vae, latents: np.ndarray, latent_h: int, latent_w: int,
                   device: torch.device, model_dtype: torch.dtype,
                   batch_size: int = 4) -> list[np.ndarray]:
    """
    latents: (T, num_patches, C) float32
    Returns: list of T uint8 frames, each (H_pixel, W_pixel, 3)
    """
    T = latents.shape[0]
    frames = []
    for i in range(0, T, batch_size):
        batch = torch.from_numpy(latents[i:i + batch_size]).to(device=device, dtype=model_dtype)
        # (B, N, C) -> (B, C, 1, H, W) for VAE decode
        z = rearrange(batch, "b (h w) c -> b c 1 h w", h=latent_h, w=latent_w)
        decoded = vae.decode(z).sample  # (B, 3, T_out, H_pixel, W_pixel)
        # clamp to [-1, 1] then map to [0, 255]
        decoded = decoded.clamp(-1, 1).add(1.0).mul(127.5)
        # (B, 3, T_out, H, W) -> iterate over batch
        for j in range(decoded.shape[0]):
            frame = decoded[j]  # (3, T_out, H, W)
            # Take first temporal frame (T_out should be 1 for single-frame input)
            frame = frame[:, 0, :, :]  # (3, H, W)
            frame = frame.float().permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # (H, W, 3)
            frames.append(frame)
    return frames[:T]


def write_video(frames: list[np.ndarray], path: str, fps: float):
    import imageio.v2 as imageio

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, format="ffmpeg")
    try:
        for f in frames:
            writer.append_data(f)
    finally:
        writer.close()


def make_side_by_side(frames_a: list[np.ndarray], frames_b: list[np.ndarray]) -> list[np.ndarray]:
    """Horizontally concatenate two frame lists, resizing to match heights."""
    out = []
    for fa, fb in zip(frames_a, frames_b):
        h = min(fa.shape[0], fb.shape[0])
        # Resize if heights differ
        if fa.shape[0] != h:
            from PIL import Image
            fa = np.array(Image.fromarray(fa).resize((int(fa.shape[1] * h / fa.shape[0]), h)))
        if fb.shape[0] != h:
            from PIL import Image
            fb = np.array(Image.fromarray(fb).resize((int(fb.shape[1] * h / fb.shape[0]), h)))
        out.append(np.concatenate([fa, fb], axis=1))
    return out


def main():
    parser = argparse.ArgumentParser(description="Decode WAN latents from HDF5 to video")
    parser.add_argument("--input-hdf5", required=True, help="HDF5 file with WAN latents")
    parser.add_argument("--output-dir", default="outputs/wan_decoded", help="Output directory")
    parser.add_argument("--model", default="ByteDance/Video-As-Prompt-Wan2.1-14B", help="WAN VAE model")
    parser.add_argument("--subfolder", default="vae")
    parser.add_argument("--dtype", default="fp32", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--front-key", default="wan_front_embd")
    parser.add_argument("--wrist-key", default="wan_wrist_embd")
    parser.add_argument("--latent-height", type=int, default=0, help="Latent grid H (0=auto)")
    parser.add_argument("--latent-width", type=int, default=0, help="Latent grid W (0=auto)")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-trajectories", type=int, default=1, help="0=all")
    parser.add_argument("--max-frames", type=int, default=0, help="0=all frames per trajectory")
    args = parser.parse_args()

    print(f"Loading WAN VAE from {args.model}/{args.subfolder} ...")
    vae, device, model_dtype = _load_wan_vae(args.model, args.subfolder, args.device, args.dtype)

    os.makedirs(args.output_dir, exist_ok=True)

    with h5py.File(args.input_hdf5, "r") as hf:
        traj_keys = sorted([k for k in hf.keys() if k.startswith("trajectory_")],
                           key=lambda k: int(k.split("_")[1]))
        if args.max_trajectories > 0:
            traj_keys = traj_keys[:args.max_trajectories]

        print(f"Found {len(traj_keys)} trajectories")

        for traj_key in tqdm(traj_keys, desc="Decoding"):
            grp = hf[traj_key]

            if args.front_key not in grp or args.wrist_key not in grp:
                print(f"  Skipping {traj_key}: missing latent keys")
                continue

            front_latents = grp[args.front_key][:]  # (T, num_patches, C)
            wrist_latents = grp[args.wrist_key][:]

            if args.max_frames > 0:
                front_latents = front_latents[:args.max_frames]
                wrist_latents = wrist_latents[:args.max_frames]

            num_patches = front_latents.shape[1]
            latent_dim = front_latents.shape[2]
            T = front_latents.shape[0]

            latent_h, latent_w = _infer_latent_hw_from_camera(
                grp, num_patches, args.latent_height, args.latent_width,
            )

            print(f"  {traj_key}: {T} frames, latent ({latent_h}x{latent_w})x{latent_dim}, "
                  f"num_patches={num_patches}")

            # Decode front and wrist latents
            front_frames = decode_latents(vae, front_latents, latent_h, latent_w,
                                          device, model_dtype, args.batch_size)
            wrist_frames = decode_latents(vae, wrist_latents, latent_h, latent_w,
                                          device, model_dtype, args.batch_size)

            # Write individual videos
            front_path = os.path.join(args.output_dir, f"{traj_key}_front_decoded.mp4")
            wrist_path = os.path.join(args.output_dir, f"{traj_key}_wrist_decoded.mp4")
            write_video(front_frames, front_path, args.fps)
            write_video(wrist_frames, wrist_path, args.fps)

            print(f"    Wrote: {front_path}")
            print(f"    Wrote: {wrist_path}")

            # If raw camera frames exist, make GT-vs-decoded comparison per camera
            has_raw = "camera_0" in grp and "camera_1" in grp
            if has_raw:
                cam0_raw = list(grp["camera_0"][:T])   # wrist (T, H, W, 3) uint8
                cam1_raw = list(grp["camera_1"][:T])   # front

                # Front: GT left, decoded right
                front_compare = make_side_by_side(cam1_raw, front_frames)
                front_cmp_path = os.path.join(args.output_dir, f"{traj_key}_front_gt_vs_decoded.mp4")
                write_video(front_compare, front_cmp_path, args.fps)
                print(f"    Wrote: {front_cmp_path}")

                # Wrist: GT left, decoded right
                wrist_compare = make_side_by_side(cam0_raw, wrist_frames)
                wrist_cmp_path = os.path.join(args.output_dir, f"{traj_key}_wrist_gt_vs_decoded.mp4")
                write_video(wrist_compare, wrist_cmp_path, args.fps)
                print(f"    Wrote: {wrist_cmp_path}")

    print(f"\nDone. Videos saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
