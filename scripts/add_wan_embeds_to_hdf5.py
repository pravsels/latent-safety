#!/usr/bin/env python3
"""
Copy an existing HDF5 dataset and add WAN VAE latents.

This mirrors scripts/add_dino_embeds_to_hdf5.py but writes WAN-backed latent keys
that can be consumed by the WAN backbone world-model flow.
"""

import argparse
import os
import sys
from typing import Iterable

import h5py
import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dino_wm.config import WAN_CONFIG


def _copy_attrs(src, dst) -> None:
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def _copy_dataset(src_ds: h5py.Dataset, dst_grp: h5py.Group, name: str, chunk_size: int = 1024):
    shape = src_ds.shape
    dtype = src_ds.dtype
    chunks = src_ds.chunks
    compression = src_ds.compression
    compression_opts = src_ds.compression_opts

    dst_ds = dst_grp.create_dataset(
        name,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        compression=compression,
        compression_opts=compression_opts,
    )

    if src_ds.ndim == 0:
        dst_ds[()] = src_ds[()]
        return

    if shape[0] == 0:
        return

    for i in range(0, shape[0], chunk_size):
        dst_ds[i:i + chunk_size] = src_ds[i:i + chunk_size]


def _iter_trajectory_keys(hf: h5py.File) -> Iterable[str]:
    keys = [k for k in hf.keys() if k.startswith("trajectory_")]
    keys.sort(key=lambda k: int(k.split("_")[1]))
    return keys


def _load_wan_vae(model_id_or_path: str, subfolder: str, device: str, dtype: str):
    from diffusers import AutoencoderKLWan

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    model_dtype = dtype_map[dtype] if dev.type == "cuda" else torch.float32
    vae = AutoencoderKLWan.from_pretrained(
        model_id_or_path,
        subfolder=subfolder,
        torch_dtype=model_dtype,
    ).to(dev).eval()
    return vae, dev, model_dtype


def _frames_to_wan_input(frames: np.ndarray, device: torch.device, model_dtype: torch.dtype,
                         input_size: int = WAN_CONFIG['input_size']) -> torch.Tensor:
    """
    frames: (B, H, W, C) uint8 -> (B, C, T=1, H, W) float in [-1, 1]
    Resizes to (input_size, input_size) so latent grid is fixed regardless of camera resolution.
    """
    x = torch.from_numpy(frames).to(device=device, dtype=torch.float32)
    x = x.permute(0, 3, 1, 2).div(127.5).sub(1.0)  # (B, C, H, W)
    x = torch.nn.functional.interpolate(x, size=(input_size, input_size), mode="bilinear", align_corners=False)
    x = x.unsqueeze(2).to(dtype=model_dtype)        # (B, C, 1, H, W)
    return x


def _flatten_wan_latents(z: torch.Tensor) -> torch.Tensor:
    """
    Accept WAN latent tensor in either shape:
      - (B, C, T, H, W) with T expected to be 1
      - (B, C, H, W)
    Return:
      - (B, H*W, C)
    """
    if z.ndim == 5:
        # Use first temporal slice for per-frame latents.
        z2d = z[:, :, 0, :, :]  # (B, C, H, W)
    elif z.ndim == 4:
        z2d = z
    else:
        raise ValueError(f"Unexpected latent shape {tuple(z.shape)}; expected 4D or 5D")
    return rearrange(z2d, "b c h w -> b (h w) c")


def _compute_wan_embeddings(
    vae,
    cam0: np.ndarray,
    cam1: np.ndarray,
    batch_size: int,
    device: torch.device,
    model_dtype: torch.dtype,
    input_size: int = 224,
):
    total = cam0.shape[0]
    for i in range(0, total, batch_size):
        w_np = cam0[i:i + batch_size]
        f_np = cam1[i:i + batch_size]

        w = _frames_to_wan_input(w_np, device, model_dtype, input_size=input_size)
        f = _frames_to_wan_input(f_np, device, model_dtype, input_size=input_size)

        with torch.no_grad():
            z_w = vae.encode(w).latent_dist.mode()
            z_f = vae.encode(f).latent_dist.mode()
            w_emb = _flatten_wan_latents(z_w).float().cpu().numpy()
            f_emb = _flatten_wan_latents(z_f).float().cpu().numpy()

        yield i, w_emb, f_emb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-hdf5", type=str, required=True)
    parser.add_argument("--output-hdf5", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model", type=str, required=True, help="WAN VAE model id/path")
    parser.add_argument("--subfolder", type=str, default="vae")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--input-size", type=int, default=WAN_CONFIG['input_size'],
                        help="Resize images to this square size before encoding.")
    parser.add_argument("--front-key", type=str, default="wan_front_embd")
    parser.add_argument("--wrist-key", type=str, default="wan_wrist_embd")
    parser.add_argument("--resume", action="store_true", help="Skip trajectories already present in output.")
    args = parser.parse_args()

    if not os.path.exists(args.input_hdf5):
        raise FileNotFoundError(f"Input HDF5 not found: {args.input_hdf5}")

    vae, device, model_dtype = _load_wan_vae(args.model, args.subfolder, args.device, args.dtype)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_hdf5)), exist_ok=True)
    out_mode = "a" if args.resume and os.path.exists(args.output_hdf5) else "w"
    if out_mode == "w" and os.path.exists(args.output_hdf5):
        raise FileExistsError(
            f"Output HDF5 exists: {args.output_hdf5}. Use --resume or delete it."
        )

    try:
        hf_out = h5py.File(args.output_hdf5, out_mode)
    except OSError as e:
        if args.resume and os.path.exists(args.output_hdf5):
            backup = f"{args.output_hdf5}.corrupt"
            os.rename(args.output_hdf5, backup)
            print(f"WARNING: Output HDF5 unreadable. Moved to {backup} and starting fresh.")
            hf_out = h5py.File(args.output_hdf5, "w")
        else:
            raise e

    with h5py.File(args.input_hdf5, "r") as hf_in, hf_out:
        in_keys = _iter_trajectory_keys(hf_in)
        out_keys = set(hf_out.keys())

        for traj_key in tqdm(in_keys, desc="Trajectories"):
            if args.resume and traj_key in out_keys:
                continue

            src_grp = hf_in[traj_key]
            dst_grp = hf_out.create_group(traj_key)
            _copy_attrs(src_grp, dst_grp)

            if "camera_0" not in src_grp or "camera_1" not in src_grp:
                raise KeyError(f"{traj_key} missing camera_0 or camera_1")

            # Copy all existing datasets.
            for name, ds in src_grp.items():
                _copy_dataset(ds, dst_grp, name)

            cam0 = src_grp["camera_0"]
            cam1 = src_grp["camera_1"]
            total = cam0.shape[0]

            front_ds = None
            wrist_ds = None
            for idx, w_emb, f_emb in _compute_wan_embeddings(
                vae=vae,
                cam0=cam0,
                cam1=cam1,
                batch_size=args.batch_size,
                device=device,
                model_dtype=model_dtype,
                input_size=args.input_size,
            ):
                if front_ds is None or wrist_ds is None:
                    num_patches = int(f_emb.shape[1])
                    dim = int(f_emb.shape[2])
                    chunk_t = min(args.batch_size, total)
                    front_ds = dst_grp.create_dataset(
                        args.front_key,
                        shape=(total, num_patches, dim),
                        dtype=np.float32,
                        chunks=(chunk_t, num_patches, dim),
                    )
                    wrist_ds = dst_grp.create_dataset(
                        args.wrist_key,
                        shape=(total, num_patches, dim),
                        dtype=np.float32,
                        chunks=(chunk_t, num_patches, dim),
                    )

                end = idx + w_emb.shape[0]
                wrist_ds[idx:end] = w_emb
                front_ds[idx:end] = f_emb

            hf_out.flush()


if __name__ == "__main__":
    main()
