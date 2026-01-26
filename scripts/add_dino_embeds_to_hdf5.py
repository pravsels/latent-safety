#!/usr/bin/env python3
"""
Copy an existing HDF5 dataset and add fresh DINO embeddings.

This is useful when labels and other fields are already correct, but the
embedded tokens need to be regenerated (e.g., DINOv3 vs DINOv2).
"""

import argparse
import os
from typing import Iterable

import h5py
import numpy as np
import torch
from tqdm import tqdm

from scripts.utils import get_dino_model, preprocess_images_for_dino
from dino_wm.config import MODEL_CONFIG, get_dino_config


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


def _ensure_dino_config(version: str):
    dino_cfg = get_dino_config(version)
    MODEL_CONFIG["dim"] = dino_cfg["dim"]
    return dino_cfg


def _compute_embeddings(
    dino_model,
    cam0: np.ndarray,
    cam1: np.ndarray,
    batch_size: int,
    device: str,
):
    total = cam0.shape[0]
    for i in range(0, total, batch_size):
        w_np = cam0[i:i + batch_size]
        f_np = cam1[i:i + batch_size]

        w = torch.from_numpy(w_np).float() / 255.0
        f = torch.from_numpy(f_np).float() / 255.0
        w = w.permute(0, 3, 1, 2).to(device)
        f = f.permute(0, 3, 1, 2).to(device)

        w_prep = preprocess_images_for_dino(w, is_front_camera=False)
        f_prep = preprocess_images_for_dino(f, is_front_camera=True)

        with torch.no_grad():
            w_emb = dino_model.forward_features(w_prep)["x_norm_patchtokens"].cpu().numpy()
            f_emb = dino_model.forward_features(f_prep)["x_norm_patchtokens"].cpu().numpy()

        yield i, w_emb, f_emb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-hdf5", type=str, required=True)
    parser.add_argument("--output-hdf5", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dino-version", type=str, choices=["v2", "v3"], default="v3")
    parser.add_argument("--resume", action="store_true", help="Skip trajectories already present in output.")
    args = parser.parse_args()

    if not os.path.exists(args.input_hdf5):
        raise FileNotFoundError(f"Input HDF5 not found: {args.input_hdf5}")

    dino_cfg = _ensure_dino_config(args.dino_version)
    num_patches = int(dino_cfg["num_patches"])
    dim = int(dino_cfg["dim"])

    device = args.device if torch.cuda.is_available() else "cpu"
    dino_model = get_dino_model(device, args.dino_version)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_hdf5)), exist_ok=True)

    out_mode = "a" if args.resume and os.path.exists(args.output_hdf5) else "w"
    if out_mode == "w" and os.path.exists(args.output_hdf5):
        raise FileExistsError(
            f"Output HDF5 exists: {args.output_hdf5}. Use --resume or delete it."
        )

    with h5py.File(args.input_hdf5, "r") as hf_in, h5py.File(args.output_hdf5, out_mode) as hf_out:
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

            # Copy all datasets except existing embeddings.
            for name, ds in src_grp.items():
                if name.endswith("_embd"):
                    continue
                _copy_dataset(ds, dst_grp, name)

            cam0 = src_grp["camera_0"]
            cam1 = src_grp["camera_1"]
            total = cam0.shape[0]

            # Create embedding datasets.
            rs_ds = dst_grp.create_dataset(
                "cam_rs_embd",
                shape=(total, num_patches, dim),
                dtype=np.float32,
                chunks=(min(args.batch_size, total), num_patches, dim),
            )
            zed_ds = dst_grp.create_dataset(
                "cam_zed_embd",
                shape=(total, num_patches, dim),
                dtype=np.float32,
                chunks=(min(args.batch_size, total), num_patches, dim),
            )

            for idx, w_emb, f_emb in _compute_embeddings(
                dino_model=dino_model,
                cam0=cam0,
                cam1=cam1,
                batch_size=args.batch_size,
                device=device,
            ):
                end = idx + w_emb.shape[0]
                rs_ds[idx:end] = w_emb
                zed_ds[idx:end] = f_emb

            hf_out.flush()


if __name__ == "__main__":
    main()

