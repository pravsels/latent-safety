#!/usr/bin/env python3
"""
Convert Hugging Face LeRobot datasets into a consolidated HDF5 file
compatible with the DINO world model code in this repo.

High‑level:
- Input  : JSON file with a list of dataset IDs (e.g. arx5_datasets.json)
- Output : Single HDF5 file with groups trajectory_0, trajectory_1, ...
           Each group contains:
             camera_0    : wrist   images, shape [T, H, W, 3], uint8
             camera_1    : front   images, shape [T, H, W, 3], uint8
             actions     : shape [T, A], float32
             states      : shape [T, S], float32 (from observation.state if available)
             cam_rs_embd : shape [T, P, D] DINO patch tokens for camera_0
             cam_zed_embd: shape [T, P, D] DINO patch tokens for camera_1

This bypasses the older "per‑episode .hdf5 + preprocess()" path and
directly writes what `dino_wm/test_loader.py::SplitTrajectoryDataset`
expects.

Example:
  python scripts/lerobot_to_hdf5.py \
    --datasets-list test_v2.json \
    --output-hdf5 test_v2.h5 \
    --max-episodes-per-dataset 3
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Ensure project root is on sys.path so we can import dino_wm.*
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Reuse DINO preprocessing from our dino_wm code
from dino_wm.hdf5_to_dataset import DINO_crop
from torchvision import transforms as T


def _to_hwc_uint8(img: torch.Tensor) -> np.ndarray:
    """
    Convert a LeRobot image tensor (CHW, float [0,1] or [0,255]) to HWC uint8.
    """
    if img is None:
        raise ValueError("Image tensor is None")

    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().numpy()
    else:
        arr = np.asarray(img)

    if arr.ndim != 3 or arr.shape[0] not in (1, 3):
        raise ValueError(f"Unexpected image shape: {arr.shape}")

    # CHW -> HWC
    arr = np.transpose(arr, (1, 2, 0))

    # Scale if necessary
    if arr.max() <= 1.0:
        arr = (arr * 255.0).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)

    return arr


def _extract_images(sample: Dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract wrist (camera_0) and front (camera_1) images from a LeRobot sample.

    We assume the ARX5 format with:
      - observation.images.wrist
      - observation.images.front
    """
    wrist_img = None
    front_img = None

    if "observation.images.wrist" in sample:
        wrist_img = _to_hwc_uint8(sample["observation.images.wrist"])
    if "observation.images.front" in sample:
        front_img = _to_hwc_uint8(sample["observation.images.front"])

    return wrist_img, front_img


def _extract_state(sample: Dict) -> Optional[np.ndarray]:
    """
    Extract a low‑dimensional state vector from the sample.
    We prioritise 'observation.state' if available.
    """
    state = None
    for key in ("observation.state",):
        if key in sample:
            state = sample[key]
            break

    if state is None:
        return None

    if isinstance(state, torch.Tensor):
        state = state.detach().cpu().numpy()
    state = np.asarray(state, dtype=np.float32).flatten()
    return state


def _extract_action(sample: Dict) -> np.ndarray:
    """
    Extract action vector as float32 1D.
    """
    act = sample.get("action", 0.0)
    if isinstance(act, torch.Tensor):
        act = act.detach().cpu().numpy()
    act = np.asarray(act, dtype=np.float32).flatten()
    return act


def _load_dino_model(device: str = "cuda:0") -> torch.nn.Module:
    """
    Load the DINOv2 ViT-S/14 model from torch.hub.
    """
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg").to(device)
    model.eval()
    return model


def _compute_dino_embeddings(
    dino: torch.nn.Module,
    wrist_frames: np.ndarray,
    front_frames: np.ndarray,
    device: str = "cuda:0",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute DINO patch tokens for wrist (camera_0) and front (camera_1) images.

    wrist_frames: [T, H, W, 3] uint8
    front_frames: [T, H, W, 3] uint8
    """
    from PIL import Image

    # For wrist images, just resize to 224x224 then normalize (no crop)
    wrist_transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    cam_rs_embds: List[np.ndarray] = []
    cam_zed_embds: List[np.ndarray] = []

    with torch.no_grad():
        for t in range(len(wrist_frames)):
            # Wrist / camera_0
            w_img = wrist_frames[t]
            img_pil = Image.fromarray(w_img).convert("RGB")
            img_tensor = wrist_transform(img_pil).to(device)
            feats = dino.forward_features(img_tensor.unsqueeze(0))
            w_emb = feats["x_norm_patchtokens"].squeeze().cpu().numpy()
            cam_rs_embds.append(w_emb)

            # Front / camera_1
            f_img = front_frames[t]
            img_pil = Image.fromarray(f_img).convert("RGB")
            img_tensor = DINO_crop(img_pil).to(device)
            feats = dino.forward_features(img_tensor.unsqueeze(0))
            f_emb = feats["x_norm_patchtokens"].squeeze().cpu().numpy()
            cam_zed_embds.append(f_emb)

    cam_rs_embds = np.stack(cam_rs_embds)  # [T, P, D]
    cam_zed_embds = np.stack(cam_zed_embds)
    return cam_rs_embds, cam_zed_embds


def convert_lerobot_to_hdf5(
    dataset_ids: List[str],
    output_hdf5: Path,
    max_datasets: Optional[int] = None,
    max_episodes_per_dataset: Optional[int] = None,
    min_episode_length: int = 2,
    device: str = "cuda:0",
) -> None:
    """
    Main conversion routine.
    """
    if max_datasets is not None:
        dataset_ids = dataset_ids[:max_datasets]

    output_hdf5.parent.mkdir(parents=True, exist_ok=True)

    dino = _load_dino_model(device=device)

    traj_counter = 0
    with h5py.File(output_hdf5, "w") as hf_out:
        for ds_idx, dataset_id in enumerate(dataset_ids, 1):
            print(f"[{ds_idx}/{len(dataset_ids)}] Loading {dataset_id}...")
            try:
                dataset = LeRobotDataset(dataset_id)
            except Exception as e:
                print(f"  Failed to load {dataset_id}: {e}")
                continue

            print(f"  Found {len(dataset)} samples")

            # First pass: index episodes
            episodes: Dict[int, List[Tuple[int, int]]] = {}
            for idx in tqdm(range(len(dataset)), desc=f"  Indexing {dataset_id}", ncols=0):
                try:
                    sample = dataset[idx]
                    ep_idx = sample.get("episode_index", 0)
                    fr_idx = sample.get("frame_index", 0)
                    if hasattr(ep_idx, "item"):
                        ep_idx = int(ep_idx.item())
                    else:
                        ep_idx = int(ep_idx)
                    if hasattr(fr_idx, "item"):
                        fr_idx = int(fr_idx.item())
                    else:
                        fr_idx = int(fr_idx)

                    episodes.setdefault(ep_idx, []).append((fr_idx, idx))
                except Exception as e:
                    # skip bad sample
                    continue

            # Second pass: build trajectories
            valid_ep = 0
            for ep_idx in sorted(episodes.keys()):
                frame_list = sorted(episodes[ep_idx])  # (frame_idx, sample_idx)
                wrist_frames: List[np.ndarray] = []
                front_frames: List[np.ndarray] = []
                actions: List[np.ndarray] = []
                states: List[np.ndarray] = []

                for _, sample_idx in frame_list:
                    sample = dataset[sample_idx]

                    wrist_img, front_img = _extract_images(sample)
                    if wrist_img is None or front_img is None:
                        # Require both cameras for DINO WM
                        continue

                    act = _extract_action(sample)
                    state = _extract_state(sample)

                    wrist_frames.append(wrist_img)
                    front_frames.append(front_img)
                    actions.append(act)
                    if state is not None:
                        states.append(state)

                if len(wrist_frames) < min_episode_length:
                    continue

                # Stack
                wrist_arr = np.stack(wrist_frames, axis=0)
                front_arr = np.stack(front_frames, axis=0)
                actions_arr = np.stack(actions, axis=0)

                if states and len(states) == len(wrist_frames):
                    states_arr = np.stack(states, axis=0)
                else:
                    states_arr = None

                # Compute DINO embeddings
                cam_rs_embd, cam_zed_embd = _compute_dino_embeddings(
                    dino, wrist_arr, front_arr, device=device
                )

                # Write group
                g = hf_out.create_group(f"trajectory_{traj_counter}")
                g.create_dataset("camera_0", data=wrist_arr, dtype="uint8")
                g.create_dataset("camera_1", data=front_arr, dtype="uint8")
                g.create_dataset("actions", data=actions_arr.astype(np.float32))
                if states_arr is not None:
                    g.create_dataset("states", data=states_arr.astype(np.float32))
                g.create_dataset("cam_rs_embd", data=cam_rs_embd.astype(np.float32))
                g.create_dataset("cam_zed_embd", data=cam_zed_embd.astype(np.float32))

                traj_counter += 1
                valid_ep += 1

                if max_episodes_per_dataset is not None and valid_ep >= max_episodes_per_dataset:
                    break

            print(f"  Wrote {valid_ep} trajectory groups from {dataset_id}")

    print(f"Done. Wrote {traj_counter} trajectories to {output_hdf5}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LeRobot HF datasets into DINO‑compatible consolidated HDF5"
    )
    parser.add_argument(
        "--datasets-list",
        type=str,
        required=True,
        help="Path to JSON file with list of dataset IDs",
    )
    parser.add_argument(
        "--output-hdf5",
        type=str,
        required=True,
        help="Output consolidated HDF5 path (e.g. wm_demos128.h5)",
    )
    parser.add_argument(
        "--max-datasets",
        type=int,
        default=None,
        help="Optional cap on number of datasets to process",
    )
    parser.add_argument(
        "--max-episodes-per-dataset",
        type=int,
        default=None,
        help="Optional cap on episodes per dataset",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for DINO model (default: cuda:0)",
    )

    args = parser.parse_args()

    with open(args.datasets_list, "r") as f:
        dataset_ids = json.load(f)
    if not isinstance(dataset_ids, list):
        raise ValueError("datasets-list JSON must contain a list of dataset IDs")

    convert_lerobot_to_hdf5(
        dataset_ids=dataset_ids,
        output_hdf5=Path(args.output_hdf5),
        max_datasets=args.max_datasets,
        max_episodes_per_dataset=args.max_episodes_per_dataset,
        device=args.device,
    )


if __name__ == "__main__":
    main()


