#!/usr/bin/env python3
"""
Verify failure labels in an HDF5 dataset.

Usage:
    python scripts/verify_labels.py fail_bin_pick_capsules_dino3.h5
    python scripts/verify_labels.py data.h5 --mark-all-safe
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


def verify_labels(hdf5_path: str):
    path = Path(hdf5_path)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    with h5py.File(path, "r") as f:
        traj_keys = sorted(
            [k for k in f.keys() if k.startswith("trajectory_")],
            key=lambda k: int(k.split("_")[1]),
        )

        print(f"File: {path.name}")
        print(f"Trajectories: {len(traj_keys)}")
        print()

        total_frames = 0
        total_safe = 0
        total_unsafe = 0
        total_weak = 0
        missing_labels = []
        empty_labels = []
        all_safe = []

        for traj_name in traj_keys:
            traj_group = f[traj_name]  # trajectory group (e.g. f["trajectory_0"])
            n_frames_this_ep = traj_group["camera_0"].shape[0] if "camera_0" in traj_group else 0

            if "labels" not in traj_group:
                missing_labels.append(traj_name)
                continue

            labels = traj_group["labels"][:]
            if len(labels) != n_frames_this_ep:
                print(f"  ⚠️  {traj_name}: label length ({len(labels)}) != frames ({n_frames_this_ep})")

            n_safe_this_ep = int(np.sum(labels == 0))
            n_unsafe_this_ep = int(np.sum(labels == 1))
            n_weak_this_ep = int(np.sum(labels == 2))

            total_frames += len(labels)
            total_safe += n_safe_this_ep
            total_unsafe += n_unsafe_this_ep
            total_weak += n_weak_this_ep

            if n_unsafe_this_ep == 0 and n_weak_this_ep == 0:
                all_safe.append(traj_name)
                continue

            if n_unsafe_this_ep == 0 and n_weak_this_ep == 0 and n_safe_this_ep == 0:
                empty_labels.append(traj_name)

            pct_unsafe = (n_unsafe_this_ep + n_weak_this_ep) / len(labels) * 100
            print(f"  {traj_name}: {len(labels)} frames | "
                  f"safe={n_safe_this_ep} unsafe={n_unsafe_this_ep} weak={n_weak_this_ep} "
                  f"({pct_unsafe:.0f}% failure)")

        # Summary
        print()
        print("=" * 50)
        print("Summary")
        print("=" * 50)
        print(f"  Total trajectories:  {len(traj_keys)}")
        print(f"  Total frames:        {total_frames}")
        print(f"  Safe frames:         {total_safe} ({total_safe / max(total_frames, 1) * 100:.1f}%)")
        print(f"  Unsafe frames:       {total_unsafe} ({total_unsafe / max(total_frames, 1) * 100:.1f}%)")
        print(f"  Weak unsafe frames:  {total_weak} ({total_weak / max(total_frames, 1) * 100:.1f}%)")

        labeled_count = len(traj_keys) - len(missing_labels)
        labeled_with_failures = labeled_count - len(all_safe)

        if missing_labels:
            print(f"\n  ❌ Labeled: {labeled_count}/{len(traj_keys)} — MISSING {len(missing_labels)}:")
            for traj_name in missing_labels:
                print(f"     {traj_name}")
        else:
            print(f"\n  ✅ All {len(traj_keys)} trajectories labeled")

        print(f"     With failures: {labeled_with_failures}")
        print(f"     Entirely safe: {len(all_safe)}")


def main():
    parser = argparse.ArgumentParser(description="Verify or mark failure labels in HDF5 dataset")
    parser.add_argument("hdf5_file", help="Path to HDF5 file")
    args = parser.parse_args()

    verify_labels(args.hdf5_file)


if __name__ == "__main__":
    main()
