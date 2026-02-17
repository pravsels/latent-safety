#!/usr/bin/env python3
"""
Verify failure labels in an HDF5 dataset.

Usage:
    python scripts/verify_labels.py fail_bin_pick_capsules_dino3.h5
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

        for k in traj_keys:
            g = f[k]
            n_frames = g["camera_0"].shape[0] if "camera_0" in g else 0

            if "labels" not in g:
                missing_labels.append(k)
                continue

            labels = g["labels"][:]
            if len(labels) != n_frames:
                print(f"  ⚠️  {k}: label length ({len(labels)}) != frames ({n_frames})")

            n_safe = int(np.sum(labels == 0))
            n_unsafe = int(np.sum(labels == 1))
            n_weak = int(np.sum(labels == 2))

            total_frames += len(labels)
            total_safe += n_safe
            total_unsafe += n_unsafe
            total_weak += n_weak

            if n_unsafe == 0 and n_weak == 0:
                all_safe.append(k)
                continue

            if n_unsafe == 0 and n_weak == 0 and n_safe == 0:
                empty_labels.append(k)

            pct_unsafe = (n_unsafe + n_weak) / len(labels) * 100
            print(f"  {k}: {len(labels)} frames | "
                  f"safe={n_safe} unsafe={n_unsafe} weak={n_weak} "
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

        if missing_labels:
            print(f"\n  ❌ Missing labels ({len(missing_labels)}):")
            for k in missing_labels:
                print(f"     {k}")

        if all_safe:
            print(f"\n  ℹ️  Entirely safe / unlabeled ({len(all_safe)}):")
            for k in all_safe:
                print(f"     {k}")

        labeled_with_failures = len(traj_keys) - len(missing_labels) - len(all_safe)
        print(f"\n  Labeled with failures: {labeled_with_failures}/{len(traj_keys)}")


def main():
    parser = argparse.ArgumentParser(description="Verify failure labels in HDF5 dataset")
    parser.add_argument("hdf5_file", help="Path to HDF5 file")
    args = parser.parse_args()
    verify_labels(args.hdf5_file)


if __name__ == "__main__":
    main()
