#!/usr/bin/env python3
"""
Verify actions_delta == actions - states (overlapping dims) for all trajectories.

QUICKSTART:
    python scripts/verify_actions_delta.py --file arx5_datasets_single.h5
"""

import argparse
from pathlib import Path
import sys

import h5py
import numpy as np


def verify_actions_delta(hdf5_path: Path) -> int:
    with h5py.File(hdf5_path, "r") as f:
        traj_keys = sorted(k for k in f.keys() if k.startswith("trajectory_"))
        total_frames = 0
        max_diff = 0.0
        mismatches = 0

        for tk in traj_keys:
            g = f[tk]
            actions = g["actions"][:]
            states = g["states"][:]
            actions_delta = g["actions_delta"][:]

            k = min(actions.shape[1], states.shape[1])
            computed = actions.copy()
            computed[:, :k] = actions[:, :k] - states[:, :k]

            diff = float(np.max(np.abs(actions_delta - computed)))
            max_diff = max(max_diff, diff)
            total_frames += actions.shape[0]

            if diff != 0.0:
                mismatches += 1
                print(f"Mismatch in {tk}: max_abs_diff={diff}")

        print(f"trajectories: {len(traj_keys)}")
        print(f"frames_checked: {total_frames}")
        print(f"max_abs_diff_overall: {max_diff}")
        print(f"trajectories_with_mismatch: {mismatches}")

    return 1 if mismatches else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to HDF5 file")
    args = parser.parse_args()

    hdf5_path = Path(args.file)
    if not hdf5_path.exists():
        print(f"Missing file: {hdf5_path}")
        return 1

    return verify_actions_delta(hdf5_path)


if __name__ == "__main__":
    sys.exit(main())
