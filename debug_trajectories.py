#!/usr/bin/env python3
"""
Inspect DINO WM HDF5 trajectories and drop into pdb for manual inspection.

Quickstart:
  python debug_trajectories.py --hdf5 test_v2.h5
"""

import argparse
import pdb

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hdf5",
        type=str,
        default="test_v2.h5",
        help="Path to DINO WM HDF5 file (default: test_v2.h5)",
    )
    args = parser.parse_args()

    f = h5py.File(args.hdf5, "r")

    # Load all trajectory_* groups into a simple Python list of dicts.
    trajectories = []
    for name in sorted(f.keys()):
        if not isinstance(f[name], h5py.Group):
            continue
        traj_group = f[name]
        data = {k: np.array(v) for k, v in traj_group.items()}
        data["name"] = name
        trajectories.append(data)

    print(f"Loaded {len(trajectories)} trajectories from {args.hdf5}")
    if trajectories:
        first = trajectories[0]
        print("First trajectory keys:", list(first.keys()))

    print("\nAvailable variables in pdb:")
    print("  trajectories - list of dicts, one per trajectory (with numpy arrays)")
    print("  f            - open h5py.File (if you want raw access)")

    pdb.set_trace()


if __name__ == "__main__":
    main()
