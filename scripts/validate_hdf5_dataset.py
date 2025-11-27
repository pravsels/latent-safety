#!/usr/bin/env python3
"""
Validate the integrity of an HDF5 dataset created by lerobot_to_hdf5.py
Checks for corruption, missing data, and reports dataset statistics.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
from tqdm import tqdm


def validate_trajectory(
    grp: h5py.Group, 
    traj_name: str, 
    verbose: bool = False
) -> Tuple[bool, List[str]]:
    """
    Validate a single trajectory group.
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    # Check required datasets
    required_datasets = ["camera_0", "camera_1", "actions", "cam_rs_embd", "cam_zed_embd"]
    optional_datasets = ["states"]
    
    for ds_name in required_datasets:
        if ds_name not in grp:
            errors.append(f"Missing required dataset: {ds_name}")
    
    # Check required attributes
    required_attrs = ["dataset_id", "original_episode_index"]
    for attr_name in required_attrs:
        if attr_name not in grp.attrs:
            errors.append(f"Missing required attribute: {attr_name}")
    
    if errors:
        return False, errors
    
    # Validate data shapes and consistency
    try:
        camera_0 = grp["camera_0"]
        camera_1 = grp["camera_1"]
        actions = grp["actions"]
        cam_rs_embd = grp["cam_rs_embd"]
        cam_zed_embd = grp["cam_zed_embd"]
        
        # Get lengths
        num_frames_cam0 = camera_0.shape[0]
        num_frames_cam1 = camera_1.shape[0]
        num_actions = actions.shape[0]
        num_embd_rs = cam_rs_embd.shape[0]
        num_embd_zed = cam_zed_embd.shape[0]
        
        # Check consistency
        if not (num_frames_cam0 == num_frames_cam1 == num_actions == num_embd_rs == num_embd_zed):
            errors.append(
                f"Inconsistent lengths: cam0={num_frames_cam0}, cam1={num_frames_cam1}, "
                f"actions={num_actions}, embd_rs={num_embd_rs}, embd_zed={num_embd_zed}"
            )
        
        # Check states if present
        if "states" in grp:
            states = grp["states"]
            num_states = states.shape[0]
            if num_states != num_frames_cam0:
                errors.append(f"States length mismatch: {num_states} vs {num_frames_cam0}")
        
        # Validate image shapes (should be HWC format)
        if len(camera_0.shape) != 4:
            errors.append(f"camera_0 has invalid shape: {camera_0.shape} (expected 4D: N,H,W,C)")
        if len(camera_1.shape) != 4:
            errors.append(f"camera_1 has invalid shape: {camera_1.shape} (expected 4D: N,H,W,C)")
        
        # Check data types
        if camera_0.dtype != np.uint8:
            errors.append(f"camera_0 has invalid dtype: {camera_0.dtype} (expected uint8)")
        if camera_1.dtype != np.uint8:
            errors.append(f"camera_1 has invalid dtype: {camera_1.dtype} (expected uint8)")
        
        # Try to read a sample to check for corruption
        try:
            _ = camera_0[0]
            _ = camera_1[0]
            _ = actions[0]
            _ = cam_rs_embd[0]
            _ = cam_zed_embd[0]
        except Exception as e:
            errors.append(f"Failed to read data: {e}")
        
        if verbose and not errors:
            print(f"  ✓ {traj_name}: {num_frames_cam0} frames, "
                  f"cam0={camera_0.shape}, cam1={camera_1.shape}, "
                  f"actions={actions.shape}, embeddings={cam_rs_embd.shape}")
    
    except Exception as e:
        errors.append(f"Validation error: {e}")
    
    return len(errors) == 0, errors


def get_dataset_stats(hf: h5py.File) -> Dict:
    """
    Gather statistics about the dataset.
    """
    stats = {
        "num_trajectories": 0,
        "total_frames": 0,
        "dataset_sources": {},
        "trajectory_lengths": [],
        "action_dims": None,
        "state_dims": None,
        "embedding_dims": None,
        "image_shapes": None,
    }
    
    for key in hf.keys():
        if key.startswith("trajectory_"):
            stats["num_trajectories"] += 1
            grp = hf[key]
            
            # Count frames
            if "camera_0" in grp:
                num_frames = grp["camera_0"].shape[0]
                stats["total_frames"] += num_frames
                stats["trajectory_lengths"].append(num_frames)
                
                # Get dimensions (first trajectory)
                if stats["action_dims"] is None and "actions" in grp:
                    stats["action_dims"] = grp["actions"].shape[1:]
                if stats["state_dims"] is None and "states" in grp:
                    stats["state_dims"] = grp["states"].shape[1:]
                if stats["embedding_dims"] is None and "cam_rs_embd" in grp:
                    stats["embedding_dims"] = grp["cam_rs_embd"].shape[1:]
                if stats["image_shapes"] is None:
                    stats["image_shapes"] = {
                        "camera_0": grp["camera_0"].shape[1:],
                        "camera_1": grp["camera_1"].shape[1:]
                    }
            
            # Track dataset sources
            if "dataset_id" in grp.attrs:
                ds_id = grp.attrs["dataset_id"]
                stats["dataset_sources"][ds_id] = stats["dataset_sources"].get(ds_id, 0) + 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Validate HDF5 dataset created by lerobot_to_hdf5.py"
    )
    parser.add_argument(
        "hdf5_file",
        type=str,
        help="Path to the HDF5 file to validate"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed information for each trajectory"
    )
    parser.add_argument(
        "--check-all-frames",
        action="store_true",
        help="Try to read all frames (slower but more thorough)"
    )
    args = parser.parse_args()
    
    hdf5_path = Path(args.hdf5_file)
    
    # Check if file exists
    if not hdf5_path.exists():
        print(f"❌ Error: File not found: {hdf5_path}")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"Validating HDF5 Dataset: {hdf5_path.name}")
    print(f"{'='*70}\n")
    
    # Open file
    try:
        hf = h5py.File(hdf5_path, "r")
    except Exception as e:
        print(f"❌ Error: Failed to open HDF5 file: {e}")
        sys.exit(1)
    
    # Get statistics
    print("📊 Gathering dataset statistics...")
    try:
        stats = get_dataset_stats(hf)
    except RuntimeError as e:
        if "addr overflow" in str(e) or "Unable to get group info" in str(e):
            print(f"\n❌ CRITICAL: HDF5 file is severely corrupted!")
            print(f"   Error: {e}")
            print(f"\n   The file metadata structure is damaged, likely due to the")
            print(f"   process being killed during writing.")
            print(f"\n   File size: {hdf5_path.stat().st_size / (1024**3):.2f} GB")
            print(f"\n   Unfortunately, this file cannot be recovered.")
            print(f"   You will need to re-run the dataset creation script.")
            hf.close()
            sys.exit(1)
        else:
            raise
    
    print(f"\n{'─'*70}")
    print("DATASET SUMMARY")
    print(f"{'─'*70}")
    print(f"Total Trajectories:  {stats['num_trajectories']}")
    print(f"Total Frames:        {stats['total_frames']}")
    
    if stats['trajectory_lengths']:
        print(f"Frames per trajectory:")
        print(f"  Min:    {min(stats['trajectory_lengths'])}")
        print(f"  Max:    {max(stats['trajectory_lengths'])}")
        print(f"  Mean:   {np.mean(stats['trajectory_lengths']):.1f}")
        print(f"  Median: {np.median(stats['trajectory_lengths']):.1f}")
    
    print(f"\nData Dimensions:")
    print(f"  Actions:     {stats['action_dims']}")
    print(f"  States:      {stats['state_dims']}")
    print(f"  Embeddings:  {stats['embedding_dims']}")
    print(f"  Camera 0:    {stats['image_shapes']['camera_0'] if stats['image_shapes'] else 'N/A'}")
    print(f"  Camera 1:    {stats['image_shapes']['camera_1'] if stats['image_shapes'] else 'N/A'}")
    
    print(f"\nDataset Sources:")
    for ds_id, count in sorted(stats['dataset_sources'].items()):
        print(f"  {ds_id}: {count} trajectories")
    
    # Validate each trajectory
    print(f"\n{'─'*70}")
    print("VALIDATING TRAJECTORIES")
    print(f"{'─'*70}\n")
    
    all_valid = True
    corrupted_trajectories = []
    
    traj_keys = sorted([k for k in hf.keys() if k.startswith("trajectory_")])
    
    for traj_name in tqdm(traj_keys, desc="Validating", ncols=80):
        grp = hf[traj_name]
        is_valid, errors = validate_trajectory(grp, traj_name, verbose=args.verbose)
        
        if not is_valid:
            all_valid = False
            corrupted_trajectories.append((traj_name, errors))
            print(f"\n❌ {traj_name} has errors:")
            for error in errors:
                print(f"   - {error}")
        
        # Optionally check all frames
        if args.check_all_frames and is_valid:
            try:
                for i in range(grp["camera_0"].shape[0]):
                    _ = grp["camera_0"][i]
                    _ = grp["camera_1"][i]
            except Exception as e:
                all_valid = False
                corrupted_trajectories.append((traj_name, [f"Frame read error: {e}"]))
                print(f"\n❌ {traj_name} frame read error: {e}")
    
    # Final report
    print(f"\n{'='*70}")
    print("VALIDATION REPORT")
    print(f"{'='*70}\n")
    
    if all_valid:
        print("✅ All trajectories are valid!")
        print(f"   Successfully validated {len(traj_keys)} trajectories")
        print(f"   Total frames: {stats['total_frames']}")
    else:
        print(f"❌ Found {len(corrupted_trajectories)} corrupted trajectory(ies):")
        for traj_name, errors in corrupted_trajectories:
            print(f"\n  {traj_name}:")
            for error in errors:
                print(f"    - {error}")
    
    # File size
    file_size_gb = hdf5_path.stat().st_size / (1024**3)
    print(f"\nFile Size: {file_size_gb:.2f} GB")
    
    hf.close()
    
    print(f"\n{'='*70}\n")
    
    sys.exit(0 if all_valid else 1)


if __name__ == "__main__":
    main()

