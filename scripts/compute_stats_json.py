#!/usr/bin/env python3
"""
Compute statistics (min, max, mean, std) for actions and states in an HDF5 dataset
and save them to a JSON file.

QUICKSTART:
    python scripts/compute_stats_json.py --file arx5_datasets.h5 --output dataset_stats.json
"""

import h5py
import numpy as np
import json
import argparse
from tqdm import tqdm

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def compute_stats(hdf5_path, output_json):
    print(f"Processing {hdf5_path}...")
    
    # Initialize accumulators for Welford's algorithm or simple sum
    # Since we have enough RAM to hold sums, we can do:
    # Mean = Sum / Count
    # Std = Sqrt( (SumSq / Count) - Mean^2 )
    
    action_sum = None
    action_sq_sum = None
    action_count = 0
    action_min = None
    action_max = None

    state_sum = None
    state_sq_sum = None
    state_count = 0
    state_min = None
    state_max = None
    
    with h5py.File(hdf5_path, 'r') as f:
        keys = [k for k in f.keys() if k.startswith('trajectory_')]
        
        for key in tqdm(keys, desc="Scanning Dataset"):
            # --- Actions ---
            acs = f[key]['actions'][:]
            # Flatten time dimension for stats: (T, D) -> D
            # Actually, we want stats per dimension across all time steps
            
            n_frames = acs.shape[0]
            
            if action_min is None:
                # Initialize shapes based on data
                dims = acs.shape[1]
                action_sum = np.zeros(dims)
                action_sq_sum = np.zeros(dims)
                action_min = np.full(dims, np.inf)
                action_max = np.full(dims, -np.inf)
            
            # Update Min/Max
            action_min = np.minimum(action_min, np.min(acs, axis=0))
            action_max = np.maximum(action_max, np.max(acs, axis=0))
            
            # Update Sums for Mean/Std
            action_sum += np.sum(acs, axis=0)
            action_sq_sum += np.sum(acs**2, axis=0)
            action_count += n_frames
            
            # --- States ---
            if 'states' in f[key]:
                sts = f[key]['states'][:]
                n_frames_st = sts.shape[0]
                
                if state_min is None:
                    dims = sts.shape[1]
                    state_sum = np.zeros(dims)
                    state_sq_sum = np.zeros(dims)
                    state_min = np.full(dims, np.inf)
                    state_max = np.full(dims, -np.inf)
                
                state_min = np.minimum(state_min, np.min(sts, axis=0))
                state_max = np.maximum(state_max, np.max(sts, axis=0))
                
                state_sum += np.sum(sts, axis=0)
                state_sq_sum += np.sum(sts**2, axis=0)
                state_count += n_frames_st

    # Final Calculations
    stats = {}
    
    # Actions
    if action_count > 0:
        stats["action_min"] = action_min
        stats["action_max"] = action_max
        stats["action_mean"] = action_sum / action_count
        stats["action_std"] = np.sqrt((action_sq_sum / action_count) - (stats["action_mean"]**2))
        
    # States
    if state_count > 0:
        stats["state_min"] = state_min
        stats["state_max"] = state_max
        stats["state_mean"] = state_sum / state_count
        stats["state_std"] = np.sqrt((state_sq_sum / state_count) - (stats["state_mean"]**2))

    print("\n✅ Stats computed!")
    print(f"Action Min: {stats['action_min']}")
    print(f"Action Max: {stats['action_max']}")
    
    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump(stats, f, cls=NumpyEncoder, indent=4)
            
    print(f"📄 Saved stats to {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to HDF5 file")
    parser.add_argument("--output", default="dataset_stats.json", help="Output JSON file")
    args = parser.parse_args()
    
    compute_stats(args.file, args.output)
