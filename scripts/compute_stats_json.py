#!/usr/bin/env python3
"""
Compute statistics (min, max, mean, std) for actions and states in an HDF5 dataset
and save them to a JSON file.

QUICKSTART:
    python scripts/compute_stats_json.py --file arx5_datasets.h5 --output dataset_stats.json
"""

import argparse
import json

import h5py
import numpy as np
from tqdm import tqdm

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def _concat_or_none(chunks):
    if not chunks:
        return None
    if len(chunks) == 1:
        return chunks[0]
    return np.concatenate(chunks, axis=0)


def _approx_quantiles_hist(chunks, q_low, q_high, min_vals, max_vals, bins=2048):
    dims = min_vals.shape[0]
    q_low_vals = np.zeros(dims)
    q_high_vals = np.zeros(dims)
    total_counts = np.zeros(dims, dtype=np.int64)
    hist_counts = np.zeros((dims, bins), dtype=np.int64)

    for chunk in chunks:
        for dim in range(dims):
            hist, _ = np.histogram(
                chunk[:, dim], bins=bins, range=(min_vals[dim], max_vals[dim])
            )
            hist_counts[dim] += hist
            total_counts[dim] += chunk.shape[0]

    for dim in range(dims):
        if total_counts[dim] == 0:
            q_low_vals[dim] = min_vals[dim]
            q_high_vals[dim] = max_vals[dim]
            continue
        edges = np.linspace(min_vals[dim], max_vals[dim], bins + 1)
        cdf = np.cumsum(hist_counts[dim])
        low_idx = np.searchsorted(cdf, q_low * total_counts[dim], side="left")
        high_idx = np.searchsorted(cdf, q_high * total_counts[dim], side="left")
        low_idx = min(max(low_idx, 0), bins - 1)
        high_idx = min(max(high_idx, 0), bins - 1)
        q_low_vals[dim] = edges[low_idx]
        q_high_vals[dim] = edges[high_idx + 1]
    return q_low_vals, q_high_vals


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
    action_chunks = []

    state_sum = None
    state_sq_sum = None
    state_count = 0
    state_min = None
    state_max = None
    state_chunks = []
    
    with h5py.File(hdf5_path, 'r') as f:
        keys = [k for k in f.keys() if k.startswith('trajectory_')]
        
        for key in tqdm(keys, desc="Scanning Dataset"):
            # --- Actions (prefer actions_delta) ---
            if 'actions_delta' in f[key]:
                acs = f[key]['actions_delta'][:]
            else:
                if 'actions' not in f[key]:
                    raise KeyError(f"Missing actions/actions_delta in {key}")
                print("⚠️  Warning: actions_delta missing; falling back to actions.")
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
            action_chunks.append(acs)
            
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
                state_chunks.append(sts)

    # Final Calculations
    stats = {}
    
    # Actions (actions_delta)
    if action_count > 0:
        stats["action_min"] = action_min
        stats["action_max"] = action_max
        stats["action_mean"] = action_sum / action_count
        stats["action_std"] = np.sqrt((action_sq_sum / action_count) - (stats["action_mean"]**2))
        try:
            action_all = _concat_or_none(action_chunks)
            q02, q98 = np.quantile(action_all, [0.02, 0.98], axis=0)
        except MemoryError:
            q02, q98 = _approx_quantiles_hist(
                action_chunks, 0.02, 0.98, action_min, action_max
            )
        stats["action_delta_q02"] = q02
        stats["action_delta_q98"] = q98
        
    # States
    if state_count > 0:
        stats["state_min"] = state_min
        stats["state_max"] = state_max
        stats["state_mean"] = state_sum / state_count
        stats["state_std"] = np.sqrt((state_sq_sum / state_count) - (stats["state_mean"]**2))
        try:
            state_all = _concat_or_none(state_chunks)
            q02, q98 = np.quantile(state_all, [0.02, 0.98], axis=0)
        except MemoryError:
            q02, q98 = _approx_quantiles_hist(
                state_chunks, 0.02, 0.98, state_min, state_max
            )
        stats["state_q02"] = q02
        stats["state_q98"] = q98

    print("\n✅ Stats computed!")
    if "action_min" in stats:
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
