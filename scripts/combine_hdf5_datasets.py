#!/usr/bin/env python3
"""
Combine multiple HDF5 datasets into a single file.
Renumbers trajectories sequentially.

Usage:
    python scripts/combine_hdf5_datasets.py \
        --inputs arx5_datasets.h5 arx5_datasets_remaining.h5 \
        --output arx5_combined.h5
"""

import argparse
import sys
from pathlib import Path
import h5py
from tqdm import tqdm
import shutil

def combine_datasets(input_paths, output_path):
    output_path = Path(output_path)
    
    # Check if output exists
    if output_path.exists():
        print(f"⚠️  Output file {output_path} already exists.")
        response = input("Overwrite? [y/N] ")
        if response.lower() != 'y':
            print("Aborting.")
            sys.exit(1)
        output_path.unlink()

    print(f"Creating merged file: {output_path}")
    
    total_trajectories = 0
    
    # Use libver='latest' for better performance/robustness
    with h5py.File(output_path, 'w', libver='latest') as f_out:
        
        for input_file in input_paths:
            input_file = Path(input_file)
            if not input_file.exists():
                print(f"❌ Input file not found: {input_file}")
                continue
                
            print(f"\nProcessing: {input_file.name}")
            
            try:
                with h5py.File(input_file, 'r') as f_in:
                    # Find all trajectory groups
                    traj_keys = [k for k in f_in.keys() if k.startswith("trajectory_")]
                    
                    # Sort by index to maintain order within the file
                    traj_keys.sort(key=lambda k: int(k.split('_')[1]))
                    
                    print(f"  Found {len(traj_keys)} trajectories")
                    
                    for old_key in tqdm(traj_keys, desc="  Copying"):
                        new_key = f"trajectory_{total_trajectories}"
                        
                        # Efficient copy from source file to dest file
                        # This copies the Group and all its contents (datasets + attributes)
                        f_out.copy(f_in[old_key], new_key)
                        
                        # Double check attributes are there (copy should handle it, but good to verify logic if debugging)
                        # attrs are copied by default with group copy
                        
                        total_trajectories += 1
                        
            except Exception as e:
                print(f"❌ Error reading {input_file}: {e}")
                sys.exit(1)

    print(f"\n{'='*50}")
    print(f"✅ Successfully created {output_path.name}")
    print(f"   Total trajectories: {total_trajectories}")
    print(f"   File size: {output_path.stat().st_size / (1024**3):.2f} GB")
    print(f"{'='*50}\n")

def main():
    parser = argparse.ArgumentParser(description="Combine multiple HDF5 datasets")
    parser.add_argument("--inputs", nargs='+', required=True, help="List of input .h5 files")
    parser.add_argument("--output", required=True, help="Output .h5 file path")
    
    args = parser.parse_args()
    
    combine_datasets(args.inputs, args.output)

if __name__ == "__main__":
    main()

