#!/usr/bin/env python3
"""
Create stratified train/eval subsets of the ARX5 dataset for quick upload/testing.

Takes a proportional sample from EACH source dataset to maintain task diversity,
then splits into train (90%) and eval (10%).

HuggingFace Upload:
    hf upload <hf_user>/arx5-robot-dataset arx5_subset_train.h5 --repo-type dataset
    hf upload <hf_user>/arx5-robot-dataset arx5_subset_eval.h5 --repo-type dataset

Usage:
    python scripts/create_subset_dataset.py \
        --input arx5_datasets.h5 \
        --output-prefix arx5_subset_train \
        --num-trajectories 100
        
    # Creates: arx5_subset_train.h5 and arx5_subset_eval.h5
"""

import argparse
import random
from pathlib import Path
from collections import defaultdict

import h5py
from tqdm import tqdm


def get_trajectories_by_dataset(hdf_file: h5py.File):
    """Group trajectories by their source dataset_id."""
    groups = defaultdict(list)
    
    for key in hdf_file.keys():
        if key.startswith("trajectory_"):
            traj = hdf_file[key]
            dataset_id = traj.attrs.get("dataset_id", "unknown")
            groups[dataset_id].append(key)
    
    return dict(groups)


def write_trajectories(src_file: h5py.File, output_path: Path, keys: list, split_name: str):
    """Write selected trajectories to a new HDF5 file."""
    print(f"\n📦 Creating {split_name}: {output_path}")
    
    with h5py.File(output_path, "w", libver='latest') as dst:
        for i, old_key in enumerate(tqdm(keys, desc=f"Copying {split_name}", ncols=80)):
            new_key = f"trajectory_{i}"
            src_file.copy(old_key, dst, name=new_key)
        
        dst.attrs['split'] = split_name
        dst.attrs['num_trajectories'] = len(keys)
        dst.flush()
    
    size_gb = output_path.stat().st_size / (1024**3)
    print(f"   ✓ {output_path.name}: {size_gb:.2f} GB, {len(keys)} trajectories")
    return size_gb


def create_stratified_subset(
    input_path: Path,
    output_prefix: str,
    num_trajectories: int,
    eval_fraction: float = 0.1,
    seed: int = 42
):
    """
    Create stratified train/eval subsets of the dataset.
    
    Takes a proportional sample from each source dataset to maintain diversity,
    then splits into train and eval.
    """
    random.seed(seed)
    
    train_path = Path(f"{output_prefix}_train.h5")
    eval_path = Path(f"{output_prefix}_eval.h5")
    
    print(f"📂 Loading dataset: {input_path}")
    
    with h5py.File(input_path, "r") as src:
        # Group by source dataset
        dataset_groups = get_trajectories_by_dataset(src)
        total_trajectories = sum(len(v) for v in dataset_groups.values())
        
        print(f"   Found {total_trajectories} trajectories across {len(dataset_groups)} source datasets:")
        for ds_id, keys in sorted(dataset_groups.items(), key=lambda x: -len(x[1])):
            print(f"      {ds_id}: {len(keys)} trajectories")
        
        # Determine how many trajectories to select
        all_traj_keys = [k for k in src.keys() if k.startswith("trajectory_")]
        target_traj = min(num_trajectories, len(all_traj_keys))
        print(f"\n📊 Target: {target_traj} trajectories")
        
        print(f"   Split: {(1-eval_fraction)*100:.0f}% train / {eval_fraction*100:.0f}% eval")
        
        # Stratified selection: gather all samples first, then split
        # Use largest remainder method to ensure we hit exact target
        allocations = []
        for ds_id, keys in dataset_groups.items():
            proportion = len(keys) / total_trajectories
            exact = target_traj * proportion
            base = int(exact)
            remainder = exact - base
            allocations.append({
                'ds_id': ds_id,
                'keys': keys,
                'base': max(1, base),  # At least 1 from each
                'remainder': remainder
            })
        
        # Calculate how many more we need to reach target
        total_base = sum(a['base'] for a in allocations)
        remaining = target_traj - total_base
        
        # Distribute remaining slots to datasets with largest remainders
        allocations.sort(key=lambda x: x['remainder'], reverse=True)
        for i in range(max(0, remaining)):
            if i < len(allocations):
                allocations[i]['base'] += 1
        
        # Now sample from each dataset
        all_selected = []
        print(f"\n🎯 Stratified sampling from each source dataset:")
        for alloc in sorted(allocations, key=lambda x: -len(x['keys'])):
            ds_id = alloc['ds_id']
            keys = alloc['keys']
            num_to_take = min(alloc['base'], len(keys))  # Don't exceed available
            
            sampled = random.sample(keys, num_to_take)
            all_selected.extend(sampled)
            
            print(f"   {ds_id}: {num_to_take}/{len(keys)} trajectories")
        
        # Now split ALL selected into train/eval (stratified across the whole selection)
        random.shuffle(all_selected)
        num_eval = max(1, int(len(all_selected) * eval_fraction))
        
        eval_keys = all_selected[:num_eval]
        train_keys = all_selected[num_eval:]
        
        print(f"\n   Total selected: {len(all_selected)} trajectories")
        print(f"   Train: {len(train_keys)} ({len(train_keys)/len(all_selected)*100:.0f}%)")
        print(f"   Eval:  {len(eval_keys)} ({len(eval_keys)/len(all_selected)*100:.0f}%)")
        
        # Write train and eval files
        train_size = write_trajectories(src, train_path, train_keys, "train")
        eval_size = write_trajectories(src, eval_path, eval_keys, "eval")
    
    # Summary
    print(f"\n✅ Created stratified train/eval subsets:")
    print(f"   Train: {train_path} ({train_size:.2f} GB, {len(train_keys)} trajectories)")
    print(f"   Eval:  {eval_path} ({eval_size:.2f} GB, {len(eval_keys)} trajectories)")
    print(f"   Total: {train_size + eval_size:.2f} GB")
    print(f"   All {len(dataset_groups)} source datasets represented in both splits!")
    
    return train_path, eval_path


def main():
    parser = argparse.ArgumentParser(description="Create stratified train/eval subsets of ARX5 dataset")
    parser.add_argument("--input", type=str, required=True, help="Input HDF5 file")
    parser.add_argument("--output-prefix", type=str, default="arx5_subset", help="Output prefix (creates _train.h5 and _eval.h5)")
    parser.add_argument("--num-trajectories", type=int, required=True, help="Number of trajectories to select")
    parser.add_argument("--eval-fraction", type=float, default=0.1, help="Fraction for eval (default: 0.1 = 10%%)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return 1
    
    train_path, eval_path = create_stratified_subset(
        input_path=input_path,
        output_prefix=args.output_prefix,
        num_trajectories=args.num_trajectories,
        eval_fraction=args.eval_fraction,
        seed=args.seed
    )
    
    print(f"\n🚀 To upload:")
    print(f"   hf upload <hf_user>/arx5-robot-dataset {train_path} --repo-type dataset")
    print(f"   hf upload <hf_user>/arx5-robot-dataset {eval_path} --repo-type dataset")
    
    return 0


if __name__ == "__main__":
    exit(main())

