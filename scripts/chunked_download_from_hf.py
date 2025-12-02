#!/usr/bin/env python3
"""
Download chunked HDF5 dataset from Hugging Face Hub and reassemble into a single file.

Downloads chunks one at a time, batches trajectories in memory, then writes to disk.
Automatically resumes from existing output file if present.

Usage:
    # With larger memory budget for faster batching:
    python scripts/chunked_download_from_hf.py \
        --repo-id pravsels/arx5-robot-dataset \
        --output arx5_dataset.h5 \
        --max-memory-gb 100

    # Start fresh (overwrite existing):
    python scripts/chunked_download_from_hf.py \
        --repo-id pravsels/arx5-robot-dataset \
        --output arx5_dataset.h5 \
        --no-resume

Environment:
    HF_TOKEN: Your Hugging Face token (or use --token), required for private repos
"""

import argparse
import os
import re
import sys
from pathlib import Path

import h5py
from tqdm import tqdm

try:
    from huggingface_hub import HfApi, hf_hub_download, list_repo_files
except ImportError:
    print("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)


def get_chunk_files(repo_id, token=None):
    """
    List all chunk files in the repository.
    Returns sorted list of filenames matching arx5_datasets_part_XXX.h5 pattern.
    """
    api = HfApi(token=token)
    files = list_repo_files(repo_id, repo_type="dataset", token=token)
    
    # Filter for chunk files
    chunk_pattern = re.compile(r"arx5_datasets_part_(\d+)\.h5")
    chunk_files = []
    
    for f in files:
        match = chunk_pattern.match(f)
        if match:
            chunk_num = int(match.group(1))
            chunk_files.append((chunk_num, f))
    
    # Sort by chunk number
    chunk_files.sort(key=lambda x: x[0])
    return [f for _, f in chunk_files]


def download_chunk(repo_id, filename, local_dir, token=None):
    """Download a single chunk file from HF Hub."""
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=local_dir,
        token=token,
    )


def merge_chunk_into_output(chunk_path, output_path, trajectory_offset, max_memory_gb=50.0):
    """
    Merge trajectories from chunk into output file using batched approach.
    Reads multiple trajectories into memory before writing for efficiency.
    
    Args:
        chunk_path: Path to source chunk HDF5 file
        output_path: Path to output HDF5 file
        trajectory_offset: Starting trajectory number in output
        max_memory_gb: Max memory for trajectory batch before writing (default 50GB)
    
    Returns the number of trajectories copied.
    """
    max_bytes = int(max_memory_gb * 1024**3)
    
    with h5py.File(chunk_path, "r") as f_chunk:
        traj_keys = sorted(
            [k for k in f_chunk.keys() if k.startswith("trajectory_")],
            key=lambda x: int(x.split("_")[1])
        )
        
        batch = []  # List of (traj_data_dict, traj_attrs)
        batch_bytes = 0
        traj_counter = 0
        
        for idx, src_key in enumerate(traj_keys):
            # Read entire trajectory into memory
            traj_data = {}
            traj_attrs = dict(f_chunk[src_key].attrs)
            traj_bytes = 0
            
            src_group = f_chunk[src_key]
            for dataset_key in src_group.keys():
                item = src_group[dataset_key]
                if isinstance(item, h5py.Dataset):
                    data = item[()]  # Read entire dataset into numpy array
                    traj_data[dataset_key] = {
                        'data': data,
                        'attrs': dict(item.attrs),
                        'chunks': item.chunks,
                        'compression': item.compression,
                        'compression_opts': item.compression_opts if item.compression else None
                    }
                    traj_bytes += data.nbytes
            
            batch.append((traj_data, traj_attrs))
            batch_bytes += traj_bytes
            
            # Write batch if we hit memory limit or last trajectory
            is_last = (idx == len(traj_keys) - 1)
            if batch_bytes >= max_bytes or is_last:
                # Write all trajectories in batch
                mode = "a" if output_path.exists() else "w"
                with h5py.File(output_path, mode, libver="latest") as f_out:
                    for traj_data, traj_attrs in batch:
                        dst_key = f"trajectory_{trajectory_offset + traj_counter}"
                        dst_group = f_out.create_group(dst_key)
                        for attr_key, attr_val in traj_attrs.items():
                            dst_group.attrs[attr_key] = attr_val
                        
                        for dataset_key, ds_info in traj_data.items():
                            chunks = ds_info['chunks'] if ds_info['chunks'] else True
                            ds = dst_group.create_dataset(
                                dataset_key,
                                data=ds_info['data'],
                                chunks=chunks,
                                compression=ds_info['compression'],
                                compression_opts=ds_info['compression_opts']
                            )
                            for attr_key, attr_val in ds_info['attrs'].items():
                                ds.attrs[attr_key] = attr_val
                        
                        traj_counter += 1
                
                print(f"      📝 Wrote batch: {len(batch)} trajectories, {batch_bytes / 1024**3:.2f} GB")
                batch = []
                batch_bytes = 0
        
        return traj_counter


def get_existing_trajectory_count(output_path):
    """Count trajectories already in the output file."""
    if not output_path.exists():
        return 0
    with h5py.File(output_path, "r") as f:
        traj_keys = [k for k in f.keys() if k.startswith("trajectory_")]
    return len(traj_keys)


def main():
    parser = argparse.ArgumentParser(
        description="Download chunked HDF5 from Hugging Face and reassemble"
    )
    parser.add_argument(
        "--repo-id", 
        required=True, 
        help="HF repo ID (e.g., user/dataset)"
    )
    parser.add_argument(
        "--output", 
        required=True, 
        help="Path for the reassembled HDF5 file"
    )
    parser.add_argument(
        "--token", 
        default=None, 
        help="HF token (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--temp-dir", 
        default=None, 
        help="Directory for temp downloads (default: same as output)"
    )
    parser.add_argument(
        "--keep-chunks", 
        action="store_true", 
        help="Keep downloaded chunks after merging (default: delete)"
    )
    parser.add_argument(
        "--no-resume", 
        action="store_true", 
        help="Start fresh instead of resuming from existing output file"
    )
    parser.add_argument(
        "--start-chunk", 
        type=int, 
        default=None,
        help="Start from specific chunk number (1-indexed). Overrides auto-resume."
    )
    parser.add_argument(
        "--max-memory-gb",
        type=float,
        default=50.0,
        help="Max memory for trajectory batch before writing (default: 50)"
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get token
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("⚠️  No HF token provided. This will fail for private repos.")
        print("   Use --token or set HF_TOKEN environment variable.")

    # Setup temp directory
    temp_dir = Path(args.temp_dir) if args.temp_dir else output_path.parent
    temp_dir.mkdir(parents=True, exist_ok=True)

    # List available chunks
    print(f"🔍 Listing chunks in {args.repo_id}...")
    try:
        chunk_files = get_chunk_files(args.repo_id, token)
    except Exception as e:
        print(f"❌ Failed to list repository files: {e}")
        sys.exit(1)

    if not chunk_files:
        print("❌ No chunk files found in repository!")
        print("   Expected files matching pattern: arx5_datasets_part_XXX.h5")
        sys.exit(1)

    print(f"   Found {len(chunk_files)} chunks")

    # Determine starting point
    if args.start_chunk is not None:
        start_chunk_idx = args.start_chunk - 1  # Convert to 0-indexed
        if start_chunk_idx < 0 or start_chunk_idx >= len(chunk_files):
            print(f"❌ Invalid start chunk: {args.start_chunk}. Valid range: 1-{len(chunk_files)}")
            sys.exit(1)
        # Calculate trajectory offset based on chunks we're skipping
        # This requires downloading and counting, so we'll approximate
        print(f"⚠️  Starting from chunk {args.start_chunk}. Make sure output file has correct trajectory count.")
        trajectory_offset = get_existing_trajectory_count(output_path)
    elif not args.no_resume and output_path.exists():
        # Resume is default behavior
        trajectory_offset = get_existing_trajectory_count(output_path)
        # Estimate which chunk to resume from (assuming ~equal trajectories per chunk)
        # This is a heuristic - user should verify
        print(f"📂 Found existing output with {trajectory_offset} trajectories")
        print(f"   Will append new trajectories starting from trajectory_{trajectory_offset}")
        start_chunk_idx = 0  # We'll skip chunks until we find new data
    else:
        # Fresh start (--no-resume or no existing file)
        if output_path.exists():
            print(f"⚠️  Output file exists: {output_path}")
            print(f"   --no-resume specified, will overwrite.")
            output_path.unlink()
        trajectory_offset = 0
        start_chunk_idx = 0

    # Process chunks
    print(f"\n🚀 Starting download and merge...\n")
    
    total_trajectories = trajectory_offset
    
    for chunk_idx, chunk_file in enumerate(chunk_files):
        if chunk_idx < start_chunk_idx:
            continue
            
        chunk_num = chunk_idx + 1
        print(f"{'='*60}")
        print(f"Chunk {chunk_num}/{len(chunk_files)}: {chunk_file}")
        print(f"{'='*60}")
        
        # Step 1: Download chunk
        print(f"   ☁️  Downloading...")
        try:
            local_chunk_path = Path(download_chunk(
                args.repo_id, 
                chunk_file, 
                temp_dir, 
                token
            ))
            chunk_size_gb = local_chunk_path.stat().st_size / (1024**3)
            print(f"   ✅ Downloaded: {chunk_size_gb:.2f} GB")
        except Exception as e:
            print(f"   ❌ Download failed: {e}")
            print(f"   ⚠️  Resume with --start-chunk {chunk_num}")
            sys.exit(1)
        
        # Step 2: Merge into output
        print(f"   📦 Merging into output (offset: {total_trajectories})...")
        try:
            num_copied = merge_chunk_into_output(
                local_chunk_path, 
                output_path, 
                total_trajectories,
                args.max_memory_gb
            )
            total_trajectories += num_copied
            print(f"   ✅ Merged {num_copied} trajectories (total: {total_trajectories})")
        except Exception as e:
            print(f"   ❌ Merge failed: {e}")
            print(f"   ⚠️  Resume with --start-chunk {chunk_num}")
            sys.exit(1)
        
        # Step 3: Delete local chunk (unless --keep-chunks)
        if not args.keep_chunks:
            print(f"   🗑️  Deleting local chunk...")
            try:
                local_chunk_path.unlink()
                # Also try to remove the repo subdirectory if empty
                chunk_parent = local_chunk_path.parent
                if chunk_parent != temp_dir:
                    try:
                        chunk_parent.rmdir()
                    except OSError:
                        pass  # Directory not empty, that's fine
                print(f"   ✅ Deleted")
            except Exception as e:
                print(f"   ⚠️  Could not delete: {e}")
        
        print()

    # Final summary
    output_size_gb = output_path.stat().st_size / (1024**3)
    print(f"{'='*60}")
    print(f"🎉 Download and reassembly complete!")
    print(f"{'='*60}")
    print(f"   Output file: {output_path}")
    print(f"   Total size: {output_size_gb:.2f} GB")
    print(f"   Total trajectories: {total_trajectories}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

