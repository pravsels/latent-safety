#!/usr/bin/env python3
"""
Chunked upload of a large HDF5 dataset to Hugging Face Hub.

Creates temporary chunks, uploads them, then deletes the local chunk to save disk space.

Usage:
    python scripts/chunked_upload_to_hf.py \
        --input /path/to/arx5_datasets_combined.h5 \
        --repo-id pravsels/arx5-robot-dataset \
        --trajectories-per-chunk 500

Environment:
    HF_TOKEN: Your Hugging Face write token (or use --token)
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
from tqdm import tqdm

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)


def get_trajectory_count(hdf5_path):
    """Count total trajectories in the HDF5 file."""
    with h5py.File(hdf5_path, "r") as f:
        traj_keys = [k for k in f.keys() if k.startswith("trajectory_")]
    return len(traj_keys)


def create_chunk(input_path, output_path, start_idx, end_idx):
    """
    Create a chunk HDF5 file containing trajectories from start_idx to end_idx (exclusive).
    Trajectories are renumbered starting from 0 in the chunk.
    """
    with h5py.File(input_path, "r") as f_in:
        with h5py.File(output_path, "w", libver="latest") as f_out:
            chunk_idx = 0
            for traj_idx in range(start_idx, end_idx):
                src_key = f"trajectory_{traj_idx}"
                if src_key not in f_in:
                    continue
                dst_key = f"trajectory_{chunk_idx}"
                f_out.copy(f_in[src_key], dst_key)
                chunk_idx += 1
            return chunk_idx


def upload_file(file_path, repo_id, path_in_repo, token):
    """Upload a single file to HF Hub."""
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Upload {path_in_repo}",
    )


def main():
    parser = argparse.ArgumentParser(description="Chunked upload of HDF5 to Hugging Face")
    parser.add_argument("--input", required=True, help="Path to the large HDF5 file")
    parser.add_argument("--repo-id", required=True, help="HF repo ID (e.g., user/dataset)")
    parser.add_argument("--trajectories-per-chunk", type=int, default=500, help="Number of trajectories per chunk")
    parser.add_argument("--token", default=None, help="HF token (or set HF_TOKEN env var)")
    parser.add_argument("--start-chunk", type=int, default=0, help="Resume from this chunk number (0-indexed)")
    parser.add_argument("--temp-dir", default=None, help="Directory for temp chunks (default: same as input)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)

    # Get token
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("❌ No HF token provided. Use --token or set HF_TOKEN environment variable.")
        sys.exit(1)

    # Setup temp directory
    temp_dir = Path(args.temp_dir) if args.temp_dir else input_path.parent
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Count trajectories
    print(f"📂 Scanning {input_path.name}...")
    total_trajectories = get_trajectory_count(input_path)
    print(f"   Found {total_trajectories} trajectories")

    chunk_size = args.trajectories_per_chunk
    num_chunks = (total_trajectories + chunk_size - 1) // chunk_size
    print(f"   Will create {num_chunks} chunks of up to {chunk_size} trajectories each")

    # Create/check repo
    print(f"\n🔗 Checking repository: {args.repo_id}")
    try:
        create_repo(args.repo_id, repo_type="dataset", exist_ok=True, token=token, private=True)
        print(f"   ✅ Repository ready")
    except Exception as e:
        print(f"   ⚠️  Repo check warning: {e}")

    # Process chunks
    print(f"\n🚀 Starting chunked upload...\n")
    
    for chunk_num in range(args.start_chunk, num_chunks):
        start_idx = chunk_num * chunk_size
        end_idx = min(start_idx + chunk_size, total_trajectories)
        
        chunk_name = f"arx5_datasets_part_{chunk_num + 1:03d}.h5"
        chunk_path = temp_dir / chunk_name
        
        print(f"{'='*60}")
        print(f"Chunk {chunk_num + 1}/{num_chunks}: trajectories {start_idx}-{end_idx - 1}")
        print(f"{'='*60}")
        
        # Step 1: Create chunk
        print(f"   📦 Creating {chunk_name}...")
        try:
            num_copied = create_chunk(input_path, chunk_path, start_idx, end_idx)
            chunk_size_gb = chunk_path.stat().st_size / (1024**3)
            print(f"   ✅ Created: {num_copied} trajectories, {chunk_size_gb:.2f} GB")
        except Exception as e:
            print(f"   ❌ Failed to create chunk: {e}")
            continue
        
        # Step 2: Upload chunk
        print(f"   ☁️  Uploading to HF...")
        try:
            upload_file(chunk_path, args.repo_id, chunk_name, token)
            print(f"   ✅ Uploaded successfully")
        except Exception as e:
            print(f"   ❌ Upload failed: {e}")
            print(f"   ⚠️  Keeping chunk file for retry. Resume with --start-chunk {chunk_num}")
            sys.exit(1)
        
        # Step 3: Delete local chunk
        print(f"   🗑️  Deleting local chunk...")
        try:
            chunk_path.unlink()
            print(f"   ✅ Deleted")
        except Exception as e:
            print(f"   ⚠️  Could not delete: {e}")
        
        print()

    print(f"{'='*60}")
    print(f"🎉 All {num_chunks} chunks uploaded successfully!")
    print(f"   Repository: https://huggingface.co/datasets/{args.repo_id}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

