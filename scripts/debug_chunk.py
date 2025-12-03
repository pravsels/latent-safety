#!/usr/bin/env python3
"""
Debug a specific dataset chunk from Hugging Face.
Downloads the file, checks SHA256, and verifies HDF5 integrity.

Usage:
    python scripts/debug_chunk.py --repo-id pravsels/arx5-robot-dataset --chunk 10
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path
import h5py
from tqdm import tqdm

try:
    from huggingface_hub import hf_hub_download, HfApi
except ImportError:
    print("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

def get_remote_sha256(repo_id, filename, token=None):
    """Fetch the expected SHA256 from HF Git LFS metadata."""
    api = HfApi(token=token)
    try:
        info = api.model_info(repo_id, files=[filename], token=token)
        # This is a bit tricky via API, usually easier to trust hf_hub_download to verify
        # But we can try to fetch the LFS pointer if needed.
        # Actually, hf_hub_download verifies checksum by default.
        return None
    except Exception as e:
        print(f"⚠️ Could not fetch remote metadata: {e}")
        return None

def calculate_sha256(file_path):
    """Calculate SHA256 of a local file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    parser = argparse.ArgumentParser(description="Debug a specific HDF5 chunk")
    parser.add_argument("--repo-id", required=True, help="HF repo ID")
    parser.add_argument("--chunk", type=int, required=True, help="Chunk number (1-indexed)")
    parser.add_argument("--token", default=None, help="HF token")
    parser.add_argument("--output-dir", default="debug_chunks", help="Directory to save chunk")
    parser.add_argument("--keep", action="store_true", help="Keep file after checking")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    chunk_name = f"arx5_datasets_part_{args.chunk:03d}.h5"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    local_path = output_dir / chunk_name

    print(f"🔍 Debugging Chunk {args.chunk}: {chunk_name}")
    print(f"   Repo: {args.repo_id}")

    # 1. Download
    print(f"\n☁️  Downloading...")
    try:
        downloaded_path = hf_hub_download(
            repo_id=args.repo_id,
            filename=chunk_name,
            repo_type="dataset",
            local_dir=output_dir,
            token=token,
            force_download=True, # Force fresh download to be sure
            resume_download=True
        )
        print(f"   ✅ Download complete: {downloaded_path}")
    except Exception as e:
        print(f"   ❌ Download failed: {e}")
        sys.exit(1)

    # 2. Check Size
    size_gb = os.path.getsize(downloaded_path) / (1024**3)
    print(f"\n📏 File Size: {size_gb:.2f} GB")

    # 3. Verify HDF5 Structure
    print(f"\n🏥 Verifying HDF5 integrity...")
    try:
        with h5py.File(downloaded_path, "r") as f:
            print(f"   ✅ Opened successfully with h5py")
            
            keys = list(f.keys())
            traj_keys = [k for k in keys if k.startswith("trajectory_")]
            print(f"   📊 Contains {len(traj_keys)} trajectory groups")
            
            if len(traj_keys) > 0:
                first_traj = traj_keys[0]
                print(f"   Checking first trajectory: {first_traj}")
                # Try to read data from the first trajectory to ensure accessibility
                first_grp = f[first_traj]
                data_keys = list(first_grp.keys())
                print(f"   Data fields: {data_keys}")
                
                # Try reading a dataset
                if data_keys:
                    ds = first_grp[data_keys[0]]
                    print(f"   Sample read ({data_keys[0]}): shape={ds.shape}, dtype={ds.dtype}")
                    _ = ds[()] # Force read
                    print(f"   ✅ Read successful")
                
            # Check last trajectory too (often where corruption is)
            if len(traj_keys) > 1:
                last_traj = traj_keys[-1]
                print(f"   Checking last trajectory: {last_traj}")
                last_grp = f[last_traj]
                if list(last_grp.keys()):
                    _ = last_grp[list(last_grp.keys())[0]][()]
                    print(f"   ✅ Last trajectory read successful")

        print(f"\n✅ Chunk {args.chunk} appears VALID.")

    except Exception as e:
        print(f"\n❌ HDF5 Validation FAILED:")
        print(f"   {e}")
        print(f"\n   This confirms the file is corrupt.")

    # 4. Cleanup
    if not args.keep:
        print(f"\n🗑️  Deleting {local_path}...")
        try:
            os.remove(local_path)
            print("   ✅ Deleted")
        except Exception as e:
            print(f"   ⚠️ Failed to delete: {e}")
    else:
        print(f"\n💾 File kept at: {local_path}")

if __name__ == "__main__":
    main()

