#!/usr/bin/env python3
"""
Upload a large file to a Hugging Face Hub dataset repository.

Usage:
    python scripts/upload_to_hf.py \
        --file /path/to/arx5_datasets_combined.h5 \
        --repo-id pravsels/arx5-robot-dataset \
        --repo-type dataset \
        --path-in-repo arx5_datasets_combined.h5
"""

import argparse
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

def upload_file(file_path, repo_id, repo_type, path_in_repo, token=None):
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    api = HfApi(token=token)

    print(f"🚀 Preparing to upload {file_path.name} ({file_path.stat().st_size / 1024**3:.2f} GB)")
    print(f"   Target: https://huggingface.co/datasets/{repo_id}/blob/main/{path_in_repo}")

    # 1. Create Repo if it doesn't exist
    try:
        create_repo(repo_id, repo_type=repo_type, exist_ok=True, token=token, private=True)
        print(f"✅ Repository {repo_id} checked/created.")
    except Exception as e:
        print(f"⚠️  Could not check/create repo (might already exist or auth issue): {e}")

    # 2. Upload
    try:
        print(f"   Starting upload...")
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=f"Upload {file_path.name}",
        )
        print(f"\n✅ Upload complete!")
        print(f"   URL: https://huggingface.co/datasets/{repo_id}/blob/main/{path_in_repo}")
    
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Upload file to Hugging Face Hub")
    parser.add_argument("--file", required=True, help="Local path to the file")
    parser.add_argument("--repo-id", required=True, help="Target HF Repo ID (e.g. user/repo)")
    parser.add_argument("--repo-type", default="dataset", choices=["model", "dataset", "space"], help="Repo type")
    parser.add_argument("--path-in-repo", default=None, help="Filename in the repo (defaults to local filename)")
    parser.add_argument("--token", default=None, help="HF Write Token (optional if logged in via cli)")

    args = parser.parse_args()

    path_in_repo = args.path_in_repo if args.path_in_repo else Path(args.file).name
    
    upload_file(args.file, args.repo_id, args.repo_type, path_in_repo, args.token)

if __name__ == "__main__":
    main()

