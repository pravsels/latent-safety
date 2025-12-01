#!/usr/bin/env python3
"""
Convert Hugging Face LeRobot datasets into a consolidated HDF5 file for DINO-WM.

QUICKSTART:
    # Fresh start
    python scripts/lerobot_to_hdf5.py \
        --datasets-list arx5_datasets.json \
        --output-hdf5 arx5_datasets.h5 \
        --batch-size 256 --resume

CRASH-RESISTANT VERSION with:
- Resume capability (can continue from interruptions)
- Flush after EVERY trajectory (protects against abrupt kills/OOM/crashes)
- Signal handling for graceful shutdown (Ctrl+C)
- Better progress tracking and validation
- Uses HDF5 libver='latest' for crash resistance

This script is designed to handle:
- Graceful interrupts (Ctrl+C, SIGTERM)
- Abrupt kills (kill -9, OOM killer, system crashes)
- Power failures (data is safe after each flush)
"""

import argparse
import json
import shutil
import signal
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Set

import h5py
import numpy as np
import torch
from tqdm import tqdm

# Import shared utilities
try:
    from scripts.utils import get_dino_model, preprocess_images_for_dino, to_hwc_uint8
    import scripts.utils
except ImportError:
    from utils import get_dino_model, preprocess_images_for_dino, to_hwc_uint8
    import utils 
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Global flag for graceful shutdown
SHUTDOWN_REQUESTED = False

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    global SHUTDOWN_REQUESTED
    print("\n\n⚠️  Interrupt received! Finishing current trajectory and shutting down gracefully...")
    print("   (Press Ctrl+C again to force quit, but this may corrupt the file)")
    SHUTDOWN_REQUESTED = True
    # Restore default handler so second Ctrl+C will force quit
    signal.signal(signal.SIGINT, signal.SIG_DFL)

# --- Worker Process ---

def data_generator(
    dataset_id: str,
    episode_indices: List[int],
    min_length: int,
    batch_size: int
):
    """
    Generator that yields (ep_idx, data_dict) or (ep_idx, None) sequentially.
    """
    try:
        dataset = LeRobotDataset(dataset_id, video_backend="pyav")
        
        for ep_idx in episode_indices:
            if SHUTDOWN_REQUESTED:
                break
                
            try:
                # Get episode range
                start_idx = dataset.episode_data_index["from"][ep_idx].item()
                end_idx = dataset.episode_data_index["to"][ep_idx].item()
                total_frames = end_idx - start_idx
                
                if total_frames < min_length:
                    yield (ep_idx, None)
                    continue
                
                # Package metadata - pass dataset ref instead of loading full batch
                data = {
                    "dataset": dataset,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "total_frames": total_frames,
                }
                yield (ep_idx, data)
                
            except Exception as e:
                print(f"⚠️ Error processing episode {ep_idx}: {e}")
                yield (ep_idx, str(e))
                
    except Exception as e:
        print(f"❌ Dataset Init Error: {e}")
        yield (-1, f"Dataset Init Error: {e}")

# --- Main Process ---

def get_processed_episodes(hdf_file: h5py.File, dataset_id: str) -> Set[int]:
    """
    Get set of episode indices already processed for a given dataset.
    """
    processed = set()
    for key in hdf_file.keys():
        if key.startswith("trajectory_"):
            grp = hdf_file[key]
            if "dataset_id" in grp.attrs and "original_episode_index" in grp.attrs:
                if grp.attrs["dataset_id"] == dataset_id:
                    processed.add(int(grp.attrs["original_episode_index"]))
    return processed

def get_next_trajectory_index(hdf_file: h5py.File) -> int:
    """
    Find the next available trajectory index.
    """
    max_idx = -1
    for key in hdf_file.keys():
        if key.startswith("trajectory_"):
            try:
                idx = int(key.split("_")[1])
                max_idx = max(max_idx, idx)
            except (ValueError, IndexError):
                pass
    return max_idx + 1

def process_dataset(
    dataset_id: str,
    hdf_file: h5py.File,
    dino_model: torch.nn.Module,
    device: str,
    batch_size: int,
    max_episodes: Optional[int],
    start_traj_idx: int,
    min_length: int,
    resume: bool = False
) -> int:
    
    print(f"\nLoading dataset metadata: {dataset_id}")
    try:
        dataset = LeRobotDataset(dataset_id, video_backend="pyav")
    except Exception as e:
        print(f"❌ Failed to init dataset {dataset_id}: {e}")
        return start_traj_idx

    num_episodes = dataset.num_episodes
    if max_episodes is not None:
        num_episodes = min(num_episodes, max_episodes)
    
    # Check for already processed episodes if resuming
    processed_episodes = set()
    if resume:
        processed_episodes = get_processed_episodes(hdf_file, dataset_id)
        if processed_episodes:
            print(f"📁 Found {len(processed_episodes)} already processed episodes, skipping them...")
    
    episodes_to_process = [i for i in range(num_episodes) if i not in processed_episodes]
    
    if not episodes_to_process:
        print(f"✅ All episodes already processed for {dataset_id}")
        return start_traj_idx
    
    print(f"Processing {len(episodes_to_process)} episodes (skipping {len(processed_episodes)})...")
    
    pbar = tqdm(total=len(episodes_to_process), desc=f"Encoding {dataset_id}", ncols=80)
    
    # Create generator
    data_iter = data_generator(dataset_id, episodes_to_process, min_length, batch_size)
    
    trajectories_saved = 0
    
    for item in data_iter:
        if SHUTDOWN_REQUESTED:
            print("\n🛑 Shutdown requested, stopping dataset processing...")
            break
            
        if item is None:
            continue
            
        ep_idx, payload = item
        
        if ep_idx == -1:
            print(f"❌ Worker Error: {payload}")
            continue
            
        if isinstance(payload, str):
            print(f"⚠️ Failed to process episode {ep_idx}: {payload}")
            pbar.update(1)
            continue
            
        if payload is None:
            pbar.update(1)
            continue
            
        # Success - Process on GPU in CHUNKS
        # Success - Process on GPU in CHUNKS
        try:
            start_idx = payload["start_idx"]
            end_idx = payload["end_idx"]
            total_frames = payload["total_frames"]
            dataset_ref = payload["dataset"]
            
            # Prepare HDF5 Group
            grp_name = f"trajectory_{start_traj_idx}"
            if grp_name in hdf_file:
                del hdf_file[grp_name]
            grp = hdf_file.create_group(grp_name)
            
            datasets_initialized = False
            
            # --- Chunked Processing ---
            for i in range(0, total_frames, batch_size):
                if SHUTDOWN_REQUESTED:
                    break
                    
                chunk_start = start_idx + i
                chunk_end = min(start_idx + i + batch_size, end_idx)
                
                # Load Chunk
                batch = dataset_ref.hf_dataset[chunk_start:chunk_end]
                
                # Get Timestamps for video query
                ts_raw = batch["timestamp"]
                batch_timestamps = [t.item() if isinstance(t, torch.Tensor) else t for t in ts_raw]
                
                query = {k: batch_timestamps for k in dataset_ref.meta.video_keys}
                video_frames = dataset_ref._query_videos(query, ep_idx)
                
                # --- Handle Wrist ---
                if "observation.images.wrist" in video_frames:
                    w_mini = video_frames["observation.images.wrist"]
                elif "observation.images.wrist" in batch:
                    wrist_data = batch["observation.images.wrist"]
                    w_mini = torch.stack(wrist_data) if isinstance(wrist_data, list) else wrist_data
                else:
                    raise ValueError("Missing observation.images.wrist")

                # --- Handle Front ---
                if "observation.images.front" in video_frames:
                    f_mini = video_frames["observation.images.front"]
                elif "observation.images.front" in batch:
                    front_data = batch["observation.images.front"]
                    f_mini = torch.stack(front_data) if isinstance(front_data, list) else front_data
                else:
                    raise ValueError("Missing observation.images.front")
                
                # Move to GPU
                w_mini = w_mini.to(device, dtype=torch.float32)
                f_mini = f_mini.to(device, dtype=torch.float32)
                
                # Store Images (uint8)
                w_uint8 = to_hwc_uint8(w_mini)
                f_uint8 = to_hwc_uint8(f_mini)
                
                # Inference
                with torch.no_grad():
                    w_prep = preprocess_images_for_dino(w_mini, is_front_camera=False)
                    f_prep = preprocess_images_for_dino(f_mini, is_front_camera=True)
                    
                    w_emb = dino_model.forward_features(w_prep)["x_norm_patchtokens"].cpu().numpy()
                    f_emb = dino_model.forward_features(f_prep)["x_norm_patchtokens"].cpu().numpy()
                
                del w_mini, f_mini, w_prep, f_prep, video_frames
                
                # --- Handle Non-Video Data (Actions/States) ---
                act = batch["action"]
                if isinstance(act, list):
                    cleaned_act = []
                    for a in act:
                        if isinstance(a, torch.Tensor):
                            if a.is_sparse:
                                a = a.to_dense()
                            cleaned_act.append(a)
                        else:
                            cleaned_act.append(torch.tensor(a))
                    act = torch.stack(cleaned_act)
                act_np = act.numpy()
            
                # States
                st_np = None
                if "observation.state" in batch:
                    st = batch["observation.state"]
                    if isinstance(st, list):
                        cleaned_st = []
                        for s in st:
                            if isinstance(s, torch.Tensor):
                                if s.is_sparse:
                                    s = s.to_dense()
                                cleaned_st.append(s)
                            else:
                                cleaned_st.append(torch.tensor(s))
                        st = torch.stack(cleaned_st)
                    st_np = st.numpy()

                # --- Write to HDF5 (Incremental) ---
                if not datasets_initialized:
                    grp.create_dataset("camera_0", data=w_uint8, maxshape=(None, *w_uint8.shape[1:]), compression="gzip", chunks=True)
                    grp.create_dataset("camera_1", data=f_uint8, maxshape=(None, *f_uint8.shape[1:]), compression="gzip", chunks=True)
                    grp.create_dataset("actions", data=act_np, maxshape=(None, *act_np.shape[1:]), chunks=True)
                    grp.create_dataset("cam_rs_embd", data=w_emb, maxshape=(None, *w_emb.shape[1:]), chunks=True)
                    grp.create_dataset("cam_zed_embd", data=f_emb, maxshape=(None, *f_emb.shape[1:]), chunks=True)
                    
                    if st_np is not None:
                        grp.create_dataset("states", data=st_np, maxshape=(None, *st_np.shape[1:]), chunks=True)
                    datasets_initialized = True
                else:
                    # Resize and Append
                    for name, arr in [
                        ("camera_0", w_uint8), ("camera_1", f_uint8), 
                        ("actions", act_np), ("cam_rs_embd", w_emb), ("cam_zed_embd", f_emb)
                    ]:
                        grp[name].resize(grp[name].shape[0] + arr.shape[0], axis=0)
                        grp[name][-arr.shape[0]:] = arr
                    
                    if st_np is not None:
                        if "states" in grp:
                            grp["states"].resize(grp["states"].shape[0] + st_np.shape[0], axis=0)
                            grp["states"][-st_np.shape[0]:] = st_np

            if SHUTDOWN_REQUESTED:
                pbar.update(1)
                continue

            # Metadata
            grp.attrs["dataset_id"] = dataset_id
            grp.attrs["original_episode_index"] = ep_idx
            
            # Flush after EVERY trajectory
            hdf_file.flush()
            
            # Quick validation: verify we can read back what we just wrote
            try:
                _ = grp["camera_0"].shape
                _ = grp.attrs["dataset_id"]
            except Exception as e:
                print(f"⚠️ Warning: Validation failed for {grp_name}: {e}")
                # Don't increment counter if validation failed
                if grp_name in hdf_file:
                    del hdf_file[grp_name]
                pbar.update(1)
                continue
            
            start_traj_idx += 1
            trajectories_saved += 1
            pbar.update(1)
            
        except Exception as e:
            print(f"⚠️ Error saving episode {ep_idx}: {e}")
            grp_name = f"trajectory_{start_traj_idx}"
            if grp_name in hdf_file:
                del hdf_file[grp_name]
            pbar.update(1)
            
    pbar.close()
    
    # Final flush after each dataset
    hdf_file.flush()
    print(f"💾 Flushed {trajectories_saved} trajectories to disk")
    
    return start_traj_idx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-list", type=str, required=True)
    parser.add_argument("--output-hdf5", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--max-episodes-per-dataset", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--resume", action="store_true", 
                       help="Resume from existing file (append mode)")
    args = parser.parse_args()

    # Setup signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Load Dataset List
    with open(args.datasets_list, "r") as f:
        dataset_ids = json.load(f)
    
    if args.max_datasets:
        dataset_ids = dataset_ids[:args.max_datasets]

    # Setup Device
    device = args.device if torch.cuda.is_available() else "cpu"
    
    # Load Model
    print("📦 Loading DINO model...")
    dino_model = get_dino_model(device)

    # Prepare Output
    output_path = Path(args.output_hdf5)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if resuming
    mode = "a" if args.resume and output_path.exists() else "w"
    
    if mode == "a":
        print(f"📂 Resuming from existing file: {output_path}")
    else:
        print(f"📝 Creating new file: {output_path}")
        if output_path.exists():
            backup_path = output_path.with_suffix(".h5.backup")
            print(f"⚠️  Backing up existing file to: {backup_path}")
            shutil.copy(output_path, backup_path)

    # Main Loop
    traj_counter = 0
    
    # Use libver='latest' for better crash resistance and modern HDF5 features
    with h5py.File(output_path, mode, libver='latest') as hf_out:
        # If resuming, find the next trajectory index
        if mode == "a":
            traj_counter = get_next_trajectory_index(hf_out)
            print(f"📊 Starting from trajectory index: {traj_counter}")
        
        for ds_id in dataset_ids:
            if SHUTDOWN_REQUESTED:
                print("\n🛑 Shutdown complete. File has been properly closed.")
                break
                
            # Check disk space
            usage = shutil.disk_usage(output_path.parent)
            if usage.free < 2 * 1024**3: # 2GB
                print("⚠️ Low disk space! Stopping.")
                break
                
            traj_counter = process_dataset(
                dataset_id=ds_id,
                hdf_file=hf_out,
                dino_model=dino_model,
                device=device,
                batch_size=args.batch_size,
                max_episodes=args.max_episodes_per_dataset,
                start_traj_idx=traj_counter,
                min_length=2,
                resume=(mode == "a")
            )
        
        # Final flush
        hf_out.flush()

    if SHUTDOWN_REQUESTED:
        print(f"\n⚠️  Interrupted! Saved {traj_counter} trajectories to {output_path}")
        print(f"   You can resume by running with --resume flag")
        sys.exit(130)  # Standard exit code for SIGINT
    else:
        print(f"\n✅ Done! Saved {traj_counter} trajectories to {output_path}")
        sys.exit(0)

if __name__ == "__main__":
    main()

