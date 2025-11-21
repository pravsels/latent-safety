#!/usr/bin/env python3
"""
Load LeRobot datasets and convert to DreamerV3 trajectory format.
Usage: python load_lerobot_datasets.py --datasets-list dataset.json --output data.pkl --max-episodes-per-dataset 10
"""

import json, argparse, pickle, numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def convert_sample(sample):
    """Convert LeRobot sample to single-frame trajectory."""
    # Get image
    img = None
    for key in ['observation.images.wrist', 'observation.images.front', 'observation.wrist', 'observation.front']:
        if key in sample:
            img = sample[key]
            break
    if img is None:
        return None

    # Convert image: CHW -> HWC, scale to 0-255
    img = img.detach().cpu().numpy()
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    # Get state
    state = sample.get('observation.state')
    if state is not None:
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        state = np.asarray(state, dtype=np.float32).flatten().tolist()

    # Get action
    action = sample.get('action', 0.0)
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy().flatten()
    elif isinstance(action, (list, np.ndarray)):
        action = np.asarray(action, dtype=np.float32).flatten()

    return {
        'obs': {
            'image': [img],
            **({'state': [state], 'priv_state': [np.asarray(state, dtype=np.float32)]} if state is not None else {}),
        },
        'actions': [action],
        'dones': [0],
    }


def load_datasets(dataset_ids, max_episodes=None, verbose=True):
    """Load and convert datasets to Dreamer format."""
    all_trajectories = []

    for dataset_id in dataset_ids:
        if verbose:
            print(f"Loading {dataset_id}...")

        try:
            dataset = LeRobotDataset(dataset_id)
            if verbose:
                print(f"  Found {len(dataset)} samples")

            # Collect episodes until we have enough valid trajectories
            episodes = {}
            valid_trajectories = 0

            for idx in tqdm(range(len(dataset)), desc=f"  Processing {dataset_id}", disable=not verbose):
                try:
                    sample = dataset[idx]

                    # Get episode/frame indices
                    ep_idx = sample.get('episode_index', 0)
                    frame_idx = sample.get('frame_index', 0)
                    if hasattr(ep_idx, 'item'):
                        ep_idx, frame_idx = ep_idx.item(), frame_idx.item()

                    ep_idx, frame_idx = int(ep_idx), int(frame_idx)

                    # Track episode
                    if ep_idx not in episodes:
                        episodes[ep_idx] = []
                    episodes[ep_idx].append((frame_idx, idx))

                except Exception as e:
                    if verbose and idx < 5:
                        print(f"  Error at sample {idx}: {e}")
                    continue

            # Process episodes
            for ep_idx in sorted(episodes.keys()):
                frames = sorted(episodes[ep_idx])  # Sort by frame_index

                traj = {'obs': {'image': []}, 'actions': [], 'dones': []}

                for frame_idx, sample_idx in frames:
                    try:
                        sample = dataset[sample_idx]
                        frame_traj = convert_sample(sample)

                        if frame_traj and frame_traj['obs']['image']:
                            traj['obs']['image'].append(frame_traj['obs']['image'][0])
                            traj['actions'].append(frame_traj['actions'][0])

                            if 'state' in frame_traj['obs']:
                                if 'state' not in traj['obs']:
                                    traj['obs']['state'] = []
                                    traj['obs']['priv_state'] = []
                                traj['obs']['state'].append(frame_traj['obs']['state'][0])
                                traj['obs']['priv_state'].append(frame_traj['obs']['priv_state'][0])

                    except Exception as e:
                        if verbose:
                            print(f"  Error converting sample {sample_idx}: {e}")
                        continue

                # Save trajectory if it has images
                if traj['obs']['image']:
                    traj['dones'] = [0] * (len(traj['obs']['image']) - 1) + [1]
                    all_trajectories.append(traj)
                    valid_trajectories += 1

                    if max_episodes and valid_trajectories >= max_episodes:
                        break

            if verbose:
                print(f"  Generated {valid_trajectories} trajectories")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    return all_trajectories


def main():
    parser = argparse.ArgumentParser(description="Load LeRobot datasets and convert to DreamerV3 format")
    parser.add_argument("--datasets-list", type=str, required=True, help="JSON file with list of dataset IDs")
    parser.add_argument("--output", type=str, required=True, help="Output pickle file")
    parser.add_argument("--max-datasets", type=int, help="Max datasets to load")
    parser.add_argument("--max-episodes-per-dataset", type=int, help="Max episodes per dataset")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args()

    # Load dataset list
    with open(args.datasets_list) as f:
        dataset_ids = json.load(f)

    if args.max_datasets:
        dataset_ids = dataset_ids[:args.max_datasets]

    # Load and convert
    trajectories = load_datasets(dataset_ids, args.max_episodes_per_dataset, verbose=not args.quiet)

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(trajectories, f)

    print(f"\nSaved {len(trajectories)} trajectories to {args.output}")
    print(f"File size: {Path(args.output).stat().st_size / (1024**2):.2f} MB")


if __name__ == "__main__":
    main()