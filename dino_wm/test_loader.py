import torch
import numpy as np
import random
import bisect
from torch.utils.data import Dataset, DataLoader
import h5py

class SplitTrajectoryDataset(Dataset):
    def __init__(
        self,
        hdf5_file,
        segment_length,
        split='train',
        num_test=100,
        seed: int = 0,
        stride: int = 1,
        action_key: str = "actions_delta",
        front_embd_key: str = "cam_zed_embd",
        wrist_embd_key: str = "cam_rs_embd",
        load_images: bool = True,
    ):
        """
        HDF5 trajectory dataset that returns fixed-length segments.
        
        Args:
            hdf5_file (str): Path to the HDF5 file containing the trajectories.
            segment_length (int): Length of the segments to sample (H timesteps).
            split (str): 'train' or 'test'.
            num_test (int): Number of trajectories to put into the test split.
            seed (int): Seed used to shuffle trajectories before splitting (for reproducible splits).
            stride (int): Step between consecutive segment start indices within a trajectory.
        """
        self.hdf5_file = hdf5_file
        self.segment_length = int(segment_length)
        self.split = split
        self.num_test = int(num_test)
        self.seed = int(seed)
        self.stride = int(stride)
        if self.segment_length <= 0:
            raise ValueError(f"segment_length must be > 0. Got {self.segment_length}")
        if self.stride <= 0:
            raise ValueError(f"stride must be > 0. Got {self.stride}")
        self._hf = None  # lazily opened per worker/process
        self.action_key = action_key
        self.front_embd_key = front_embd_key
        self.wrist_embd_key = wrist_embd_key
        self.load_images = load_images
        self._missing_action_key_warned = False
        
        # Open HDF5 file to get a list of trajectory groups
        with h5py.File(self.hdf5_file, 'r') as hf:
            trajectory_ids = sorted(list(hf.keys()))
        
        # Shuffle trajectories before splitting (reproducible).
        rng = random.Random(self.seed)
        rng.shuffle(trajectory_ids)

        # Split the dataset based on the specified split
        if self.split == 'train':
            self.trajectory_ids = trajectory_ids[self.num_test:]
        elif self.split == 'test':
            self.trajectory_ids = trajectory_ids[:self.num_test]
        else:
            raise ValueError("split must be 'train' or 'test'.")
        
        # Precompute how many valid slices each trajectory contributes (prefix-sum index).
        self._cum_slices = [0]  # length = num_traj + 1; last element = total slices
        with h5py.File(self.hdf5_file, 'r') as hf:
            for traj_id in self.trajectory_ids:
                trajectory = hf[traj_id]
                action_key = self._select_action_key(trajectory)
                traj_len = int(trajectory[action_key].shape[0])
                max_start = traj_len - self.segment_length
                if max_start < 0:
                    num_slices = 0
                else:
                    num_slices = 1 + (max_start // self.stride)
                self._cum_slices.append(self._cum_slices[-1] + num_slices)

        if self._cum_slices[-1] <= 0:
            raise ValueError(
                f"No valid segments found for split='{self.split}' "
                f"with segment_length={self.segment_length} in {self.hdf5_file}."
            )

    def __len__(self):
        """Returns the number of segments in the selected split."""
        return int(self._cum_slices[-1])

    def _get_hf(self):
        if self._hf is None:
            self._hf = h5py.File(self.hdf5_file, 'r')
        return self._hf

    def __del__(self):
        try:
            if getattr(self, "_hf", None) is not None:
                self._hf.close()
        except Exception:
            pass

    def __getitem__(self, idx):
        """Return a segment from the global segment index space."""
        idx = int(idx)
        if idx < 0 or idx >= self.__len__():
            raise IndexError(idx)

        # Find which trajectory this idx falls into.
        traj_pos = bisect.bisect_right(self._cum_slices, idx) - 1
        if traj_pos < 0 or traj_pos >= len(self.trajectory_ids):
            raise IndexError(idx)
        local_idx = idx - self._cum_slices[traj_pos]
        start_idx = int(local_idx * self.stride)
        end_idx = start_idx + self.segment_length

        hf = self._get_hf()
        traj_id = self.trajectory_ids[traj_pos]
        trajectory = hf[traj_id]

        segment_obs_tensor = {}
        if self.load_images:
            segment_obs_tensor["robot0_eye_in_hand_image"] = torch.tensor(trajectory["camera_0"][start_idx:end_idx], dtype=torch.uint8)
            segment_obs_tensor["agentview_image"] = torch.tensor(trajectory["camera_1"][start_idx:end_idx], dtype=torch.uint8)
        segment_obs_tensor["cam_rs_embd"] = torch.tensor(
            trajectory[self.wrist_embd_key][start_idx:end_idx], dtype=torch.float32
        )
        segment_obs_tensor["cam_zed_embd"] = torch.tensor(
            trajectory[self.front_embd_key][start_idx:end_idx], dtype=torch.float32
        )
        segment_obs_tensor["state"] = torch.tensor(trajectory["states"][start_idx:end_idx], dtype=torch.float32)
        action_key = self._select_action_key(trajectory)
        segment_obs_tensor["action"] = torch.tensor(trajectory[action_key][start_idx:end_idx], dtype=torch.float32)
        segment_obs_tensor["traj_id"] = traj_id
        segment_obs_tensor["start_idx"] = start_idx
        if "labels" in trajectory.keys():
            segment_obs_tensor["failure"] = torch.tensor(trajectory["labels"][start_idx:end_idx], dtype=torch.float32)
        segment_obs_tensor["is_first"] = torch.zeros(self.segment_length)
        segment_obs_tensor["is_last"] = torch.zeros(self.segment_length)
        segment_obs_tensor["is_first"][0] = 1.
        segment_obs_tensor["is_terminal"] = segment_obs_tensor["is_last"]
        segment_obs_tensor["discount"] = torch.ones(self.segment_length, dtype=torch.float32)
        return segment_obs_tensor

    def _select_action_key(self, trajectory):
        if self.action_key in trajectory:
            return self.action_key
        if not self._missing_action_key_warned:
            print(
                f"⚠️ Warning: '{self.action_key}' missing in trajectory; falling back to 'actions'."
            )
            self._missing_action_key_warned = True
        return "actions"
    
if __name__ == '__main__':
    # Path to your HDF5 file
    hdf5_file = '/home/kensuke/data/skittles_trajectories_dreamer.h5'
    segment_length = 32  # Number of timesteps per segment
    batch_size = 32      # Number of trajectories per batch

    # Create the dataset
    train_dataset = SplitTrajectoryDataset(hdf5_file, segment_length, split='train', num_test=100, seed=0)
    test_dataset = SplitTrajectoryDataset(hdf5_file, segment_length, split='test', num_test=100, seed=0)


    # Create the DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Example usage:
    for batch_idx, data in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(data.keys())
        print(data['agentview_image'].shape)
        print(data['agentview_image'].max())
        print(f"Observations: {data['cam_zed_right_embd'].shape}")
        print(f"Actions: {data['action'].shape}")

        break  # Just print one batch

    for batch_idx, data in enumerate(test_loader):
        print(f"Batch {batch_idx}:")
        print(data.keys())
        print(f"Observations: {data['cam_zed_right_embd'].shape}")
        print(f"Actions: {data['action'].shape}")


        
        break  # Just print one batch