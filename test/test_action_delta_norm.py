import importlib.util
from pathlib import Path
import json

import numpy as np
import h5py
import torch


def _load_data_utils():
    module_path = Path(__file__).resolve().parents[1] / "dino_wm" / "data_utils.py"
    spec = importlib.util.spec_from_file_location("dino_wm.data_utils", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load dino_wm.data_utils module spec.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_test_loader():
    module_path = Path(__file__).resolve().parents[1] / "dino_wm" / "test_loader.py"
    spec = importlib.util.spec_from_file_location("dino_wm.test_loader", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load dino_wm.test_loader module spec.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_compute_stats():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "compute_stats_json.py"
    spec = importlib.util.spec_from_file_location("scripts.compute_stats_json", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load scripts.compute_stats_json module spec.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_dino_models():
    module_path = Path(__file__).resolve().parents[1] / "dino_wm" / "dino_models.py"
    spec = importlib.util.spec_from_file_location("dino_wm.dino_models", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load dino_wm.dino_models module spec.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compute_action_deltas_shared_dims():
    actions = np.array([[2.0, 3.0, 9.0], [4.0, 5.0, 8.0]], dtype=np.float32)
    states = np.array([[1.0, 1.5], [2.0, 2.5]], dtype=np.float32)
    # shared dims = 2, last dim stays unchanged
    data_utils = _load_data_utils()
    deltas = data_utils.compute_action_deltas(actions, states)
    expected = np.array([[1.0, 1.5, 9.0], [2.0, 2.5, 8.0]], dtype=np.float32)
    assert np.allclose(deltas, expected)


def test_quantile_normalize_to_minus1_plus1():
    x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    q02 = np.array([0.0], dtype=np.float32)
    q98 = np.array([2.0], dtype=np.float32)
    data_utils = _load_data_utils()
    out = data_utils.quantile_normalize(x, q02, q98)
    assert np.allclose(out, np.array([-1.0, 0.0, 1.0], dtype=np.float32))


def test_lerobot_writer_adds_actions_delta(tmp_path):
    data_utils = _load_data_utils()
    actions = np.array([[2.0, 3.0, 9.0], [4.0, 5.0, 8.0]], dtype=np.float32)
    states = np.array([[1.0, 1.5], [2.0, 2.5]], dtype=np.float32)
    expected = data_utils.compute_action_deltas(actions, states)

    hdf5_path = tmp_path / "test_actions_delta.h5"
    with h5py.File(hdf5_path, "w") as h5f:
        grp = h5f.create_group("trajectory_0")
        data_utils.write_actions_delta(grp, actions, states, initialized=False)
        assert "actions_delta" in grp
        assert np.allclose(grp["actions_delta"][:], expected)


def test_loader_prefers_actions_delta(tmp_path):
    test_loader = _load_test_loader()
    data_utils = _load_data_utils()
    actions = np.array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]], dtype=np.float32)
    states = np.array([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]], dtype=np.float32)
    actions_delta = data_utils.compute_action_deltas(actions, states)

    hdf5_path = tmp_path / "test_loader_actions_delta.h5"
    with h5py.File(hdf5_path, "w") as h5f:
        grp = h5f.create_group("trajectory_0")
        grp.create_dataset("camera_0", data=np.zeros((3, 2, 2, 3), dtype=np.uint8))
        grp.create_dataset("camera_1", data=np.zeros((3, 2, 2, 3), dtype=np.uint8))
        grp.create_dataset("cam_rs_embd", data=np.zeros((3, 1, 1), dtype=np.float32))
        grp.create_dataset("cam_zed_embd", data=np.zeros((3, 1, 1), dtype=np.float32))
        grp.create_dataset("states", data=states)
        grp.create_dataset("actions", data=actions)
        grp.create_dataset("actions_delta", data=actions_delta)

    dataset = test_loader.SplitTrajectoryDataset(
        str(hdf5_path), segment_length=2, split="train", num_test=0, seed=0
    )
    sample = dataset[0]
    assert torch.allclose(sample["action"], torch.tensor(actions_delta[:2], dtype=torch.float32))


def test_loader_supports_custom_latent_keys(tmp_path):
    test_loader = _load_test_loader()

    hdf5_path = tmp_path / "test_loader_custom_latents.h5"
    with h5py.File(hdf5_path, "w") as h5f:
        grp = h5f.create_group("trajectory_0")
        grp.create_dataset("camera_0", data=np.zeros((3, 2, 2, 3), dtype=np.uint8))
        grp.create_dataset("camera_1", data=np.zeros((3, 2, 2, 3), dtype=np.uint8))
        grp.create_dataset("wan_front_embd", data=np.ones((3, 4, 16), dtype=np.float32))
        grp.create_dataset("wan_wrist_embd", data=np.full((3, 4, 16), 2.0, dtype=np.float32))
        grp.create_dataset("states", data=np.zeros((3, 2), dtype=np.float32))
        grp.create_dataset("actions", data=np.zeros((3, 2), dtype=np.float32))

    dataset = test_loader.SplitTrajectoryDataset(
        str(hdf5_path),
        segment_length=2,
        split="train",
        num_test=0,
        seed=0,
        front_embd_key="wan_front_embd",
        wrist_embd_key="wan_wrist_embd",
    )
    sample = dataset[0]
    assert sample["cam_zed_embd"].shape == (2, 4, 16)
    assert sample["cam_rs_embd"].shape == (2, 4, 16)
    assert torch.allclose(sample["cam_zed_embd"], torch.ones((2, 4, 16), dtype=torch.float32))
    assert torch.allclose(sample["cam_rs_embd"], torch.full((2, 4, 16), 2.0, dtype=torch.float32))


def test_compute_stats_quantiles(tmp_path):
    compute_stats = _load_compute_stats()
    hdf5_path = tmp_path / "test_stats_quantiles.h5"
    out_json = tmp_path / "stats.json"
    actions_delta = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    states = np.array([[10.0], [11.0], [12.0]], dtype=np.float32)

    with h5py.File(hdf5_path, "w") as h5f:
        grp = h5f.create_group("trajectory_0")
        grp.create_dataset("actions_delta", data=actions_delta)
        grp.create_dataset("states", data=states)

    compute_stats.compute_stats(str(hdf5_path), str(out_json))

    with open(out_json, "r") as f:
        stats = json.load(f)

    assert "action_delta_q02" in stats
    assert "action_delta_q98" in stats
    assert "state_q02" in stats
    assert "state_q98" in stats


def test_normalize_uses_quantiles():
    dino_models = _load_dino_models()
    acs = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
    states = torch.tensor([10.0, 11.0, 12.0], dtype=torch.float32)
    q02 = torch.tensor([0.0], dtype=torch.float32)
    q98 = torch.tensor([2.0], dtype=torch.float32)
    state_q02 = torch.tensor([10.0], dtype=torch.float32)
    state_q98 = torch.tensor([12.0], dtype=torch.float32)

    norm_acs = dino_models.normalize_acs(acs, None, None, q02=q02, q98=q98)
    norm_states = dino_models.normalize_states(states, None, None, q02=state_q02, q98=state_q98)

    assert torch.allclose(norm_acs, torch.tensor([-1.0, 0.0, 1.0]))
    assert torch.allclose(norm_states, torch.tensor([-1.0, 0.0, 1.0]))
