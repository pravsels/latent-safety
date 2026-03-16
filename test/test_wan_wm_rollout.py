import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "wan_wm_rollout.py"
    spec = importlib.util.spec_from_file_location("scripts.wan_wm_rollout", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load scripts/wan_wm_rollout.py module spec.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_args_defaults_dataset_stats_to_hpc_file(monkeypatch):
    mod = _load_module()
    monkeypatch.setattr(sys, "argv", ["wan_wm_rollout.py", "--dataset", "dummy/repo"])
    args = mod.parse_args()
    assert args.dataset_stats == "arx5_datasets_6Feb_26_stats.json"


def test_wan_episode_provider_is_lazy(monkeypatch):
    mod = _load_module()
    query_calls = []
    encode_calls = []

    class _FakeHFDataset:
        def __init__(self, batch):
            self._batch = batch

        def with_format(self, _fmt):
            return self

        def __getitem__(self, item):
            assert isinstance(item, slice)
            return self._batch

    class _FakeDataset:
        def __init__(self, batch):
            self.hf_dataset = _FakeHFDataset(batch)
            self.meta = type(
                "Meta",
                (),
                {"video_keys": ["observation.images.front", "observation.images.wrist"]},
            )()

        def _query_videos(self, query, episode):
            del episode
            timestamps = query["observation.images.front"]
            query_calls.append(tuple(float(ts) for ts in timestamps))
            n = len(timestamps)
            front = torch.zeros((n, 3, 2, 2), dtype=torch.float32)
            wrist = torch.ones((n, 3, 2, 2), dtype=torch.float32)
            return {
                "observation.images.front": front,
                "observation.images.wrist": wrist,
            }

    batch = {
        "timestamp": np.arange(6, dtype=np.float64),
        "action": np.zeros((6, 2), dtype=np.float32),
        "observation.state": np.zeros((6, 2), dtype=np.float32),
    }
    fake_dataset = _FakeDataset(batch)
    monkeypatch.setattr(mod, "load_lerobot_episode", lambda dataset_id, episode: (fake_dataset, 0, 6))

    def _fake_encode_frame(_vae, frame_hwc_uint8, _dev, _dtype):
        encode_calls.append(frame_hwc_uint8.copy())
        return torch.zeros((1, 1, 1), dtype=torch.float32)

    monkeypatch.setattr(mod, "encode_frame", _fake_encode_frame)

    provider = mod.WanEpisodeProvider(
        dataset_id="dummy",
        episode=0,
        max_frames=6,
        video_query_batch=2,
        vae=None,
        vae_device=torch.device("cpu"),
        vae_dtype=torch.float32,
    )

    assert query_calls == []
    assert encode_calls == []

    provider.get_latent_pair(0)
    assert query_calls == [(0.0, 1.0)]
    assert len(encode_calls) == 2

    provider.get_latent_pair(0)
    assert query_calls == [(0.0, 1.0)]
    assert len(encode_calls) == 2


def test_wan_episode_provider_reuses_recent_chunk(monkeypatch):
    mod = _load_module()
    query_calls = []

    class _FakeHFDataset:
        def __init__(self, batch):
            self._batch = batch

        def with_format(self, _fmt):
            return self

        def __getitem__(self, item):
            assert isinstance(item, slice)
            return self._batch

    class _FakeDataset:
        def __init__(self, batch):
            self.hf_dataset = _FakeHFDataset(batch)
            self.meta = type(
                "Meta",
                (),
                {"video_keys": ["observation.images.front", "observation.images.wrist"]},
            )()

        def _query_videos(self, query, episode):
            del episode
            timestamps = query["observation.images.front"]
            query_calls.append(tuple(float(ts) for ts in timestamps))
            n = len(timestamps)
            front = torch.zeros((n, 3, 2, 2), dtype=torch.float32)
            wrist = torch.ones((n, 3, 2, 2), dtype=torch.float32)
            return {
                "observation.images.front": front,
                "observation.images.wrist": wrist,
            }

    batch = {
        "timestamp": np.arange(6, dtype=np.float64),
        "action": np.zeros((6, 2), dtype=np.float32),
        "observation.state": np.zeros((6, 2), dtype=np.float32),
    }
    fake_dataset = _FakeDataset(batch)
    monkeypatch.setattr(mod, "load_lerobot_episode", lambda dataset_id, episode: (fake_dataset, 0, 6))
    monkeypatch.setattr(
        mod,
        "encode_frame",
        lambda _vae, _frame_hwc_uint8, _dev, _dtype: torch.zeros((1, 1, 1), dtype=torch.float32),
    )

    provider = mod.WanEpisodeProvider(
        dataset_id="dummy",
        episode=0,
        max_frames=6,
        video_query_batch=2,
        vae=None,
        vae_device=torch.device("cpu"),
        vae_dtype=torch.float32,
    )

    provider.get_frame_pair(0)
    provider.get_frame_pair(2)
    provider.get_frame_pair(1)

    assert query_calls == [(0.0, 1.0), (2.0, 3.0)]


def test_plot_state_rollout_writes_png(tmp_path):
    mod = _load_module()
    gt_states = torch.tensor([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype=torch.float32)
    pred_states = torch.tensor([[0.0, 1.0], [1.5, 2.5], [2.5, 3.5]], dtype=torch.float32)
    output_path = tmp_path / "state_rollout.png"

    mod.plot_state_rollout(gt_states, pred_states, str(output_path))

    assert output_path.exists()
    assert output_path.stat().st_size > 0
