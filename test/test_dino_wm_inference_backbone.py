import importlib.util
from pathlib import Path

import numpy as np
import torch


def _load_inference_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "dino-wm_inference.py"
    spec = importlib.util.spec_from_file_location("scripts.dino_wm_inference", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load scripts.dino-wm_inference.py module spec.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_args_accepts_episode_flow_required_args():
    mod = _load_inference_module()
    args = mod.parse_args(
        [
            "--wm-checkpoint",
            "wm.pth",
            "--decoder-checkpoint",
            "decoder.pth",
            "--dataset",
            "villekuosmanen/bin_pick_pack_coffee_capsules_eval",
            "--dataset-stats",
            "stats.json",
        ]
    )
    assert args.dataset == "villekuosmanen/bin_pick_pack_coffee_capsules_eval"
    assert args.decoder_checkpoint == "decoder.pth"


def test_parse_args_defaults_dataset_stats_to_hpc_file():
    mod = _load_inference_module()
    args = mod.parse_args(
        [
            "--wm-checkpoint",
            "wm.pth",
            "--decoder-checkpoint",
            "decoder.pth",
            "--dataset",
            "villekuosmanen/bin_pick_pack_coffee_capsules_eval",
        ]
    )
    assert args.dataset_stats == "arx5_datasets_6Feb_26_stats.json"


def test_validate_args_requires_decoder_checkpoint():
    mod = _load_inference_module()
    try:
        mod.parse_args(
            [
                "--wm-checkpoint",
                "wm.pth",
                "--dataset",
                "villekuosmanen/bin_pick_pack_coffee_capsules_eval",
            ]
        )
    except SystemExit:
        pass
    else:
        raise AssertionError("Expected parse_args to require --decoder-checkpoint")


def test_parse_args_accepts_future_action_steps():
    mod = _load_inference_module()
    args = mod.parse_args(
        [
            "--wm-checkpoint",
            "wm.pth",
            "--decoder-checkpoint",
            "decoder.pth",
            "--dataset",
            "villekuosmanen/bin_pick_pack_coffee_capsules_eval",
            "--dataset-stats",
            "stats.json",
            "--future-action-steps",
            "30",
        ]
    )
    assert args.future_action_steps == 30


def test_validate_args_rejects_negative_future_action_steps():
    mod = _load_inference_module()
    args = mod.parse_args(
        [
            "--wm-checkpoint",
            "wm.pth",
            "--decoder-checkpoint",
            "decoder.pth",
            "--dataset",
            "villekuosmanen/bin_pick_pack_coffee_capsules_eval",
            "--dataset-stats",
            "stats.json",
            "--future-action-steps",
            "-1",
        ]
    )
    try:
        mod.validate_args(args)
    except ValueError as e:
        assert "--future-action-steps must be >= 0" in str(e)
    else:
        raise AssertionError("Expected ValueError for negative --future-action-steps")


def test_validate_args_accepts_episode_flow():
    mod = _load_inference_module()
    args = mod.parse_args(
        [
            "--wm-checkpoint",
            "wm.pth",
            "--decoder-checkpoint",
            "decoder.pth",
            "--dataset-stats",
            "stats.json",
            "--dataset",
            "villekuosmanen/bin_pick_pack_coffee_capsules_eval",
            "--episode",
            "0",
        ]
    )
    mod.validate_args(args)


def test_validate_args_rejects_negative_reset_interval():
    mod = _load_inference_module()
    args = mod.parse_args(
        [
            "--wm-checkpoint",
            "wm.pth",
            "--decoder-checkpoint",
            "decoder.pth",
            "--dataset",
            "villekuosmanen/bin_pick_pack_coffee_capsules_eval",
            "--dataset-stats",
            "stats.json",
            "--reset-interval",
            "-2",
        ]
    )
    try:
        mod.validate_args(args)
    except ValueError as e:
        assert "--reset-interval must be >= 0" in str(e)
    else:
        raise AssertionError("Expected ValueError for negative --reset-interval")


def test_episode_frame_provider_queries_video_lazily(monkeypatch):
    mod = _load_inference_module()
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

    provider = mod.EpisodeFrameProvider(
        dataset_id="dummy",
        episode=0,
        max_frames=6,
        video_query_batch=2,
    )

    assert query_calls == []

    provider.get_frame_pair(0)
    assert query_calls == [(0.0, 1.0)]

    provider.get_frame_pair(1)
    assert query_calls == [(0.0, 1.0)]

    provider.get_frame_pair(2)
    assert query_calls == [(0.0, 1.0), (2.0, 3.0)]


def test_episode_frame_provider_reuses_recent_chunk(monkeypatch):
    mod = _load_inference_module()
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

    provider = mod.EpisodeFrameProvider(
        dataset_id="dummy",
        episode=0,
        max_frames=6,
        video_query_batch=2,
    )

    provider.get_frame_pair(0)
    provider.get_frame_pair(2)
    provider.get_frame_pair(1)

    assert query_calls == [(0.0, 1.0), (2.0, 3.0)]
