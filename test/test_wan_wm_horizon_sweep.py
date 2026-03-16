import importlib.util
from pathlib import Path

import numpy as np
import torch


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "wan_wm_horizon_sweep.py"
    spec = importlib.util.spec_from_file_location("scripts.wan_wm_horizon_sweep", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load scripts/wan_wm_horizon_sweep.py module spec.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeTransition:
    def __init__(self, state_scale: float):
        self.state_scale = state_scale
        self.future_actions_seen = []

    def eval(self):
        return self

    def __call__(self, in_front, in_wrist, in_state, in_acs, future_actions):
        if future_actions is None:
            raise AssertionError("evaluate_horizon_wan should always pass future action tensors.")
        self.future_actions_seen.append(future_actions.detach().cpu())

        # Predict target token using "last context index + sum(future actions) + 1".
        # This matches synthetic ground truth when future actions are all ones.
        last = in_front[:, -1]
        delta = future_actions.sum(dim=1).view(in_front.shape[0], 1, 1)
        pred_front_last = last + delta + 1.0
        pred_wrist_last = pred_front_last.clone()
        pred_state_last = pred_front_last.view(in_front.shape[0], 1) * self.state_scale

        pred_front = in_front.clone()
        pred_wrist = in_wrist.clone()
        pred_state = in_state.clone()
        pred_front[:, -1] = pred_front_last
        pred_wrist[:, -1] = pred_wrist_last
        pred_state[:, -1] = pred_state_last
        return pred_front, pred_wrist, pred_state, None


def _make_episode_data(T: int):
    # Embedding token at time i is exactly i. Shape: (T, P=1, D=1).
    idx = torch.arange(T, dtype=torch.float32).view(T, 1, 1)
    # State is normalized index in [0,1].
    state = idx.view(T, 1) / float(T - 1)
    action = torch.ones((T, 1), dtype=torch.float32)
    return {"front_latent": idx, "wrist_latent": idx.clone(), "actions": action, "states": state}


def _make_stats():
    return {
        "action_min": torch.tensor([0.0], dtype=torch.float32),
        "action_max": torch.tensor([1.0], dtype=torch.float32),
        "state_min": torch.tensor([0.0], dtype=torch.float32),
        "state_max": torch.tensor([1.0], dtype=torch.float32),
    }


def test_wan_horizon_sweep_uses_future_actions_when_enabled():
    mod = _load_module()
    episode = _make_episode_data(T=40)
    stats = _make_stats()
    transition = _FakeTransition(state_scale=1.0 / 39.0)

    out = mod.evaluate_horizon_wan(
        transition=transition,
        episode_data=episode,
        stats=stats,
        context_length=3,
        pred_step=1,
        max_horizon=5,
        stride=1,
        device="cpu",
        no_future_actions=False,
    )

    assert out["latent_mse_mean"][0] < 1e-8
    assert out["state_mse_mean"][0] < 1e-8
    assert all(float(v.mean()) > 0.5 for v in transition.future_actions_seen)


def test_wan_horizon_sweep_zeroes_future_actions_in_ablation():
    mod = _load_module()
    episode = _make_episode_data(T=40)
    stats = _make_stats()
    transition = _FakeTransition(state_scale=1.0 / 39.0)

    out = mod.evaluate_horizon_wan(
        transition=transition,
        episode_data=episode,
        stats=stats,
        context_length=3,
        pred_step=1,
        max_horizon=5,
        stride=1,
        device="cpu",
        no_future_actions=True,
    )

    assert out["latent_mse_mean"][0] > 0.5
    assert out["state_mse_mean"][0] > 1e-4
    assert all(float(v.abs().max()) == 0.0 for v in transition.future_actions_seen)


class _SliceableHF:
    def __init__(self, payload):
        self.payload = payload

    def with_format(self, _fmt):
        return self

    def __getitem__(self, slc):
        return {k: v[slc] for k, v in self.payload.items()}


class _FakeDataset:
    def __init__(self, T: int):
        self.meta = type("Meta", (), {"video_keys": ["observation.images.front", "observation.images.wrist"]})()
        self.hf_dataset = _SliceableHF(
            {
                "timestamp": [float(i) for i in range(T)],
                "action": [[0.0] for _ in range(T)],
                "observation.state": [[0.0] for _ in range(T)],
            }
        )
        self.query_calls = 0

    def _query_videos(self, query, _episode_idx):
        ts = query["observation.images.front"]
        if len(ts) > 2:
            raise AssertionError("query chunk too large")
        self.query_calls += 1
        n = len(ts)
        front = torch.full((n, 3, 224, 224), 0.5, dtype=torch.float32)
        wrist = torch.full((n, 3, 224, 224), 0.5, dtype=torch.float32)
        return {
            "observation.images.front": front,
            "observation.images.wrist": wrist,
        }


def test_wan_latent_loader_queries_video_in_chunks(monkeypatch):
    mod = _load_module()
    fake_ds = _FakeDataset(T=5)

    monkeypatch.setattr(mod, "load_lerobot_episode", lambda _repo, _ep: (fake_ds, 0, 5))

    def _fake_encode_frame(_vae, frame_hwc_uint8, _dev, _dtype):
        assert isinstance(frame_hwc_uint8, np.ndarray)
        return torch.zeros((1, 1, 1), dtype=torch.float32)

    monkeypatch.setattr(mod, "encode_frame", _fake_encode_frame)

    out = mod.load_episode_wan_latents(
        "dummy/repo",
        0,
        vae=None,
        vae_device=torch.device("cpu"),
        vae_dtype=torch.float32,
        device="cpu",
        video_query_batch=2,
    )

    assert fake_ds.query_calls == 3
    assert tuple(out["front_latent"].shape) == (5, 1, 1)
    assert tuple(out["wrist_latent"].shape) == (5, 1, 1)
