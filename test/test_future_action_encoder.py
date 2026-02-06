import pytest
import torch


def test_future_action_encoder_shapes(monkeypatch):
    import dino_wm.dino_models as dino_models

    class DummyDino(torch.nn.Module):
        def to(self, device):
            return self

    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: DummyDino())

    with pytest.raises(ValueError):
        dino_models.VideoTransformer(
            image_size=(224, 224),
            dim=32,
            action_embed_dim=8,
            state_embed_dim=6,
            state_dim=5,
            action_dim=4,
            action_horizon=5,
            depth=1,
            heads=2,
            mlp_dim=64,
            num_frames=3,
            device="cpu",
            dino_version="v3",
        )

    trajectory_summary_dim = 16
    model = dino_models.VideoTransformer(
        image_size=(224, 224),
        dim=32,
        action_embed_dim=8,
        state_embed_dim=6,
        state_dim=5,
        action_dim=4,
        action_horizon=100,
        trajectory_summary_dim=trajectory_summary_dim,
        depth=1,
        heads=2,
        mlp_dim=64,
        num_frames=3,
        device="cpu",
        dino_version="v3",
    )

    batch_size = 2
    horizon = 100
    future_actions = torch.zeros(batch_size, horizon, 4)

    future_emb, future_broadcast = model._encode_future_actions(
        future_actions,
        num_frames=3,
    )

    assert future_emb.shape == (batch_size, trajectory_summary_dim)
    assert future_broadcast.shape == (
        batch_size,
        3,
        model.num_patches,
        trajectory_summary_dim,
    )

    wrong_horizon = torch.zeros(batch_size, 101, 4)
    with pytest.raises(ValueError):
        model._encode_future_actions(wrong_horizon, num_frames=3)


def test_future_action_encoder_masks_padding(monkeypatch):
    import dino_wm.dino_models as dino_models

    class DummyDino(torch.nn.Module):
        def to(self, device):
            return self

    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: DummyDino())

    model = dino_models.VideoTransformer(
        image_size=(224, 224),
        dim=32,
        action_embed_dim=8,
        state_embed_dim=6,
        state_dim=5,
        action_dim=4,
        action_horizon=100,
        trajectory_summary_dim=8,
        depth=1,
        heads=2,
        mlp_dim=64,
        num_frames=3,
        device="cpu",
        dino_version="v3",
    )

    batch_size = 1
    horizon = 100
    action_dim = 4
    length = 5
    lengths = torch.tensor([length])

    future_actions_tail = torch.zeros(batch_size, horizon, action_dim)
    future_actions_tail[:, length:, :] = 1.0
    future_actions_zero = torch.zeros_like(future_actions_tail)

    summary_tail, _ = model._encode_future_actions(
        future_actions_tail,
        num_frames=3,
        future_action_lengths=lengths,
    )
    summary_zero, _ = model._encode_future_actions(
        future_actions_zero,
        num_frames=3,
        future_action_lengths=lengths,
    )

    assert torch.allclose(summary_tail, summary_zero, atol=1e-6)


def test_future_action_encoder_pads_short_horizon(monkeypatch):
    import dino_wm.dino_models as dino_models

    class DummyDino(torch.nn.Module):
        def to(self, device):
            return self

    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: DummyDino())

    model = dino_models.VideoTransformer(
        image_size=(224, 224),
        dim=32,
        action_embed_dim=8,
        state_embed_dim=6,
        state_dim=5,
        action_dim=4,
        action_horizon=100,
        trajectory_summary_dim=8,
        depth=1,
        heads=2,
        mlp_dim=64,
        num_frames=3,
        device="cpu",
        dino_version="v3",
    )

    batch_size = 2
    short_horizon = 5
    action_dim = 4
    future_actions_short = torch.randn(batch_size, short_horizon, action_dim)
    lengths = torch.full((batch_size,), short_horizon)

    summary_short, _ = model._encode_future_actions(
        future_actions_short,
        num_frames=3,
    )

    padded = torch.zeros(batch_size, model.action_horizon, action_dim)
    padded[:, :short_horizon] = future_actions_short
    summary_padded, _ = model._encode_future_actions(
        padded,
        num_frames=3,
        future_action_lengths=lengths,
    )

    assert torch.allclose(summary_short, summary_padded, atol=1e-6)
