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

    model = dino_models.VideoTransformer(
        image_size=(224, 224),
        dim=32,
        action_embed_dim=8,
        state_embed_dim=6,
        state_dim=5,
        action_dim=4,
        action_horizon=100,
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

    assert future_emb.shape == (batch_size, 100, 8)
    assert future_broadcast.shape == (batch_size, 3, model.num_patches, 800)

    wrong_horizon = torch.zeros(batch_size, 3, 4)
    with pytest.raises(ValueError):
        model._encode_future_actions(wrong_horizon, num_frames=3)
