import pytest
import torch


def _make_dummy_dino():
    class DummyDino(torch.nn.Module):
        def to(self, device):
            return self

    return DummyDino()


def test_trajectory_encoder_output_shape():
    import dino_wm.dino_models as dino_models

    encoder = dino_models.TrajectoryEncoder(action_dim=4, trajectory_summary_dim=8)
    future_actions = torch.zeros(2, 10, 4)
    summary = encoder(future_actions)

    assert summary.shape == (2, 8)


def test_trajectory_encoder_variable_length_runs():
    import dino_wm.dino_models as dino_models

    encoder = dino_models.TrajectoryEncoder(action_dim=4, trajectory_summary_dim=8)
    short_actions = torch.zeros(2, 5, 4)
    long_actions = torch.zeros(2, 25, 4)

    short_summary = encoder(short_actions)
    long_summary = encoder(long_actions)

    assert short_summary.shape == (2, 8)
    assert long_summary.shape == (2, 8)


def test_encode_future_actions_broadcast_shape(monkeypatch):
    import dino_wm.dino_models as dino_models

    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: _make_dummy_dino())

    model = dino_models.VideoTransformer(
        image_size=(224, 224),
        dim=32,
        action_embed_dim=8,
        state_embed_dim=6,
        state_dim=5,
        action_dim=4,
        action_horizon=dino_models.FUTURE_ACTION_HORIZON_MAX,
        trajectory_summary_dim=12,
        depth=1,
        heads=2,
        mlp_dim=64,
        num_frames=3,
        device="cpu",
        dino_version="v3",
    )

    future_actions = torch.zeros(2, 7, 4)
    summary, broadcast = model._encode_future_actions(future_actions, num_frames=3)

    assert summary.shape == (2, 12)
    assert broadcast.shape == (2, 3, model.num_patches, 12)


def test_encode_future_actions_rejects_over_max(monkeypatch):
    import dino_wm.dino_models as dino_models

    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: _make_dummy_dino())

    model = dino_models.VideoTransformer(
        image_size=(224, 224),
        dim=32,
        action_embed_dim=8,
        state_embed_dim=6,
        state_dim=5,
        action_dim=4,
        action_horizon=dino_models.FUTURE_ACTION_HORIZON_MAX,
        trajectory_summary_dim=12,
        depth=1,
        heads=2,
        mlp_dim=64,
        num_frames=3,
        device="cpu",
        dino_version="v3",
    )

    future_actions = torch.zeros(1, dino_models.FUTURE_ACTION_HORIZON_MAX + 1, 4)
    with pytest.raises(ValueError):
        model._encode_future_actions(future_actions, num_frames=3)


def test_video_transformer_forward_shapes(monkeypatch):
    import dino_wm.dino_models as dino_models

    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: _make_dummy_dino())

    model = dino_models.VideoTransformer(
        image_size=(224, 224),
        dim=32,
        action_embed_dim=8,
        state_embed_dim=6,
        state_dim=5,
        action_dim=4,
        action_horizon=dino_models.FUTURE_ACTION_HORIZON_MAX,
        trajectory_summary_dim=12,
        depth=1,
        heads=2,
        mlp_dim=64,
        num_frames=3,
        device="cpu",
        dino_version="v3",
    )

    batch = 2
    num_frames = 3
    num_patches = model.num_patches
    dim = 32

    video1 = torch.zeros(batch, num_frames, num_patches, dim)
    video2 = torch.zeros(batch, num_frames, num_patches, dim)
    states = torch.zeros(batch, num_frames, 5)
    actions = torch.zeros(batch, num_frames, 4)
    future_actions = torch.zeros(batch, 7, 4)

    pred1, pred2, state_preds, failure_preds = model(
        video1, video2, states, actions, future_actions
    )

    assert pred1.shape == (batch, num_frames, num_patches, dim)
    assert pred2.shape == (batch, num_frames, num_patches, dim)
    assert state_preds.shape == (batch, num_frames, 5)
    assert failure_preds.shape == (batch, num_frames, 1)


def test_multihead_attention_mask_blocks_future_frames():
    import dino_wm.dino_models as dino_models

    attn = dino_models.MultiHeadAttention(
        dim=8,
        heads=2,
        dim_head=4,
        dropout=0.0,
        num_frames=3,
        patches_per_frame=4,
    )
    mask = attn._create_causal_mask(num_frames=3, patches_per_frame=4)

    # Frame 0 (patches 0..3) cannot attend to frame 1+ (patches 4..)
    assert torch.all(mask[0:4, 4:] == 0)
    # Frame 1 (patches 4..7) can attend to frames 0 and 1, but not frame 2
    assert torch.all(mask[4:8, 8:] == 0)
    assert torch.all(mask[4:8, 0:8] == 1)


def test_multihead_attention_expands_mask_for_more_frames():
    import dino_wm.dino_models as dino_models

    attn = dino_models.MultiHeadAttention(
        dim=8,
        heads=2,
        dim_head=4,
        dropout=0.0,
        num_frames=2,
        patches_per_frame=4,
    )
    # seq_len=12 -> 3 frames (mask should expand)
    x = torch.zeros(1, 12, 8)
    _ = attn(x)
    assert attn.mask.shape[0] >= 12


def test_multihead_attention_rejects_bad_seq_len():
    import dino_wm.dino_models as dino_models

    attn = dino_models.MultiHeadAttention(
        dim=8,
        heads=2,
        dim_head=4,
        dropout=0.0,
        num_frames=2,
        patches_per_frame=4,
    )
    x = torch.zeros(1, 10, 8)  # 10 not divisible by 4
    with pytest.raises(ValueError):
        attn(x)


def test_video_transformer_wan_backbone_uses_provided_latent_shape(monkeypatch):
    import dino_wm.dino_models as dino_models

    def _fail_load(*args, **kwargs):
        raise AssertionError("torch.hub.load should not be called for WAN backbone")

    monkeypatch.setattr(torch.hub, "load", _fail_load)

    model = dino_models.VideoTransformer(
        image_size=(224, 224),
        dim=16,
        action_embed_dim=8,
        state_embed_dim=6,
        state_dim=5,
        action_dim=4,
        action_horizon=dino_models.FUTURE_ACTION_HORIZON_MAX,
        trajectory_summary_dim=12,
        depth=1,
        heads=2,
        mlp_dim=64,
        num_frames=3,
        device="cpu",
        backbone="wan",
        num_patches=64,
    )

    batch = 2
    num_frames = 3
    video1 = torch.zeros(batch, num_frames, 64, 16)
    video2 = torch.zeros(batch, num_frames, 64, 16)
    states = torch.zeros(batch, num_frames, 5)
    actions = torch.zeros(batch, num_frames, 4)
    future_actions = torch.zeros(batch, 7, 4)

    pred1, pred2, state_preds, failure_preds = model(
        video1, video2, states, actions, future_actions
    )

    assert model.num_patches == 64
    assert pred1.shape == (batch, num_frames, 64, 16)
    assert pred2.shape == (batch, num_frames, 64, 16)
    assert state_preds.shape == (batch, num_frames, 5)
    assert failure_preds.shape == (batch, num_frames, 1)
