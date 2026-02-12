import torch
from torch import nn

from dino_wm.config import MODEL_CONFIG
from dino_wm.dino_models import FUTURE_ACTION_HORIZON_MAX, VideoTransformer
from dino_wm.train_wm_common import build_wm_optimizer, freeze_failure_head_for_wm_training


class _ToyTransition(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Linear(8, 8)
        self.state_head = nn.Linear(8, 8)
        self.front_head = nn.Linear(8, 8)
        self.wrist_head = nn.Linear(8, 8)
        self.failure_head = nn.Linear(8, 8)
        self.action_encoder = nn.Linear(8, 8)
        self.state_encoder = nn.Linear(8, 8)
        self.trajectory_encoder = nn.Linear(8, 8)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, 8))
        self.temp_embedding = nn.Parameter(torch.randn(1, 1, 8))
        # New module that is not part of explicit optimizer groups.
        self.extra_head = nn.Linear(8, 8)


def _optimizer_coverage(model: nn.Module):
    opt = build_wm_optimizer(model)
    grouped_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    trainable_named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    trainable_ids = {id(p) for _, p in trainable_named}
    missing = [n for n, p in trainable_named if id(p) not in grouped_ids]
    return grouped_ids, trainable_ids, missing


def test_build_wm_optimizer_covers_all_video_transformer_trainable_params():
    model = VideoTransformer(
        state_dim=7,
        action_dim=7,
        num_frames=3,
        action_horizon=FUTURE_ACTION_HORIZON_MAX,
        backbone="wan",
        dino_version="v3",
        num_patches=196,
        **dict(MODEL_CONFIG),
    )
    # WM-phase policy: failure_head is trained separately by classifier scripts.
    freeze_failure_head_for_wm_training(model)
    grouped_ids, trainable_ids, missing = _optimizer_coverage(model)
    assert not missing
    assert grouped_ids == trainable_ids


def test_build_wm_optimizer_detects_unassigned_trainable_params():
    model = _ToyTransition()
    freeze_failure_head_for_wm_training(model)
    _, _, missing = _optimizer_coverage(model)
    assert "extra_head.weight" in missing
    assert "extra_head.bias" in missing


def test_build_wm_optimizer_ignores_frozen_unassigned_params():
    model = _ToyTransition()
    freeze_failure_head_for_wm_training(model)
    # extra_head is intentionally not part of build_wm_optimizer groups.
    # Freezing it should make coverage checks pass.
    for p in model.extra_head.parameters():
        p.requires_grad = False
    grouped_ids, trainable_ids, missing = _optimizer_coverage(model)
    assert not missing
    assert grouped_ids == trainable_ids


def test_freeze_failure_head_for_wm_training_only_affects_failure_head():
    model = _ToyTransition()
    before_other = [p.requires_grad for p in model.transformer.parameters()]
    freeze_failure_head_for_wm_training(model)
    assert all(not p.requires_grad for p in model.failure_head.parameters())
    assert [p.requires_grad for p in model.transformer.parameters()] == before_other
