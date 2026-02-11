import importlib.util
import sys
from pathlib import Path

import torch


def _load_train_module():
    repo_root = Path(__file__).resolve().parents[1]
    dino_wm_dir = repo_root / "dino_wm"
    if str(dino_wm_dir) not in sys.path:
        sys.path.insert(0, str(dino_wm_dir))
    train_path = dino_wm_dir / "train_dino_wm.py"
    spec = importlib.util.spec_from_file_location("wm_train", train_path)
    assert spec is not None and spec.loader is not None
    wm_train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wm_train)
    return wm_train


class _DummyTransition(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.Linear(4, 4)
        self.state_head = torch.nn.Linear(4, 4)
        self.front_head = torch.nn.Linear(4, 4)
        self.wrist_head = torch.nn.Linear(4, 4)
        self.action_encoder = torch.nn.Linear(4, 4)
        self.state_encoder = torch.nn.Linear(4, 4)
        self.trajectory_encoder = torch.nn.Linear(4, 4)
        self.pos_embedding = torch.nn.Parameter(torch.randn(1, 1, 4))
        self.temp_embedding = torch.nn.Parameter(torch.randn(1, 1, 4))


def test_wm_optimizer_includes_trajectory_encoder_params():
    wm_train = _load_train_module()
    transition = _DummyTransition()

    optimizer = wm_train.build_wm_optimizer(transition)

    all_optimized_param_ids = {
        id(p)
        for group in optimizer.param_groups
        for p in group["params"]
    }
    trajectory_param_ids = {id(p) for p in transition.trajectory_encoder.parameters()}

    assert trajectory_param_ids.issubset(all_optimized_param_ids)
