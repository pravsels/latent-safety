import os
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_DINO_WM_DIR = _ROOT / "dino_wm"
if str(_DINO_WM_DIR) not in sys.path:
    sys.path.insert(0, str(_DINO_WM_DIR))

from train_dino_wm import init_distributed_from_env


def test_ddp_env_parse(monkeypatch):
    calls = {}

    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "1")

    def _fake_set_device(idx):
        calls["set_device"] = idx

    class _FakeDistributed:
        def init_process_group(self, backend, rank, world_size):
            calls["init_process_group"] = (backend, rank, world_size)

    monkeypatch.setattr(torch.cuda, "set_device", _fake_set_device)
    monkeypatch.setattr(torch, "distributed", _FakeDistributed())

    rank, world_size, local_rank, is_distributed = init_distributed_from_env()

    assert (rank, world_size, local_rank, is_distributed) == (2, 4, 1, True)
    assert calls["init_process_group"] == ("nccl", 2, 4)
    assert calls["set_device"] == 1
