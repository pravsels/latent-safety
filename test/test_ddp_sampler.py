import sys
from pathlib import Path

from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

_ROOT = Path(__file__).resolve().parents[1]
_DINO_WM_DIR = _ROOT / "dino_wm"
if str(_DINO_WM_DIR) not in sys.path:
    sys.path.insert(0, str(_DINO_WM_DIR))

from train_dino_wm import build_train_loader


class _DummyDataset(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return idx


def test_ddp_uses_distributed_sampler():
    loader, sampler = build_train_loader(
        _DummyDataset(),
        batch_size=2,
        is_distributed=True,
        rank=1,
        world_size=4,
    )

    assert isinstance(sampler, DistributedSampler)
    assert loader.sampler is sampler
