import sys
from pathlib import Path
import random

_ROOT = Path(__file__).resolve().parents[1]
_DINO_WM_DIR = _ROOT / "dino_wm"
if str(_DINO_WM_DIR) not in sys.path:
    sys.path.insert(0, str(_DINO_WM_DIR))

from train_dino_wm import sample_future_action_window


def test_future_action_window_sampling():
    rng = random.Random(0)
    length = sample_future_action_window(
        action_horizon=100,
        future_action_steps_train=50,
        rng=rng,
    )
    assert 1 <= length <= 50

    rng = random.Random(1)
    length = sample_future_action_window(
        action_horizon=3,
        future_action_steps_train=50,
        rng=rng,
    )
    assert 1 <= length <= 3


def test_future_action_window_prefers_small():
    rng = random.Random(0)
    samples = [
        sample_future_action_window(
            action_horizon=100,
            future_action_steps_train=50,
            rng=rng,
        )
        for _ in range(1000)
    ]
    small = sum(1 for s in samples if s <= 20)
    assert 1 <= min(samples) <= max(samples) <= 50
