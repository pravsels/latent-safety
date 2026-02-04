import importlib.util
import sys
from pathlib import Path


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


def test_action_horizon_slicing():
    wm_train = _load_train_module()
    ctx_idx, future_slice, target_idx, segment_length = wm_train.compute_action_horizon_indices(
        context_length=3,
        pred_step=2,
        action_horizon=4,
        device="cpu",
    )

    assert ctx_idx.tolist() == [0, 2, 4]
    assert future_slice.start == 4
    assert future_slice.stop == 8
    assert target_idx == 8
    assert segment_length == 9

    (
        ctx_idx_ar,
        future_slice_ar,
        target_idx_ar,
        ar_future_slice,
        ar_target_idx,
        segment_length_ar,
    ) = wm_train.compute_action_horizon_ar_indices(
        context_length=3,
        pred_step=2,
        action_horizon=4,
        device="cpu",
    )
    assert ctx_idx_ar.tolist() == [0, 2, 4]
    assert future_slice_ar.start == 4
    assert future_slice_ar.stop == 8
    assert target_idx_ar == 8
    assert ar_future_slice.start == 5
    assert ar_future_slice.stop == 9
    assert ar_target_idx == 9
    assert segment_length_ar == 10
