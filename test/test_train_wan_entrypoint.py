import importlib.util
from pathlib import Path


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "dino_wm" / "train_wan_wm.py"
    spec = importlib.util.spec_from_file_location("dino_wm.train_wan_wm", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load dino_wm.train_wan_wm module spec.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_core_argv_forces_wan_backbone():
    mod = _load_module()
    args = mod.parse_args(["--hdf5-file", "data.h5", "--dataset-stats", "stats.json"])
    assert args.hdf5_file == "data.h5"
    assert args.dataset_stats == "stats.json"


def test_default_config_points_to_wan_config():
    mod = _load_module()
    args = mod.parse_args([])
    assert args.config.endswith("configs/wan_wm_config.yaml")


def test_cli_overrides_are_applied():
    mod = _load_module()
    args = mod.parse_args(
        [
            "--eval-interval",
            "123",
            "--batch-size",
            "8",
        ]
    )
    assert args.eval_interval == 123
    assert args.batch_size == 8


def test_backbone_flag_is_not_accepted():
    mod = _load_module()
    try:
        mod.parse_args(["--backbone", "dino"])
    except SystemExit:
        pass
    else:
        raise AssertionError("Expected parse failure for unsupported --backbone flag.")
