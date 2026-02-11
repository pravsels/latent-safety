import importlib.util
from pathlib import Path


def _load_inference_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "dino-wm_inference.py"
    spec = importlib.util.spec_from_file_location("scripts.dino_wm_inference", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load scripts.dino-wm_inference.py module spec.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_args_accepts_wan_backbone():
    mod = _load_inference_module()
    args = mod.parse_args(
        [
            "--wm-checkpoint",
            "wm.pth",
            "--hdf5-file",
            "data.h5",
            "--dataset-stats",
            "stats.json",
            "--backbone",
            "wan",
            "--wan-vae-model",
            "ByteDance/Video-As-Prompt-Wan2.1-14B",
        ]
    )
    assert args.backbone == "wan"
    assert args.wan_vae_model == "ByteDance/Video-As-Prompt-Wan2.1-14B"


def test_validate_args_requires_decoder_checkpoint_for_dino():
    mod = _load_inference_module()
    args = mod.parse_args(
        [
            "--wm-checkpoint",
            "wm.pth",
            "--hdf5-file",
            "data.h5",
            "--dataset-stats",
            "stats.json",
            "--backbone",
            "dino",
        ]
    )
    try:
        mod.validate_args(args)
    except ValueError as e:
        assert "--decoder-checkpoint is required when --backbone=dino" in str(e)
    else:
        raise AssertionError("Expected ValueError when dino backbone has no decoder checkpoint")
