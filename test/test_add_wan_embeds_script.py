import importlib.util
from pathlib import Path

import torch


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "add_wan_embeds_to_hdf5.py"
    spec = importlib.util.spec_from_file_location("scripts.add_wan_embeds_to_hdf5", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load scripts.add_wan_embeds_to_hdf5 module spec.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_flatten_latents_supports_5d():
    mod = _load_module()
    # (B, C, T, H, W) with T=1
    z = torch.randn(2, 16, 1, 3, 5)
    flat = mod._flatten_wan_latents(z)
    assert flat.shape == (2, 15, 16)


def test_flatten_latents_supports_4d():
    mod = _load_module()
    # (B, C, H, W)
    z = torch.randn(2, 16, 3, 5)
    flat = mod._flatten_wan_latents(z)
    assert flat.shape == (2, 15, 16)
