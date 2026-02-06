import importlib.util
import sys
import types
from pathlib import Path


def _ensure_dummy_robocandywrapper():
    if "robocandywrapper" in sys.modules:
        return
    robocandywrapper = types.ModuleType("robocandywrapper")
    dataformats = types.ModuleType("robocandywrapper.dataformats")
    lerobot_21 = types.ModuleType("robocandywrapper.dataformats.lerobot_21")

    class LeRobot21DatasetMetadata:
        def __init__(self, *args, **kwargs):
            pass

    lerobot_21.LeRobot21DatasetMetadata = LeRobot21DatasetMetadata
    dataformats.lerobot_21 = lerobot_21
    robocandywrapper.dataformats = dataformats

    sys.modules["robocandywrapper"] = robocandywrapper
    sys.modules["robocandywrapper.dataformats"] = dataformats
    sys.modules["robocandywrapper.dataformats.lerobot_21"] = lerobot_21


def _load_module():
    _ensure_dummy_robocandywrapper()
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "filter_hf_datasets.py"
    spec = importlib.util.spec_from_file_location("filter_hf_datasets", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DummyMeta:
    def __init__(self, camera_keys, codebase_version):
        self.camera_keys = camera_keys
        self.codebase_version = codebase_version


def test_is_supported_version_str():
    mod = _load_module()
    assert mod._is_supported_version_str(None) is False
    assert mod._is_supported_version_str("2.0") is False
    assert mod._is_supported_version_str("2.1") is True
    assert mod._is_supported_version_str("3.0") is True
    assert mod._is_supported_version_str("unknown") is False


def test_check_robot_type_accepts_single_arm_arx5(monkeypatch):
    mod = _load_module()
    meta = DummyMeta(
        camera_keys={"observation.images.front", "observation.images.wrist"},
        codebase_version="2.1",
    )
    monkeypatch.setattr(mod, "_load_metadata", lambda repo_id: meta)
    assert mod.check_robot_type("owner/ds", "arx5", verbose=False) is True


def test_check_robot_type_rejects_unsupported_version(monkeypatch):
    mod = _load_module()
    meta = DummyMeta(
        camera_keys={"observation.images.front", "observation.images.wrist"},
        codebase_version="2.0",
    )
    monkeypatch.setattr(mod, "_load_metadata", lambda repo_id: meta)
    assert mod.check_robot_type("owner/ds", "arx5", verbose=False) is False


def test_check_robot_type_rejects_multi_arm(monkeypatch):
    mod = _load_module()
    meta = DummyMeta(
        camera_keys={
            "observation.images.front",
            "observation.images.wrist",
            "observation.images.left_wrist",
        },
        codebase_version="3.0",
    )
    monkeypatch.setattr(mod, "_load_metadata", lambda repo_id: meta)
    assert mod.check_robot_type("owner/ds", "arx5", verbose=False) is False


def test_check_robot_type_allows_other_robot_type(monkeypatch):
    mod = _load_module()
    meta = DummyMeta(camera_keys=set(), codebase_version="3.0")
    monkeypatch.setattr(mod, "_load_metadata", lambda repo_id: meta)
    assert mod.check_robot_type("owner/ds", "ur5", verbose=False) is True
