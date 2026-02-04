import importlib.util
import sys
from pathlib import Path


def test_action_horizon_cli_parsing():
    repo_root = Path(__file__).resolve().parents[1]
    dino_wm_dir = repo_root / "dino_wm"
    if str(dino_wm_dir) not in sys.path:
        sys.path.insert(0, str(dino_wm_dir))
    cli_path = dino_wm_dir / "train_dino_wm.py"
    spec = importlib.util.spec_from_file_location("wm_train", cli_path)
    assert spec is not None and spec.loader is not None
    wm_train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wm_train)
    args = wm_train.parse_args(["--action-horizon", "5"])
    assert args.action_horizon == 5
