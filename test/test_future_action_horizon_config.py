import importlib
from pathlib import Path


def test_future_action_horizon_reads_config():
    config_path = (
        Path(__file__).resolve().parents[1] / "configs" / "wm_config.yaml"
    )
    original = config_path.read_text()
    new_val = 123

    try:
        lines = original.splitlines()
        replaced = False
        for i, line in enumerate(lines):
            if line.lstrip().startswith("action_horizon:"):
                indent = line.split("action_horizon:")[0]
                lines[i] = f"{indent}action_horizon: {new_val}"
                replaced = True
                break
        if not replaced:
            lines.append(f"action_horizon: {new_val}")
        config_path.write_text("\n".join(lines) + "\n")

        import dino_wm.dino_models as dino_models

        importlib.reload(dino_models)
        assert dino_models.FUTURE_ACTION_HORIZON_MAX == new_val
    finally:
        config_path.write_text(original)
        import dino_wm.dino_models as dino_models

        importlib.reload(dino_models)
