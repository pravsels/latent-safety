import numpy as np


def compute_action_deltas(actions: np.ndarray, states: np.ndarray) -> np.ndarray:
    if actions.ndim != 2 or states.ndim != 2:
        raise ValueError("actions and states must be 2D arrays (T, D).")
    k = min(actions.shape[1], states.shape[1])
    deltas = actions.copy()
    deltas[:, :k] = actions[:, :k] - states[:, :k]
    return deltas


def quantile_normalize(
    x: np.ndarray, q02: np.ndarray, q98: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    denom = q98 - q02
    if np.any(denom == 0):
        denom = denom + eps
    return (x - q02) / denom * 2.0 - 1.0


def write_actions_delta(
    group, actions: np.ndarray, states: np.ndarray, initialized: bool
) -> None:
    if states is None:
        raise ValueError("states required to compute actions_delta; missing observation.state")
    actions_delta = compute_action_deltas(actions, states)
    if not initialized:
        group.create_dataset(
            "actions_delta",
            data=actions_delta,
            maxshape=(None, *actions_delta.shape[1:]),
            chunks=True,
        )
    else:
        group["actions_delta"].resize(
            group["actions_delta"].shape[0] + actions_delta.shape[0], axis=0
        )
        group["actions_delta"][-actions_delta.shape[0] :] = actions_delta
