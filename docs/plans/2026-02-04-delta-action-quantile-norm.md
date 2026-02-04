# Delta Action Quantile Norm Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Store `actions_delta = actions - state` in HDF5, use it by default, and normalize both `actions_delta` and `state` with global 2–98% quantiles.

**Architecture:** Add delta-action generation during dataset creation/consolidation, update loaders to prefer `actions_delta`, and switch normalization/stats to quantile-based scaling with fallback to existing min/max where needed.

**Tech Stack:** Python, h5py, NumPy, PyTorch, pytest.
---

### Task 1: Add delta-action helper + tests

**Files:**
- Create: `dino_wm/data_utils.py`
- Test: `test/test_action_delta_norm.py`

**Step 1: Write the failing test**

```python
import numpy as np
from dino_wm.data_utils import compute_action_deltas, quantile_normalize


def test_compute_action_deltas_shared_dims():
    actions = np.array([[2.0, 3.0, 9.0], [4.0, 5.0, 8.0]], dtype=np.float32)
    states = np.array([[1.0, 1.5], [2.0, 2.5]], dtype=np.float32)
    # shared dims = 2, last dim stays unchanged
    deltas = compute_action_deltas(actions, states)
    expected = np.array([[1.0, 1.5, 9.0], [2.0, 2.5, 8.0]], dtype=np.float32)
    assert np.allclose(deltas, expected)


def test_quantile_normalize_to_minus1_plus1():
    x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    q02 = np.array([0.0], dtype=np.float32)
    q98 = np.array([2.0], dtype=np.float32)
    out = quantile_normalize(x, q02, q98)
    assert np.allclose(out, np.array([-1.0, 0.0, 1.0], dtype=np.float32))
```

**Step 2: Run test to verify it fails**

Run: `pytest test/test_action_delta_norm.py -v`  
Expected: FAIL with `ModuleNotFoundError: dino_wm.data_utils`

**Step 3: Write minimal implementation**

```python
# dino_wm/data_utils.py
import numpy as np


def compute_action_deltas(actions: np.ndarray, states: np.ndarray) -> np.ndarray:
    if actions.ndim != 2 or states.ndim != 2:
        raise ValueError("actions and states must be 2D arrays (T, D).")
    k = min(actions.shape[1], states.shape[1])
    deltas = actions.copy()
    deltas[:, :k] = actions[:, :k] - states[:, :k]
    return deltas


def quantile_normalize(x: np.ndarray, q02: np.ndarray, q98: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return (x - q02) / (q98 - q02 + eps) * 2.0 - 1.0
```

**Step 4: Run test to verify it passes**

Run: `pytest test/test_action_delta_norm.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add dino_wm/data_utils.py test/test_action_delta_norm.py
git commit -m "feat: add delta action + quantile helpers"
```

---

### Task 2: Write `actions_delta` into HDF5 during dataset creation

**Files:**
- Modify: `scripts/lerobot_to_hdf5.py`
- Modify: `dino_wm/hdf5_to_dataset.py`

**Step 1: Write the failing test**

Add a focused unit test that constructs a tiny in-memory HDF5 (or uses a temporary file) and verifies that `actions_delta` is written when `actions` and `states` are present. Example:

```python
def test_lerobot_writer_adds_actions_delta(tmp_path):
    # Create minimal HDF5 and call the helper that writes datasets
    # Expect actions_delta to exist and match compute_action_deltas
    ...
```

**Step 2: Run test to verify it fails**

Run: `pytest test/test_action_delta_norm.py::test_lerobot_writer_adds_actions_delta -v`  
Expected: FAIL (missing actions_delta)

**Step 3: Implement delta creation**

In `scripts/lerobot_to_hdf5.py`, after `act_np` and `st_np` are computed:

```python
from dino_wm.data_utils import compute_action_deltas

if st_np is None:
    raise ValueError("states required to compute actions_delta; missing observation.state")
actions_delta = compute_action_deltas(act_np, st_np)
```

Write `actions_delta` alongside `actions` (both on init and append).

In `dino_wm/hdf5_to_dataset.py`, during consolidation:
- If `actions_delta` exists, copy it.
- Else if `actions` and `states` exist, compute and create `actions_delta` in the consolidated file.

**Step 4: Run test to verify it passes**

Run: `pytest test/test_action_delta_norm.py::test_lerobot_writer_adds_actions_delta -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/lerobot_to_hdf5.py dino_wm/hdf5_to_dataset.py test/test_action_delta_norm.py
git commit -m "feat: write actions_delta during dataset build"
```

---

### Task 3: Update loader + validation to default to `actions_delta`

**Files:**
- Modify: `dino_wm/test_loader.py`
- Modify: `dino_wm/train_dino_wm.py`
- Modify: `dino_wm/train_dino_classifier.py`
- Modify: `dino_wm/train_dino_classifier_gp.py`
- Modify: `scripts/validate_hdf5_dataset.py`
- Modify: `configs/wm_config.yaml`

**Step 1: Write the failing test**

Extend `test_action_delta_norm.py` to ensure loader prefers `actions_delta` when present:

```python
def test_loader_prefers_actions_delta(tmp_path):
    # Build tiny HDF5 with actions + actions_delta
    # Load dataset and assert "action" equals actions_delta
    ...
```

**Step 2: Run test to verify it fails**

Run: `pytest test/test_action_delta_norm.py::test_loader_prefers_actions_delta -v`  
Expected: FAIL (loader uses actions)

**Step 3: Implement loader changes**

In `SplitTrajectoryDataset`, add an `action_key` parameter defaulting to `"actions_delta"`; fall back to `"actions"` if missing but log a warning.  
In training scripts, pass `action_key` from config; default to `actions_delta` in `configs/wm_config.yaml`.

In `validate_hdf5_dataset.py`, treat `actions_delta` as required (or required when `states` exist), and validate its length matches `actions`.

**Step 4: Run test to verify it passes**

Run: `pytest test/test_action_delta_norm.py::test_loader_prefers_actions_delta -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add dino_wm/test_loader.py dino_wm/train_dino_wm.py dino_wm/train_dino_classifier*.py scripts/validate_hdf5_dataset.py configs/wm_config.yaml
git commit -m "feat: default to actions_delta in loaders"
```

---

### Task 4: Quantile stats computation (2% / 98%) for state + actions_delta

**Files:**
- Modify: `scripts/compute_stats_json.py`

**Step 1: Write the failing test**

Add a small test that runs `compute_stats_json` on a tiny HDF5 and asserts `action_delta_q02/q98` and `state_q02/q98` keys exist with expected values.

**Step 2: Run test to verify it fails**

Run: `pytest test/test_action_delta_norm.py::test_compute_stats_quantiles -v`  
Expected: FAIL (keys missing)

**Step 3: Implement quantile stats**

Update the stats script to:
- Read `actions_delta` (fallback to `actions` if missing but warn)
- Compute global quantiles for `actions_delta` and `states`
- Save `action_delta_q02`, `action_delta_q98`, `state_q02`, `state_q98` keys

Implementation detail: if memory allows, concatenate and call `np.quantile`. If not, implement a two-pass histogram approach (pass 1 min/max, pass 2 histogram bins) to approximate quantiles.

**Step 4: Run test to verify it passes**

Run: `pytest test/test_action_delta_norm.py::test_compute_stats_quantiles -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/compute_stats_json.py test/test_action_delta_norm.py
git commit -m "feat: compute quantile stats for actions_delta and state"
```

---

### Task 5: Use quantile stats for normalization (actions_delta + state)

**Files:**
- Modify: `dino_wm/dino_models.py`
- Modify: `dino_wm/train_dino_wm.py`
- Modify: `dino_wm/train_dino_classifier.py`
- Modify: `dino_wm/train_dino_classifier_gp.py`
- Modify: `dino_wm/eval_dino_brt.py`
- Modify: `dino_wm/eval_dino_classifier.py`
- Modify: `scripts/dino-wm_inference.py`
- Modify: `scripts/benchmark_latent_safety_realtime.py`

**Step 1: Write the failing test**

Add a unit test validating quantile normalization in `normalize_acs` and `normalize_states` when quantile stats are provided.

**Step 2: Run test to verify it fails**

Run: `pytest test/test_action_delta_norm.py::test_normalize_uses_quantiles -v`  
Expected: FAIL

**Step 3: Implement normalization update**

Update `normalize_acs`/`normalize_states` to accept either:
- `q02/q98` tensors for quantile normalization, or
- fallback to min/max when quantiles missing.

Update training/eval scripts to read `action_delta_q02/q98` and `state_q02/q98` from the stats JSON and pass those to the normalization functions.

**Step 4: Run test to verify it passes**

Run: `pytest test/test_action_delta_norm.py::test_normalize_uses_quantiles -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add dino_wm/dino_models.py dino_wm/train_dino_wm.py dino_wm/train_dino_classifier*.py dino_wm/eval_dino_* scripts/dino-wm_inference.py scripts/benchmark_latent_safety_realtime.py test/test_action_delta_norm.py
git commit -m "feat: normalize actions_delta and state with quantiles"
```

---

### Task 6: Smoke check + docs update

**Files:**
- Modify: `README.md` (optional)
- Modify: `docs/decoder_training_guide.md` or `docs/wm_training_guide.md` (optional)

**Step 1: Run dataset validation**

Run: `python scripts/validate_hdf5_dataset.py <new_dataset>.h5`  
Expected: No missing datasets, actions_delta present.

**Step 2: Recompute stats**

Run: `python scripts/compute_stats_json.py --file <new_dataset>.h5 --output <new_stats>.json`  
Expected: stats include `action_delta_q02/q98`, `state_q02/q98`

**Step 3: Add short doc note**

Document that loaders default to `actions_delta` and normalization is quantile-based.

**Step 4: Commit**

```bash
git add README.md docs/*.md
git commit -m "docs: note actions_delta default and quantile norms"
```

---

## Execution Handoff

Plan complete and saved to `docs/plans/2026-02-04-delta-action-quantile-norm.md`. Two execution options:

1. **Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration  
2. **Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
