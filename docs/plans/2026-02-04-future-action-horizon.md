# Future Action Horizon Implementation Plan
 
> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
 
**Goal:** Add a future-action horizon input (raw-frame actions) while keeping existing checkpoints loadable.
 
**Architecture:** Keep the existing per-frame past-action flow; add a separate future-action encoder that summarizes the K-step action sequence into one embedding, broadcast it across frames/patches, and concatenate alongside existing features. Update training/eval slicing to predict only the t+K target.
 
**Tech Stack:** PyTorch, einops, HDF5 dataset loader, wandb.
 
---
 
### Task 1: Add action_horizon config + CLI plumbing
 
**Files:**
- Modify: `configs/wm_config.yaml`
- Modify: `dino_wm/train_dino_wm.py`
 
**Step 1: Write the failing test**
 
```python
def test_action_horizon_cli_parsing():
    # TODO: construct argv with --action-horizon 5 and assert args.action_horizon == 5
    # Expected to fail because the argument does not exist yet.
    assert False
```
 
**Step 2: Run test to verify it fails**
 
Run: `pytest tests/test_action_horizon.py::test_action_horizon_cli_parsing -v`  
Expected: FAIL with "argument not recognized" or assertion failure.
 
**Step 3: Write minimal implementation**
 
- Add `action_horizon: 1` to `configs/wm_config.yaml`.
- Add `--action-horizon` argument in `dino_wm/train_dino_wm.py`.
 
**Step 4: Run test to verify it passes**
 
Run: `pytest tests/test_action_horizon.py::test_action_horizon_cli_parsing -v`  
Expected: PASS.
 
**Step 5: Commit**
 
```bash
git add configs/wm_config.yaml dino_wm/train_dino_wm.py tests/test_action_horizon.py
git commit -m "feat: add action_horizon CLI/config"
```
 
---
 
### Task 2: Add future-action encoder to VideoTransformer
 
**Files:**
- Modify: `dino_wm/dino_models.py`
- Create: `tests/test_future_action_encoder.py`
 
**Step 1: Write the failing test**
 
```python
def test_future_action_encoder_shapes():
    # Given actions shape (B, K, A), encoder returns (B, action_embed_dim)
    # and broadcast shape (B, H, num_patches, action_embed_dim)
    assert False
```
 
**Step 2: Run test to verify it fails**
 
Run: `pytest tests/test_future_action_encoder.py::test_future_action_encoder_shapes -v`  
Expected: FAIL because encoder is not implemented.
 
**Step 3: Write minimal implementation**
 
- Add `future_action_encoder` to `VideoTransformer.__init__`:
  - input `(B, K, A)` → output `(B, action_embed_dim)`
  - choose a simple GRU + linear head or 1D conv + pooling.
- In `forward_features()`:
  - compute `future_action_emb = future_action_encoder(future_actions)`
  - broadcast to `(B, num_frames, num_patches, action_embed_dim)`
  - concatenate with existing `(video1, video2, past_action_emb, state_emb)` features.
 
**Step 4: Run test to verify it passes**
 
Run: `pytest tests/test_future_action_encoder.py::test_future_action_encoder_shapes -v`  
Expected: PASS.
 
**Step 5: Commit**
 
```bash
git add dino_wm/dino_models.py tests/test_future_action_encoder.py
git commit -m "feat: add future action encoder for horizon input"
```
 
---
 
### Task 3: Update training/eval slicing for future actions
 
**Files:**
- Modify: `dino_wm/train_dino_wm.py`
- Modify (if needed): `dino_wm/test_loader.py`
- Create: `tests/test_action_horizon_slicing.py`
 
**Step 1: Write the failing test**
 
```python
def test_action_horizon_slicing():
    # Build a tiny fake sequence and ensure:
    # - context uses H past frames
    # - future actions are actions[t:t+K]
    # - target is frame/state at t+K
    assert False
```
 
**Step 2: Run test to verify it fails**
 
Run: `pytest tests/test_action_horizon_slicing.py::test_action_horizon_slicing -v`  
Expected: FAIL because slicing logic is not implemented.
 
**Step 3: Write minimal implementation**
 
- In `train_dino_wm.py`, compute:
  - `ctx_idx` as before (using `pred_step` for past context)
  - `t = ctx_idx[-1]`
  - `future_actions = actions[:, t : t + action_horizon]` (raw frames)
  - `target` at `t + action_horizon` for front/wrist/state
  - increase `segment_length` to cover `t + action_horizon`
- Update loss to compare only the t+K target.
 
**Step 4: Run test to verify it passes**
 
Run: `pytest tests/test_action_horizon_slicing.py::test_action_horizon_slicing -v`  
Expected: PASS.
 
**Step 5: Commit**
 
```bash
git add dino_wm/train_dino_wm.py tests/test_action_horizon_slicing.py
git commit -m "feat: add fixed 100-step future action horizon"
```
 
---
 
### Task 4: Eval/logging adjustments
 
**Files:**
- Modify: `dino_wm/train_dino_wm.py`
 
**Step 1: Write the failing test**
 
```python
def test_eval_uses_t_plus_k_target_only():
    # Ensure eval compares predicted t+K to GT t+K without rollouts
    assert False
```
 
**Step 2: Run test to verify it fails**
 
Run: `pytest tests/test_eval_t_plus_k.py::test_eval_uses_t_plus_k_target_only -v`  
Expected: FAIL because eval still uses rollouts.
 
**Step 3: Write minimal implementation**
 
- Replace rollout loop with a single-step comparison at `t+K`.
- Optionally log a single predicted frame vs GT instead of a video.
 
**Step 4: Run test to verify it passes**
 
Run: `pytest tests/test_eval_t_plus_k.py::test_eval_uses_t_plus_k_target_only -v`  
Expected: PASS.
 
**Step 5: Commit**
 
```bash
git add dino_wm/train_dino_wm.py tests/test_eval_t_plus_k.py
git commit -m "feat: condition WM on fixed 100-step future actions"
```
 
---
 
## Execution Handoff
Plan complete and saved to `docs/plans/2026-02-04-future-action-horizon.md`. Two execution options:
 
1. Subagent-Driven (this session) — use superpowers:subagent-driven-development  
2. Parallel Session (separate) — open new session and use superpowers:executing-plans  
 
Which approach?
