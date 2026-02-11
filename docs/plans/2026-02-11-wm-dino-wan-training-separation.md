# WM DINO/WAN Training Separation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split world-model training into clear DINO-only and WAN-only workflows while keeping shared training logic in a reusable common module.

**Architecture:** Extract shared training runtime (dataset/stats loading, train/eval loop, checkpointing, logging) into a common module. Keep `train_dino_wm.py` responsible only for DINO-specific wiring and `train_wan_wm.py` responsible only for WAN-specific wiring/CLI. Preserve behavior and checkpoint compatibility.

**Tech Stack:** Python, PyTorch, argparse, HDF5, WandB, pytest.

---

### Task 1: Add common training module shell and backward-compatible entrypoint behavior

**Files:**
- Create: `dino_wm/train_wm_common.py`
- Modify: `dino_wm/train_dino_wm.py`
- Test: `test/test_train_wan_entrypoint.py`

**Step 1: Write the failing test**

Add a test asserting core trainer entrypoint can be invoked with explicit argv list and still parses config overrides correctly.

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n latent_safety python -m pytest -q test/test_train_wan_entrypoint.py`

Expected: FAIL due to missing/incorrect common entrypoint glue.

**Step 3: Write minimal implementation**

- Create `dino_wm/train_wm_common.py` with minimal public API surface:
  - `load_stats(...)`
  - `build_datasets(...)`
  - `run_training_loop(...)` (initial scaffold)
- Keep `train_dino_wm.py` callable via `main(argv=None)` and ensure existing CLI behavior remains unchanged.

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n latent_safety python -m pytest -q test/test_train_wan_entrypoint.py`

Expected: PASS.

**Step 5: Commit**

```bash
git add dino_wm/train_wm_common.py dino_wm/train_dino_wm.py test/test_train_wan_entrypoint.py
git commit -m "refactor: add common wm training module scaffold"
```

---

### Task 2: Move shared train/eval/checkpoint loop into common module

**Files:**
- Modify: `dino_wm/train_wm_common.py`
- Modify: `dino_wm/train_dino_wm.py`
- Test: `test/test_video_transformer.py`
- Test: `test/test_action_delta_norm.py`

**Step 1: Write the failing test**

Add/adjust tests to ensure:
- DINO trainer still supports existing CLI/config combinations.
- Core training utility path does not alter model/data shapes expected by existing tests.

**Step 2: Run test to verify it fails**

Run:
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n latent_safety python -m pytest -q test/test_video_transformer.py test/test_action_delta_norm.py`

Expected: FAIL after introducing test assertions around shared runner integration.

**Step 3: Write minimal implementation**

- Move shared logic from `train_dino_wm.py` into `train_wm_common.py`:
  - stats loading/validation
  - dataset split and loader setup
  - optimizer/scheduler setup
  - checkpoint load/save helpers
  - shared train/eval loop
- Keep DINO specifics in DINO script:
  - DINO config selection
  - DINO decoder loading
  - backbone-specific runtime options

**Step 4: Run test to verify it passes**

Run:
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n latent_safety python -m pytest -q test/test_video_transformer.py test/test_action_delta_norm.py`

Expected: PASS.

**Step 5: Commit**

```bash
git add dino_wm/train_wm_common.py dino_wm/train_dino_wm.py test/test_video_transformer.py test/test_action_delta_norm.py
git commit -m "refactor: extract shared world model training loop"
```

---

### Task 3: Make WAN trainer fully explicit and workflow-focused

**Files:**
- Modify: `dino_wm/train_wan_wm.py`
- Modify: `configs/wan_wm_config.yaml`
- Modify: `slurm/train_wan_wm.sh`
- Test: `test/test_train_wan_entrypoint.py`

**Step 1: Write the failing test**

Add tests asserting WAN trainer:
- exposes WAN-focused defaults (`wan_wm_config.yaml`)
- enforces WAN backbone
- forwards unknown core args cleanly to shared runner.

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n latent_safety python -m pytest -q test/test_train_wan_entrypoint.py`

Expected: FAIL until CLI passthrough and WAN defaults are finalized.

**Step 3: Write minimal implementation**

- Update `train_wan_wm.py` to be an explicit WAN workflow entrypoint with:
  - clear WAN-centric argument help text
  - passthrough for advanced shared/core flags
  - explicit forcing of WAN backbone
- Align `slurm/train_wan_wm.sh` and `wan_wm_config.yaml` with the new WAN entrypoint contract.

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n latent_safety python -m pytest -q test/test_train_wan_entrypoint.py`

Expected: PASS.

**Step 5: Commit**

```bash
git add dino_wm/train_wan_wm.py configs/wan_wm_config.yaml slurm/train_wan_wm.sh test/test_train_wan_entrypoint.py
git commit -m "refactor: make wan trainer explicit and self-contained"
```

---

### Task 4: Validate end-to-end regression and document workflow split

**Files:**
- Modify: `docs/wan_vae_reading_guide.md`
- Modify: `docs/wm_training_guide.md`
- Test: `test/test_dino_wm_inference_backbone.py`
- Test: `test/test_add_wan_embeds_script.py`
- Test: `test/test_video_transformer.py`
- Test: `test/test_action_delta_norm.py`
- Test: `test/test_train_wan_entrypoint.py`

**Step 1: Write the failing test**

Add a lightweight doc/command consistency assertion test if project style permits; otherwise add explicit checklist assertions in existing test docs section.

**Step 2: Run test to verify it fails**

Run targeted tests to capture any behavioral drift before docs are finalized.

**Step 3: Write minimal implementation**

- Update docs to show:
  - DINO path: `train_dino_wm.py` + `dino_wm_config.yaml`
  - WAN path: `train_wan_wm.py` + `wan_wm_config.yaml`
  - latent-generation + SLURM command order for WAN

**Step 4: Run test to verify it passes**

Run:
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n latent_safety python -m pytest -q test/test_video_transformer.py test/test_action_delta_norm.py test/test_add_wan_embeds_script.py test/test_dino_wm_inference_backbone.py test/test_train_wan_entrypoint.py`

Expected: PASS.

**Step 5: Commit**

```bash
git add docs/wan_vae_reading_guide.md docs/wm_training_guide.md test/test_dino_wm_inference_backbone.py test/test_add_wan_embeds_script.py test/test_video_transformer.py test/test_action_delta_norm.py test/test_train_wan_entrypoint.py
git commit -m "docs: clarify separated dino and wan training workflows"
```
