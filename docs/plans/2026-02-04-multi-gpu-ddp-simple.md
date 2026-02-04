# Multi-GPU DDP (Simple) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable single-node 4‑GPU training via `srun` with minimal code changes.

**Architecture:** Initialize torch distributed from Slurm env vars, wrap the model in `DistributedDataParallel`, shard the training loader with `DistributedSampler`, and gate eval/logging/checkpointing to rank 0. Update the Slurm script to launch 4 ranks.

**Tech Stack:** PyTorch DDP, Slurm `srun`.

---

### Task 1: Minimal DDP init + model wrap

**Files:**
- Modify: `dino_wm/train_dino_wm.py`

**Step 1: Write the failing test**

```python
def test_ddp_env_parse():
    assert False
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n latent_safety pytest test/test_ddp_env_parse.py::test_ddp_env_parse -v`  
Expected: FAIL (helper missing).

**Step 3: Write minimal implementation**

- Add `init_distributed_from_env()` to read `RANK`, `WORLD_SIZE`, `LOCAL_RANK`.
- If `WORLD_SIZE>1`, call `torch.distributed.init_process_group` and `torch.cuda.set_device(local_rank)`.
- Wrap `VideoTransformer` in `DistributedDataParallel` when distributed.
- Ensure `wandb.init`, checkpoint writes, and plots run only on rank 0.

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n latent_safety pytest test/test_ddp_env_parse.py::test_ddp_env_parse -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add dino_wm/train_dino_wm.py test/test_ddp_env_parse.py
git commit -m "feat: add minimal DDP setup for Slurm"
```

---

### Task 2: Shard training data with DistributedSampler

**Files:**
- Modify: `dino_wm/train_dino_wm.py`

**Step 1: Write the failing test**

```python
def test_ddp_uses_distributed_sampler():
    assert False
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n latent_safety pytest test/test_ddp_sampler.py::test_ddp_uses_distributed_sampler -v`  
Expected: FAIL.

**Step 3: Write minimal implementation**

- Use `DistributedSampler` for the **training** dataset when world>1.
- Call `sampler.set_epoch(i)` when re‑creating/iterating the loader.
- Leave eval loader rank‑0 only (simple path).

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n latent_safety pytest test/test_ddp_sampler.py::test_ddp_uses_distributed_sampler -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add dino_wm/train_dino_wm.py test/test_ddp_sampler.py
git commit -m "feat: shard training data for DDP"
```

---

### Task 3: Update Slurm script to 4 GPUs via srun

**Files:**
- Modify: `slurm/train_dinowm.sh`

**Step 1: Write the failing test**

```python
def test_slurm_requests_4_gpus():
    assert False
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n latent_safety pytest test/test_slurm_multi_gpu.py::test_slurm_requests_4_gpus -v`  
Expected: FAIL.

**Step 3: Write minimal implementation**

- Set `#SBATCH --gres=gpu:4` and `#SBATCH --ntasks-per-node=4`.
- Launch with `srun --ntasks=4 --gpus-per-task=1`.

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n latent_safety pytest test/test_slurm_multi_gpu.py::test_slurm_requests_4_gpus -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add slurm/train_dinowm.sh test/test_slurm_multi_gpu.py
git commit -m "feat: run DINO WM with 4 GPUs on Slurm"
```

---

## Execution Handoff

Plan complete and saved to `docs/plans/2026-02-04-multi-gpu-ddp-simple.md`. Two execution options:

1. Subagent-Driven (this session) — use superpowers:subagent-driven-development  
2. Parallel Session (separate) — open new session and use superpowers:executing-plans

Which approach?
