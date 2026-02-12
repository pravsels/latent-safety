# DDP Issues in This Repo

**Glossary of acronyms:**
- **DDP** -- DistributedDataParallel (PyTorch's multi-GPU training wrapper)
- **NCCL** -- NVIDIA Collective Communications Library (handles GPU-to-GPU communication: all-reduce, broadcast, etc.)
- **FSDP** -- Fully Sharded Data Parallel (alternative to DDP that shards model parameters across GPUs)
- **SLURM** -- Simple Linux Utility for Resource Management (HPC job scheduler)
- **NVLink** -- NVIDIA's high-bandwidth GPU-to-GPU interconnect on the same node
- **PCIe** -- Peripheral Component Interconnect Express (fallback GPU interconnect, slower than NVLink)

Audit of multi-GPU training compared to [openpi](../openpi), which produces
clean single-stream logs and a single wandb run even across 3 GPUs.

---

## Background: How PyTorch DDP Works

PyTorch DDP launches **one OS process per GPU**. Each process runs the full
training script independently. They coordinate via NCCL to synchronise
gradients, but everything else -- printing, logging, file I/O -- happens
independently in every process unless you explicitly guard it.

Five environment variables must be set for `torch.distributed.init_process_group`
to connect the processes:

| Variable | What it does |
|---|---|
| `RANK` | Global rank of this process across **all nodes** (0, 1, 2, ..., `WORLD_SIZE-1`). The process with `RANK=0` is the "main" process and is the only one that should log, save checkpoints, and initialise wandb. |
| `WORLD_SIZE` | Total number of processes participating in training. Each process needs this so it knows how many peers to wait for during gradient all-reduce. |
| `LOCAL_RANK` | GPU index **on this specific node** (0, 1, 2, ...). Resets to 0 on each node. Used by `torch.cuda.set_device(local_rank)` so each process binds to a different GPU. Without this, all processes try to use GPU 0. |
| `MASTER_ADDR` | IP or hostname of the rank-0 node. All processes connect to this address to establish the NCCL communication group. For single-node training it's usually `localhost` or the node hostname. |
| `MASTER_PORT` | Port the rank-0 process listens on for the initial rendezvous. Must be a free port; conventionally `29500`. All processes must use the same port. |

`RANK` and `LOCAL_RANK` are the same in single-node training but diverge
in multi-node setups. Example with 2 nodes, 3 GPUs each (6 processes):

| Node | Process | `RANK` | `LOCAL_RANK` |
|------|---------|--------|--------------|
| Node 0 | GPU 0 | 0 | 0 |
| Node 0 | GPU 1 | 1 | 1 |
| Node 0 | GPU 2 | 2 | 2 |
| Node 1 | GPU 0 | 3 | 0 |
| Node 1 | GPU 1 | 4 | 1 |
| Node 1 | GPU 2 | 5 | 2 |

`LOCAL_RANK` is used for CUDA device binding (per-machine GPU index).
`RANK` is used for logic guards (`if rank == 0: log(...)`) and NCCL
process identification.

**Caveat with SLURM cgroup isolation:** The table above shows `LOCAL_RANK`
as reported by `SLURM_LOCALID`. But when using `--gpus-per-task=1`, SLURM
isolates GPUs via cgroups so each task only **sees 1 GPU**. From each
task's perspective `torch.cuda.device_count() == 1` and only `cuda:0`
exists, even though `SLURM_LOCALID` might be 1 or 2. That's why
`init_distributed_from_env()` clamps with `local_rank % device_count()`
-- see Issue 11 for details.

When any of these are missing, `init_distributed_from_env()` falls through to
`WORLD_SIZE=1` and DDP is never initialised. Each process runs as an
independent single-GPU training job.

### How DDP Actually Works (Peer-to-Peer, No Master)

There is no master process during training. All ranks are equals.

**Startup (rendezvous):** `MASTER_ADDR`/`MASTER_PORT` are used *only once*
for the initial handshake. Rank 0 opens a TCP socket on that address/port,
ranks 1 and 2 connect to it, they exchange GPU topology info, and the
NCCL communication group is formed. After this, `MASTER_ADDR` is never
used again.

**During training, each process independently:**
1. Loads its own batch (different data, thanks to `DistributedSampler`)
2. Runs forward pass on its own GPU
3. Computes its own loss and gradients

**Gradient all-reduce:** NCCL then performs an **all-reduce** across all
GPUs. This is not a "master collects and redistributes" pattern -- it's a
ring algorithm where each GPU sends a chunk to its neighbor, receives a
chunk from the other side, accumulates, and passes it on:

```
GPU 0 ──gradients──> GPU 1 ──gradients──> GPU 2
  ^                                         |
  └─────────────gradients───────────────────┘
```

After one full loop, all 3 GPUs have the exact same **averaged** gradient.
No single GPU did all the work.

**After all-reduce:** Each process applies the identical averaged gradient
to its own copy of the model weights via `optimizer.step()`. Since they
all started with the same weights and applied the same gradient, the
models stay in sync.

**Rank 0 is only "special" for I/O:** logging, wandb, checkpoint saving.
That's a convention we enforce with `if is_rank0:`. From NCCL's
perspective, rank 0 is no different from rank 1 or 2. `RANK` doesn't mean
"rank 0 is the boss" -- it just means "you are process number N in a
group of peers."

---

## How openpi Avoids All These Problems

openpi uses **JAX**, which has a fundamentally different parallelism model.
JAX sees all GPUs as devices within a **single OS process**. Their SLURM
script launches:

```bash
srun --ntasks=1 --gpus-per-task=3 ...
```

One process, three GPUs. JAX's `jax.jit` with mesh shardings (FSDP)
automatically distributes data and parameters across the device mesh.
Because there's only one process, every `print()`, `logging.info()`,
`wandb.init()`, and `wandb.log()` call executes exactly once. No rank
guards needed.

Key functions in openpi:
- `sharding.make_mesh()` -- creates JAX device mesh across GPUs
- `jax.jit(train_step, in_shardings=..., out_shardings=...)` -- auto-shards computation
- `pbar.write()` -- single progress bar from the single process
- `wandb.log()` -- called once, no guard needed

---

## Issue 1: `slurm/train_dinowm.sh` Missing DDP Environment Variables

### The problem

`train_dinowm.sh` launches 3 SLURM tasks but **never exports the DDP env vars**:

```bash
# slurm/train_dinowm.sh, lines 76-88
srun --ntasks=3 --gpus-per-task=1 --cpu-bind=cores \
apptainer exec --nv \
    ...
    bash -c "export PYTHONPATH=... && \
        export WANDB_DIR=... && \
        export CUDA_VISIBLE_DEVICES=0,1,2 && \
        ... && ${TRAIN_CMD}"
```

Compare with `train_wan_wm.sh` which correctly does:

```bash
# slurm/train_wan_wm.sh, lines 79-81
export RANK=${SLURM_PROCID} WORLD_SIZE=${SLURM_NTASKS} LOCAL_RANK=${SLURM_LOCALID} && \
export MASTER_ADDR=$(scontrol show hostnames ${SLURM_NODELIST} | head -n 1) && \
export MASTER_PORT=${MASTER_PORT:-29500} && \
```

### The consequence

`init_distributed_from_env()` in `train_wm_common.py` reads:

```python
rank = int(os.environ.get("RANK", "0"))           # -> 0 for all 3
world_size = int(os.environ.get("WORLD_SIZE", "1"))  # -> 1 for all 3
local_rank = int(os.environ.get("LOCAL_RANK", "0"))  # -> 0 for all 3
is_distributed = world_size > 1                       # -> False for all 3
```

All 3 processes think they are a standalone single-GPU job. DDP is never
initialised. Each runs the full training loop independently with its own
optimizer, its own wandb run, its own checkpoint writes -- all competing
for the same GPU memory and writing to the same files.

### Files to fix

- `slurm/train_dinowm.sh` -- add the same `RANK`/`WORLD_SIZE`/`LOCAL_RANK`/`MASTER_ADDR`/`MASTER_PORT` exports as `train_wan_wm.sh`

---

## Issue 2: Unguarded `print()` Calls (Duplicate Output Even When DDP Works)

Even when DDP **is** correctly initialised (e.g., via `train_wan_wm.sh`),
many `print()` calls run on every rank, producing 3x duplicate output.

### Every-iteration training progress (worst offender)

```python
# train_wm_common.py, line 479 (inside run_train_eval_loop)
print(
    f"\rIter {i} | lr {optimizer.param_groups[0]['lr']:.2e} | TF {loss_tf:.4f} | AR {loss_ar:.4f} | grad {grad_norm:.2f} | weight {weight_norm:.2f}",
    end="",
    flush=True,
)
```

This runs on **all ranks, every iteration**. It's the source of the garbled
interleaved output you see in the SLURM logs. Must be guarded with
`if is_rank0:`.

### Startup prints in `train_dino_wm.py`

All of these run on every rank:

| Line(s) | What it prints |
|---------|----------------|
| 345 | `"Backbone flow: dino | ..."` |
| 349 | `f"Loading dataset stats from {stats_path}"` |
| 360-361 | `f"Loaded state normalization stats..."` and `f"Inferred state_dim=..."` |
| 385-387 | `f"Dataset: ..."`, `f"  Train: ..."`, `f"  Eval: ..."` |
| 420-424 | Decoder checkpoint warnings (missing/unexpected/mismatched keys) |
| 426 | `f"Loaded DINO decoder from ..."` |
| 461 | `f"Loaded previous best eval: ..."` |
| 508 | `f"Resuming from checkpoint: ..."` |
| 517-547 | All checkpoint loading warnings |
| 625 | `f"\nTraining complete. Best eval loss: ..."` |

### Startup prints in `train_wan_wm.py`

Same pattern -- all unguarded:

| Line(s) | What it prints |
|---------|----------------|
| 326-328 | WAN VAE decoder loaded / skipped messages |
| 367-369 | Checkpoint fallback warnings |
| 379 | `f"Resuming from checkpoint: ..."` |
| 388-397 | All checkpoint loading warnings |
| 475 | `f"\nTraining complete. Best eval loss: ..."` |

### What's already correctly guarded

These are the `is_rank0`-guarded calls that work properly:

| Location | What |
|----------|------|
| `train_dino_wm.py:288` | `wandb.init()` |
| `train_wan_wm.py:215` | `wandb.init()` |
| `train_wm_common.py:419-423` | Initial debug index prints (first iter only) |
| `train_wm_common.py:484` | `wandb.log()` (training metrics) |
| `train_wm_common.py:496` | Periodic latest checkpoint save |
| `train_wm_common.py:504-586` | Entire eval block |

### Files to fix

- `dino_wm/train_wm_common.py` -- guard line 479 with `if is_rank0:`
- `dino_wm/train_dino_wm.py` -- guard all startup/resume prints with `if is_rank0:`
- `dino_wm/train_wan_wm.py` -- guard all startup/resume prints with `if is_rank0:`

---

## Issue 3: Triple wandb Runs

### The problem

When DDP env vars are missing (Issue 1), all 3 processes have `is_rank0 = True`
because `rank` defaults to 0. So the `if is_rank0: wandb.init(...)` guard
passes for all 3, creating 3 separate wandb runs:

```
wandb: Run data is saved locally in .../offline-run-...-jg8egmzo
wandb: Run data is saved locally in .../offline-run-...-a8af6657
wandb: Run data is saved locally in .../offline-run-...-nen82o09
```

This is visible in the SLURM logs you pasted.

### The fix

Fixing Issue 1 (setting env vars) fixes this automatically -- only rank 0
will pass the `if is_rank0:` guard and the other two will skip
`wandb.init()`.

---

## Issue 4: `tqdm` Progress Bar Not Guarded

```python
# train_wm_common.py, line 381
for i in tqdm(range(start_iter, train_iter), desc="Training", unit="iter"):
```

This creates a `tqdm` progress bar on **every rank**, producing the garbled
overlapping progress bars visible in `slurm-2273682.err`. Should either:
- disable on non-rank0: `tqdm(..., disable=not is_rank0)`
- or replace with the rank0-only print approach

---

## Issue 5: `plt.savefig()` Race Condition

```python
# train_dino_wm.py, lines 619-622
if is_rank0:
    plt.legend()
os.makedirs(args.checkpoint_dir, exist_ok=True)
plt.savefig(os.path.join(args.checkpoint_dir, 'training_curve.png'))
```

`plt.savefig()` runs on all ranks but only rank 0 called `plt.legend()`.
All 3 processes write to the same file concurrently. The `plt.savefig()`
should be inside the `if is_rank0:` block.

Same pattern in `train_wan_wm.py` lines 471-474.

---

## Issue 6: Checkpoint File Writes Are Not Rank-Guarded Everywhere

### Latest checkpoint save (OK)

```python
# train_wm_common.py, line 496
if is_rank0 and args.save_every and (i % args.save_every == 0):
    torch.save(_make_ckpt_dict(i, current_best_eval), latest_ckpt_path)
```

This is correctly guarded.

### Eval checkpoint saves (OK)

Lines 564-568 in `train_wm_common.py` are inside the `if is_rank0:` eval
block, so they're guarded.

### Final `os.makedirs` (minor)

```python
# train_dino_wm.py, line 621
os.makedirs(args.checkpoint_dir, exist_ok=True)
```

This is harmless (idempotent), but it runs on all ranks unnecessarily.

---

## Issue 7: Eval DataLoaders Not Distributed

```python
# train_wm_common.py, lines 366-367
expert_loader_eval = iter(DataLoader(expert_data_eval, batch_size=args.batch_size, shuffle=True))
expert_loader_imagine = iter(DataLoader(expert_data_imagine, batch_size=1, shuffle=True))
```

Only the training loader uses `DistributedSampler`. The eval loaders use
plain `shuffle=True`, which means every rank loads the same eval data.
Since eval currently only runs on rank 0 (guarded), this isn't a
correctness bug, but it means non-rank-0 processes still read the HDF5
file to construct these datasets unnecessarily.

---

## Issue 8: `CUDA_VISIBLE_DEVICES` Conflicts With DDP `LOCAL_RANK`

```bash
# slurm/train_dinowm.sh, line 85
export CUDA_VISIBLE_DEVICES=0,1,2
```

When you set `CUDA_VISIBLE_DEVICES=0,1,2`, every process sees 3 GPUs. With
DDP properly configured, `LOCAL_RANK=0` maps to physical GPU 0,
`LOCAL_RANK=1` to physical GPU 1, etc., which happens to be correct.

But SLURM with `--gpus-per-task=1` already assigns one GPU per task via
cgroup isolation. Setting `CUDA_VISIBLE_DEVICES` manually can **override**
SLURM's GPU binding, causing multiple processes to land on the same
physical GPU or causing GPU index confusion.

The safer approach: remove `CUDA_VISIBLE_DEVICES` entirely and let SLURM
handle GPU assignment, or set it per-task:
`export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}`.

---

## Issue 9: No `dist.destroy_process_group()` Cleanup

Neither `train_dino_wm.py` nor `train_wan_wm.py` call
`torch.distributed.destroy_process_group()` at the end of training.
While not always required (the process exits anyway), it's good practice
and avoids hanging processes if NCCL cleanup fails.

---

## Issue 10: `scontrol` Not Available Inside Apptainer Container

The SLURM scripts originally resolved `MASTER_ADDR` inside the container:

```bash
export MASTER_ADDR=\$(scontrol show hostnames \${SLURM_NODELIST} | head -n 1)
```

`scontrol` is a SLURM host-side command and does not exist inside the
Apptainer container. The command substitution silently failed, leaving
`MASTER_ADDR` empty, which caused `init_process_group` to crash with:

```
ValueError: Error initializing torch.distributed using env:// rendezvous:
environment variable MASTER_ADDR expected, but not set
```

The fix: resolve `MASTER_ADDR` on the **host** before `srun`, then pass
the literal value into the container's `bash -c` string.

Note: `SLURM_PROCID`, `SLURM_NTASKS`, and `SLURM_LOCALID` do **not** have
this problem -- they are environment variables injected by `srun` into
each task, and they propagate into the container automatically. Only
`scontrol` (a command, not a variable) is unavailable inside the container.

---

## Issue 11: `LOCAL_RANK` vs SLURM cgroup GPU Isolation

With `--gpus-per-task=1`, SLURM uses cgroup isolation so each task only
sees **one GPU**. From each task's perspective, `torch.cuda.device_count()`
returns 1 and only `cuda:0` exists. But `LOCAL_RANK` (from `SLURM_LOCALID`)
is 0, 1, 2 -- so ranks 1 and 2 tried `torch.cuda.set_device(1)` and
`torch.cuda.set_device(2)`, which don't exist:

```
RuntimeError: CUDA error: invalid device ordinal
```

The root cause is that `LOCAL_RANK` means "which GPU on this node" in the
multi-GPU-visible sense, but with cgroup isolation each task's "node" only
has 1 GPU.

| Task | `RANK` | `LOCAL_RANK` (SLURM) | Visible GPUs | Physical GPU | `local_rank` after fix |
|------|--------|----------------------|-------------|-------------|----------------------|
| 0 | 0 | 0 | 1 (`cuda:0`) | GPU 0 | 0 |
| 1 | 1 | 1 | 1 (`cuda:0`) | GPU 1 | 0 |
| 2 | 2 | 2 | 1 (`cuda:0`) | GPU 2 | 0 |

Each task sees only its own GPU as `cuda:0`, but NCCL still knows about
all physical GPUs at the hardware level for inter-GPU communication.

The fix in `init_distributed_from_env()`:

```python
if torch.cuda.is_available():
    local_rank = local_rank % torch.cuda.device_count()
```

This works for both setups:
- **SLURM cgroup isolation** (`device_count=1`): `local_rank % 1 = 0` for all tasks
- **All GPUs visible** (torchrun, no isolation, `device_count=3`): `local_rank % 3` stays 0, 1, 2

---

## Issue 12: DDP Needs `find_unused_parameters=True`

The training loop crashes on the second iteration with:

```
RuntimeError: Expected to have finished reduction in the prior iteration
before starting a new one. This error indicates that your module has
parameters that were not used in producing loss.
```

Two reasons:

1. **Frozen `failure_head`:** `freeze_failure_head_for_wm_training()` sets
   `requires_grad=False` on all failure_head parameters. DDP expects every
   parameter to receive gradients during backward. The frozen parameters
   never do.

2. **Two forward passes per iteration:** The training loop runs two forward
   passes through the DDP model (teacher forcing + autoregressive) before
   a single `backward()`. DDP hooks into each forward pass to track which
   parameters need reduction. When the second forward starts before the
   first's reduction completes, DDP raises an error.

`find_unused_parameters=True` tells DDP to dynamically figure out which
parameters participate in the backward pass, rather than assuming all of
them do. This handles both the frozen head and the double-forward pattern.

```python
DistributedDataParallel(transition, device_ids=[local_rank], find_unused_parameters=True)
```

Note: this has a small performance overhead (DDP must traverse the
autograd graph after each backward to mark unused params). If we later
remove the double-forward pattern or unfreeze the failure_head, we could
remove this flag.

---

## Issue 13: NCCL Topology Detection Fails With `--gpus-per-task=1`

With `srun --gpus-per-task=1`, SLURM uses cgroup isolation so each task
only sees 1 GPU. NCCL uses NVML to discover GPU topology (NVLink vs PCIe
layout), but the cgroup hides the other GPUs from NVML:

```
nvmlDeviceGetHandleByPciBusId() failed: Not Found
```

The fix: remove `--gpus-per-task=1` from `srun` so all tasks see all GPUs
from the job-level `--gres=gpu:4`. Each task picks its own GPU via
`LOCAL_RANK`, and NCCL can see the full topology for inter-GPU
communication. GPU 3 sits idle.

---

## Fix Checklist

- [x] **1 (P0):** Add DDP env vars (`RANK`/`WORLD_SIZE`/`LOCAL_RANK`/`MASTER_ADDR`/`MASTER_PORT`) to `slurm/train_dinowm.sh`
- [x] **2 (P0):** Guard training progress print with `if is_rank0:` in `dino_wm/train_wm_common.py:479`
- [x] **3 (P0):** Guard all unguarded startup/resume prints with `if is_rank0:` in `dino_wm/train_dino_wm.py`
- [x] **4 (P0):** Guard all unguarded startup/resume prints with `if is_rank0:` in `dino_wm/train_wan_wm.py`
- [x] **5 (P1):** Disable tqdm on non-rank0 in `dino_wm/train_wm_common.py:381`
- [x] **6 (P1):** Move `plt.savefig()` inside `is_rank0` guard in `dino_wm/train_dino_wm.py:619-622` and `dino_wm/train_wan_wm.py:471-474`
- [x] **7 (P2):** Remove or fix `CUDA_VISIBLE_DEVICES` in `slurm/train_dinowm.sh`
- [x] **8 (P2):** Add `dist.destroy_process_group()` cleanup in `dino_wm/train_dino_wm.py` and `dino_wm/train_wan_wm.py`
- [x] **9 (P3):** ~Avoid constructing eval datasets on non-rank0~ -- **deferred**: datasets are lightweight index wrappers; actual data is only loaded on rank0 during eval. Adding conditional construction would add complexity for negligible savings.
- [x] **10 (P0):** Resolve `MASTER_ADDR` on the host before `srun` (not inside the container where `scontrol` is unavailable) in `slurm/train_dinowm.sh` and `slurm/train_wan_wm.sh`
- [x] **11 (P0):** Clamp `local_rank` to `% torch.cuda.device_count()` in `init_distributed_from_env()` to handle SLURM cgroup GPU isolation (`dino_wm/train_wm_common.py`)
- [x] **12 (P0):** Add `find_unused_parameters=True` to DDP wrapper in `dino_wm/train_dino_wm.py` and `dino_wm/train_wan_wm.py` (frozen failure_head + double forward pass)
- [x] **13 (P0):** Remove `--gpus-per-task=1` from `srun` in `slurm/train_dinowm.sh` and `slurm/train_wan_wm.sh` (NCCL can't detect topology with cgroup isolation)
