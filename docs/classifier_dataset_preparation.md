# Classifier Dataset Preparation

How to go from a HuggingFace LeRobot dataset to a labeled HDF5 file ready for failure classifier training.

## Overview

```
HuggingFace dataset ──► HDF5 with embeddings ──► Label failures ──► Verify ──► Train classifier
     (lerobot_to_hdf5)                        (label_trajectories)  (verify_labels)
```

## Step 1: Convert LeRobot dataset to HDF5

Download a dataset from HuggingFace and convert it to HDF5 with camera images, actions, states, and embeddings.

```bash
# Single dataset (DINO v3 embeddings by default)
python scripts/lerobot_to_hdf5.py \
    --dataset villekuosmanen/fail_bil_pick_capsules_drop_on_table \
    --output-hdf5 fail_bil_pick_capsules_dino3.h5 \
    --batch-size 32

# Multiple datasets
python scripts/lerobot_to_hdf5.py \
    --dataset user/dataset_a user/dataset_b \
    --output-hdf5 combined.h5

# From a JSON list
python scripts/lerobot_to_hdf5.py \
    --datasets-list my_datasets.json \
    --output-hdf5 output.h5 --resume

# With WAN VAE embeddings instead of DINO
python scripts/lerobot_to_hdf5.py \
    --dataset villekuosmanen/fail_bil_pick_capsules_drop_on_table \
    --output-hdf5 fail_bil_pick_capsules_wan.h5 \
    --embed-type wan
```

**Tip:** Use `--max-episodes-per-dataset N` to do a quick test run first.

### What this produces

Each trajectory group in the HDF5 contains:

| Dataset | Shape | Description |
|---------|-------|-------------|
| `camera_0` | (T, H, W, 3) uint8 | Wrist camera |
| `camera_1` | (T, H, W, 3) uint8 | Front camera |
| `cam_rs_embd` | (T, 196, 384) float32 | Wrist DINO embeddings (or `wan_wrist_embd` for WAN) |
| `cam_zed_embd` | (T, 196, 384) float32 | Front DINO embeddings (or `wan_front_embd` for WAN) |
| `actions` | (T, D) float32 | Raw actions |
| `actions_delta` | (T, D) float32 | Actions minus states |
| `states` | (T, D) float32 | Robot state |

## Step 2: Label failure regions

Open the labeling UI to mark unsafe frames in each trajectory.

```bash
python scripts/label_trajectories.py --hdf5 fail_bil_pick_capsules_dino3.h5
```

### Controls

| Key | Action |
|-----|--------|
| Space | Play / Pause |
| Left / Right | Previous / Next frame (accelerates on hold) |
| Up / Down | Next / Previous trajectory |
| U | Toggle Unsafe region (press to start, press again to end) |
| W | Toggle Weak Unsafe region |
| Z | Undo |
| C | Clear all labels for current trajectory |
| Q | Save & Quit |

### Label values

- **0** = Safe (default, unmarked frames)
- **1** = Unsafe (hard failure)
- **2** = Weak Unsafe (borderline / approaching failure)

Progress is saved to a `.session.json` file so you can quit and resume later.

## Step 3: Verify labels

Check that all trajectories have been labeled and inspect class balance.

```bash
python scripts/verify_labels.py fail_bil_pick_capsules_dino3.h5
```

Example output:

```
Trajectories: 102
  trajectory_0: 178 frames | safe=120 unsafe=58 weak=0 (33% failure)
  trajectory_1: 158 frames | safe=100 unsafe=50 weak=8 (37% failure)
  ...

Summary
  Total trajectories:  102
  Total frames:        19282
  Safe frames:         13996 (72.6%)
  Unsafe frames:       5016 (26.0%)
  Weak unsafe frames:  270 (1.4%)

  Labeled with failures: 102/102
```

**What to look for:**
- No trajectories with missing labels
- Reasonable failure percentage (15-40% is typical)
- No trajectories that should have failures but show as all-safe

## Step 4: Train the classifier

Once verified, proceed to classifier training (see `docs/classifier_training_guide.md`):

```bash
python dino_wm/train_dino_classifier.py \
    --hdf5-file fail_bil_pick_capsules_dino3.h5 \
    --dino-version v3 \
    --wm-checkpoint dino3_wm_checkpoints/best_wm.pth \
    --decoder-checkpoint dino3_decoder_checkpoints/best_decoder.pth \
    --checkpoint-dir dino3_classifier_checkpoints
```

## Scripts reference

| Script | Purpose |
|--------|---------|
| `scripts/lerobot_to_hdf5.py` | Convert HuggingFace datasets to HDF5 with embeddings |
| `scripts/label_trajectories.py` | GUI for labeling safe/unsafe frame regions |
| `scripts/verify_labels.py` | Verify label completeness and class balance |
| `scripts/validate_hdf5_dataset.py` | General HDF5 integrity check |
| `scripts/add_dino_embeds_to_hdf5.py` | Add/replace DINO embeddings on existing HDF5 |
| `scripts/add_wan_embeds_to_hdf5.py` | Add/replace WAN embeddings on existing HDF5 |
| `scripts/combine_hdf5_datasets.py` | Merge multiple HDF5 files into one |
