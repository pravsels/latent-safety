# Latent Safety - Project Guide

## Overview

This repository implements HJ (Hamilton-Jacobi) reachability analysis using PyTorch, with a focus on learning safety value functions through reinforcement learning. The main innovation is **DINO-WM** - a DINO-based world model for latent safety filtering.

Key papers:
- [Discounted Safety/Reach-avoid Bellman equation](https://ieeexplore.ieee.org/abstract/document/8794107)
- [Learning Reachability](https://arxiv.org/abs/2112.12288)

## Project Structure

```
latent_safety/
├── PyHJ/                    # Core RL library (based on Tianshou 0.5.1)
│   ├── data/                # Replay buffers, collectors, batch utilities
│   ├── env/                 # Environment wrappers (Gym/Gymnasium)
│   ├── policy/              # RL policies (DDPG, SAC, reach-avoid variants)
│   ├── trainer/             # Training loops (on/off-policy, offline)
│   ├── reach_rl_gym_envs/   # Dubins car environments for reachability
│   └── utils/               # Networks, logging, schedulers
├── dino_wm/                 # DINO World Model implementation
│   ├── train_dino_decoder.py    # Train image decoder
│   ├── train_dino_wm.py         # Train world model
│   ├── train_dino_classifier.py # Train failure classifier
│   ├── eval_dino_*.py           # Evaluation scripts
│   ├── dino_models.py           # Model architectures
│   └── config.py                # Configuration
├── scripts/                 # Training and utility scripts
│   ├── run_training_ddpg-dinowm.py  # Main BRT training script
│   └── lerobot_to_hdf5.py           # Data conversion utilities
└── test/                    # Unit tests
```

## Setup

Python 3.12 recommended.

```bash
pip install -e .
conda install -c conda-forge ffmpeg
```

## DINO-WM Training Pipeline

1. **Collect data**: Store trajectories as `traj_XXXX.hdf5` in `/data`
2. **Label trajectories**: `cd dino_wm && python label.py`
3. **Consolidate data**: `python hdf5_to_dataset.py`
4. **Train decoder**: `python train_dino_decoder.py`
5. **Train world model**: `python train_dino_wm.py`
6. **Train classifier**: `python train_dino_classifier.py`
7. **Evaluate classifier**: `python eval_dino_classifier.py`
8. **Train safety filter**: `cd ../scripts && python run_training_ddpg-dinowm.py`
9. **Evaluate BRT**: `cd ../dino_wm && python eval_dino_brt.py`

## Key Components

### PyHJ Library
- **Policies**: `PyHJ/policy/modelfree/` contains DDPG and SAC with reach-avoid variants
- **Buffers**: Prioritized replay, vectorized buffers in `PyHJ/data/buffer/`
- **Environments**: Dubins car variants in `PyHJ/reach_rl_gym_envs/`

### DINO-WM
- Uses DINOv2 embeddings for visual representation
- World model predicts future latent states
- Failure classifier identifies unsafe states
- Integrated with DDPG for BRT (Backward Reachable Tube) learning

## Data Format

Trajectories stored as HDF5 files with structure:
- Images/observations
- Actions
- Failure labels (added via labeling script)

## Preferences

- **Commits**: Use concise, single-line commit messages
