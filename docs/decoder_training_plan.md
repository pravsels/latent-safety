# Decoder Training Implementation Plan

This document tracks the instrumentation improvements for decoder training.

## Status Summary

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Logging & Visual Diagnostics | ✅ Done |
| 2 | Optimization Schedule | ✅ Done |
| 3 | Loss Upgrades (Perceptual + DINO cycle) | ✅ Done |
| 4 | Failure-Mode Surfacing | ⬚ Not started |

---

## Phase 1: Logging & Visual Diagnostics ✅

**Goal:** Make training/eval behavior legible and debuggable in WandB.

- [x] Log `eval_psnr` and `eval_ssim` (in addition to `eval_loss`)
- [x] Log difference maps (`abs(pred - gt)`) for front + wrist cameras
- [x] Log `grad_norm` and `weight_norm` for stability monitoring

**WandB outputs:**
- Time-series: `train_loss`, `eval_loss`, `eval_psnr`, `eval_ssim`, `grad_norm`, `weight_norm`
- Images: `ground_truth_front`, `pred_front`, `diff_front`, `ground_truth_wrist`, `pred_wrist`, `diff_wrist`

---

## Phase 2: Optimization Schedule ✅

**Goal:** Improve convergence and reduce sensitivity to LR choice.

- [x] Warmup + cosine decay schedule (`--lr-schedule cosine`)
- [x] ReduceLROnPlateau option (`--lr-schedule plateau`)
- [x] Log `lr` every step

**Config knobs:** `lr`, `lr_min`, `lr_warmup_iters`, `lr_schedule`, `plateau_*`

---

## Phase 3: Loss Upgrades ✅

**Goal:** Reduce blur and ensure reconstructions preserve task-relevant structure.

- [x] VGG16 perceptual loss (`--perceptual-kind vgg16`, weight via `--perceptual-weight`)
- [x] DINO perceptual loss (`--perceptual-kind dino`)
- [x] DINO cycle-consistency loss (`--dino-cycle-weight`, metric via `--dino-cycle-metric`)
- [x] Two-stage schedule: MSE-only until plateau, then enable perceptual + cycle losses

**WandB outputs:** `train_perceptual_loss`, `train_dino_cycle_loss`, `eval_dino_cycle_loss`, `extra_losses_enabled`

---

## Phase 4: Failure-Mode Surfacing ⬚

**Goal:** Quickly inspect and understand worst-case behavior.

- [ ] Log WandB Table of top-K highest `eval_loss` samples (with GT/pred/diff images + trajectory id / timestep)

**Done criteria:**
- WandB table shows worst examples with visuals and identifiers
