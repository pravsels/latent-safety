# World Model Training Implementation Plan

This document tracks the instrumentation improvements for World Model (VideoTransformer) training.

## Status Summary

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Logging & Visual Diagnostics | ✅ Completed |
| 2 | Optimization Schedule | ✅ Completed |
| 3 | Failure-Mode Surfacing | ✅ Completed |

---

## Phase 1: Logging & Visual Diagnostics ✅

**Goal:** Make training/eval behavior legible and debuggable in WandB.

- [x] Log `grad_norm` and `weight_norm` for stability monitoring
- [x] Implement robust auto-resume (model + optimizer + iter)
- [x] Save iterating checkpoints and persistent "best" model

**WandB outputs:**
- Time-series: `train_loss`, `train_loss_ar`, `eval_loss`, `grad_norm`, `weight_norm`, `lr`

---

## Phase 2: Optimization Schedule ✅

**Goal:** Improve convergence and reduce sensitivity to LR choice.

- [x] Warmup + cosine decay schedule (`--lr-schedule cosine`)
- [x] Apply schedule to multiple parameter groups proportionally
- [x] Log effective `lr` for each group

---

## Phase 3: Failure-Mode Surfacing ✅

**Goal:** Quickly inspect and understand worst-case behavior.

- [x] Log WandB Table of top-K highest `eval_loss` samples (with GT/pred videos)
- [x] Add trajectory and timestep metadata to evaluation samples
