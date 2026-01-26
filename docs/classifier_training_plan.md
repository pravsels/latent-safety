# Classifier Training Implementation Plan

This document tracks training and instrumentation improvements for the failure classifier head.

## Status Summary

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | DINOv3 Alignment + Defaults | ✅ Done |
| 2 | Metrics & Calibration | ⬚ Not started |
| 3 | Data Balancing + Hard Cases | ⬚ Not started |

---

## Phase 1: DINOv3 Alignment + Defaults ✅

**Goal:** Make classifier training version-aware and safe by default.

- [x] Add `--dino-version` flag (v2 / v3)
- [x] Version-aware defaults for WM/decoder checkpoints and save directory
- [x] Resize eval visuals to DINO-specific decoder output size

---

## Phase 2: Metrics & Calibration ✅

**Goal:** Make evaluation actionable beyond a single loss scalar.

- [x] Log confusion stats (TP/FN/FP/TN) per eval interval
- [x] Log precision/recall/F1 by threshold sweep
- [ ] Add a simple threshold calibration script for deployment

---

## Phase 3: Data Balancing + Hard Cases ⬚

**Goal:** Improve generalization on rare failures.

- [ ] Add label-balancing sampler or class-weighted loss
- [ ] Log top-K worst trajectories for manual review

