# Classifier Training Guide

This guide covers training the failure classifier head attached to a frozen DINO World Model.

## 1. Quickstart (DINOv3)

```bash
python dino_wm/train_dino_classifier.py \
  --hdf5-file /data/labeled/train.h5 \
  --dino-version v3 \
  --wm-checkpoint dino3_wm_checkpoints/best_wm.pth \
  --decoder-checkpoint dino3_decoder_checkpoints/best_decoder.pth \
  --checkpoint-dir dino3_wm_checkpoints
```

Optional: gradient-penalty variant (more stable, slower):

```bash
python dino_wm/train_dino_classifier_gp.py \
  --hdf5-file /data/labeled/train.h5 \
  --dino-version v3 \
  --wm-checkpoint dino3_wm_checkpoints/best_wm.pth \
  --decoder-checkpoint dino3_decoder_checkpoints/best_decoder.pth \
  --checkpoint-dir dino3_wm_checkpoints
```

## 2. Critical DINOv3 Requirements

- Embeddings in the HDF5 must match the DINO version (`cam_*_embd` patch count).
- The classifier uses a frozen World Model; only `failure_head` is trainable.
- The decoder is only used for visualization; it must match the same DINO version.

## 3. Key Metrics to Track

- `train_loss`: hinge-style failure loss on the current batch.
- `eval_loss`: same loss computed on held-out data.
- `video`: qualitative rollouts with predicted failures highlighted.

## 4. Best Practices

- Keep `--sequence-length` and `--context-length` consistent with the World Model.
- Use a small LR (default `5e-5` or `1e-4` for GP) to avoid overfitting the head.
- If `eval_loss` is flat, inspect label balance or increase dataset size.
- If loss collapses to zero quickly, verify label correctness and class balance.

## 5. What to Expect in WandB

- `train_loss` should drop quickly in the first 1-2k iters, then flatten.
- `eval_loss` should track `train_loss` and stay within ~10-30% gap.
- `eval/precision` will often start high while `eval/recall` lags; recall should improve steadily.
- Threshold sweep metrics (`eval/*@thr`) should show a clear trade-off; pick a threshold based on desired recall.
- `eval/f1` should peak around the threshold where false positives and false negatives balance.

## 6. What to Inspect in Videos

- Failure frames should tint red in the predicted rollouts.
- Compare timing: predicted failure should align with ground-truth failure moments.
- Look for systematic delay (head is too conservative) or early triggers (too aggressive).

