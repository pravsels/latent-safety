# Rigorous Decoder Training Guide

This guide outlines the best practices for training and evaluating the DINO-to-Pixel decoder. When training a decoder to serve as the foundation for a World Model or Policy, "loss going down" is insufficient. We must ensure semantic fidelity and structural precision.

## 1. Key Metrics to Track in WandB

### Semantic Fidelity (DINO Cycle Consistency)
The input to the decoder consists of DINO patch embeddings. A perfect decoder should reconstruct an image that, when passed back through the same DINO encoder, produces the original embeddings.
*   **Metric:** `dino_cycle_loss` (L2 or Cosine distance between original and re-encoded DINO features).
*   **What to look for:** If this distance remains high even as MSE decreases, the decoder is creating "hallucinated" pixels that look okay to humans but lack the semantic information the World Model requires.

### Structural Precision
Raw MSE is often biased toward low-frequency information (smooth areas), causing the model to ignore sharp edges or small objects (like the robot gripper).
*   **Metric:** `PSNR` (Peak Signal-to-Noise Ratio) and `SSIM` (Structural Similarity Index).
*   **Target:** For high-quality robotics tasks, aim for `SSIM > 0.85` and `PSNR > 25`.
*   **Visual Check:** Inspect the robot's gripper and object edges in `pred_front` vs `ground_truth_front`. They should be sharp, not "ghosting" or blurry.

### Training Stability
*   **Metric:** `grad_norm` and `weight_norm`.
*   **What to look for:** Sudden spikes in `grad_norm` often precede training instability. Increasing `weight_norm` suggests the model is starting to overfit by creating high-frequency weights to memorize specific training samples.

## 2. Visualizations for Deep Insight

### The Difference Map
Don't just log predicted images. Log the **error map**: `abs(ground_truth - prediction)`.
*   **Interpretation:** 
    *   **Dark background, bright objects:** The model has successfully learned the static environment but is struggling with dynamic objects.
    *   **Uniform gray/noisy:** The model has a general reconstruction error but isn't biased toward specific scene elements.
    *   **Edge highlights:** The model is "close" but has slight spatial misalignment.

### Hardest Sample Mining
Use WandB tables to sort by the highest `eval_loss`.
*   **Why:** The model's average performance is less important than its failure modes. If the model fails specifically during fast movements or contact events, the downstream World Model will fail at the most critical moments of the task.

## 3. Best Practices for Convergence

### Learning Rate Scheduling
Using a fixed learning rate is often sub-optimal.
*   **Recommended:** Use a **Cosine Annealing** schedule with a short warmup period. This helps the model settle into a sharper local minimum in the final stages of training.
*   **Caveat (cosine needs a horizon):** Cosine decay assumes you know (or can bound) the total training length. If you frequently stop early or extend runs, consider **ReduceLROnPlateau** (decays LR based on eval loss stagnation) instead.

### Perceptual Losses
If MSE results in blurry outputs, consider incorporating a **perceptual loss** (e.g., VGG16 feature matching). This forces the decoder to match the "features" of the image (edges, textures) rather than just the pixel-wise average color. The trainer supports `--perceptual-kind vgg16` (default) or `--perceptual-kind dino`.

### Generalization Baseline
Always compare `train_loss` vs `eval_loss` across different trajectories. A significant gap indicates that the decoder is memorizing background details of specific runs rather than learning a general-purpose mapping from DINO features to pixels.

