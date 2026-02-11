# World Model Training Guide

This guide outlines best practices for training and evaluating the DINO-based VideoTransformer World Model.

## 1. Quickstart

Train the World Model using the default configuration:

```bash
python dino_wm/train_dino_wm.py --config configs/dino_wm_config.yaml
```

The script supports both YAML configuration and CLI flags. CLI flags will override values specified in the YAML file.

### Data Normalization Defaults

The loaders default to `actions_delta` (actions minus state for shared dims) when present. Stats now include global 2%/98% quantiles (`action_delta_q02/q98`, `state_q02/q98`) and normalization uses those quantiles when available, with a min/max fallback for older stats files.

## 2. Key Metrics to Track

### Prediction Accuracy
*   **Teacher-Forcing Loss (`train_loss`):** How well the model predicts $t+1$ given ground-truth history up to $t$.
*   **Auto-Regressive Loss (`train_loss_ar`):** How well the model predicts $t+2$ given its own prediction of $t+1$. This is crucial for long-horizon stability.

### Training Stability
*   **Gradient Norm (`grad_norm`):** Spikes in gradient norm often indicate that the Transformer is struggling with specific transitions or is about to diverge.
*   **Weight Norm (`weight_norm`):** Steady increase may indicate overfitting or weights pushing into saturating regions of activations.

### Visual Quality
*   **Imagine Videos:** Periodically log auto-regressive rollouts. Check for:
    *   **Drift:** Do objects melt or disappear over time?
    *   **Action Conditionality:** Does the predicted movement match the provided actions?
    *   **Consistency:** Are the front and wrist views spatially consistent?

## 2. Best Practices

### Multi-Group Learning Rates
The World Model uses different learning rates for different components (e.g., higher LR for encoders and embeddings, lower LR for the Transformer backbone).
*   **Recommendation:** Use a **Cosine Annealing** schedule with warmup that scales all groups proportionally from their base LRs.

### Auto-Regressive Warmup
Start with a higher weight on Teacher-Forcing loss and gradually increase the importance of AR loss or increase the AR horizon as training progresses.

### Handling Preemption
Use `--auto-resume` to ensure that training can be interrupted and resumed without losing the optimizer state or restarting the LR schedule and data shuffle.

## 3. Evaluating Training Runs

Once training is underway, monitor the following trends in WandB to ensure convergence and stability.

### Expected Log Trends

1.  **Teacher-Forcing Loss (`train_loss`):**
    *   **Trend:** Should drop sharply in the first 5-10k iterations and then enter a slower, steady decline.
    *   **Baseline:** For DINOv3 embeddings, a loss below **0.05** usually indicates the model has learned the basic spatial structure.

2.  **Auto-Regressive Loss (`train_loss_ar`):**
    *   **Trend:** Usually 2-5x higher than TF loss. It should trend downward in sync with TF loss.
    *   **Warning Sign:** If `train_loss_ar` stays flat while `train_loss` drops, the model is likely "cheating" by relying too heavily on ground-truth history and failing to transition to its own predictions.

3.  **Gradient Norm (`grad_norm`):**
    *   **Trend:** Should stabilize after the warmup period.
    *   **Warning Sign:** Frequent spikes (e.g., > 5.0) may indicate that the learning rate is too high for the Transformer backbone or that there are corrupted samples in the dataset.

4.  **Evaluation Loss (`eval_loss`):**
    *   **Trend:** Should closely track `train_loss`.
    *   **Overfitting:** A gap that widens significantly after 50k iterations suggests the model is overfitting to specific trajectories. Consider increasing data augmentation or reducing model capacity.

### Qualitative Analysis (WandB Table)

The training script logs an `eval_failures_table` containing the highest-loss trajectories from the evaluation set.

*   **Check the "Worst" Samples:** Inspect the videos for the trajectories with the highest `rollout_mse`.
*   **Failure Modes:**
    *   **Melting:** Objects lose their shape over time. This often means the AR horizon or weight is too low.
    *   **Action Neglection:** The gripper moves in the predicted video regardless of the input action. This suggests the action encoder needs a higher learning rate.
    *   **Spatial Drift:** The front and wrist views diverge (e.g., the gripper is visible in the wrist cam but its shadow is missing in the front cam).
