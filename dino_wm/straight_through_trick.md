# Straight-through estimator (VQ / discretization)

This repo uses the classic **straight-through estimator** to train through a non-differentiable quantization step (argmin + embedding lookup).

You’ll see this pattern in `Quantize.forward`:

```python
quantize = input + (quantize - input).detach()
```

## What problem it solves

The true quantization path:

- pick nearest code index via `argmin`
- map index → embedding vector

is **not differentiable**, so gradients from the decoder can’t reach the encoder latents.

## How the trick works

Let:

- \(x\) = `input` (encoder latents)
- \(q\) = `quantize` (true quantized vectors from the codebook)

Define:

\[
  y = x + \text{stopgrad}(q - x)
\]

where `stopgrad(·)` is `.detach()`.

- **Forward pass (values):** `detach()` does not change values, only gradients
  - \(y = x + (q - x) = q\)  → decoder sees the **real quantized** vectors
- **Backward pass (gradients):** `stopgrad(q - x)` has zero gradient
  - \(\frac{\partial y}{\partial x} = 1\) → encoder gets gradients **as if \(y=x\)** (identity)

Intuition:

- forward: “use discrete codebook vectors”
- backward: “pretend quantization was identity so training works”

