# WAN VAE Reading Guide

This note is a focused guide for understanding how WAN VAE works before integrating it into the `latent_safety` world-model pipeline.

## What to read first

1. `../Wan2.1/wan/modules/vae.py`  
   Source of truth for temporal behavior, chunking, and encode/decode APIs.
2. `../Wan2.1/tools/roundtrip_compare.py`  
   Practical usage (load video, encode, decode, measure PSNR).
3. `../Wan2.1/README.md`  
   High-level context and links to the technical report.

## Key questions and where to answer them

### 1) "What tensor shape does WAN VAE take?"

Read `WanVAE.encode()` and docstring in `../Wan2.1/wan/modules/vae.py`.

- Input videos are described as shape `[C, T, H, W]` (list of clips).
- Internally, encode/decode work with batched tensors shaped `[B, C, T, H, W]`.

### 2) "Why do people say 1 + 4k frames?"

Read `WanVAE_.encode()` in `../Wan2.1/wan/modules/vae.py`.

- It computes `iter_ = 1 + (t - 1) // 4`.
- It feeds the encoder in chunks:
  - first chunk: `x[:, :, :1, :, :]` (one frame),
  - then chunks of 4 frames:
    `x[:, :, 1 + 4*(i-1) : 1 + 4*i, :, :]`.
- This yields the natural pattern `T in {1, 5, 9, 13, ...}` for clean coverage.

### 3) "Does encoding use future frames?"

Read `CausalConv3d` in `../Wan2.1/wan/modules/vae.py`.

- WAN VAE uses causal 3D convolutions.
- Temporal padding/caching is designed so a timestep depends on past + present, not future.
- This makes it suitable for causal world-model inputs when windows are chosen correctly.

### 4) "How is temporal compression handled?"

Read `Resample` + encoder/decoder construction in `../Wan2.1/wan/modules/vae.py`.

- There is temporal down/upsampling configured by `temperal_downsample`.
- The common behavior is roughly 4x temporal compression in latent time.
- Verify exact `T -> Tz` mapping for your selected checkpoint/config by running a short shape probe.

### 5) "How do I quickly validate roundtrip quality?"

Read and run `../Wan2.1/tools/roundtrip_compare.py`.

- It does encode -> decode on a local video or LeRobot episode.
- It writes side-by-side video and PSNR metrics.
- Use this first to sanity-check your environment and selected WAN model path.

## Recommended quick checks before integration

1. Run WAN roundtrip on a short clip and confirm output quality.
2. Print shape mapping for several lengths:
   - `T = 1, 5, 9, 13`
   - and one "non-ideal" length like `T = 6` to see behavior.
3. Decide whether your WM dataset slicing should enforce `1 + 4k` windows.
4. Confirm causal input windows for WM training are past-only (no future leakage).

## Suggested integration reading order (in this repo)

After understanding WAN VAE internals, read:

1. `scripts/roundtrip_compare.py` (current DINO roundtrip baseline)
2. `scripts/stitch_compare.py` (existing WAN vs DINO visual compare path)
3. `dino_wm/dino_models.py` (where latent adapter path would be inserted)
4. `dino_wm/test_loader.py` (where WAN latent keys/windowing will matter)
5. `dino_wm/train_dino_wm.py` (training-time data flow and target construction)

This sequence makes the WAN-to-VideoTransformer integration work concrete before code changes.
