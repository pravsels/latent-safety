"""
Global configuration for DINO World Model.
"""

import math

DINOV2_CONFIG = {
    'version': 'v2',
    'hub_repo': 'facebookresearch/dinov2',
    'model_name': 'dinov2_vits14_reg',
    'dim': 384,
    'num_patches': 256,  # 16x16 grid for 224x224 input
    'patch_size': 14,
}

DINOV3_CONFIG = {
    'version': 'v3',
    'hub_repo': '../dinov3',
    'hub_source': 'local',
    'model_name': 'dinov3_vits16plus',
    'weights_path': './weights/dinov3_vits16plus.pth',
    'dim': 384,
    'num_patches': 196,  # 14x14 grid for 224x224 input
    'patch_size': 16,
}

DINO_CONFIGS = {
    'v2': DINOV2_CONFIG,
    'v3': DINOV3_CONFIG,
}

# Default version (used when get_dino_config() called without argument)
DEFAULT_DINO_VERSION = 'v3'

def get_dino_config(version: str | None = None):
    """Get DINO config by version. Falls back to DEFAULT_DINO_VERSION if not specified."""
    v = version if version is not None else DEFAULT_DINO_VERSION
    if v not in DINO_CONFIGS:
        raise ValueError(f"Unknown DINO version: {v}. Must be one of {list(DINO_CONFIGS.keys())}")
    return DINO_CONFIGS[v]

# Decoder architecture constant: total spatial upsampling factor (stride=4 twice -> 4*4=16)
DECODER_UPSAMPLE_FACTOR = 16

def compute_decoder_image_size(dino_cfg: dict) -> tuple[int, int]:
    """
    Decoder output size implied by the DINO patch grid and the decoder architecture.

    The decoder operates on a sqrt(num_patches) x sqrt(num_patches) grid and upsamples
    spatially by DECODER_UPSAMPLE_FACTOR (stride=4 decoder twice -> 4*4=16).

    - DINOv2: 16x16 patches -> 256x256
    - DINOv3: 14x14 patches -> 224x224
    """
    num_patches = int(dino_cfg["num_patches"])
    side = int(math.isqrt(num_patches))
    if side * side != num_patches:
        raise ValueError(f"num_patches must be a perfect square. Got {num_patches}")
    out = side * DECODER_UPSAMPLE_FACTOR
    return (out, out)

def get_decoder_image_size(version: str | None = None) -> tuple[int, int]:
    """Convenience wrapper using the specified or default DINO version."""
    return compute_decoder_image_size(get_dino_config(version))

# Model Architecture
MODEL_CONFIG = {
    'image_size': (224, 224),         # Input image size for DINO / world model
    'dim': get_dino_config()['dim'],  # DINO feature dimension
    'action_embed_dim': 10,            # Action embedding dimension
    'state_embed_dim': 10,             # State embedding dimension
    'depth': 6,                        # Number of transformer blocks
    'heads': 16,                       # Number of attention heads
    'mlp_dim': 2048,                   # Hidden dimension of feedforward network
    'dropout': 0.1                     # Dropout rate
}

# Decoder-specific configuration
DECODER_CONFIG = {
    # Decoder output size (kept consistent with selected DINO version)
    'decoder_image_size': get_decoder_image_size(),
    'codebook_size': 2048,             # VQ-VAE codebook size (number of embeddings in the discrete vocabulary)
}

# WAN VAE configuration
_WAN_INPUT_SIZE = 224               # Must be multiple of 8 (spatial_downsample)
WAN_CONFIG = {
    'input_size': _WAN_INPUT_SIZE,
    'spatial_downsample': 8,                            # VAE spatial compression factor
    'latent_dim': 16,                                   # Latent channels per spatial position
    'latent_side': _WAN_INPUT_SIZE // 8,                # = 28
    'num_patches': (_WAN_INPUT_SIZE // 8) ** 2,         # = 784 (28x28 grid)
}

# Training Defaults
TRAIN_CONFIG = {
    'sequence_length': 4,      # Total sequence length (context + prediction)
    'context_length': 3,       # Number of frames for context
    'batch_size': 16,
}
