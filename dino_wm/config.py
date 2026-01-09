"""
Global configuration for DINO World Model.
"""

import math

# DINO Version Selection
DINO_VERSION = 'v2'  # 'v2' or 'v3'

DINOV2_CONFIG = {
    'hub_repo': 'facebookresearch/dinov2',
    'model_name': 'dinov2_vits14_reg',
    'dim': 384,
    'num_patches': 256,  # 16x16 grid for 224x224 input
    'patch_size': 14,
}

DINOV3_CONFIG = {
    'hub_repo': '../dinov3',
    'hub_source': 'local',
    'model_name': 'dinov3_vits16plus',
    'weights_path': './weights/dinov3_vits16plus.pth',
    'dim': 384,
    'num_patches': 196,  # 14x14 grid for 224x224 input
    'patch_size': 16,
}

def get_dino_config():
    return DINOV3_CONFIG if DINO_VERSION == 'v3' else DINOV2_CONFIG

def get_decoder_image_size() -> tuple[int, int]:
    """
    Decoder output size implied by the DINO patch grid and the decoder architecture.

    The decoder operates on a sqrt(num_patches) x sqrt(num_patches) grid and upsamples
    spatially by 16x (stride=4 decoder twice -> 4*4).

    - DINOv2: 16x16 patches -> 256x256
    - DINOv3: 14x14 patches -> 224x224
    """
    num_patches = int(get_dino_config()["num_patches"])
    side = int(math.isqrt(num_patches))
    if side * side != num_patches:
        raise ValueError(f"num_patches must be a perfect square. Got {num_patches}")
    out = side * 16
    return (out, out)

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
    # Decoder output size (kept consistent with selected DINO_VERSION)
    'decoder_image_size': get_decoder_image_size(),
    'codebook_size': 2048,             # VQ-VAE codebook size (number of embeddings in the discrete vocabulary)
}

# Training Defaults
TRAIN_CONFIG = {
    'sequence_length': 4,      # Total sequence length (context + prediction)
    'context_length': 3,       # Number of frames for context
    'batch_size': 16,
}
