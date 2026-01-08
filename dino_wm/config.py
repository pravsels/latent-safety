"""
Global configuration for DINO World Model.
"""

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
    'decoder_image_size': (256, 256),  # Decoder output size
    'codebook_size': 2048,             # VQ-VAE codebook size (number of embeddings in the discrete vocabulary)
}

# Training Defaults
TRAIN_CONFIG = {
    'sequence_length': 4,      # Total sequence length (context + prediction)
    'context_length': 3,       # Number of frames for context
    'batch_size': 16,
}
