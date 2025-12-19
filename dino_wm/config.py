"""
Global configuration for DINO World Model.
"""

# Model Architecture
MODEL_CONFIG = {
    'image_size': (224, 224),         # Input image size for DINO / world model
    'dim': 384,                        # DINOv2 feature dimension
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
