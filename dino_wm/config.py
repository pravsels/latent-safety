"""
Global configuration for DINO World Model.
"""

# Model Architecture
MODEL_CONFIG = {
    'image_size': (224, 224),
    'dim': 384,                # DINOv2 feature dimension
    'action_embed_dim': 10,    # Action embedding dimension
    'state_embed_dim': 10,     # State embedding dimension
    'depth': 6,                # Number of transformer blocks
    'heads': 16,               # Number of attention heads
    'mlp_dim': 2048,           # Hidden dimension of feedforward network
    'dropout': 0.1             # Dropout rate
}

# Training Defaults
TRAIN_CONFIG = {
    'sequence_length': 4,      # Total sequence length (context + prediction)
    'context_length': 3,       # Number of frames for context
    'batch_size': 16,
}
