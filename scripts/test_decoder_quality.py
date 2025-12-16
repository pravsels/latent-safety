#!/usr/bin/env python3
"""
Test decoder quality by comparing original images with decoded DINOv2 embeddings.

Usage:
    python scripts/test_decoder_quality.py \
            --decoder-checkpoint dino_decoder_checkpoints/testing_decoder.pth \
            --image-path path/to/image.jpg \
            --output-dir test_decoder_results \
            --num-images 10

Or test with images from HDF5:
    python scripts/test_decoder_quality.py \
            --decoder-checkpoint dino_decoder_checkpoints/testing_decoder.pth \
            --hdf5-file arx5_10traj.h5 \
            --output-dir test_decoder_results \
            --num-images 10
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from einops import rearrange

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from dino_wm.dino_decoder import VQVAE
from dino_wm.config import MODEL_CONFIG
from scripts.utils import get_dino_model, preprocess_images_for_dino
from dino_wm.test_loader import SplitTrajectoryDataset
from torch.utils.data import DataLoader


def extract_dino_features(images, dino_model, device, is_front_camera=False):
    """
    Extract DINOv2 features from images.
    
    Args:
        images: (B, 3, H, W) tensor in [0, 1] range
        dino_model: DINOv2 model
        device: Device to run on
        is_front_camera: Whether these are front camera images (affects preprocessing)
    
    Returns:
        features: (B, num_patches, dim) DINO patch tokens
    """
    # Preprocess images for DINOv2
    preprocessed = preprocess_images_for_dino(images, is_front_camera=is_front_camera)
    
    # Extract features
    with torch.no_grad():
        features_dict = dino_model.forward_features(preprocessed)
        # Get patch tokens: (B, num_patches, dim)
        patch_tokens = features_dict['x_norm_patchtokens']
    
    return patch_tokens


def load_image_from_path(image_path, device):
    """Load and preprocess a single image from file path."""
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(MODEL_CONFIG['image_size']),
        transforms.ToTensor(),  # Converts to [0, 1] range
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)  # (1, 3, H, W)
    return img_tensor


def test_decoder_on_images(decoder, dino_model, images, device, is_front_camera=False, save_path=None):
    """
    Test decoder by encoding images with DINO and decoding back.
    
    Args:
        decoder: VQVAE decoder
        dino_model: DINOv2 model
        images: (B, 3, H, W) tensor in [0, 1] range
        device: Device to run on
        is_front_camera: Whether these are front camera images
        save_path: Path to save comparison image (optional)
    
    Returns:
        original_images: (B, H, W, 3) numpy array
        decoded_images: (B, H, W, 3) numpy array
    """
    B = images.shape[0]
    
    # Extract DINO features
    print("Extracting DINO features...")
    dino_features = extract_dino_features(images, dino_model, device, is_front_camera=is_front_camera)
    # dino_features: (B, num_patches, dim)
    
    # VQVAE expects input in format (b, t, num_patches, emb_dim)
    # Add time dimension: (B, num_patches, dim) -> (B, 1, num_patches, dim)
    dino_features = dino_features.unsqueeze(1)  # (B, 1, num_patches, dim)
    
    # Decode features
    print("Decoding features...")
    decoder.eval()
    with torch.no_grad():
        decoded, _ = decoder(dino_features)
        # decoded: (B*T, C, H, W) where T=1
    
    # Rearrange decoded to (B, H, W, C)
    decoded = rearrange(decoded, "(b t) c h w -> b t c h w", t=1, b=B)
    decoded = decoded.squeeze(1).permute(0, 2, 3, 1)  # (B, H, W, C)
    
    # Decoder outputs values that should be in [0, 1] range (trained with MSE on normalized images)
    # Clip to [0, 1] to handle any out-of-range values
    decoded = torch.clamp(decoded, 0, 1)
    
    # Convert to numpy
    original_np = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    decoded_np = (decoded.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    
    # Save comparison if requested
    if save_path is not None:
        fig, axes = plt.subplots(B, 2, figsize=(10, 5*B))
        if B == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(B):
            axes[i, 0].imshow(original_np[i])
            axes[i, 0].set_title('Original')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(decoded_np[i])
            axes[i, 1].set_title('Decoded from DINO')
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison to {save_path}")
    
    return original_np, decoded_np


def main():
    parser = argparse.ArgumentParser(description="Test decoder quality by comparing original vs decoded images")
    parser.add_argument("--decoder-checkpoint", type=str, required=True,
                       help="Path to decoder checkpoint")
    parser.add_argument("--image-path", type=str, default=None,
                       help="Path to single image file (optional)")
    parser.add_argument("--hdf5-file", type=str, default=None,
                       help="Path to HDF5 file to sample images from (optional)")
    parser.add_argument("--num-images", type=int, default=5,
                       help="Number of images to test (default: 5)")
    parser.add_argument("--output-dir", type=str, default="test_decoder_results",
                       help="Directory to save results (default: test_decoder_results)")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use (default: cuda:0)")
    parser.add_argument("--camera-type", type=str, default="wrist", choices=["front", "wrist"],
                       help="Camera type for preprocessing (default: wrist)")
    parser.add_argument("--quantize", action="store_true",
                       help="Enable VQ codebook quantization (must match training setting)")
    
    args = parser.parse_args()
    
    device = args.device
    is_front_camera = (args.camera_type == "front")
    
    # Load decoder
    print("Loading decoder...")
    decoder = VQVAE(quantize=args.quantize).to(device)
    if args.quantize:
        print("VQ codebook quantization enabled")
    else:
        print("VQ codebook quantization disabled (standard autoencoder)")
    decoder.load_state_dict(torch.load(args.decoder_checkpoint, map_location=device, weights_only=False))
    decoder.eval()
    
    # Load DINO model
    print("Loading DINO model...")
    dino_model = get_dino_model(device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load images
    if args.image_path is not None:
        # Single image mode
        print(f"Loading image from {args.image_path}")
        images = load_image_from_path(args.image_path, device)
        original_np, decoded_np = test_decoder_on_images(
            decoder, dino_model, images, device, is_front_camera=is_front_camera,
            save_path=os.path.join(args.output_dir, "comparison_single.png")
        )
        
    elif args.hdf5_file is not None:
        # HDF5 dataset mode
        print(f"Loading images from {args.hdf5_file}")
        dataset = SplitTrajectoryDataset(
            args.hdf5_file,
            segment_length=1,  # Just get single frames
            split='train',
            num_test=0
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        all_originals = []
        all_decoded = []
        
        for i in range(args.num_images):
            data = next(iter(dataloader))
            
            # Get images based on camera type
            # Data shape: (batch=1, segment_length=1, H, W, C)
            if is_front_camera:
                img = data['agentview_image'][0, 0].to(device).float() / 255.0  # (H, W, C)
            else:
                img = data['robot0_eye_in_hand_image'][0, 0].to(device).float() / 255.0  # (H, W, C)
            
            # Convert to (1, 3, H, W)
            images = img.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            # Resize to MODEL_CONFIG['image_size']
            images = F.interpolate(images, size=MODEL_CONFIG['image_size'], mode='bilinear', align_corners=False)
            
            original_np, decoded_np = test_decoder_on_images(
                decoder, dino_model, images, device, is_front_camera=is_front_camera,
                save_path=os.path.join(args.output_dir, f"comparison_{i:03d}.png")
            )
            
            all_originals.append(original_np[0])
            all_decoded.append(decoded_np[0])
        
        # Create a grid of all comparisons
        fig, axes = plt.subplots(args.num_images, 2, figsize=(10, 5*args.num_images))
        if args.num_images == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(args.num_images):
            axes[i, 0].imshow(all_originals[i])
            axes[i, 0].set_title(f'Original {i+1}')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(all_decoded[i])
            axes[i, 1].set_title(f'Decoded {i+1}')
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "comparison_grid.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved grid comparison to {args.output_dir}/comparison_grid.png")
        
    else:
        parser.error("Either --image-path or --hdf5-file must be specified")
    
    print(f"Done! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()

