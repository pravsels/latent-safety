
import warnings
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

# --- Monkeypatch for LeRobot Dataset ---
# Fixes "ValueError: too many dimensions 'str'" when loading datasets with string columns
import lerobot.datasets.lerobot_dataset

def safe_hf_transform_to_torch(items_dict: dict) -> dict:
    for key in items_dict:
        first_item = items_dict[key][0]
        if isinstance(first_item, Image.Image):
            to_tensor = transforms.ToTensor()
            items_dict[key] = [to_tensor(img) for img in items_dict[key]]
        elif first_item is None:
            pass
        elif isinstance(first_item, (list, tuple, np.ndarray)) and len(first_item) > 0 and isinstance(first_item[0], str):
            # Skip conversion for list of strings
            pass
        else:
            # Convert everything else to tensors, skipping individual strings
            items_dict[key] = [x if isinstance(x, str) else torch.tensor(x) for x in items_dict[key]]
    return items_dict

lerobot.datasets.lerobot_dataset.hf_transform_to_torch = safe_hf_transform_to_torch

# --- Transforms & Model ---

def get_dino_model(device: str):
    print(f"Loading DINOv2 model on {device}...")
    # Suppress xformers warning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="xFormers is not available")
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg").to(device)
    model.eval()
    return model

def preprocess_images_for_dino(images: torch.Tensor, is_front_camera: bool) -> torch.Tensor:
    """
    Prepares a batch of images for DINO inference.
    Input: [B, 3, H, W] float32 tensor (0-1)
    Output: [B, 3, 224, 224] normalized tensor
    """
    # Standard ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    out = images
    
    if is_front_camera:
        # Specific transforms for front camera (crop top middle, etc.)
        # Blur -> Crop -> Resize
        # Gaussian Blur (kernel 5, sigma 0.1)
        out = transforms.functional.gaussian_blur(out, kernel_size=(5, 5), sigma=(0.1, 0.1))
        # Crop Top Middle: top=30, left=46, h=180, w=180
        out = transforms.functional.crop(out, top=30, left=46, height=180, width=180)
    
    # Resize to 224x224 for ViT
    out = transforms.functional.resize(out, (224, 224), antialias=True)
    
    # Normalize
    out = transforms.functional.normalize(out, mean=mean, std=std)
    
    return out

def to_hwc_uint8(images: torch.Tensor) -> np.ndarray:
    """
    Convert batch of [B, 3, H, W] float (0-1) tensors to [B, H, W, 3] uint8 numpy.
    """
    # Permute to HWC
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    # Clip and Scale
    images = (np.clip(images, 0, 1) * 255).astype(np.uint8)
    return images
