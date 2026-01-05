"""
Smoke test for DINOv2 and DINOv3 vision encoders.

Usage:
    python scripts/smoke_test_dino.py
    python scripts/smoke_test_dino.py --device cpu
    python scripts/smoke_test_dino.py --version v3
"""

import argparse
import sys
import torch


def smoke_test_dino(
    device: str,
    batch_size: int,
    h: int,
    w: int,
    version: str,
) -> None:
    """Test DINO model loading via get_dino_model()."""
    from dino_wm import config
    from scripts.utils import get_dino_model

    config.DINO_VERSION = version
    dino_cfg = config.get_dino_config()

    print(f"\n[DINO {version}] Loading {dino_cfg['model_name']}...")

    model = get_dino_model(device)

    img = torch.randn(batch_size, 3, h, w, device=device)

    with torch.no_grad():
        features = model.forward_features(img)

    patch_tokens = features['x_norm_patchtokens']
    cls_token = features['x_norm_clstoken']

    expected_patches = dino_cfg['num_patches']
    expected_dim = dino_cfg['dim']

    assert patch_tokens.shape == (batch_size, expected_patches, expected_dim), \
        f"Expected {(batch_size, expected_patches, expected_dim)}, got {patch_tokens.shape}"
    assert cls_token.shape == (batch_size, expected_dim), \
        f"Expected {(batch_size, expected_dim)}, got {cls_token.shape}"

    print(f"[DINO {version}] patch_tokens: {tuple(patch_tokens.shape)}")
    print(f"[DINO {version}] cls_token: {tuple(cls_token.shape)}")
    print(f"[DINO {version}] PASSED")


def smoke_test_compare(
    device: str,
    batch_size: int,
    h: int,
    w: int,
) -> None:
    """Compare DINOv2 and DINOv3 outputs side by side."""
    from dino_wm import config
    from scripts.utils import get_dino_model

    print("\n[Compare] Loading both models...")

    config.DINO_VERSION = 'v2'
    v2 = get_dino_model(device)

    config.DINO_VERSION = 'v3'
    v3 = get_dino_model(device)

    img = torch.randn(batch_size, 3, h, w, device=device)

    with torch.no_grad():
        feat_v2 = v2.forward_features(img)
        feat_v3 = v3.forward_features(img)

    print(f"\n{'='*50}")
    print(f"{'Comparison':^50}")
    print(f"{'='*50}")
    print(f"{'Metric':<25} {'DINOv2':<12} {'DINOv3':<12}")
    print(f"{'-'*50}")
    print(f"{'Patch tokens shape':<25} {str(tuple(feat_v2['x_norm_patchtokens'].shape)):<12} {str(tuple(feat_v3['x_norm_patchtokens'].shape)):<12}")
    print(f"{'CLS token shape':<25} {str(tuple(feat_v2['x_norm_clstoken'].shape)):<12} {str(tuple(feat_v3['x_norm_clstoken'].shape)):<12}")
    print(f"{'Num patches':<25} {feat_v2['x_norm_patchtokens'].shape[1]:<12} {feat_v3['x_norm_patchtokens'].shape[1]:<12}")
    print(f"{'Feature dim':<25} {feat_v2['x_norm_patchtokens'].shape[2]:<12} {feat_v3['x_norm_patchtokens'].shape[2]:<12}")
    print(f"{'='*50}")
    print("[Compare] PASSED")


def smoke_test_decoder(device: str) -> None:
    """Test Decoder for both v2 and v3 patch configurations."""
    from dino_wm.dino_models import Decoder
    from dino_wm import config

    print("\n[Decoder] Testing both v2 and v3 configurations...")

    # Test v2: 256 patches, 16x16 grid
    original_version = config.DINO_VERSION
    config.DINO_VERSION = 'v2'
    decoder_v2 = Decoder().to(device)
    x_v2 = torch.randn(2, 256, 384, device=device)
    out_v2 = decoder_v2(x_v2)
    assert out_v2.shape == (2, 3, 224, 224), f"v2 expected (2, 3, 224, 224), got {out_v2.shape}"
    print(f"[Decoder v2] grid_size: {decoder_v2.grid_size}, input: (2, 256, 384), output: {tuple(out_v2.shape)}")

    # Test v3: 196 patches, 14x14 grid
    config.DINO_VERSION = 'v3'
    decoder_v3 = Decoder().to(device)
    x_v3 = torch.randn(2, 196, 384, device=device)
    out_v3 = decoder_v3(x_v3)
    assert out_v3.shape == (2, 3, 224, 224), f"v3 expected (2, 3, 224, 224), got {out_v3.shape}"
    print(f"[Decoder v3] grid_size: {decoder_v3.grid_size}, input: (2, 196, 384), output: {tuple(out_v3.shape)}")

    # Restore original version
    config.DINO_VERSION = original_version
    print("[Decoder] PASSED")


def smoke_test_video_transformer(device: str) -> None:
    """Test VideoTransformer for both v2 and v3 patch configurations."""
    from dino_wm.dino_models import VideoTransformer
    from dino_wm import config

    print("\n[VideoTransformer] Testing both v2 and v3 configurations...")

    original_version = config.DINO_VERSION

    for version in ['v2', 'v3']:
        config.DINO_VERSION = version
        patches = config.get_dino_config()['num_patches']

        model = VideoTransformer(
            image_size=(224, 224),
            dim=384,
            action_embed_dim=10,
            state_embed_dim=10,
            state_dim=8,
            action_dim=7,
            depth=6,
            heads=16,
            mlp_dim=2048,
            num_frames=4,
            device=device
        ).to(device)

        video1 = torch.randn(2, 4, patches, 384, device=device)
        video2 = torch.randn(2, 4, patches, 384, device=device)
        states = torch.randn(2, 4, 8, device=device)
        actions = torch.randn(2, 4, 7, device=device)

        out = model.forward_features(video1, video2, states, actions)
        expected = (2, 4, patches, 788)
        assert out.shape == expected, f"{version} expected {expected}, got {out.shape}"
        print(f"[VideoTransformer for DINO{version}] num_patches: {model.num_patches}, output: {tuple(out.shape)}")

    config.DINO_VERSION = original_version
    print("[VideoTransformer] PASSED")


def main():
    parser = argparse.ArgumentParser(description="Smoke test DINO vision encoders")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--height", type=int, default=224, help="Image height")
    parser.add_argument("--width", type=int, default=224, help="Image width")
    parser.add_argument("--version", type=str, default="all", choices=["v2", "v3", "all", "compare", "decoder", "transformer"],
                        help="Which version to test")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.height}x{args.width}")

    try:
        if args.version in ("v2", "all"):
            smoke_test_dino(args.device, args.batch_size, args.height, args.width, "v2")

        if args.version in ("v3", "all"):
            smoke_test_dino(args.device, args.batch_size, args.height, args.width, "v3")

        if args.version == "compare":
            smoke_test_compare(args.device, args.batch_size, args.height, args.width)

        if args.version == "decoder":
            smoke_test_decoder(args.device)

        if args.version == "transformer":
            smoke_test_video_transformer(args.device)

        print("\n" + "="*50)
        print("ALL SMOKE TESTS PASSED")
        print("="*50)

    except Exception as e:
        print(f"\nSMOKE TEST FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
