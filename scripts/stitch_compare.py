#!/usr/bin/env python3
"""
Stitch two side-by-side roundtrip videos into a 3-panel comparison:

    Original  |  Wan 2.1 VAE  |  DINOv3

Each input video is assumed to be a side-by-side (original | decoded) produced
by a roundtrip_compare.py script.  The left half of one video is used as the
"Original" panel; the right halves become the two model panels.

Examples:
  # Compare front camera, episode 0
  python scripts/stitch_compare.py \
    --vae_video  ./outputs/roundtrip_vae/side_by_side_front_ep0.mp4 \
    --dino_video ./outputs/roundtrip_dino/side_by_side_front_ep0.mp4 \
    --out ./outputs/compare/compare_front_ep0.mp4

  # Compare wrist camera, episode 0
  python scripts/stitch_compare.py \
    --vae_video  ./outputs/roundtrip_vae/side_by_side_wrist_ep0.mp4 \
    --dino_video ./outputs/roundtrip_dino/side_by_side_wrist_ep0.mp4 \
    --out ./outputs/compare/compare_wrist_ep0.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def add_label(frame: np.ndarray, text: str, font_scale: float = 0.6) -> np.ndarray:
    """Burn a white-on-black label into the top-left corner of a frame."""
    try:
        import cv2

        frame = frame.copy()
        thickness = max(1, int(font_scale * 2))
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        # Background rectangle
        cv2.rectangle(frame, (0, 0), (tw + 12, th + 14), (0, 0, 0), -1)
        # Text
        cv2.putText(
            frame, text, (6, th + 8),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )
        return frame
    except ImportError:
        # Fallback: no labels if cv2 is unavailable
        return frame


def read_frames(path: str) -> tuple[list[np.ndarray], float]:
    reader = imageio.get_reader(path, format="ffmpeg")
    fps = 16.0
    meta = reader.get_meta_data() if hasattr(reader, "get_meta_data") else {}
    if isinstance(meta, dict):
        try:
            fps = float(meta["fps"])
        except (KeyError, TypeError, ValueError):
            pass
    frames = [f[..., :3].astype(np.uint8) for f in reader]
    reader.close()
    return frames, fps


def split_halves(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a side-by-side frame into left and right halves."""
    w = frame.shape[1]
    mid = w // 2
    return frame[:, :mid], frame[:, mid:]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Stitch two roundtrip side-by-side videos into a 3-panel comparison",
    )
    p.add_argument("--vae_video", required=True, help="Side-by-side video from Wan 2.1 VAE roundtrip")
    p.add_argument("--dino_video", required=True, help="Side-by-side video from DINOv3 roundtrip")
    p.add_argument("--out", default="./outputs/compare/compare.mp4", help="Output video path")
    p.add_argument("--fps", type=float, default=0.0, help="Override fps (0 = auto from vae_video)")
    p.add_argument("--no_labels", action="store_true", help="Skip text labels on panels")
    args = p.parse_args()

    # Read both videos
    print(f"Reading VAE video:  {args.vae_video}")
    vae_frames, vae_fps = read_frames(args.vae_video)
    print(f"Reading DINO video: {args.dino_video}")
    dino_frames, dino_fps = read_frames(args.dino_video)

    fps = args.fps if args.fps > 0 else vae_fps
    n = min(len(vae_frames), len(dino_frames))
    if n == 0:
        raise RuntimeError("No frames to process")
    print(f"Stitching {n} frames @ {fps:.1f} fps")

    # Write output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8, format="ffmpeg")

    # Determine uniform panel size (use the larger of the two so nothing gets squished)
    _, sample_vae = split_halves(vae_frames[0])
    _, sample_dino = split_halves(dino_frames[0])
    panel_h = max(sample_vae.shape[0], sample_dino.shape[0])
    panel_w = max(sample_vae.shape[1], sample_dino.shape[1])
    print(f"Panel size: {panel_w}x{panel_h}")

    def resize_panel(frame: np.ndarray) -> np.ndarray:
        if frame.shape[0] == panel_h and frame.shape[1] == panel_w:
            return frame
        return np.array(Image.fromarray(frame).resize((panel_w, panel_h), Image.LANCZOS))

    try:
        for i in range(n):
            _, vae_dec = split_halves(vae_frames[i])
            _, dino_dec = split_halves(dino_frames[i])

            vae_dec = resize_panel(vae_dec)
            dino_dec = resize_panel(dino_dec)

            if not args.no_labels:
                vae_dec = add_label(vae_dec, "Wan 2.1 VAE")
                dino_dec = add_label(dino_dec, "DINOv3")

            writer.append_data(np.concatenate([vae_dec, dino_dec], axis=1))
    finally:
        writer.close()

    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
