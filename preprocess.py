#!/usr/bin/env python3
"""
Traced Pipeline — Stage 0.5: Preprocess Reference Image
Crops/fits reference photo to exactly 1080×1920 (9:16 portrait).
All downstream steps then work in native canvas coordinates.

Usage:
    python preprocess.py --image photo.jpg --output preprocessed.jpg
    python preprocess.py --image photo.jpg --output preprocessed.jpg --mode fill
"""

import argparse
from pathlib import Path
import cv2
import numpy as np

CANVAS_W = 1080
CANVAS_H = 1920


def preprocess(image_path: str, output_path: str, mode: str = "fill"):
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Cannot read {image_path}")
        return
    
    h, w = img.shape[:2]
    target_aspect = CANVAS_W / CANVAS_H  # 0.5625
    src_aspect = w / h
    
    print(f"Source: {w}×{h} (aspect {src_aspect:.3f})")
    print(f"Target: {CANVAS_W}×{CANVAS_H} (aspect {target_aspect:.3f})")
    
    if mode == "fill":
        # Crop to 9:16, keeping center
        if src_aspect > target_aspect:
            # Too wide — crop sides
            new_w = int(h * target_aspect)
            x_start = (w - new_w) // 2
            cropped = img[:, x_start:x_start + new_w]
        else:
            # Too tall — crop top/bottom
            new_h = int(w / target_aspect)
            y_start = (h - new_h) // 2
            cropped = img[y_start:y_start + new_h, :]
        
        result = cv2.resize(cropped, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_LANCZOS4)
    
    elif mode == "fit":
        # Letterbox to 9:16
        canvas = np.full((CANVAS_H, CANVAS_W, 3), 242, dtype=np.uint8)  # cream background
        if src_aspect > target_aspect:
            new_w = CANVAS_W
            new_h = int(CANVAS_W / src_aspect)
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            y_offset = (CANVAS_H - new_h) // 2
            canvas[y_offset:y_offset + new_h, :] = resized
        else:
            new_h = CANVAS_H
            new_w = int(CANVAS_H * src_aspect)
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            x_offset = (CANVAS_W - new_w) // 2
            canvas[:, x_offset:x_offset + new_w] = resized
        result = canvas
    
    else:
        # Stretch (not recommended)
        result = cv2.resize(img, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_LANCZOS4)
    
    cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Output: {CANVAS_W}×{CANVAS_H} → {output_path}")
    print(f"Mode: {mode}")


def main():
    parser = argparse.ArgumentParser(description="Traced: Preprocess reference to 1080×1920")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="preprocessed.jpg")
    parser.add_argument("--mode", default="fill", choices=["fill", "fit", "stretch"])
    args = parser.parse_args()
    preprocess(args.image, args.output, args.mode)


if __name__ == "__main__":
    main()
