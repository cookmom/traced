#!/usr/bin/env python3
"""
Traced — Training Data Collector
Runs alongside the pipeline to accumulate labeled examples for LoRA training.

Every extraction run saves:
- Cropped mask images per element
- Shape classification labels
- Curve family labels (from optimizer)
- Style labels (from research)
- Proportional relationships

This builds a training dataset over time. No API cost — runs locally.

Usage:
    python collect_training.py --extraction extraction.json --image photo.jpg --output-dir training_data/
    
    # Or auto-collect from existing pipeline outputs
    python collect_training.py --auto --output-dir training_data/
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2


def collect_from_extraction(extraction_path: str, image_path: str, output_dir: str,
                             optimized_path: str = None):
    """Collect training examples from a pipeline run."""
    
    extraction = json.loads(Path(extraction_path).read_text())
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read {image_path}")
        return 0
    
    img_h, img_w = img.shape[:2]
    
    # Load optimized curve families if available
    optimized = None
    if optimized_path and Path(optimized_path).exists():
        optimized = json.loads(Path(optimized_path).read_text())
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    shapes_dir = Path(output_dir) / "shapes"
    os.makedirs(shapes_dir, exist_ok=True)
    
    # Collect per-element crops + labels
    examples = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source = Path(image_path).stem
    
    for el in extraction.get("elements", []):
        bbox = el.get("primitives", {}).get("bbox", {})
        if not bbox or bbox.get("w", 0) < 10 or bbox.get("h", 0) < 10:
            continue
        
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        # Pad slightly
        pad = max(5, min(w, h) // 10)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_w, x + w + pad)
        y2 = min(img_h, y + h + pad)
        
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        
        # Resize to standard size for training (128x128)
        crop_resized = cv2.resize(crop, (128, 128))
        
        # Shape label
        shape_type = el.get("shape", {}).get("type", "unknown")
        
        # Curve family from optimizer (if available)
        curve_family = None
        if optimized:
            opt_params = optimized.get("params", {})
            if el["name"] in opt_params:
                curve_family = opt_params[el["name"]].get("type", shape_type)
        
        # Save crop
        crop_filename = f"{source}_{timestamp}_{el['name']}.jpg"
        shape_subdir = shapes_dir / shape_type
        os.makedirs(shape_subdir, exist_ok=True)
        cv2.imwrite(str(shape_subdir / crop_filename), crop_resized)
        
        example = {
            "filename": crop_filename,
            "source_image": image_path,
            "element_name": el["name"],
            "shape_type": shape_type,
            "curve_family": curve_family,
            "confidence": el.get("shape", {}).get("confidence", 0),
            "area_pct": el.get("area_pct", 0),
            "bbox": {"x": x, "y": y, "w": w, "h": h},
            "position_pct": el.get("position_pct", {}),
            "circularity": el.get("shape", {}).get("circularity", 0),
            "solidity": el.get("shape", {}).get("solidity", 0),
            "aspect": el.get("shape", {}).get("aspect", 0),
            "depth_layer": el.get("depth", {}).get("layer", "unknown"),
            "timestamp": timestamp,
        }
        examples.append(example)
    
    # Save labels
    labels_file = Path(output_dir) / "labels.jsonl"
    with open(labels_file, "a") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    
    # Save style labels (one per image)
    styles = extraction.get("knowledge", {}).get("styles", [])
    if styles:
        style_file = Path(output_dir) / "styles.jsonl"
        with open(style_file, "a") as f:
            f.write(json.dumps({
                "source_image": image_path,
                "styles": styles,
                "timestamp": timestamp,
            }) + "\n")
    
    # Save proportional relationships (for ratio predictor training)
    relationships = extraction.get("relationships", [])
    if relationships:
        ratios_file = Path(output_dir) / "ratios.jsonl"
        with open(ratios_file, "a") as f:
            for r in relationships:
                if r.get("quality") in ("strong", "possible"):
                    f.write(json.dumps({
                        "a": r["a"], "b": r["b"],
                        "type": r["type"],
                        "value": r["value"],
                        "match": r["match"],
                        "error_pct": r["error_pct"],
                        "quality": r["quality"],
                        "source_image": image_path,
                        "timestamp": timestamp,
                    }) + "\n")
    
    print(f"Collected {len(examples)} training examples")
    print(f"  Shape crops saved to: {shapes_dir}/")
    print(f"  Labels appended to: {labels_file}")
    
    # Print distribution
    from collections import Counter
    type_counts = Counter(ex["shape_type"] for ex in examples)
    print(f"  Distribution:")
    for shape, count in type_counts.most_common():
        print(f"    {shape:20s} {count:4d}")
    
    return len(examples)


def print_stats(output_dir: str):
    """Print training data statistics."""
    labels_file = Path(output_dir) / "labels.jsonl"
    if not labels_file.exists():
        print("No training data collected yet.")
        return
    
    from collections import Counter
    
    examples = []
    with open(labels_file) as f:
        for line in f:
            examples.append(json.loads(line))
    
    print(f"\n{'='*50}")
    print(f"TRAINING DATA STATS")
    print(f"{'='*50}")
    print(f"Total examples: {len(examples)}")
    print(f"Source images: {len(set(ex['source_image'] for ex in examples))}")
    
    type_counts = Counter(ex["shape_type"] for ex in examples)
    print(f"\nShape type distribution:")
    for shape, count in type_counts.most_common():
        bar = "█" * min(40, count)
        ready = "✓ READY" if count >= 50 else f"need {50-count} more"
        print(f"  {shape:20s} {count:4d} {bar} {ready}")
    
    # Curve families
    curve_counts = Counter(ex.get("curve_family", "none") for ex in examples if ex.get("curve_family"))
    if curve_counts:
        print(f"\nCurve family distribution:")
        for curve, count in curve_counts.most_common():
            print(f"  {curve:20s} {count:4d}")
    
    # Check readiness for LoRA training
    ready_types = sum(1 for count in type_counts.values() if count >= 50)
    total_types = len(type_counts)
    print(f"\nLoRA readiness: {ready_types}/{total_types} types have 50+ examples")
    if ready_types >= 10:
        print("  ✓ READY FOR TRAINING — enough data for element classifier LoRA")
    else:
        print(f"  Need more data — run pipeline on {max(1, (10 - ready_types) * 3)} more buildings")


def main():
    parser = argparse.ArgumentParser(description="Traced: Collect LoRA training data")
    parser.add_argument("--extraction", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--optimized", default=None)
    parser.add_argument("--output-dir", default="training_data")
    parser.add_argument("--stats", action="store_true", help="Print training data stats")
    args = parser.parse_args()
    
    if args.stats:
        print_stats(args.output_dir)
        return
    
    if not args.extraction or not args.image:
        print("Usage: python collect_training.py --extraction extraction.json --image photo.jpg")
        print("       python collect_training.py --stats")
        return
    
    collect_from_extraction(args.extraction, args.image, args.output_dir, args.optimized)
    print_stats(args.output_dir)


if __name__ == "__main__":
    main()
