#!/usr/bin/env python3
"""
Traced Pipeline — SAM 2 Automatic Architectural Extraction
Uses SAM 2's automatic mask generator to find ALL segments in an image,
then classifies them by shape (dome, arch, rectangle, etc.),
fits geometric primitives, and computes proportional analysis.

No text prompts needed — SAM 2 finds everything automatically.

Usage:
    python extract-sam2.py --image photo.jpg --output extraction.json
    python extract-sam2.py --image photo.jpg --output extraction.json --checkpoint /path/to/sam2.1_hiera_large.pt
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

# ============================================================
# CONSTANTS
# ============================================================
PHI = (1 + math.sqrt(5)) / 2
CANONICAL_RATIOS = {
    "φ": PHI, "1/φ": 1/PHI, "φ²": PHI**2, "1/φ²": 1/PHI**2,
    "√2": math.sqrt(2), "1/√2": 1/math.sqrt(2),
    "√3": math.sqrt(3), "√3/2": math.sqrt(3)/2,
    "√5": math.sqrt(5),
    "2": 2.0, "3": 3.0, "1/2": 0.5, "1/3": 1/3,
    "2/3": 2/3, "3/4": 0.75, "1/4": 0.25,
}


def find_closest_ratio(value: float) -> dict:
    if value <= 0:
        return {"match": "none", "error_pct": 100.0, "quality": "weak"}
    best_name, best_error = "none", float("inf")
    for name, const in CANONICAL_RATIOS.items():
        err = abs(value / const - 1) * 100
        if err < best_error:
            best_error, best_name = err, name
    return {
        "match": best_name,
        "error_pct": round(best_error, 2),
        "quality": "strong" if best_error < 2 else ("possible" if best_error < 5 else "weak"),
    }


def classify_shape(mask: np.ndarray) -> dict:
    """Classify a mask's shape as dome, arch, rectangle, etc."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"type": "unknown", "circularity": 0}
    
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)
    circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    
    x, y, w, h = cv2.boundingRect(largest)
    aspect = w / h if h > 0 else 1
    solidity = area / (w * h) if w * h > 0 else 0
    
    # Convex hull
    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)
    convexity = area / hull_area if hull_area > 0 else 0
    
    # Classify
    shape_type = "unknown"
    confidence = 0.5
    
    if circularity > 0.75 and convexity > 0.9:
        shape_type = "dome"  # Near-circular, convex
        confidence = circularity
    elif circularity > 0.5 and aspect > 1.5 and solidity > 0.6:
        shape_type = "arch"  # Semi-circular top, wider than tall
        confidence = 0.7
    elif circularity > 0.5 and aspect < 0.7 and solidity > 0.6:
        shape_type = "tall_arch"  # Taller than wide arch
        confidence = 0.7
    elif convexity > 0.85 and 0.3 < circularity < 0.75:
        if aspect > 2:
            shape_type = "horizontal_band"
            confidence = 0.8
        elif aspect < 0.4:
            shape_type = "column"
            confidence = 0.8
        else:
            shape_type = "panel"
            confidence = 0.6
    elif solidity > 0.8 and circularity < 0.5:
        shape_type = "wall"
        confidence = 0.6
    elif solidity < 0.5:
        shape_type = "openwork"  # Grille, lattice
        confidence = 0.5
    
    # Check for dome-like top profile (upper half more curved than lower)
    if shape_type == "unknown" and len(largest) > 10:
        mid_y = y + h // 2
        upper_pts = [p for p in largest if p[0][1] < mid_y]
        if len(upper_pts) > 5:
            upper_contour = np.array(upper_pts)
            upper_hull = cv2.convexHull(upper_contour)
            upper_convexity = len(upper_pts) / len(upper_hull) if len(upper_hull) > 0 else 0
            if upper_convexity > 0.7:
                shape_type = "dome_like"
                confidence = 0.6
    
    return {
        "type": shape_type,
        "circularity": round(circularity, 4),
        "solidity": round(solidity, 4),
        "convexity": round(convexity, 4),
        "aspect": round(aspect, 4),
        "confidence": round(confidence, 3),
    }


def fit_primitives(mask: np.ndarray, shape_info: dict) -> dict:
    """Fit geometric primitives based on shape classification."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {}
    
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    
    result = {
        "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
        "center": {"x": float(x + w/2), "y": float(y + h/2)},
    }
    
    # Fit ellipse for dome-like shapes
    if shape_info["type"] in ("dome", "dome_like", "arch", "tall_arch") and len(largest) >= 5:
        el = cv2.fitEllipse(largest)
        result["ellipse"] = {
            "center": {"x": float(el[0][0]), "y": float(el[0][1])},
            "axes": {"w": float(el[1][0]), "h": float(el[1][1])},
            "angle": float(el[2]),
        }
    
    # Fit minimum enclosing circle for domes
    if shape_info["type"] in ("dome", "dome_like") and len(largest) >= 3:
        (cx, cy), radius = cv2.minEnclosingCircle(largest)
        result["circle"] = {
            "center": {"x": float(cx), "y": float(cy)},
            "radius": float(radius),
        }
    
    # Simplified contour for drawing
    epsilon = max(2.0, min(w, h) * 0.02)
    simplified = cv2.approxPolyDP(largest, epsilon, True)
    result["contour"] = [[int(pt[0][0]), int(pt[0][1])] for pt in simplified[:80]]
    
    return result


def run_sam2_extraction(image_path: str, checkpoint: str = None) -> dict:
    """Run SAM 2 automatic mask generation on an image."""
    
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        import torch
    except ImportError:
        try:
            # Alternative import paths
            from segment_anything_2.build_sam import build_sam2
            from segment_anything_2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            import torch
        except ImportError:
            print("SAM 2 not installed. Falling back to OpenCV.")
            return run_opencv_extraction(image_path)
    
    # Find checkpoint
    if checkpoint is None:
        script_dir = Path(__file__).parent
        search_paths = [
            script_dir / "checkpoints" / "sam2.1_hiera_large.pt",
            script_dir / "checkpoints" / "sam2.1_hiera_base_plus.pt",
            Path.home() / "sam2-checkpoints" / "sam2.1_hiera_large.pt",
        ]
        for p in search_paths:
            if Path(p).exists():
                checkpoint = str(p)
                break
    
    if checkpoint is None:
        print("No SAM 2 checkpoint found. Run setup-sam2.sh first.")
        return run_opencv_extraction(image_path)
    
    print(f"Loading SAM 2 from {checkpoint}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Determine model config based on checkpoint name
    if "large" in checkpoint:
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    elif "base_plus" in checkpoint:
        model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    else:
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    sam2 = build_sam2(model_cfg, checkpoint, device=device)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=32,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.85,
        min_mask_region_area=500,
    )
    
    # Load image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w = img.shape[:2]
    print(f"Image: {img_w}×{img_h}")
    
    # Generate masks
    print("Generating masks (this may take 10-30 seconds)...")
    masks = mask_generator.generate(img_rgb)
    print(f"Found {len(masks)} segments")
    
    # Sort by area (largest first)
    masks.sort(key=lambda m: m["area"], reverse=True)
    
    # Process each mask
    elements = []
    min_area_pct = 0.005  # Skip segments smaller than 0.5% of image
    min_area = img_w * img_h * min_area_pct
    
    for i, mask_data in enumerate(masks):
        if mask_data["area"] < min_area:
            continue
        if len(elements) >= 30:  # Cap at 30 elements
            break
        
        mask = mask_data["segmentation"].astype(np.uint8)
        
        # Classify shape
        shape = classify_shape(mask)
        
        # Fit primitives
        primitives = fit_primitives(mask, shape)
        if not primitives:
            continue
        
        bbox = primitives["bbox"]
        center = primitives["center"]
        
        element = {
            "name": f"{shape['type']}_{i}",
            "index": i,
            "shape": shape,
            "primitives": primitives,
            "sam2_score": round(float(mask_data.get("predicted_iou", 0)), 4),
            "stability": round(float(mask_data.get("stability_score", 0)), 4),
            "area_pct": round(mask_data["area"] / (img_w * img_h), 4),
            "position_pct": {
                "x": round(center["x"] / img_w, 4),
                "y": round(center["y"] / img_h, 4),
            },
            "size_pct": {
                "w": round(bbox["w"] / img_w, 4),
                "h": round(bbox["h"] / img_h, 4),
            },
            "position_analysis": {
                "x_ratio": find_closest_ratio(center["x"] / img_w),
                "y_ratio": find_closest_ratio(center["y"] / img_h),
                "w_ratio": find_closest_ratio(bbox["w"] / img_w),
                "h_ratio": find_closest_ratio(bbox["h"] / img_h),
            },
        }
        
        elements.append(element)
        print(f"  [{i}] {shape['type']} (conf={shape['confidence']:.2f}) "
              f"pos=({element['position_pct']['x']:.3f},{element['position_pct']['y']:.3f}) "
              f"size={element['area_pct']:.3f}")
    
    return {
        "image": str(image_path),
        "image_size": {"w": img_w, "h": img_h},
        "method": "sam2",
        "total_masks": len(masks),
        "elements": elements,
    }


def run_opencv_extraction(image_path: str) -> dict:
    """Fallback: OpenCV contour detection."""
    print("Running OpenCV fallback...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Cannot read {image_path}")
        sys.exit(1)
    
    img_h, img_w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive threshold + Canny
    edges = cv2.Canny(blurred, 30, 100)
    
    # Dilate to connect nearby edges
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    min_area = img_w * img_h * 0.005
    significant = [c for c in contours if cv2.contourArea(c) > min_area]
    significant.sort(key=cv2.contourArea, reverse=True)
    
    elements = []
    for i, contour in enumerate(significant[:30]):
        # Create mask from contour
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 1, -1)
        
        shape = classify_shape(mask)
        primitives = fit_primitives(mask, shape)
        if not primitives:
            continue
        
        bbox = primitives["bbox"]
        center = primitives["center"]
        area = cv2.contourArea(contour)
        
        element = {
            "name": f"{shape['type']}_{i}",
            "index": i,
            "shape": shape,
            "primitives": primitives,
            "area_pct": round(area / (img_w * img_h), 4),
            "position_pct": {
                "x": round(center["x"] / img_w, 4),
                "y": round(center["y"] / img_h, 4),
            },
            "size_pct": {
                "w": round(bbox["w"] / img_w, 4),
                "h": round(bbox["h"] / img_h, 4),
            },
            "position_analysis": {
                "x_ratio": find_closest_ratio(center["x"] / img_w),
                "y_ratio": find_closest_ratio(center["y"] / img_h),
            },
        }
        elements.append(element)
    
    return {
        "image": str(image_path),
        "image_size": {"w": img_w, "h": img_h},
        "method": "opencv",
        "elements": elements,
    }


def compute_relationships(extraction: dict) -> dict:
    """Compute pairwise proportional relationships."""
    elements = extraction["elements"]
    img_w, img_h = extraction["image_size"]["w"], extraction["image_size"]["h"]
    
    relationships = []
    for i, a in enumerate(elements):
        for j, b in enumerate(elements):
            if j <= i:
                continue
            
            a_bb = a.get("primitives", {}).get("bbox", {})
            b_bb = b.get("primitives", {}).get("bbox", {})
            if not a_bb or not b_bb:
                continue
            
            # Size ratios
            for dim in ["w", "h"]:
                av, bv = a_bb.get(dim, 0), b_bb.get(dim, 0)
                if av > 10 and bv > 10:
                    ratio = max(av, bv) / min(av, bv)
                    m = find_closest_ratio(ratio)
                    if m["quality"] != "weak":
                        relationships.append({
                            "a": a["name"], "b": b["name"],
                            "type": f"{dim}_ratio", "value": round(ratio, 4), **m
                        })
            
            # Vertical position ratio (important for architectural hierarchy)
            ay = a.get("position_pct", {}).get("y", 0)
            by = b.get("position_pct", {}).get("y", 0)
            if ay > 0.01 and by > 0.01:
                ratio = max(ay, by) / min(ay, by)
                m = find_closest_ratio(ratio)
                if m["quality"] != "weak":
                    relationships.append({
                        "a": a["name"], "b": b["name"],
                        "type": "y_position_ratio", "value": round(ratio, 4), **m
                    })
    
    quality_order = {"strong": 0, "possible": 1, "weak": 2}
    relationships.sort(key=lambda r: (quality_order[r["quality"]], r["error_pct"]))
    extraction["relationships"] = relationships
    return extraction


def generate_js(extraction: dict) -> str:
    """Generate JavaScript proportional constants."""
    lines = [
        "// === Auto-generated by Traced Pipeline ===",
        f"// Source: {extraction.get('image', 'unknown')}",
        f"// Method: {extraction.get('method', 'unknown')}",
        f"// Elements: {len(extraction['elements'])}",
        "",
        "var W = 1080, H = 1920;",
        "var cx = W / 2;",
        "var PHI = (1 + Math.sqrt(5)) / 2;",
        "",
    ]
    
    # Group by shape type
    by_type = {}
    for el in extraction["elements"]:
        t = el.get("shape", {}).get("type", "unknown")
        by_type.setdefault(t, []).append(el)
    
    for shape_type, els in by_type.items():
        lines.append(f"// --- {shape_type.upper()} elements ---")
        for el in els:
            name = el["name"]
            pos = el.get("position_pct", {})
            size = el.get("size_pct", {})
            pa = el.get("position_analysis", {})
            
            y_info = pa.get("y_ratio", {})
            y_comment = ""
            if y_info.get("quality") in ("strong", "possible"):
                y_comment = f"  // ≈ {y_info['match']} (err {y_info['error_pct']}%)"
            
            lines.append(f"var {name}_cy = H * {pos.get('y', 0):.4f};{y_comment}")
            lines.append(f"var {name}_cx = W * {pos.get('x', 0):.4f};")
            lines.append(f"var {name}_w = W * {size.get('w', 0):.4f};")
            lines.append(f"var {name}_h = H * {size.get('h', 0):.4f};")
            lines.append("")
    
    # Strong relationships
    strong = [r for r in extraction.get("relationships", []) if r["quality"] == "strong"]
    if strong:
        lines.append("// === Strong proportional relationships (<2% error) ===")
        for r in strong[:15]:
            lines.append(f"// {r['a']} ↔ {r['b']}: {r['type']} = {r['value']} ≈ {r['match']} ({r['error_pct']}%)")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Traced: SAM 2 architectural extraction")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="extraction.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--js-output", default=None)
    args = parser.parse_args()
    
    extraction = run_sam2_extraction(args.image, args.checkpoint)
    extraction = compute_relationships(extraction)
    
    js_code = generate_js(extraction)
    extraction["generated_js"] = js_code
    
    Path(args.output).write_text(json.dumps(extraction, indent=2, default=str))
    
    strong = [r for r in extraction.get("relationships", []) if r["quality"] == "strong"]
    possible = [r for r in extraction.get("relationships", []) if r["quality"] == "possible"]
    
    print(f"\n{'='*60}")
    print(f"Elements: {len(extraction['elements'])}")
    print(f"Relationships: {len(extraction.get('relationships', []))} "
          f"({len(strong)} strong, {len(possible)} possible)")
    print(f"Saved to: {args.output}")
    
    if args.js_output:
        Path(args.js_output).write_text(js_code)
        print(f"JS saved to: {args.js_output}")
    
    print(f"\n{'='*60}")
    print("GENERATED JAVASCRIPT:")
    print(f"{'='*60}")
    print(js_code)


if __name__ == "__main__":
    main()
