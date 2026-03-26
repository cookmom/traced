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
    """Classify a mask's shape into 17 architectural types."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"type": "unknown", "circularity": 0, "solidity": 0, "convexity": 0, "aspect": 0, "confidence": 0, "vertices": 0}
    
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)
    circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    
    x, y, w, h = cv2.boundingRect(largest)
    aspect = w / h if h > 0 else 1
    solidity = area / (w * h) if w * h > 0 else 0
    
    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)
    convexity = area / hull_area if hull_area > 0 else 0
    
    # Vertex count for polygon detection
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(largest, epsilon, True)
    n_vertices = len(approx)
    
    # Analyze upper vs lower half curvature (for arch type detection)
    mid_y = y + h // 2
    upper_pts = [p for p in largest if p[0][1] < mid_y]
    lower_pts = [p for p in largest if p[0][1] >= mid_y]
    upper_circularity = 0
    if len(upper_pts) > 5:
        up_cont = np.array(upper_pts)
        up_area = cv2.contourArea(up_cont) if len(up_cont) >= 3 else 0
        up_peri = cv2.arcLength(up_cont, False) if len(up_cont) >= 3 else 0
        upper_circularity = 4 * math.pi * up_area / (up_peri * up_peri) if up_peri > 0 else 0
    
    shape_type = "unknown"
    confidence = 0.5
    
    # === CIRCLE (near-perfect) ===
    if circularity > 0.85 and convexity > 0.95 and 0.85 < aspect < 1.15:
        shape_type = "circle"
        confidence = circularity
    
    # === DOME (circular, convex, wider-than-tall or equal) ===
    elif circularity > 0.7 and convexity > 0.88 and aspect >= 0.7:
        shape_type = "dome"
        confidence = circularity
    
    # === OCTAGON (7-9 vertices, high solidity) ===
    elif 7 <= n_vertices <= 9 and solidity > 0.75 and convexity > 0.85:
        shape_type = "octagon"
        confidence = 0.8
    
    # === CRESCENT (thin, curved, low solidity) ===
    elif solidity < 0.35 and convexity < 0.6 and circularity > 0.2:
        shape_type = "crescent"
        confidence = 0.6
    
    # === STAR POLYGON (many vertices, moderate solidity) ===
    elif n_vertices >= 10 and 0.4 < solidity < 0.75 and convexity < 0.8:
        shape_type = "star_polygon"
        confidence = 0.65
    
    # === OPENWORK / GRILLE (low solidity, complex outline) ===
    elif solidity < 0.45:
        shape_type = "openwork"
        confidence = 0.55
    
    # === MINARET (extremely tall and thin, at edge of image) ===
    elif aspect < 0.15 and h > 3 * w and convexity > 0.7:
        shape_type = "minaret"
        confidence = 0.75
    
    # === COLUMN (tall and thin) ===
    elif aspect < 0.3 and h > 2.5 * w and convexity > 0.7:
        shape_type = "column"
        confidence = 0.75
    
    # === HORIZONTAL BAND (very wide, short) ===
    elif aspect > 3.0 and solidity > 0.7:
        shape_type = "horizontal_band"
        confidence = 0.8
    
    # === SPANDREL (triangular, 3 vertices) ===
    elif n_vertices == 3 and solidity > 0.6:
        shape_type = "spandrel"
        confidence = 0.7
    
    # === ARCH TYPES (curved top, open bottom or sides) ===
    elif circularity > 0.4 and upper_circularity > 0.3:
        if aspect < 0.7:
            # Taller than wide
            if upper_circularity > 0.5 and solidity > 0.7:
                shape_type = "pointed_arch"
                confidence = 0.7
            else:
                shape_type = "tall_arch"
                confidence = 0.65
        elif aspect > 1.2:
            # Wider than tall — could be horseshoe
            if convexity > 0.85 and circularity > 0.55:
                shape_type = "horseshoe_arch"
                confidence = 0.65
            else:
                shape_type = "arch"
                confidence = 0.6
        else:
            # Near-square arch opening — could be ogee
            if convexity < 0.85 and upper_circularity > 0.3:
                shape_type = "ogee_arch"
                confidence = 0.55
            else:
                shape_type = "arch"
                confidence = 0.6
    
    # === SQUARE (rectangle with near-equal sides) ===
    elif solidity > 0.8 and 0.85 < aspect < 1.15 and circularity < 0.7:
        shape_type = "square"
        confidence = 0.75
    
    # === RECTANGLE (high solidity, clear corners) ===
    elif solidity > 0.75 and circularity < 0.7 and n_vertices <= 6:
        shape_type = "rectangle"
        confidence = 0.7
    
    # === WALL (large area, blocky) ===
    elif solidity > 0.7 and circularity < 0.5:
        shape_type = "wall"
        confidence = 0.6
    
    # === DOME-LIKE (curved upper profile) ===
    elif upper_circularity > 0.4 and convexity > 0.7:
        shape_type = "dome_like"
        confidence = 0.55
    
    # === PANEL (catch-all for remaining convex shapes) ===
    elif convexity > 0.7:
        shape_type = "panel"
        confidence = 0.5
    
    return {
        "type": shape_type,
        "circularity": round(circularity, 4),
        "solidity": round(solidity, 4),
        "convexity": round(convexity, 4),
        "aspect": round(aspect, 4),
        "confidence": round(confidence, 3),
        "vertices": n_vertices,
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


def generate_analysis_card(extraction: dict, building_name: str = "Unknown Building") -> str:
    """Generate architectural analysis card with scores and style markers."""
    elements = extraction.get("elements", [])
    relationships = extraction.get("relationships", [])
    
    strong = [r for r in relationships if r["quality"] == "strong"]
    possible = [r for r in relationships if r["quality"] == "possible"]
    
    # --- Proportional Coherence Score ---
    n_pairs = max(1, len(elements) * (len(elements) - 1) // 2)
    max_possible = n_pairs * 3  # w, h, y ratios per pair
    coherence = min(100, int((len(strong) * 3 + len(possible) * 1.5) * 100 / max(1, max_possible)))
    
    # --- System Detection ---
    phi_family = {"φ", "1/φ", "φ²", "1/φ²"}
    sqrt2_family = {"√2", "1/√2"}
    sqrt3_family = {"√3", "√3/2"}
    
    phi_count = sum(1 for r in strong + possible if r.get("match") in phi_family)
    sqrt2_count = sum(1 for r in strong + possible if r.get("match") in sqrt2_family)
    sqrt3_count = sum(1 for r in strong + possible if r.get("match") in sqrt3_family)
    
    has_phi = phi_count > 0
    has_sqrt2 = sqrt2_count > 0
    has_sqrt3 = sqrt3_count > 0
    
    if has_phi and has_sqrt2 and has_sqrt3:
        system = "Islamic Synthesis (φ + √2 + √3)"
    elif phi_count >= sqrt2_count and phi_count >= sqrt3_count and has_phi:
        system = "Golden Section dominant"
    elif sqrt2_count >= phi_count and sqrt2_count >= sqrt3_count and has_sqrt2:
        system = "Ad Quadratum dominant"
    elif sqrt3_count >= phi_count and sqrt3_count >= sqrt2_count and has_sqrt3:
        system = "Ad Triangulum dominant"
    elif len(strong) + len(possible) > 0:
        system = "Mixed / Simple ratios"
    else:
        system = "No clear system detected"
    
    # --- Symmetry Score ---
    bilateral_pairs = 0
    checked = set()
    for i, a in enumerate(elements):
        ax = a.get("position_pct", {}).get("x", 0)
        for j, b in enumerate(elements):
            if j <= i or (i, j) in checked:
                continue
            checked.add((i, j))
            bx = b.get("position_pct", {}).get("x", 0)
            # Check if mirrored around 0.5 (center)
            if abs((1 - ax) - bx) < 0.05 and abs(ax - 0.5) > 0.05:
                # Also check similar y position and size
                ay = a.get("position_pct", {}).get("y", 0)
                by = b.get("position_pct", {}).get("y", 0)
                if abs(ay - by) < 0.05:
                    bilateral_pairs += 1
    
    non_center = sum(1 for e in elements if abs(e.get("position_pct", {}).get("x", 0.5) - 0.5) > 0.05)
    symmetry = min(100, int(bilateral_pairs * 2 * 100 / max(1, non_center)))
    
    # --- Average error of strong matches ---
    avg_error = sum(r["error_pct"] for r in strong) / max(1, len(strong))
    
    # --- Shape type counts ---
    shape_counts = {}
    for e in elements:
        t = e.get("shape", {}).get("type", "unknown")
        shape_counts[t] = shape_counts.get(t, 0) + 1
    
    dome_count = sum(v for k, v in shape_counts.items() if "dome" in k)
    
    # --- Build card ---
    w = 58
    lines = []
    lines.append("╔" + "═" * w + "╗")
    lines.append("║" + f"  TRACED ANALYSIS — {building_name}".ljust(w) + "║")
    lines.append("╠" + "═" * w + "╣")
    
    # Proportional Coherence
    bar = "█" * (coherence // 5) + "░" * (20 - coherence // 5)
    lines.append("║" + f"  PROPORTIONAL COHERENCE        {coherence:3d}/100".ljust(w) + "║")
    lines.append("║" + f"  [{bar}]".ljust(w) + "║")
    lines.append("║" + f"    φ relationships:          {phi_count:5d}".ljust(w) + "║")
    lines.append("║" + f"    √2 relationships:         {sqrt2_count:5d}".ljust(w) + "║")
    lines.append("║" + f"    √3 relationships:         {sqrt3_count:5d}".ljust(w) + "║")
    lines.append("║" + f"    Strong matches (<2%):     {len(strong):5d}".ljust(w) + "║")
    lines.append("║" + f"    Average error:            {avg_error:5.2f}%".ljust(w) + "║")
    lines.append("║" + f"    System: {system}".ljust(w) + "║")
    lines.append("║" + " " * w + "║")
    
    # Symmetry
    sym_bar = "█" * (symmetry // 5) + "░" * (20 - symmetry // 5)
    lines.append("║" + f"  SYMMETRY SCORE                {symmetry:3d}/100".ljust(w) + "║")
    lines.append("║" + f"  [{sym_bar}]".ljust(w) + "║")
    lines.append("║" + f"    Bilateral pairs found:    {bilateral_pairs:5d}".ljust(w) + "║")
    lines.append("║" + " " * w + "║")
    
    # Elements detected
    lines.append("║" + f"  ELEMENTS DETECTED             {len(elements):5d}".ljust(w) + "║")
    for shape_type, count in sorted(shape_counts.items(), key=lambda x: -x[1]):
        lines.append("║" + f"    {shape_type:25s} {count:5d}".ljust(w) + "║")
    lines.append("║" + " " * w + "║")
    
    # Architectural Style Markers
    lines.append("║" + "  ARCHITECTURAL STYLE MARKERS".ljust(w) + "║")
    if has_sqrt2:
        lines.append("║" + "    ✓ Ad Quadratum (√2) — Mamluk/Roman".ljust(w) + "║")
    if has_sqrt3:
        lines.append("║" + "    ✓ Ad Triangulum (√3) — Gothic/pointed arch".ljust(w) + "║")
    if has_phi:
        lines.append("║" + "    ✓ Golden Section (φ) — Classical/Ottoman".ljust(w) + "║")
    if any("horseshoe" in e.get("shape", {}).get("type", "") for e in elements):
        lines.append("║" + "    ✓ Horseshoe arch — Moorish/Maghrebi".ljust(w) + "║")
    if any("octagon" in e.get("shape", {}).get("type", "") for e in elements):
        lines.append("║" + "    ✓ Octagonal geometry — Islamic Ad Quadratum".ljust(w) + "║")
    if dome_count >= 3:
        lines.append("║" + "    ✓ Dome cascade — Mughal/Ottoman hierarchy".ljust(w) + "║")
    if any("star" in e.get("shape", {}).get("type", "") for e in elements):
        lines.append("║" + "    ✓ Star polygon — Islamic geometric art".ljust(w) + "║")
    if any("pointed" in e.get("shape", {}).get("type", "") for e in elements):
        lines.append("║" + "    ✓ Pointed arch — Islamic/Gothic tradition".ljust(w) + "║")
    lines.append("║" + " " * w + "║")
    
    # Notable
    lines.append("║" + "  NOTABLE".ljust(w) + "║")
    if has_phi and has_sqrt2 and has_sqrt3:
        lines.append("║" + "    All three canonical proportional systems".ljust(w) + "║")
        lines.append("║" + "    found — characteristic of pan-Islamic".ljust(w) + "║")
        lines.append("║" + "    synthesis architecture.".ljust(w) + "║")
    if avg_error < 1.0 and len(strong) > 5:
        lines.append("║" + "    Extremely precise proportional construction".ljust(w) + "║")
        lines.append("║" + "    — suggests careful geometric planning.".ljust(w) + "║")
    if symmetry > 90:
        lines.append("║" + "    Near-perfect bilateral symmetry.".ljust(w) + "║")
    if dome_count >= 3:
        lines.append("║" + f"    {dome_count} dome elements at varying scales —".ljust(w) + "║")
        lines.append("║" + "    hierarchical dome cascade detected.".ljust(w) + "║")
    
    lines.append("╚" + "═" * w + "╝")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Traced: SAM 2 architectural extraction")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="extraction.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--js-output", default=None)
    parser.add_argument("--name", default="Unknown Building", help="Building name for analysis card")
    args = parser.parse_args()
    
    extraction = run_sam2_extraction(args.image, args.checkpoint)
    extraction = compute_relationships(extraction)
    
    js_code = generate_js(extraction)
    extraction["generated_js"] = js_code
    
    card = generate_analysis_card(extraction, args.name)
    extraction["analysis_card"] = card
    
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
    
    print(f"\n")
    print(card)


if __name__ == "__main__":
    main()
