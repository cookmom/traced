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


def classify_shape(mask: np.ndarray, knowledge: dict = None) -> dict:
    """Classify a mask's shape into 17 architectural types.
    If knowledge is provided, uses building style to bias classification."""
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
    
    # === ADDITIONAL SHAPE DETECTION (doors, windows, modern + Islamic elements) ===
    
    # WINDOW — rectangular void within a wall, moderate size, high solidity
    if shape_type in ("rectangle", "panel", "unknown") and 0.5 < aspect < 2.0:
        # Windows tend to have clear edges and moderate area relative to parent
        if solidity > 0.7 and area < (w * h * 0.3):  # Not too big
            # Check if mostly in upper half of image (windows above ground)
            cy_pct = (y + h/2) / max(1, mask.shape[0])
            if cy_pct < 0.75:  # Above lower quarter
                shape_type = "window"
                confidence = 0.6
    
    # DOOR — tall rectangle in lower portion of image
    if shape_type in ("rectangle", "panel", "unknown") and aspect < 0.55:
        cy_pct = (y + h/2) / max(1, mask.shape[0])
        if cy_pct > 0.5 and solidity > 0.75:  # Lower half, solid
            shape_type = "door"
            confidence = 0.6
    
    # ROSETTE / ROSE WINDOW — circular decorative element, smaller than dome
    if shape_type in ("circle", "dome") and area < (mask.shape[0] * mask.shape[1] * 0.02):
        shape_type = "rosette"
        confidence = 0.65
    
    # BALUSTRADE / RAILING — very wide, very short, near top of walls
    if shape_type == "horizontal_band" and (h / max(1, w)) < 0.08:
        shape_type = "balustrade"
        confidence = 0.6
    
    # BUTTRESS — thick vertical element at edge, wider than column
    if shape_type in ("column", "rectangle") and aspect < 0.4 and 0.3 < (x / max(1, mask.shape[1])):
        if w > 20 and solidity > 0.8:
            cx_pct = (x + w/2) / max(1, mask.shape[1])
            if cx_pct < 0.15 or cx_pct > 0.85:  # Near edges
                shape_type = "buttress"
                confidence = 0.6
    
    # MIHRAB — pointed arch niche, typically centered, smaller than main arches
    if shape_type in ("pointed_arch", "arch") and area < (mask.shape[0] * mask.shape[1] * 0.02):
        cx_pct = (x + w/2) / max(1, mask.shape[1])
        if 0.4 < cx_pct < 0.6:  # Centered
            shape_type = "mihrab"
            confidence = 0.6
    
    # MUQARNAS — stalactite vaulting, complex shape with many vertices
    if n_vertices > 15 and solidity < 0.6 and convexity < 0.7:
        shape_type = "muqarnas"
        confidence = 0.55
    
    # MASHRABIYA — lattice screen, very low solidity, rectangular
    if solidity < 0.35 and 0.4 < aspect < 2.5 and n_vertices > 8:
        shape_type = "mashrabiya"
        confidence = 0.55
    
    # PARAPET — low wall along roofline, wide and short at top
    if shape_type == "horizontal_band":
        cy_pct = (y + h/2) / max(1, mask.shape[0])
        if cy_pct < 0.3:  # Upper portion
            shape_type = "parapet"
            confidence = 0.6
    
    # VAULT / BARREL VAULT — wide curved ceiling element
    if shape_type in ("arch", "dome_like") and aspect > 2.0 and circularity > 0.3:
        shape_type = "vault"
        confidence = 0.55
    
    # CURTAIN WALL — modern: large flat glass panel, high solidity, very large
    if shape_type in ("wall", "rectangle") and solidity > 0.9 and area > (mask.shape[0] * mask.shape[1] * 0.1):
        shape_type = "curtain_wall"
        confidence = 0.55
    
    # CANOPY / OVERHANG — wide horizontal element projecting outward
    if shape_type in ("horizontal_band", "rectangle") and aspect > 4.0:
        shape_type = "canopy"
        confidence = 0.5
    
    # Knowledge-biased reclassification
    if knowledge and shape_type in ("arch", "tall_arch", "unknown"):
        known_arches = knowledge.get("arch_types", [])
        if "horseshoe" in known_arches and circularity > 0.35 and aspect > 0.8:
            shape_type = "horseshoe_arch"
            confidence = max(confidence, 0.65)
        elif "cusped" in known_arches and n_vertices > 8:
            shape_type = "cusped_arch"
            confidence = max(confidence, 0.6)
        elif "pointed" in known_arches and aspect < 0.8:
            shape_type = "pointed_arch"
            confidence = max(confidence, 0.65)
    
    if knowledge and shape_type in ("dome", "dome_like"):
        known_domes = knowledge.get("dome_types", [])
        if "hemisphere" in known_domes and 0.4 < (h/max(1,w)) < 0.65:
            shape_type = "dome"
            confidence = max(confidence, 0.75)
    
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
    """Fit geometric primitives + curve profiles based on shape classification."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {}
    
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    
    result = {
        "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
        "center": {"x": float(x + w/2), "y": float(y + h/2)},
    }
    
    # Fit ellipse for curved shapes
    curved_types = ("dome", "dome_like", "arch", "tall_arch", "pointed_arch",
                    "horseshoe_arch", "ogee_arch", "circle")
    if shape_info["type"] in curved_types and len(largest) >= 5:
        el = cv2.fitEllipse(largest)
        result["ellipse"] = {
            "center": {"x": float(el[0][0]), "y": float(el[0][1])},
            "axes": {"w": float(el[1][0]), "h": float(el[1][1])},
            "angle": float(el[2]),
        }
    
    # Fit minimum enclosing circle for domes
    if shape_info["type"] in ("dome", "dome_like", "circle") and len(largest) >= 3:
        (cx_c, cy_c), radius = cv2.minEnclosingCircle(largest)
        result["circle"] = {
            "center": {"x": float(cx_c), "y": float(cy_c)},
            "radius": float(radius),
        }
    
    # === CURVE PROFILE EXTRACTION ===
    shape_t = shape_info["type"]
    
    # Extract upper profile (top half of contour — the curved part of domes/arches)
    if shape_t in curved_types:
        mid_y = y + h // 2
        # Sort contour points by x for profile extraction
        all_pts = [(int(p[0][0]), int(p[0][1])) for p in largest]
        upper_pts = sorted([p for p in all_pts if p[1] <= mid_y], key=lambda p: p[0])
        lower_pts = sorted([p for p in all_pts if p[1] > mid_y], key=lambda p: p[0])
        
        if len(upper_pts) >= 4:
            # Normalize upper profile to 0-1 range for portability
            min_x = min(p[0] for p in upper_pts)
            max_x = max(p[0] for p in upper_pts)
            min_y = min(p[1] for p in upper_pts)
            max_y = max(p[1] for p in upper_pts)
            span_x = max(1, max_x - min_x)
            span_y = max(1, max_y - min_y)
            
            # Sample profile at regular intervals (20 points)
            profile_pts = []
            n_samples = min(20, len(upper_pts))
            for i in range(n_samples):
                idx = i * (len(upper_pts) - 1) // max(1, n_samples - 1)
                px, py = upper_pts[idx]
                profile_pts.append([
                    round((px - min_x) / span_x, 4),  # normalized x: 0-1
                    round((py - min_y) / span_y, 4),   # normalized y: 0-1 (0=top)
                ])
            
            result["upper_profile"] = {
                "points": profile_pts,
                "span": {"x": span_x, "y": span_y},
                "origin": {"x": min_x, "y": min_y},
            }
        
        if len(lower_pts) >= 4:
            min_x = min(p[0] for p in lower_pts)
            max_x = max(p[0] for p in lower_pts)
            min_y = min(p[1] for p in lower_pts)
            max_y = max(p[1] for p in lower_pts)
            span_x = max(1, max_x - min_x)
            span_y = max(1, max_y - min_y)
            
            profile_pts = []
            n_samples = min(20, len(lower_pts))
            for i in range(n_samples):
                idx = i * (len(lower_pts) - 1) // max(1, n_samples - 1)
                px, py = lower_pts[idx]
                profile_pts.append([
                    round((px - min_x) / span_x, 4),
                    round((py - min_y) / span_y, 4),
                ])
            
            result["lower_profile"] = {
                "points": profile_pts,
                "span": {"x": span_x, "y": span_y},
                "origin": {"x": min_x, "y": min_y},
            }
    
    # === ARCH-SPECIFIC: springing points + apex ===
    if shape_t in ("arch", "tall_arch", "pointed_arch", "horseshoe_arch", "ogee_arch"):
        all_pts = [(int(p[0][0]), int(p[0][1])) for p in largest]
        if all_pts:
            # Apex = topmost point
            apex = min(all_pts, key=lambda p: p[1])
            # Springing points = leftmost and rightmost points near the widest horizontal
            left_spring = min(all_pts, key=lambda p: p[0])
            right_spring = max(all_pts, key=lambda p: p[0])
            
            result["arch_geometry"] = {
                "apex": {"x": apex[0], "y": apex[1]},
                "spring_left": {"x": left_spring[0], "y": left_spring[1]},
                "spring_right": {"x": right_spring[0], "y": right_spring[1]},
                "span": right_spring[0] - left_spring[0],
                "rise": max(left_spring[1], right_spring[1]) - apex[1],
                "rise_to_span": round(
                    (max(left_spring[1], right_spring[1]) - apex[1]) / 
                    max(1, right_spring[0] - left_spring[0]), 4
                ),
            }
            # Rise-to-span ratio tells us the arch type:
            # 0.5 = semicircular, 0.866 = equilateral pointed, >1 = tall pointed
            # <0.5 = segmental, >0.5 with horseshoe = horseshoe
            r2s = result["arch_geometry"]["rise_to_span"]
            result["arch_geometry"]["profile_type"] = (
                "segmental" if r2s < 0.45 else
                "semicircular" if r2s < 0.55 else
                "slightly_pointed" if r2s < 0.75 else
                "equilateral_pointed" if r2s < 0.95 else
                "lancet" if r2s < 1.5 else
                "stilted"
            )
    
    # === DOME-SPECIFIC: base diameter + height ratio ===
    if shape_t in ("dome", "dome_like"):
        all_pts = [(int(p[0][0]), int(p[0][1])) for p in largest]
        if all_pts:
            apex = min(all_pts, key=lambda p: p[1])
            # Base = widest horizontal extent at bottom quarter
            bottom_quarter_y = y + h * 3 // 4
            base_pts = [p for p in all_pts if p[1] >= bottom_quarter_y]
            if base_pts:
                base_left = min(base_pts, key=lambda p: p[0])
                base_right = max(base_pts, key=lambda p: p[0])
                base_diameter = base_right[0] - base_left[0]
                dome_height = max(base_left[1], base_right[1]) - apex[1]
                
                result["dome_geometry"] = {
                    "apex": {"x": apex[0], "y": apex[1]},
                    "base_left": {"x": base_left[0], "y": base_left[1]},
                    "base_right": {"x": base_right[0], "y": base_right[1]},
                    "base_diameter": base_diameter,
                    "height": dome_height,
                    "height_to_diameter": round(dome_height / max(1, base_diameter), 4),
                    "profile_type": (
                        "onion" if dome_height / max(1, base_diameter) > 0.8 else
                        "hemisphere" if dome_height / max(1, base_diameter) > 0.4 else
                        "saucer"
                    ),
                }
    
    # Simplified contour for drawing (more points for curved shapes)
    max_pts = 120 if shape_t in curved_types else 60
    epsilon = max(1.0, min(w, h) * 0.015)
    simplified = cv2.approxPolyDP(largest, epsilon, True)
    result["contour"] = [[int(pt[0][0]), int(pt[0][1])] for pt in simplified[:max_pts]]
    
    return result


def run_depth_estimation(image_path: str) -> np.ndarray | None:
    """Run Depth Anything V2 on an image, return normalized depth map."""
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
        import torch
    except ImportError:
        print("  Depth Anything V2 not installed — skipping depth estimation")
        return None
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Try loading the model
        model_configs = {
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        }
        
        # Try largest first, fall back to smaller
        model = None
        for size in ['vitl', 'vitb', 'vits']:
            try:
                model = DepthAnythingV2(**model_configs[size])
                ckpt = f'depth_anything_v2_{size}.pth'
                script_dir = Path(__file__).parent
                ckpt_path = script_dir / "checkpoints" / ckpt
                if ckpt_path.exists():
                    model.load_state_dict(torch.load(str(ckpt_path), map_location=device, weights_only=True))
                else:
                    # Try HuggingFace hub
                    from huggingface_hub import hf_hub_download
                    ckpt_path = hf_hub_download(f"depth-anything/Depth-Anything-V2-{size.upper()}", f"{ckpt}", local_dir=str(script_dir / "checkpoints"))
                    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
                model = model.to(device).eval()
                print(f"  Loaded Depth Anything V2 ({size}) on {device}")
                break
            except Exception as e:
                model = None
                continue
        
        if model is None:
            print("  Could not load any Depth Anything V2 model")
            return None
        
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        with torch.no_grad():
            depth = model.infer_image(img)
        
        # Normalize to 0-1
        depth = depth.astype(np.float32)
        depth = (depth - depth.min()) / max(0.001, depth.max() - depth.min())
        print(f"  Depth map: {depth.shape}, range [{depth.min():.3f}, {depth.max():.3f}]")
        return depth
        
    except Exception as e:
        print(f"  Depth estimation failed: {e}")
        return None


def extract_edge_map(image_path: str) -> np.ndarray | None:
    """Extract Canny edge map from image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # Bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(img, 9, 75, 75)
    
    # Multi-scale Canny for both fine and coarse edges
    edges_fine = cv2.Canny(filtered, 30, 80)
    edges_coarse = cv2.Canny(filtered, 60, 150)
    
    # Combine: fine edges capture ornamental detail, coarse edges capture structure
    combined = cv2.bitwise_or(edges_fine, edges_coarse)
    
    print(f"  Edge map: {combined.shape}, {np.count_nonzero(combined)} edge pixels")
    return combined


def extract_dominant_lines(image_path: str) -> list:
    """Extract dominant structural lines using Hough transform."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []
    
    edges = cv2.Canny(img, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=min(img.shape) * 0.1,
                            maxLineGap=10)
    if lines is None:
        return []
    
    h, w = img.shape
    result = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        angle = math.degrees(math.atan2(y2-y1, x2-x1)) % 180
        
        # Classify line orientation
        if angle < 10 or angle > 170:
            orientation = "horizontal"
        elif 80 < angle < 100:
            orientation = "vertical"
        else:
            orientation = "diagonal"
        
        result.append({
            "start": {"x": round(x1/w, 4), "y": round(y1/h, 4)},
            "end": {"x": round(x2/w, 4), "y": round(y2/h, 4)},
            "length_pct": round(length / math.sqrt(w**2 + h**2), 4),
            "angle": round(angle, 1),
            "orientation": orientation,
        })
    
    # Sort by length, keep top 30
    result.sort(key=lambda l: -l["length_pct"])
    return result[:30]


def extract_symmetry(extraction: dict) -> dict:
    """Analyze bilateral symmetry of the composition."""
    elements = extraction.get("elements", [])
    img_w = extraction["image_size"]["w"]
    
    # Find symmetry axis (default: image center)
    x_positions = [e.get("position_pct", {}).get("x", 0.5) for e in elements]
    # Test if center of mass is near 0.5
    avg_x = sum(x_positions) / max(1, len(x_positions))
    axis = 0.5  # Could refine by optimizing mirror matches
    
    # Find bilateral pairs
    pairs = []
    used = set()
    for i, a in enumerate(elements):
        if i in used:
            continue
        ax = a.get("position_pct", {}).get("x", 0.5)
        ay = a.get("position_pct", {}).get("y", 0.5)
        a_type = a.get("shape", {}).get("type", "")
        
        if abs(ax - axis) < 0.03:  # On axis — no pair needed
            continue
        
        best_match = None
        best_dist = float("inf")
        for j, b in enumerate(elements):
            if j <= i or j in used:
                continue
            bx = b.get("position_pct", {}).get("x", 0.5)
            by = b.get("position_pct", {}).get("y", 0.5)
            b_type = b.get("shape", {}).get("type", "")
            
            mirror_x = 2 * axis - ax
            dist = abs(bx - mirror_x) + abs(by - ay) * 0.5
            
            if dist < 0.08 and b_type == a_type and dist < best_dist:
                best_match = j
                best_dist = dist
        
        if best_match is not None:
            pairs.append({
                "left": a["name"],
                "right": elements[best_match]["name"],
                "deviation": round(best_dist, 4),
            })
            used.add(i)
            used.add(best_match)
    
    return {
        "axis_x": round(axis, 4),
        "center_of_mass_x": round(avg_x, 4),
        "bilateral_pairs": pairs,
        "symmetry_score": min(100, int(len(pairs) * 2 * 100 / max(1, len(elements)))),
    }


def extract_hierarchy(extraction: dict) -> list:
    """Determine element nesting/containment hierarchy."""
    elements = extraction.get("elements", [])
    hierarchy = []
    
    for i, a in enumerate(elements):
        a_bb = a.get("primitives", {}).get("bbox", {})
        if not a_bb:
            continue
        
        children = []
        for j, b in enumerate(elements):
            if j == i:
                continue
            b_bb = b.get("primitives", {}).get("bbox", {})
            if not b_bb:
                continue
            
            # Check if b is contained within a
            if (b_bb["x"] >= a_bb["x"] and 
                b_bb["y"] >= a_bb["y"] and
                b_bb["x"] + b_bb["w"] <= a_bb["x"] + a_bb["w"] and
                b_bb["y"] + b_bb["h"] <= a_bb["y"] + a_bb["h"]):
                children.append(b["name"])
        
        if children:
            hierarchy.append({
                "parent": a["name"],
                "children": children,
                "depth": 0,  # Will be computed below
            })
    
    # Compute depth (elements contained in more parents are deeper)
    depth_map = {}
    for el in elements:
        name = el["name"]
        depth = sum(1 for h in hierarchy if name in h["children"])
        depth_map[name] = depth
    
    for h in hierarchy:
        h["depth"] = depth_map.get(h["parent"], 0)
    
    hierarchy.sort(key=lambda h: h["depth"])
    return hierarchy


def run_sam2_extraction(image_path: str, checkpoint: str = None, knowledge: dict = None) -> dict:
    """Run SAM 2 automatic mask generation on an image.
    If knowledge is provided, uses it to bias shape classification."""
    
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
    # High-density segmentation for architectural detail
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=64,          # 4x more seed points (was 32)
        pred_iou_thresh=0.6,         # Lower threshold to catch more (was 0.7)
        stability_score_thresh=0.80,  # More permissive (was 0.85)
        min_mask_region_area=200,     # Catch smaller features (was 500)
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
    min_area_pct = 0.001  # Catch features as small as 0.1% of image (was 0.5%)
    min_area = img_w * img_h * min_area_pct
    
    for i, mask_data in enumerate(masks):
        if mask_data["area"] < min_area:
            continue
        if len(elements) >= 50:  # Cap at 50 elements (was 30)
            break
        
        mask = mask_data["segmentation"].astype(np.uint8)
        
        # Classify shape
        shape = classify_shape(mask, knowledge)
        
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
    
    # === SUB-ELEMENT EXTRACTION ===
    # For large elements (>5% of image), run SAM again inside the region
    # to find windows, grilles, columns, ornament within
    print("Running sub-element extraction...")
    sub_elements = []
    large_elements = [e for e in elements if e.get("area_pct", 0) > 0.05]
    
    for parent in large_elements:
        bb = parent.get("primitives", {}).get("bbox", {})
        if not bb or bb.get("w", 0) < 50 or bb.get("h", 0) < 50:
            continue
        
        # Crop image to parent's bounding box
        x1, y1 = max(0, bb["x"]), max(0, bb["y"])
        x2, y2 = min(img_w, bb["x"] + bb["w"]), min(img_h, bb["y"] + bb["h"])
        crop = img_rgb[y1:y2, x1:x2]
        
        if crop.shape[0] < 30 or crop.shape[1] < 30:
            continue
        
        try:
            # Use tighter settings for sub-elements
            sub_generator = SAM2AutomaticMaskGenerator(
                model=sam2,
                points_per_side=32,
                pred_iou_thresh=0.65,
                stability_score_thresh=0.82,
                min_mask_region_area=100,
            )
            sub_masks = sub_generator.generate(crop)
            
            # Filter: keep only sub-elements that are significantly smaller than parent
            parent_area = (x2 - x1) * (y2 - y1)
            for sm in sub_masks:
                sub_area_pct = sm["area"] / parent_area
                if sub_area_pct > 0.5 or sub_area_pct < 0.01:
                    continue  # Skip too-big (parent itself) or too-small (noise)
                
                sub_mask = sm["segmentation"].astype(np.uint8)
                sub_shape = classify_shape(sub_mask, knowledge)
                sub_prims = fit_primitives(sub_mask, sub_shape)
                
                if not sub_prims:
                    continue
                
                # Offset coordinates back to full image space
                sub_bb = sub_prims["bbox"]
                sub_prims["bbox"]["x"] += x1
                sub_prims["bbox"]["y"] += y1
                sub_prims["center"]["x"] += x1
                sub_prims["center"]["y"] += y1
                
                sub_el = {
                    "name": f"sub_{parent['name']}_{sub_shape['type']}_{len(sub_elements)}",
                    "parent": parent["name"],
                    "shape": sub_shape,
                    "primitives": sub_prims,
                    "area_pct": round(sm["area"] / (img_w * img_h), 4),
                    "position_pct": {
                        "x": round(sub_prims["center"]["x"] / img_w, 4),
                        "y": round(sub_prims["center"]["y"] / img_h, 4),
                    },
                    "size_pct": {
                        "w": round(sub_bb["w"] / img_w, 4),
                        "h": round(sub_bb["h"] / img_h, 4),
                    },
                }
                sub_elements.append(sub_el)
            
        except Exception as e:
            print(f"    Sub-extraction failed for {parent['name']}: {e}")
    
    if sub_elements:
        print(f"  Found {len(sub_elements)} sub-elements within {len(large_elements)} large elements")
        elements.extend(sub_elements)
    else:
        print(f"  No sub-elements found")
    
    result = {
        "image": str(image_path),
        "image_size": {"w": img_w, "h": img_h},
        "method": "sam2",
        "total_masks": len(masks),
        "sub_elements_found": len(sub_elements),
        "elements": elements,
    }
    
    # Additional structural analysis
    print("Extracting dominant lines...")
    result["dominant_lines"] = extract_dominant_lines(image_path)
    print(f"  Found {len(result['dominant_lines'])} structural lines")
    
    print("Analyzing symmetry...")
    result["symmetry"] = extract_symmetry(result)
    print(f"  Symmetry score: {result['symmetry']['symmetry_score']}/100, "
          f"{len(result['symmetry']['bilateral_pairs'])} bilateral pairs")
    
    print("Building hierarchy...")
    result["hierarchy"] = extract_hierarchy(result)
    print(f"  {len(result['hierarchy'])} parent-child relationships")
    
    # Depth estimation
    print("Running depth estimation...")
    depth_map = run_depth_estimation(image_path)
    if depth_map is not None:
        # Assign depth layer to each element
        print("Assigning depth layers...")
        for el in elements:
            bb = el.get("primitives", {}).get("bbox", {})
            if bb:
                x, y, w, h = bb["x"], bb["y"], bb["w"], bb["h"]
                # Sample depth in element's region
                region = depth_map[y:y+h, x:x+w]
                if region.size > 0:
                    mean_depth = float(np.mean(region))
                    median_depth = float(np.median(region))
                    el["depth"] = {
                        "mean": round(mean_depth, 4),
                        "median": round(median_depth, 4),
                    }
        
        # Sort elements by depth and assign layers using k-means clustering
        depth_elements = [e for e in elements if "depth" in e]
        if depth_elements:
            depths = np.array([e["depth"]["median"] for e in depth_elements])
            min_d, max_d = float(depths.min()), float(depths.max())
            depth_range = max(0.001, max_d - min_d)
            
            # Normalize
            for el in depth_elements:
                el["depth"]["normalized"] = round((el["depth"]["median"] - min_d) / depth_range, 4)
            
            # K-means clustering into 4 layers (or fewer if not enough elements)
            from scipy.cluster.vq import kmeans, vq
            norm_depths = np.array([e["depth"]["normalized"] for e in depth_elements]).reshape(-1, 1)
            n_clusters = min(4, len(depth_elements))
            
            if n_clusters >= 2:
                centroids, _ = kmeans(norm_depths.astype(float), n_clusters)
                labels, _ = vq(norm_depths.astype(float), centroids)
                
                # Sort clusters by depth (0 = nearest/foreground)
                cluster_order = np.argsort(centroids.flatten())
                label_to_rank = {old: new for new, old in enumerate(cluster_order)}
                
                layer_names = ["foreground_frame", "mid_facade", "background_domes", "distant_sky"]
                layer_weights = [2.5, 1.8, 1.2, 0.6]
                
                for i, el in enumerate(depth_elements):
                    rank = label_to_rank[labels[i]]
                    # Map rank to layer name (handle < 4 clusters)
                    if n_clusters == 2:
                        layer_idx = 0 if rank == 0 else 3
                    elif n_clusters == 3:
                        layer_idx = [0, 1, 3][rank]
                    else:
                        layer_idx = rank
                    
                    el["depth"]["layer"] = layer_names[layer_idx]
                    el["depth"]["line_weight"] = layer_weights[layer_idx]
                    el["depth"]["cluster"] = int(rank)
                    
                    print(f"    {el['name']:25s} depth={el['depth']['normalized']:.3f} → {el['depth']['layer']} (weight={el['depth']['line_weight']})")
            else:
                # Only 1 element — assign to mid
                depth_elements[0]["depth"]["layer"] = "mid_facade"
                depth_elements[0]["depth"]["line_weight"] = 1.8
                print(f"    {depth_elements[0]['name']:25s} → mid_facade (single element)")
        
        result["depth_available"] = True
    else:
        result["depth_available"] = False
    
    # Edge extraction per element
    print("Extracting edge contours per element...")
    edge_map = extract_edge_map(image_path)
    if edge_map is not None:
        for el in elements:
            bb = el.get("primitives", {}).get("bbox", {})
            if bb:
                x, y, w, h = bb["x"], bb["y"], bb["w"], bb["h"]
                # Crop edge map to element's region
                region = edge_map[y:y+h, x:x+w]
                # Find contours in the cropped edge region
                contours, _ = cv2.findContours(region, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                # Keep significant contours (>2% of element area)
                min_len = max(10, (w + h) * 0.04)
                sig_contours = [c for c in contours if cv2.arcLength(c, False) > min_len]
                
                # Convert to normalized drawing paths (0-1 within element bbox)
                edge_paths = []
                for c in sorted(sig_contours, key=lambda c: -cv2.arcLength(c, False))[:15]:
                    epsilon = max(1.0, min(w, h) * 0.01)
                    simplified = cv2.approxPolyDP(c, epsilon, False)
                    if len(simplified) >= 2:
                        path = []
                        for pt in simplified:
                            px = round(pt[0][0] / max(1, w), 4)
                            py = round(pt[0][1] / max(1, h), 4)
                            path.append([px, py])
                        edge_paths.append(path)
                
                if edge_paths:
                    el["edge_paths"] = edge_paths
                    el["edge_path_count"] = len(edge_paths)
                    # Detail density = edges per pixel area
                    el["detail_density"] = round(sum(len(p) for p in edge_paths) / max(1, w * h) * 10000, 2)
        
        # Report detail density
        detailed = sorted([e for e in elements if "detail_density" in e], key=lambda e: -e["detail_density"])
        if detailed:
            print("  Detail density (high = more ornamental detail):")
            for e in detailed[:5]:
                print(f"    {e['name']:25s} density={e['detail_density']:6.2f} paths={e.get('edge_path_count', 0)}")
        
        result["edges_available"] = True
    else:
        result["edges_available"] = False
    
    return result


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
        
        shape = classify_shape(mask, knowledge)
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
    
    # Depth layers (if available)
    depth_layers = {}
    for e in elements:
        layer = e.get("depth", {}).get("layer", "unknown")
        if layer != "unknown":
            depth_layers[layer] = depth_layers.get(layer, 0) + 1
    if depth_layers:
        lines.append("║" + "  DEPTH LAYERS".ljust(w) + "║")
        layer_order = ["foreground_frame", "mid_facade", "background_domes", "distant_sky"]
        for layer in layer_order:
            if layer in depth_layers:
                weight = {"foreground_frame": 2.5, "mid_facade": 1.8, "background_domes": 1.2, "distant_sky": 0.6}.get(layer, 1.0)
                lines.append("║" + f"    {layer:25s} {depth_layers[layer]:2d} elements (wt={weight})".ljust(w) + "║")
        lines.append("║" + " " * w + "║")
    
    # Detail hotspots
    detailed = sorted([e for e in elements if "detail_density" in e], key=lambda e: -e["detail_density"])
    if detailed:
        lines.append("║" + "  DETAIL HOTSPOTS (draw more detail here)".ljust(w) + "║")
        for e in detailed[:3]:
            lines.append("║" + f"    {e['name']:20s} density={e['detail_density']:6.2f}".ljust(w) + "║")
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
    parser.add_argument("--knowledge", default=None, help="Path to knowledge.json from research.py")
    args = parser.parse_args()
    
    # Load research knowledge if provided
    knowledge = None
    if args.knowledge and Path(args.knowledge).exists():
        knowledge = json.loads(Path(args.knowledge).read_text())
        print(f"Loaded research knowledge: {len(knowledge.get('style_influences', []))} styles, "
              f"{len(knowledge.get('arch_types', []))} arch types, "
              f"{len(knowledge.get('dome_types', []))} dome types")
    
    extraction = run_sam2_extraction(args.image, args.checkpoint, knowledge)
    extraction = compute_relationships(extraction)
    
    js_code = generate_js(extraction)
    extraction["generated_js"] = js_code
    if knowledge:
        extraction["knowledge"] = {
            "styles": knowledge.get("style_influences", []),
            "arch_types": knowledge.get("arch_types", []),
            "dome_types": knowledge.get("dome_types", []),
            "dimensions": knowledge.get("dimensions", {}),
            "traditions": list(knowledge.get("traditions", {}).keys()),
        }
    
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
    
    # Structural summary
    sym = extraction.get("symmetry", {})
    hier = extraction.get("hierarchy", [])
    dom_lines = extraction.get("dominant_lines", [])
    
    print(f"\n{'='*60}")
    print("STRUCTURAL ANALYSIS:")
    print(f"{'='*60}")
    
    print(f"\n  SYMMETRY — score: {sym.get('symmetry_score', 0)}/100")
    for pair in sym.get("bilateral_pairs", []):
        print(f"    {pair['left']} ↔ {pair['right']} (dev: {pair['deviation']:.4f})")
    
    print(f"\n  HIERARCHY — {len(hier)} containment relationships")
    for h in hier[:8]:
        children_str = ", ".join(h["children"][:4])
        if len(h["children"]) > 4:
            children_str += f" +{len(h['children'])-4} more"
        print(f"    {h['parent']} contains [{children_str}]")
    
    print(f"\n  DOMINANT LINES — {len(dom_lines)} structural lines")
    orientations = {}
    for l in dom_lines:
        orientations[l["orientation"]] = orientations.get(l["orientation"], 0) + 1
    for orient, count in sorted(orientations.items(), key=lambda x: -x[1]):
        print(f"    {orient:12s} {count:3d} lines")
    
    # Depth layers
    depth_layers = {}
    for e in extraction["elements"]:
        layer = e.get("depth", {}).get("layer")
        if layer:
            depth_layers.setdefault(layer, []).append(e["name"])
    if depth_layers:
        print(f"\n  DEPTH LAYERS")
        for layer in ["foreground_frame", "mid_facade", "background_domes", "distant_sky"]:
            if layer in depth_layers:
                print(f"    {layer:25s} [{', '.join(depth_layers[layer][:4])}]")
    
    # Detail hotspots
    detailed = sorted([e for e in extraction["elements"] if "detail_density" in e], key=lambda e: -e["detail_density"])
    if detailed:
        print(f"\n  DETAIL HOTSPOTS")
        for e in detailed[:5]:
            print(f"    {e['name']:25s} density={e['detail_density']:6.2f} ({e.get('edge_path_count', 0)} paths)")
    
    print(f"\n  CURVE GEOMETRY")
    for e in extraction["elements"]:
        p = e.get("primitives", {})
        ag = p.get("arch_geometry")
        dg = p.get("dome_geometry")
        up = p.get("upper_profile")
        parts = []
        if ag:
            parts.append(f"arch rise/span={ag['rise_to_span']} ({ag['profile_type']})")
        if dg:
            parts.append(f"dome h/d={dg['height_to_diameter']} ({dg['profile_type']})")
        if up:
            parts.append(f"profile: {len(up['points'])} pts")
        if parts:
            print(f"    {e['name']:25s} {' | '.join(parts)}")
    
    print(f"\n")
    print(card)


if __name__ == "__main__":
    main()
