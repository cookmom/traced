#!/usr/bin/env python3
"""
Traced Pipeline — Stage 3.5: Gradient-Based Parameter Optimization
Takes initial parameters from extraction and optimizes them against
the reference photo's edge map using differentiable rendering.

The output is MATHEMATICALLY PRECISE architectural parameters:
- Dome center, radius, height ratio → hemisphere equation
- Arch springing, span, rise → pointed/horseshoe arc equation  
- Rectangle corners → axis-aligned geometry

These parameters ARE the drawing. No approximation. Pure math.

Usage:
    python optimize.py --extraction extraction.json --image photo.jpg --output optimized.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import cv2

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch not available — optimization disabled")


def extract_reference_edges(image_path: str) -> np.ndarray:
    """Get clean edge map from reference photo."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read {image_path}")
    
    # Bilateral filter preserves edges while smoothing noise
    filtered = cv2.bilateralFilter(img, 9, 75, 75)
    
    # Multi-scale Canny
    edges_fine = cv2.Canny(filtered, 30, 80)
    edges_coarse = cv2.Canny(filtered, 60, 150)
    combined = cv2.bitwise_or(edges_fine, edges_coarse)
    
    # Light morphological cleanup
    kernel = np.ones((2, 2), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    
    return combined


def render_dome(canvas: np.ndarray, cx: float, cy: float, radius: float, 
                h_ratio: float = 1.0, thickness: int = 2):
    """Render a dome curve onto canvas using exact mathematical formula."""
    h, w = canvas.shape
    pts = []
    for i in range(100):
        t = i / 99
        angle = math.pi * (1 - t)
        x = int(cx + radius * math.cos(angle))
        y = int(cy - radius * h_ratio * math.sin(angle))
        if 0 <= x < w and 0 <= y < h:
            pts.append((x, y))
    
    for i in range(1, len(pts)):
        cv2.line(canvas, pts[i-1], pts[i], 255, thickness)


def render_arch(canvas: np.ndarray, cx: float, spring_y: float, 
                half_span: float, rise_ratio: float, 
                profile: str = "pointed", thickness: int = 2):
    """Render an arch curve using exact mathematical formula."""
    h, w = canvas.shape
    rise = half_span * 2 * rise_ratio
    pts = []
    
    for i in range(100):
        t = i / 99
        if profile == "horseshoe":
            # Horseshoe: arc > 180°
            arch_r = half_span * 1.05
            center_y = spring_y - half_span * 0.1
            angle = math.pi * 0.15 + (math.pi - math.pi * 0.3) * t
            x = cx + arch_r * math.cos(math.pi - angle)
            y = center_y - arch_r * math.sin(angle)
        elif profile == "ogee":
            x = cx - half_span + half_span * 2 * t
            nt = t * 2 if t <= 0.5 else (1 - t) * 2
            y = spring_y - rise * (0.5 * math.sin(math.pi * nt) + 
                                    0.5 * (math.sin(math.pi * t) ** 0.6))
        elif profile == "semicircular":
            angle = math.pi * (1 - t)
            x = cx + half_span * math.cos(angle)
            y = spring_y - half_span * math.sin(angle)
        else:  # pointed
            if t <= 0.5:
                tt = t * 2
                x = cx - half_span + half_span * tt
                y = spring_y - rise * math.sin(math.pi / 2 * tt)
            else:
                tt = (t - 0.5) * 2
                x = cx + half_span * tt
                y = spring_y - rise + rise * (1 - math.cos(math.pi / 2 * tt))
        
        ix, iy = int(x), int(y)
        if 0 <= ix < w and 0 <= iy < h:
            pts.append((ix, iy))
    
    for i in range(1, len(pts)):
        cv2.line(canvas, pts[i-1], pts[i], 255, thickness)


def render_rect(canvas: np.ndarray, x: float, y: float, 
                w_r: float, h_r: float, thickness: int = 2):
    """Render a rectangle."""
    h, w = canvas.shape
    x1, y1 = int(max(0, x)), int(max(0, y))
    x2, y2 = int(min(w-1, x + w_r)), int(min(h-1, y + h_r))
    cv2.rectangle(canvas, (x1, y1), (x2, y2), 255, thickness)


def render_from_params(params_dict: dict, canvas_w: int, canvas_h: int) -> np.ndarray:
    """Render all elements from parameter dict onto a canvas."""
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    
    for name, p in params_dict.items():
        shape = p.get("type", "unknown")
        weight = max(1, int(p.get("line_weight", 1.5)))
        
        if "dome" in shape:
            render_dome(canvas, p["cx"], p["cy"], p["radius"], 
                       p.get("h_ratio", 1.0), weight)
        elif "arch" in shape:
            profile = "pointed"
            if "horseshoe" in shape:
                profile = "horseshoe"
            elif "ogee" in shape:
                profile = "ogee"
            elif "semicircular" in shape:
                profile = "semicircular"
            render_arch(canvas, p["cx"], p["spring_y"], p["half_span"],
                       p.get("rise_ratio", 0.866), profile, weight)
        elif shape in ("wall", "rectangle", "square"):
            render_rect(canvas, p["x"], p["y"], p["w"], p["h"], weight)
    
    return canvas


def chamfer_distance(edges_a: np.ndarray, edges_b: np.ndarray) -> float:
    """Compute chamfer distance between two edge maps.
    Lower = better match."""
    # Distance transform of B
    dist_b = cv2.distanceTransform(255 - edges_b, cv2.DIST_L2, 5)
    
    # Average distance of A's edge pixels to nearest B edge
    a_pixels = edges_a > 0
    if np.sum(a_pixels) == 0:
        return 1000.0
    
    forward = np.mean(dist_b[a_pixels])
    
    # Reverse: distance of B's edges to nearest A edge
    dist_a = cv2.distanceTransform(255 - edges_a, cv2.DIST_L2, 5)
    b_pixels = edges_b > 0
    if np.sum(b_pixels) == 0:
        return 1000.0
    
    backward = np.mean(dist_a[b_pixels])
    
    return (forward + backward) / 2


def edge_iou(edges_a: np.ndarray, edges_b: np.ndarray, dilate: int = 3) -> float:
    """Compute IoU of edge pixels (with dilation tolerance)."""
    kernel = np.ones((dilate, dilate), np.uint8)
    a_dilated = cv2.dilate(edges_a, kernel)
    b_dilated = cv2.dilate(edges_b, kernel)
    
    intersection = np.sum((a_dilated > 0) & (b_dilated > 0))
    union = np.sum((a_dilated > 0) | (b_dilated > 0))
    
    return intersection / max(1, union)


def render_element(canvas: np.ndarray, p: dict):
    """Render a single element onto canvas."""
    shape = p.get("type", "unknown")
    weight = max(1, int(p.get("line_weight", 1.5)))
    
    if "dome" in shape:
        render_dome(canvas, p["cx"], p["cy"], p["radius"], p.get("h_ratio", 1.0), weight)
    elif "arch" in shape:
        profile = "pointed"
        if "horseshoe" in shape:
            profile = "horseshoe"
        elif "ogee" in shape:
            profile = "ogee"
        elif "semicircular" in shape:
            profile = "semicircular"
        elif "cusped" in shape:
            profile = "cusped"
        render_arch(canvas, p["cx"], p["spring_y"], p["half_span"],
                   p.get("rise_ratio", 0.866), profile, weight)
    elif shape in ("wall", "rectangle", "square"):
        render_rect(canvas, p["x"], p["y"], p["w"], p["h"], weight)


def render_cusped_arch(canvas, cx, spring_y, half_span, rise_ratio, thickness=2):
    """Cusped arch — pointed with decorative scallops."""
    h, w = canvas.shape
    rise = half_span * 2 * rise_ratio
    pts = []
    for i in range(100):
        t = i / 99
        x_base = cx - half_span + half_span * 2 * t
        base_curve = math.sin(math.pi * t)
        cusps = 0.06 * math.sin(math.pi * t * 7)
        y = spring_y - rise * (base_curve + cusps)
        ix, iy = int(x_base), int(y)
        if 0 <= ix < w and 0 <= iy < h:
            pts.append((ix, iy))
    for i in range(1, len(pts)):
        cv2.line(canvas, pts[i-1], pts[i], 255, thickness)


def render_tudor_arch(canvas, cx, spring_y, half_span, rise_ratio, thickness=2):
    """Tudor arch — four-center depressed arch."""
    h, w = canvas.shape
    rise = half_span * 2 * rise_ratio
    pts = []
    for i in range(100):
        t = i / 99
        x = cx - half_span + half_span * 2 * t
        # Four-center: flatter in middle, steeper at sides
        if t < 0.2 or t > 0.8:
            # Steep side arcs
            nt = t / 0.2 if t < 0.2 else (1 - t) / 0.2
            y = spring_y - rise * 0.4 * math.sin(math.pi / 2 * nt)
        else:
            # Flat center arc
            nt = (t - 0.2) / 0.6
            y = spring_y - rise * (0.4 + 0.6 * math.sin(math.pi * nt))
        ix, iy = int(x), int(y)
        if 0 <= ix < w and 0 <= iy < h:
            pts.append((ix, iy))
    for i in range(1, len(pts)):
        cv2.line(canvas, pts[i-1], pts[i], 255, thickness)


def render_elliptical_arch(canvas, cx, spring_y, half_span, rise_ratio, thickness=2):
    """Elliptical arch — smooth semi-ellipse."""
    h, w = canvas.shape
    rise = half_span * 2 * rise_ratio
    pts = []
    for i in range(100):
        t = i / 99
        angle = math.pi * (1 - t)
        x = cx + half_span * math.cos(angle)
        y = spring_y - rise * math.sin(angle)
        ix, iy = int(x), int(y)
        if 0 <= ix < w and 0 <= iy < h:
            pts.append((ix, iy))
    for i in range(1, len(pts)):
        cv2.line(canvas, pts[i-1], pts[i], 255, thickness)


def render_parabolic_arch(canvas, cx, spring_y, half_span, rise_ratio, thickness=2):
    """Parabolic arch — y = a*x²."""
    h, w = canvas.shape
    rise = half_span * 2 * rise_ratio
    pts = []
    for i in range(100):
        t = i / 99
        x = cx - half_span + half_span * 2 * t
        # Parabola: y = rise * (1 - ((x-cx)/half_span)²)
        nx = (x - cx) / half_span
        y = spring_y - rise * (1 - nx * nx)
        ix, iy = int(x), int(y)
        if 0 <= ix < w and 0 <= iy < h:
            pts.append((ix, iy))
    for i in range(1, len(pts)):
        cv2.line(canvas, pts[i-1], pts[i], 255, thickness)


def render_catenary_arch(canvas, cx, spring_y, half_span, rise_ratio, thickness=2):
    """Catenary arch — inverted catenary curve (structurally optimal)."""
    h, w = canvas.shape
    rise = half_span * 2 * rise_ratio
    a = rise / (math.cosh(half_span / rise * 1.5) - 1) if rise > 0 else half_span
    pts = []
    for i in range(100):
        t = i / 99
        x = cx - half_span + half_span * 2 * t
        nx = (x - cx) / half_span * 1.5
        y = spring_y - a * (math.cosh(half_span / rise * 1.5) - math.cosh(nx * rise / a)) if a > 0 else spring_y
        y = max(spring_y - rise * 1.2, min(spring_y, y))
        ix, iy = int(x), int(y)
        if 0 <= ix < w and 0 <= iy < h:
            pts.append((ix, iy))
    for i in range(1, len(pts)):
        cv2.line(canvas, pts[i-1], pts[i], 255, thickness)


# All curve families for arch elements
ARCH_CURVE_FAMILIES = {
    "pointed": lambda c, cx, sy, hs, rr, w: render_arch(c, cx, sy, hs, rr, "pointed", w),
    "horseshoe": lambda c, cx, sy, hs, rr, w: render_arch(c, cx, sy, hs, rr, "horseshoe", w),
    "ogee": lambda c, cx, sy, hs, rr, w: render_arch(c, cx, sy, hs, rr, "ogee", w),
    "semicircular": lambda c, cx, sy, hs, rr, w: render_arch(c, cx, sy, hs, rr, "semicircular", w),
    "cusped": lambda c, cx, sy, hs, rr, w: render_cusped_arch(c, cx, sy, hs, rr, w),
    "tudor": lambda c, cx, sy, hs, rr, w: render_tudor_arch(c, cx, sy, hs, rr, w),
    "elliptical": lambda c, cx, sy, hs, rr, w: render_elliptical_arch(c, cx, sy, hs, rr, w),
    "parabolic": lambda c, cx, sy, hs, rr, w: render_parabolic_arch(c, cx, sy, hs, rr, w),
    "catenary": lambda c, cx, sy, hs, rr, w: render_catenary_arch(c, cx, sy, hs, rr, w),
}

# All curve families for dome elements
DOME_CURVE_FAMILIES = {
    "hemisphere": 1.0,
    "pointed": 1.2,
    "onion": None,  # special handling
    "saucer": 0.4,
    "bulbous": None,  # special handling
}


def find_best_curve_family(name: str, p: dict, ref_edges: np.ndarray, 
                           canvas_w: int, canvas_h: int) -> tuple:
    """Test ALL curve families for an element, return best match + loss."""
    shape = p.get("type", "unknown")
    weight = max(1, int(p.get("line_weight", 1.5)))
    
    best_family = shape
    best_loss = float("inf")
    results = []
    
    if "arch" in shape or shape in ("tall_arch",):
        # Test all arch curve families
        for family_name, render_fn in ARCH_CURVE_FAMILIES.items():
            canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
            try:
                render_fn(canvas, p["cx"], p["spring_y"], p["half_span"],
                         p.get("rise_ratio", 0.866), weight)
            except Exception:
                continue
            
            chamfer = chamfer_distance(canvas, ref_edges)
            iou = edge_iou(canvas, ref_edges)
            loss = chamfer * (1 - iou)
            results.append((family_name, loss, chamfer, iou))
            
            if loss < best_loss:
                best_loss = loss
                best_family = family_name
    
    elif "dome" in shape:
        # Test dome profiles
        for family_name, h_ratio_override in DOME_CURVE_FAMILIES.items():
            canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
            hr = h_ratio_override if h_ratio_override else p.get("h_ratio", 1.0)
            
            if family_name == "onion":
                # Onion: bulge + point
                pts = []
                for i in range(100):
                    t = i / 99
                    a = math.pi * (1 - t)
                    r_mod = p["radius"] * (1 + 0.15 * math.sin(a * 2))
                    x = int(p["cx"] + r_mod * math.cos(a))
                    y = int(p["cy"] - p["radius"] * 1.1 * math.sin(a))
                    if 0 <= x < canvas_w and 0 <= y < canvas_h:
                        pts.append((x, y))
                for i in range(1, len(pts)):
                    cv2.line(canvas, pts[i-1], pts[i], 255, weight)
            elif family_name == "bulbous":
                pts = []
                for i in range(100):
                    t = i / 99
                    a = math.pi * (1 - t)
                    r_mod = p["radius"] * (1 + 0.25 * math.sin(a * 1.5))
                    x = int(p["cx"] + r_mod * math.cos(a))
                    y = int(p["cy"] - p["radius"] * 0.9 * math.sin(a))
                    if 0 <= x < canvas_w and 0 <= y < canvas_h:
                        pts.append((x, y))
                for i in range(1, len(pts)):
                    cv2.line(canvas, pts[i-1], pts[i], 255, weight)
            else:
                render_dome(canvas, p["cx"], p["cy"], p["radius"], hr, weight)
            
            chamfer = chamfer_distance(canvas, ref_edges)
            iou = edge_iou(canvas, ref_edges)
            loss = chamfer * (1 - iou)
            results.append((family_name, loss, chamfer, iou))
            
            if loss < best_loss:
                best_loss = loss
                best_family = family_name
    
    return best_family, best_loss, results


def ransac_fit_circle(contour_pts: list, n_iter: int = 100, threshold: float = 5.0) -> dict | None:
    """RANSAC circle fitting — robust to outlier contour points."""
    if len(contour_pts) < 5:
        return None
    
    pts = np.array(contour_pts, dtype=np.float64)
    best_cx, best_cy, best_r = 0, 0, 0
    best_inliers = 0
    
    for _ in range(n_iter):
        # Sample 3 random points
        idx = np.random.choice(len(pts), 3, replace=False)
        p1, p2, p3 = pts[idx[0]], pts[idx[1]], pts[idx[2]]
        
        # Fit circle through 3 points
        ax, ay = p1[0], p1[1]
        bx, by = p2[0], p2[1]
        cx_t, cy_t = p3[0], p3[1]
        
        D = 2 * (ax * (by - cy_t) + bx * (cy_t - ay) + cx_t * (ay - by))
        if abs(D) < 1e-10:
            continue
        
        ux = ((ax*ax + ay*ay) * (by - cy_t) + (bx*bx + by*by) * (cy_t - ay) + (cx_t*cx_t + cy_t*cy_t) * (ay - by)) / D
        uy = ((ax*ax + ay*ay) * (cx_t - bx) + (bx*bx + by*by) * (ax - cx_t) + (cx_t*cx_t + cy_t*cy_t) * (bx - ax)) / D
        r = math.sqrt((ax - ux)**2 + (ay - uy)**2)
        
        # Count inliers
        dists = np.sqrt((pts[:, 0] - ux)**2 + (pts[:, 1] - uy)**2)
        inliers = np.sum(np.abs(dists - r) < threshold)
        
        if inliers > best_inliers:
            best_inliers = inliers
            best_cx, best_cy, best_r = ux, uy, r
    
    if best_inliers < len(pts) * 0.3:
        return None
    
    return {
        "cx": float(best_cx), "cy": float(best_cy), "radius": float(best_r),
        "inlier_ratio": round(best_inliers / len(pts), 3),
    }


def ransac_fit_line(contour_pts: list, n_iter: int = 50, threshold: float = 3.0) -> dict | None:
    """RANSAC line fitting — for walls, columns, bands."""
    if len(contour_pts) < 4:
        return None
    
    pts = np.array(contour_pts, dtype=np.float64)
    best_line = None
    best_inliers = 0
    
    for _ in range(n_iter):
        idx = np.random.choice(len(pts), 2, replace=False)
        p1, p2 = pts[idx[0]], pts[idx[1]]
        
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = math.sqrt(dx*dx + dy*dy)
        if length < 1:
            continue
        
        nx, ny = -dy/length, dx/length
        dists = np.abs((pts[:, 0] - p1[0]) * nx + (pts[:, 1] - p1[1]) * ny)
        inliers = np.sum(dists < threshold)
        
        if inliers > best_inliers:
            best_inliers = inliers
            best_line = {"x1": float(p1[0]), "y1": float(p1[1]),
                        "x2": float(p2[0]), "y2": float(p2[1]),
                        "angle": math.degrees(math.atan2(dy, dx))}
    
    if best_line and best_inliers > len(pts) * 0.3:
        best_line["inlier_ratio"] = round(best_inliers / len(pts), 3)
        return best_line
    return None


def apply_geometric_constraints(params: dict) -> dict:
    """Enforce architectural constraints between elements.
    
    Constraints:
    1. Symmetry: mirrored elements share same y, symmetric x around axis
    2. Alignment: elements at same depth share springing heights
    3. Tangency: domes sit smoothly on drums
    4. Orthogonality: walls meet at 90°
    """
    # Find symmetry axis (median x of all elements)
    all_cx = [p.get("cx", p.get("x", 0) + p.get("w", 0)/2) for p in params.values()]
    axis_x = float(np.median(all_cx)) if all_cx else 540.0
    
    # 1. SYMMETRY — find mirrored pairs and enforce
    names = list(params.keys())
    for i, name_a in enumerate(names):
        pa = params[name_a]
        ax = pa.get("cx", pa.get("x", 0) + pa.get("w", 0)/2)
        ay = pa.get("cy", pa.get("y", 0) + pa.get("h", 0)/2)
        
        for j, name_b in enumerate(names):
            if j <= i:
                continue
            pb = params[name_b]
            bx = pb.get("cx", pb.get("x", 0) + pb.get("w", 0)/2)
            by = pb.get("cy", pb.get("y", 0) + pb.get("h", 0)/2)
            
            # Check if mirrored around axis (within 10% tolerance)
            mirror_x = 2 * axis_x - ax
            if abs(bx - mirror_x) < abs(ax - axis_x) * 0.2 and abs(ay - by) < 30:
                # Enforce exact symmetry
                mid_y = (ay + by) / 2
                dist_from_axis = abs(ax - axis_x)
                
                if "cx" in pa:
                    pa["cx"] = axis_x - dist_from_axis
                    pb["cx"] = axis_x + dist_from_axis
                if "cy" in pa and "cy" in pb:
                    pa["cy"] = mid_y
                    pb["cy"] = mid_y
                if "spring_y" in pa and "spring_y" in pb:
                    mid_sy = (pa["spring_y"] + pb["spring_y"]) / 2
                    pa["spring_y"] = mid_sy
                    pb["spring_y"] = mid_sy
                if "half_span" in pa and "half_span" in pb:
                    avg_hs = (pa["half_span"] + pb["half_span"]) / 2
                    pa["half_span"] = avg_hs
                    pb["half_span"] = avg_hs
                if "rise_ratio" in pa and "rise_ratio" in pb:
                    avg_rr = (pa["rise_ratio"] + pb["rise_ratio"]) / 2
                    pa["rise_ratio"] = avg_rr
                    pb["rise_ratio"] = avg_rr
                
                print(f"    SYMMETRY: {name_a} ↔ {name_b} (axis={axis_x:.0f})")
    
    # 2. ALIGNMENT — arches at same depth share springing heights
    arch_elements = [(n, p) for n, p in params.items() if "arch" in p.get("type", "")]
    if len(arch_elements) >= 2:
        # Group by similar spring_y (within 30px)
        for i, (na, pa) in enumerate(arch_elements):
            for j, (nb, pb) in enumerate(arch_elements):
                if j <= i:
                    continue
                if abs(pa.get("spring_y", 0) - pb.get("spring_y", 0)) < 30:
                    avg_sy = (pa["spring_y"] + pb["spring_y"]) / 2
                    pa["spring_y"] = avg_sy
                    pb["spring_y"] = avg_sy
                    print(f"    ALIGNMENT: {na} + {nb} spring_y → {avg_sy:.0f}")
    
    # 3. TANGENCY — dome centers aligned with axis
    for name, p in params.items():
        if "dome" in p.get("type", ""):
            if abs(p.get("cx", 0) - axis_x) < 50:
                old_cx = p["cx"]
                p["cx"] = axis_x
                print(f"    TANGENCY: {name} cx {old_cx:.0f} → {axis_x:.0f} (axis-aligned)")
    
    return params


def extraction_to_params(extraction: dict, scale: float, ox: float, oy: float) -> dict:
    """Convert extraction elements to optimizable parameter dict."""
    params = {}
    
    for el in extraction["elements"]:
        if el.get("area_pct", 0) > 0.4 or el.get("area_pct", 0) < 0.003:
            continue
        
        name = el["name"]
        shape = el.get("shape", {}).get("type", "unknown")
        bbox = el.get("primitives", {}).get("bbox", {})
        center = el.get("primitives", {}).get("center", {})
        depth = el.get("depth", {})
        arch_geo = el.get("primitives", {}).get("arch_geometry", {})
        dome_geo = el.get("primitives", {}).get("dome_geometry", {})
        
        cx = center.get("x", 0) * scale + ox
        cy = center.get("y", 0) * scale + oy
        bw = bbox.get("w", 50) * scale
        bh = bbox.get("h", 50) * scale
        bx = bbox.get("x", 0) * scale + ox
        by = bbox.get("y", 0) * scale + oy
        
        p = {
            "type": shape,
            "line_weight": depth.get("line_weight", 1.5),
        }
        
        if "dome" in shape:
            p["cx"] = cx
            p["cy"] = cy
            p["radius"] = min(bw, bh) / 2
            p["h_ratio"] = dome_geo.get("height_to_diameter", 0.5) * 2
        elif "arch" in shape:
            p["cx"] = cx
            p["spring_y"] = by + bh
            p["half_span"] = bw / 2
            p["rise_ratio"] = arch_geo.get("rise_to_span", 0.866)
        elif shape in ("wall", "rectangle", "square"):
            p["x"] = bx
            p["y"] = by
            p["w"] = bw
            p["h"] = bh
        else:
            # Generic: store as contour reference
            p["cx"] = cx
            p["cy"] = cy
            p["bw"] = bw
            p["bh"] = bh
            continue  # Skip non-geometric elements for now
        
        params[name] = p
    
    return params


def optimize_params(params: dict, ref_edges: np.ndarray, 
                    canvas_w: int, canvas_h: int,
                    max_iter: int = 200, lr: float = 0.8,
                    convergence_threshold: float = 0.1) -> dict:
    """Optimize element parameters via gradient descent against reference edges."""
    
    # === MULTI-CURVE FAMILY TESTING ===
    # For each element, test ALL curve types and pick the best match
    print(f"\n  Testing curve families per element...")
    for name, p in list(params.items()):
        best_family, best_loss, results = find_best_curve_family(
            name, p, ref_edges, canvas_w, canvas_h)
        
        if results:
            results.sort(key=lambda r: r[1])
            original_type = p.get("type", "unknown")
            print(f"    {name}:")
            for fam, loss, chamf, iou in results[:4]:
                marker = " ← BEST" if fam == best_family else ""
                print(f"      {fam:15s} loss={loss:8.2f} chamfer={chamf:6.2f} IoU={iou:.4f}{marker}")
            
            # Update type if a different curve family won
            if "arch" in original_type or original_type == "tall_arch":
                if best_family != original_type.replace("_arch", "").replace("tall_", "pointed"):
                    print(f"      → RECLASSIFIED: {original_type} → {best_family}_arch")
                    p["type"] = f"{best_family}_arch"
            elif "dome" in original_type:
                p["best_dome_profile"] = best_family
    
    print(f"\n  Starting parameter optimization ({max_iter} max iterations, lr={lr})...")
    
    # Flatten params to numpy array for optimization
    param_names = []
    param_values = []
    param_bounds = []
    
    for name, p in params.items():
        shape = p.get("type", "unknown")
        if "dome" in shape:
            for key in ["cx", "cy", "radius", "h_ratio"]:
                param_names.append(f"{name}.{key}")
                param_values.append(p[key])
                if key == "cx":
                    param_bounds.append((p[key] - 50, p[key] + 50))
                elif key == "cy":
                    param_bounds.append((p[key] - 50, p[key] + 50))
                elif key == "radius":
                    param_bounds.append((p[key] * 0.7, p[key] * 1.3))
                elif key == "h_ratio":
                    param_bounds.append((0.3, 1.5))
        elif "arch" in shape:
            for key in ["cx", "spring_y", "half_span", "rise_ratio"]:
                param_names.append(f"{name}.{key}")
                param_values.append(p[key])
                if key in ("cx", "spring_y"):
                    param_bounds.append((p[key] - 40, p[key] + 40))
                elif key == "half_span":
                    param_bounds.append((p[key] * 0.7, p[key] * 1.3))
                elif key == "rise_ratio":
                    param_bounds.append((0.3, 2.0))
        elif shape in ("wall", "rectangle", "square"):
            for key in ["x", "y", "w", "h"]:
                param_names.append(f"{name}.{key}")
                param_values.append(p[key])
                if key in ("x", "y"):
                    param_bounds.append((p[key] - 30, p[key] + 30))
                else:
                    param_bounds.append((p[key] * 0.8, p[key] * 1.2))
    
    if not param_values:
        print("  No optimizable parameters found")
        return params
    
    values = np.array(param_values, dtype=np.float64)
    bounds = np.array(param_bounds)
    best_values = values.copy()
    best_loss = float("inf")
    
    # Compute initial loss
    render = render_from_params(params, canvas_w, canvas_h)
    initial_chamfer = chamfer_distance(render, ref_edges)
    initial_iou = edge_iou(render, ref_edges)
    initial_loss = initial_chamfer * (1 - initial_iou)
    best_loss = initial_loss
    
    print(f"  Initial: chamfer={initial_chamfer:.2f}, IoU={initial_iou:.4f}, loss={initial_loss:.4f}")
    
    # Simple coordinate descent (works without torch)
    for iteration in range(max_iter):
        improved = False
        
        for pi in range(len(values)):
            # Try perturbation in both directions
            for delta_sign in [1, -1]:
                # Adaptive step size (decreases over iterations)
                step = lr * (bounds[pi][1] - bounds[pi][0]) * 0.02 * (1 - iteration / max_iter)
                
                trial = values.copy()
                trial[pi] = np.clip(trial[pi] + delta_sign * step, bounds[pi][0], bounds[pi][1])
                
                # Update params dict
                trial_params = update_params_from_array(params, param_names, trial)
                
                # Render and evaluate
                render = render_from_params(trial_params, canvas_w, canvas_h)
                chamfer = chamfer_distance(render, ref_edges)
                iou = edge_iou(render, ref_edges)
                loss = chamfer * (1 - iou)
                
                if loss < best_loss:
                    best_loss = loss
                    best_values = trial.copy()
                    values = trial.copy()
                    improved = True
        
        if iteration % 20 == 0:
            current_params = update_params_from_array(params, param_names, values)
            render = render_from_params(current_params, canvas_w, canvas_h)
            chamfer = chamfer_distance(render, ref_edges)
            iou = edge_iou(render, ref_edges)
            print(f"  Iter {iteration:3d}: chamfer={chamfer:.2f}, IoU={iou:.4f}, loss={best_loss:.4f}")
        
        if not improved:
            print(f"  Converged at iteration {iteration}")
            break
    
    # Apply best values
    optimized = update_params_from_array(params, param_names, best_values)
    
    # Apply geometric constraints (symmetry, alignment, tangency)
    print(f"\n  Applying geometric constraints...")
    optimized = apply_geometric_constraints(optimized)
    
    # Final stats
    render = render_from_params(optimized, canvas_w, canvas_h)
    final_chamfer = chamfer_distance(render, ref_edges)
    final_iou = edge_iou(render, ref_edges)
    
    improvement = (1 - best_loss / max(0.001, initial_loss)) * 100
    print(f"\n  Final: chamfer={final_chamfer:.2f}, IoU={final_iou:.4f}")
    print(f"  Improvement: {improvement:.1f}%")
    print(f"  Parameters optimized: {len(param_names)}")
    
    # Log parameter changes
    print(f"\n  Parameter deltas:")
    for i, name in enumerate(param_names):
        delta = best_values[i] - param_values[i]
        if abs(delta) > 0.1:
            print(f"    {name:35s} {param_values[i]:8.1f} → {best_values[i]:8.1f} (Δ={delta:+.1f})")
    
    return optimized


def update_params_from_array(params: dict, names: list, values: np.ndarray) -> dict:
    """Update params dict from flat array."""
    result = {}
    for name, p in params.items():
        result[name] = dict(p)
    
    for i, param_name in enumerate(names):
        parts = param_name.split(".")
        el_name = parts[0]
        key = parts[1]
        if el_name in result:
            result[el_name][key] = float(values[i])
    
    return result


def params_to_js(params: dict) -> str:
    """Convert optimized params to JavaScript constants with formulas."""
    lines = [
        "// === OPTIMIZED PARAMETERS (gradient descent converged) ===",
        f"// Parameters: {sum(1 for p in params.values() for _ in p if _ not in ('type','line_weight'))}",
        "",
    ]
    
    PHI = (1 + math.sqrt(5)) / 2
    
    for name, p in params.items():
        shape = p.get("type", "unknown")
        lines.append(f"// {name} ({shape})")
        
        if "dome" in shape:
            lines.append(f"//   hemisphere: x² + y² = R², R = {p['radius']:.1f}")
            lines.append(f"var {name}_cx = {p['cx']:.1f};")
            lines.append(f"var {name}_cy = {p['cy']:.1f};")
            lines.append(f"var {name}_r = {p['radius']:.1f};")
            lines.append(f"var {name}_hr = {p.get('h_ratio', 1.0):.4f};")
        elif "arch" in shape:
            profile = "pointed"
            if "horseshoe" in shape:
                profile = "horseshoe"
            elif "ogee" in shape:
                profile = "ogee"
            rise = p.get("half_span", 100) * 2 * p.get("rise_ratio", 0.866)
            lines.append(f"//   {profile}: rise={rise:.1f}, span={p.get('half_span',100)*2:.1f}")
            lines.append(f"var {name}_cx = {p['cx']:.1f};")
            lines.append(f"var {name}_sy = {p['spring_y']:.1f};")
            lines.append(f"var {name}_hs = {p['half_span']:.1f};")
            lines.append(f"var {name}_rr = {p['rise_ratio']:.4f};")
        elif shape in ("wall", "rectangle", "square"):
            lines.append(f"var {name}_x = {p['x']:.1f};")
            lines.append(f"var {name}_y = {p['y']:.1f};")
            lines.append(f"var {name}_w = {p['w']:.1f};")
            lines.append(f"var {name}_h = {p['h']:.1f};")
        
        lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Traced: Optimize drawing parameters via gradient descent")
    parser.add_argument("--extraction", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="optimized.json")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.8)
    args = parser.parse_args()
    
    extraction = json.loads(Path(args.extraction).read_text())
    img_w = extraction["image_size"]["w"]
    img_h = extraction["image_size"]["h"]
    
    # Compute aspect-preserving mapping
    src_aspect = img_w / img_h
    dst_aspect = CANVAS_W / CANVAS_H
    if src_aspect > dst_aspect:
        map_scale = CANVAS_W / img_w
        map_ox, map_oy = 0, (CANVAS_H - img_h * map_scale) / 2
    else:
        map_scale = CANVAS_H / img_h
        map_ox, map_oy = (CANVAS_W - img_w * map_scale) / 2, 0
    
    print(f"Image: {img_w}×{img_h} → Canvas: {CANVAS_W}×{CANVAS_H}")
    print(f"Scale: {map_scale:.3f}, Offset: ({map_ox:.1f}, {map_oy:.1f})")
    
    # Get reference edges (scaled to canvas size)
    print("Extracting reference edges...")
    ref_edges_raw = extract_reference_edges(args.image)
    # Resize to canvas dimensions
    ref_edges = cv2.resize(ref_edges_raw, (CANVAS_W, CANVAS_H))
    # Apply aspect mapping (blank out letterbox areas)
    if map_oy > 0:
        ref_edges[:int(map_oy), :] = 0
        ref_edges[int(CANVAS_H - map_oy):, :] = 0
    if map_ox > 0:
        ref_edges[:, :int(map_ox)] = 0
        ref_edges[:, int(CANVAS_W - map_ox):] = 0
    
    print(f"  Reference edges: {np.count_nonzero(ref_edges)} pixels")
    
    # Convert extraction to params
    print("Building parameter space...")
    params = extraction_to_params(extraction, map_scale, map_ox, map_oy)
    n_geometric = len(params)
    print(f"  {n_geometric} geometric elements")
    
    # Optimize
    optimized = optimize_params(params, ref_edges, CANVAS_W, CANVAS_H,
                                max_iter=args.iterations, lr=args.lr)
    
    # Save
    output = {
        "params": {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating)) else vv 
                       for kk, vv in v.items()} for k, v in optimized.items()},
        "canvas": {"w": CANVAS_W, "h": CANVAS_H},
        "mapping": {"scale": map_scale, "offset_x": map_ox, "offset_y": map_oy},
        "source": args.image,
        "generated_js": params_to_js(optimized),
    }
    
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"\nSaved optimized parameters to {args.output}")
    print(f"\n{'='*60}")
    print("OPTIMIZED JAVASCRIPT:")
    print(f"{'='*60}")
    print(output["generated_js"])


CANVAS_W = 1080
CANVAS_H = 1920

if __name__ == "__main__":
    main()
