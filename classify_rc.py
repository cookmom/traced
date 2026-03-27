#!/usr/bin/env python3
"""
Ruler-and-Compass shape classifier.

Two primitives only:
  - LINE (ruler): straight segments
  - ARC (compass): circular arcs

Everything is classified by decomposing the contour into these two primitives,
then naming the composition.
"""

import cv2
import numpy as np
import math


def fit_circle(points):
    """Fit a circle to 2D points using algebraic fit. Returns (center, radius) or (None, 0)."""
    if len(points) < 3:
        return None, 0
    pts = points.astype(float)
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2*x, 2*y, np.ones_like(x)])
    b_vec = x**2 + y**2
    try:
        result, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
        cx, cy = result[0], result[1]
        r = math.sqrt(max(0, result[2] + cx**2 + cy**2))
        return (cx, cy), r
    except:
        return None, 0


def decompose_contour(contour, min_arc_radius=5):
    """Decompose a contour into LINE and ARC primitives.
    
    Strategy: simplify the contour to vertices, then classify each EDGE
    between consecutive vertices as either a line or part of an arc.
    Groups of consecutive arc-edges that share a common center become one arc.
    """
    if len(contour) < 3:
        return []
    
    perimeter = cv2.arcLength(contour, True)
    
    # Two levels of simplification:
    # Coarse (2%) for vertex detection (corners, inflection points)
    # Fine (0.5%) for arc fitting
    coarse_eps = 0.02 * perimeter
    coarse = cv2.approxPolyDP(contour, coarse_eps, True).reshape(-1, 2)
    
    fine_eps = 0.005 * perimeter
    fine = cv2.approxPolyDP(contour, fine_eps, True).reshape(-1, 2)
    
    n_coarse = len(coarse)
    primitives = []
    
    if n_coarse < 3:
        if n_coarse == 2:
            primitives.append({
                'type': 'line', 'p1': tuple(int(v) for v in coarse[0]),
                'p2': tuple(int(v) for v in coarse[1]),
                'length': float(np.linalg.norm(coarse[1] - coarse[0]))
            })
        return primitives
    
    # For each edge in the coarse polygon, check if the ACTUAL contour
    # between those vertices is straight or curved
    edges = []
    for i in range(n_coarse):
        p1 = coarse[i]
        p2 = coarse[(i + 1) % n_coarse]
        seg_len = np.linalg.norm(p2 - p1)
        
        # Find fine points between p1 and p2
        # (points closest to the line segment p1→p2)
        fine_between = []
        for fp in fine:
            if seg_len > 0:
                t = np.dot(fp - p1, p2 - p1) / (seg_len * seg_len)
                if 0.05 < t < 0.95:
                    proj = p1 + t * (p2 - p1)
                    perp_dist = np.linalg.norm(fp - proj)
                    # Only count points actually near this edge segment
                    if perp_dist < seg_len * 0.5:
                        fine_between.append((fp, perp_dist))
        
        # Is this edge straight? Check max deviation of fine points
        max_deviation = max((d for _, d in fine_between), default=0)
        
        if max_deviation < 5 or seg_len < 15:
            # Straight line
            edges.append({
                'type': 'line', 'p1': p1, 'p2': p2,
                'length': float(seg_len), 'max_dev': max_deviation
            })
        else:
            # Curved — fit arc through p1, midpoints, p2
            arc_pts = np.array([p1] + [fp for fp, _ in fine_between] + [p2])
            center, radius = fit_circle(arc_pts)
            if center is not None and radius > min_arc_radius:
                edges.append({
                    'type': 'arc_edge', 'p1': p1, 'p2': p2,
                    'center': center, 'radius': radius,
                    'arc_pts': arc_pts, 'max_dev': max_deviation
                })
            else:
                edges.append({
                    'type': 'line', 'p1': p1, 'p2': p2,
                    'length': float(seg_len), 'max_dev': max_deviation
                })
    
    # Now merge consecutive arc_edges with similar center/radius into single arcs
    i = 0
    while i < len(edges):
        e = edges[i]
        if e['type'] == 'line':
            primitives.append({
                'type': 'line',
                'p1': tuple(int(v) for v in e['p1']),
                'p2': tuple(int(v) for v in e['p2']),
                'length': e['length']
            })
            i += 1
        elif e['type'] == 'arc_edge':
            # Collect consecutive arc edges with similar center
            group = [e]
            j = i + 1
            while j < len(edges) and edges[j]['type'] == 'arc_edge':
                dist = np.linalg.norm(np.array(edges[j]['center']) - np.array(e['center']))
                r_diff = abs(edges[j]['radius'] - e['radius'])
                if dist < e['radius'] * 0.3 and r_diff < e['radius'] * 0.3:
                    group.append(edges[j])
                    j += 1
                else:
                    break
            
            # Fit one arc to all points in the group
            all_pts = np.vstack([g['arc_pts'] for g in group])
            center, radius = fit_circle(all_pts)
            if center is not None:
                angles = [math.atan2(p[1] - center[1], p[0] - center[0]) for p in all_pts]
                # Sort angles and compute sweep
                start = angles[0]
                end = angles[-1]
                sweep = end - start
                # For closed curves, compute sweep correctly
                if len(group) > n_coarse * 0.5:
                    # Probably a full circle
                    sweep = 2 * math.pi * (len(group) / n_coarse)
                else:
                    while sweep > math.pi: sweep -= 2 * math.pi
                    while sweep < -math.pi: sweep += 2 * math.pi
                
                primitives.append({
                    'type': 'arc',
                    'center': (float(center[0]), float(center[1])),
                    'radius': float(radius),
                    'start_angle': float(start),
                    'sweep': float(sweep),
                    'n_points': len(all_pts)
                })
            i = j
        else:
            i += 1
    
    # Special case: if ALL edges are arcs with same center → circle
    if not primitives:
        # All edges were arc_edges — try fitting entire contour as one arc
        center, radius = fit_circle(fine)
        if center is not None and radius > min_arc_radius:
            dists = np.array([np.linalg.norm(p - np.array(center)) for p in fine])
            if np.std(dists) / radius < 0.1:  # good circle fit
                primitives.append({
                    'type': 'arc',
                    'center': (float(center[0]), float(center[1])),
                    'radius': float(radius),
                    'start_angle': 0,
                    'sweep': 2 * math.pi,
                    'n_points': len(fine)
                })
    
    return primitives


def classify_from_primitives(primitives, contour, mask_shape):
    """Classify shape based on its ruler-and-compass decomposition."""
    lines = [p for p in primitives if p['type'] == 'line']
    arcs = [p for p in primitives if p['type'] == 'arc']
    
    total_line_length = sum(l['length'] for l in lines)
    total_arc_sweep = sum(abs(a['sweep']) for a in arcs)
    
    area = cv2.contourArea(contour) if len(contour) >= 3 else 0
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    aspect = w / h if h > 0 else 1
    circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    solidity = area / (w * h) if w * h > 0 else 0
    
    result = {
        'primitives': {'lines': len(lines), 'arcs': len(arcs)},
        'total_line_length': total_line_length,
        'total_arc_sweep_deg': math.degrees(total_arc_sweep),
        'circularity': circularity,
        'solidity': solidity,
        'aspect': aspect,
        'bbox': (x, y, w, h),
    }
    
    # CIRCLE: one arc sweeping ~360°, OR high circularity + near-square aspect
    if len(arcs) >= 1:
        largest_arc = max(arcs, key=lambda a: abs(a['sweep']))
        if abs(largest_arc['sweep']) > math.radians(300) and total_line_length < perimeter * 0.15:
            result['type'] = 'circle'
            result['center'] = largest_arc['center']
            result['radius'] = largest_arc['radius']
            result['confidence'] = min(0.95, abs(largest_arc['sweep']) / (2 * math.pi))
            return result
    
    # CIRCLE fallback: moderate+ circularity, near-square, dominated by arcs (filled circle masks)
    if circularity > 0.55 and 0.80 < aspect < 1.20 and len(lines) == 0:
        # Fit circle to the full contour
        center, radius = fit_circle(contour.reshape(-1, 2))
        if center is not None and radius > 5:
            result['type'] = 'circle'
            result['center'] = (float(center[0]), float(center[1]))
            result['radius'] = float(radius)
            result['confidence'] = circularity
            return result
    
    # ELLIPSE
    if circularity > 0.80 and total_line_length < perimeter * 0.2:
        if aspect < 0.75 or aspect > 1.25:
            result['type'] = 'ellipse'
            result['confidence'] = circularity * 0.9
            return result
    
    # SQUARE: ~4 lines of equal length, near-square aspect
    if 3 <= len(lines) <= 5 and len(arcs) <= 1:
        if solidity > 0.80 and 0.80 < aspect < 1.20:
            result['type'] = 'square'
            result['confidence'] = 0.75
            return result
    
    # RECTANGLE
    if 3 <= len(lines) <= 6 and len(arcs) <= 1 and solidity > 0.75:
        result['type'] = 'rectangle'
        result['confidence'] = 0.7
        return result
    
    # TRIANGLE: 2-3 lines, 0-1 arcs, low solidity (~0.5 for equilateral)
    if 2 <= len(lines) <= 3 and len(arcs) <= 1 and 0.4 < solidity < 0.7:
        result['type'] = 'triangle'
        result['confidence'] = 0.7
        return result
    
    # POLYGON
    if len(lines) >= 5 and len(arcs) <= 1 and solidity > 0.70:
        result['type'] = 'polygon'
        result['vertices'] = len(lines)
        result['confidence'] = 0.65
        return result
    
    # MIXED: arcs + lines — just report what was found
    if len(arcs) >= 1 and len(lines) >= 1:
        result['type'] = 'mixed'
        result['confidence'] = 0.5
        return result
    
    # ARC-ONLY
    if len(arcs) >= 1:
        result['type'] = 'arc'
        result['confidence'] = 0.5
        return result
    
    # LINE-ONLY (didn't match square/rect/triangle above)
    if len(lines) >= 1:
        result['type'] = 'lines'
        result['confidence'] = 0.5
        return result
    
    # NOTHING DETECTED
    result['type'] = 'unknown'
    result['confidence'] = 0.3
    return result


def classify_shape_rc(mask, knowledge=None):
    """Main entry: ruler-and-compass shape classification."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"type": "unknown", "circularity": 0, "solidity": 0, "convexity": 0, "aspect": 0, "confidence": 0, "vertices": 0}
    
    largest = max(contours, key=cv2.contourArea)
    primitives = decompose_contour(largest)
    result = classify_from_primitives(primitives, largest, mask.shape)
    
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)
    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)
    x, y, w, h = cv2.boundingRect(largest)
    
    result.setdefault('circularity', 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0)
    result.setdefault('solidity', area / (w * h) if w * h > 0 else 0)
    result.setdefault('convexity', area / hull_area if hull_area > 0 else 0)
    result.setdefault('aspect', w / h if h > 0 else 1)
    result.setdefault('confidence', 0.5)
    
    eps = 0.02 * perimeter
    approx = cv2.approxPolyDP(largest, eps, True)
    result['vertices'] = result.get('vertices', len(approx))
    
    return result


if __name__ == '__main__':
    import sys
    from PIL import Image
    
    img_path = sys.argv[1] if len(sys.argv) > 1 else 'cal2-pre.jpg'
    img = np.array(Image.open(img_path).convert('L'))
    
    _, thresh = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    print(f"Found {len(contours)} contours\n")
    
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 500:
            continue
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], 0, 255, -1)
        
        result = classify_shape_rc(mask)
        x, y, w, h = cv2.boundingRect(cnt)
        prims = result.get('primitives', {})
        
        print(f"Shape {i}: {result['type']} (conf={result['confidence']:.2f})")
        print(f"  bbox: ({x},{y}) {w}x{h}")
        print(f"  primitives: {prims.get('lines',0)} lines, {prims.get('arcs',0)} arcs")
        print(f"  circularity={result['circularity']:.3f} solidity={result['solidity']:.3f} aspect={result['aspect']:.3f}")
        if 'center' in result:
            print(f"  center={result['center']}, radius={result.get('radius',0):.1f}")
        print()
