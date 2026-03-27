#!/usr/bin/env python3
"""
Aristotelian primitive detection.

Input: image
Output: lines and arcs

Method: find edge pixels, then RANSAC fit lines and circles
until all edge pixels are explained.

No SAM. No classification. No labels. Just geometry from points.
"""

import argparse
import json
import math
import numpy as np
import cv2
from pathlib import Path


def find_edge_pixels(image_path):
    """Truth 1: edges exist where intensity changes."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 50, 150)
    ey, ex = np.where(edges > 0)
    return np.column_stack([ex, ey]).astype(float), img.shape[:2]


def _thin_fallback(edges):
    """Skeleton thinning without ximgproc — morphological skeleton."""
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skeleton = np.zeros_like(edges)
    temp = edges.copy()
    while True:
        eroded = cv2.erode(temp, element)
        opened = cv2.dilate(eroded, element)
        diff = cv2.subtract(temp, opened)
        skeleton = cv2.bitwise_or(skeleton, diff)
        temp = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break
    return skeleton


def fit_line_ransac(points, n_iter=200, threshold=3.0):
    """Fit a line to points using RANSAC. Returns (p1, p2, inlier_mask) or None."""
    if len(points) < 2:
        return None
    
    best_inliers = None
    best_count = 0
    
    n = len(points)
    for _ in range(n_iter):
        # Sample 2 random points
        idx = np.random.choice(n, 2, replace=False)
        p1, p2 = points[idx[0]], points[idx[1]]
        
        d = p2 - p1
        seg_len = np.linalg.norm(d)
        if seg_len < 5:
            continue
        
        # Normal to the line
        normal = np.array([-d[1], d[0]]) / seg_len
        
        # Distance of all points to this line
        diffs = points - p1
        dists = np.abs(diffs @ normal)
        
        inlier_mask = dists < threshold
        count = np.sum(inlier_mask)
        
        if count > best_count:
            best_count = count
            best_inliers = inlier_mask
    
    if best_inliers is None or best_count < 10:
        return None
    
    # Refit line to all inliers
    inlier_pts = points[best_inliers]
    
    # Project inliers onto the line direction to find endpoints
    mean = inlier_pts.mean(axis=0)
    centered = inlier_pts - mean
    
    # PCA to find line direction
    cov = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    direction = eigenvectors[:, 1]  # largest eigenvalue
    
    # Project to find extent
    projections = centered @ direction
    t_min, t_max = projections.min(), projections.max()
    
    p1 = mean + direction * t_min
    p2 = mean + direction * t_max
    
    return p1, p2, best_inliers


def fit_circle_ransac(points, n_iter=300, threshold=3.0, min_radius=15, max_radius=500):
    """Fit a circle to points using RANSAC. Returns (center, radius, inlier_mask) or None."""
    if len(points) < 3:
        return None
    
    best_inliers = None
    best_count = 0
    best_center = None
    best_radius = 0
    
    n = len(points)
    for _ in range(n_iter):
        # Sample 3 random points
        idx = np.random.choice(n, 3, replace=False)
        p1, p2, p3 = points[idx[0]], points[idx[1]], points[idx[2]]
        
        # Fit circle through 3 points
        ax, ay = p1
        bx, by = p2
        cx, cy = p3
        
        D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(D) < 1e-10:
            continue
        
        ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / D
        uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / D
        
        center = np.array([ux, uy])
        radius = np.linalg.norm(p1 - center)
        
        if radius < min_radius or radius > max_radius:
            continue
        
        # Distance of all points to this circle
        dists_to_center = np.linalg.norm(points - center, axis=1)
        dists_to_circle = np.abs(dists_to_center - radius)
        
        inlier_mask = dists_to_circle < threshold
        count = np.sum(inlier_mask)
        
        if count > best_count:
            best_count = count
            best_inliers = inlier_mask
            best_center = center
            best_radius = radius
    
    if best_inliers is None or best_count < 15:
        return None
    
    # Refit circle to all inliers
    inlier_pts = points[best_inliers]
    center, radius = _algebraic_circle_fit(inlier_pts)
    if center is None:
        return None
    
    # Recompute inliers with refined circle
    dists = np.abs(np.linalg.norm(points - np.array(center), axis=1) - radius)
    best_inliers = dists < threshold
    
    # Compute arc sweep from inlier angles
    inlier_pts = points[best_inliers]
    angles = np.arctan2(inlier_pts[:, 1] - center[1], inlier_pts[:, 0] - center[0])
    angles_sorted = np.sort(angles)
    
    # Check if it's a full circle or partial arc
    gaps = np.diff(angles_sorted)
    gaps = np.append(gaps, angles_sorted[0] + 2*math.pi - angles_sorted[-1])
    max_gap = np.max(gaps)
    
    if max_gap < math.radians(30):
        # Full circle (no big gap)
        sweep = 2 * math.pi
        start = 0.0
    else:
        # Partial arc — find the gap and set start/sweep
        gap_idx = np.argmax(gaps)
        if gap_idx < len(angles_sorted) - 1:
            start = angles_sorted[gap_idx + 1]
        else:
            start = angles_sorted[0]
        sweep = 2 * math.pi - max_gap
    
    return center, radius, start, sweep, best_inliers


def _algebraic_circle_fit(points):
    """Fit circle using algebraic method."""
    if len(points) < 3:
        return None, 0
    pts = points.astype(float)
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2*x, 2*y, np.ones_like(x)])
    b = x**2 + y**2
    try:
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy = result[0], result[1]
        r = math.sqrt(max(0, result[2] + cx**2 + cy**2))
        return (cx, cy), r
    except:
        return None, 0


def _is_continuous(points, threshold=8.0, is_circular=False):
    """Check if points form a continuous chain (not scattered across the image).
    
    For lines: sort along dominant axis, check consecutive gaps.
    For arcs: use spatial nearest-neighbor connectivity.
    
    Phantom shapes have points from multiple disconnected regions.
    """
    if len(points) < 5:
        return True
    
    # Build a nearest-neighbor graph and check connectivity
    from scipy.spatial import KDTree
    tree = KDTree(points)
    
    # For each point, find its nearest neighbor distance
    dists, _ = tree.query(points, k=2)  # k=2 because first match is self
    nn_dists = dists[:, 1]
    
    median_nn = np.median(nn_dists)
    
    # What fraction of points have a close neighbor?
    close_count = np.sum(nn_dists < max(threshold, median_nn * 3))
    connectivity = close_count / len(points)
    
    # If less than 80% of points have a close neighbor, it's scattered
    if connectivity < 0.80:
        return False
    
    # Also check: is there a single connected component?
    # BFS from first point, using threshold as connectivity radius
    connect_radius = max(threshold, median_nn * 4)
    visited = np.zeros(len(points), dtype=bool)
    queue = [0]
    visited[0] = True
    while queue:
        idx = queue.pop(0)
        neighbors = tree.query_ball_point(points[idx], connect_radius)
        for ni in neighbors:
            if not visited[ni]:
                visited[ni] = True
                queue.append(ni)
    
    # If the largest connected component has less than 70% of points, it's fragmented
    if np.sum(visited) / len(points) < 0.70:
        return False
    
    return True


def detect_primitives(edge_points, image_shape, min_line_length=30):
    """
    The core algorithm:
    1. Try to fit a line (RANSAC)
    2. Try to fit a circle (RANSAC)
    3. Keep whichever explains more points
    4. Remove those points
    5. Repeat until no more fits
    """
    H, W = image_shape
    remaining = edge_points.copy()
    primitives = []
    
    max_iterations = 50  # safety limit
    
    for iteration in range(max_iterations):
        if len(remaining) < 10:
            break
        
        # Try line
        line_result = fit_line_ransac(remaining)
        line_count = np.sum(line_result[2]) if line_result else 0
        
        # Try circle
        circle_result = fit_circle_ransac(remaining, max_radius=max(W, H) * 0.4)
        circle_count = np.sum(circle_result[4]) if circle_result else 0
        
        # Parsimony: prefer lines over circles (fewer parameters = simpler truth)
        # A line wins if it explains at least 60% as many points as the circle,
        # because a large arc can always approximate a line but not vice versa.
        min_points = max(15, len(remaining) * 0.02)
        
        line_wins = False
        if line_result and line_count >= min_points:
            p1, p2, mask = line_result
            length = np.linalg.norm(p2 - p1)
            if length >= min_line_length:
                # Line wins if: more inliers than circle, OR
                # at least 60% of circle's inliers (parsimony preference)
                if circle_result is None or line_count >= circle_count * 0.6:
                    line_wins = True
        
        if line_wins:
            p1, p2, mask = line_result
            # Continuity check: are inliers a continuous chain?
            inlier_pts = remaining[mask]
            if not _is_continuous(inlier_pts, threshold=8.0):
                # Scattered points — not a real line, skip and remove anyway
                remaining = remaining[~mask]
                continue
            length = np.linalg.norm(p2 - p1)
            primitives.append({
                'type': 'line',
                'params': {
                    'x1': float(p1[0]), 'y1': float(p1[1]),
                    'x2': float(p2[0]), 'y2': float(p2[1])
                },
                'n_inliers': int(line_count),
                'name': f'line_{len(primitives)}'
            })
            remaining = remaining[~mask]
            
        elif circle_count >= min_points:
            center, radius, start, sweep, mask = circle_result
            # Continuity check
            inlier_pts = remaining[mask]
            if not _is_continuous(inlier_pts, threshold=8.0):
                remaining = remaining[~mask]
                continue
            primitives.append({
                'type': 'arc',
                'params': {
                    'cx': float(center[0]), 'cy': float(center[1]),
                    'radius': float(radius),
                    'start_angle': float(start),
                    'sweep': float(sweep)
                },
                'n_inliers': int(circle_count),
                'name': f'arc_{len(primitives)}'
            })
            remaining = remaining[~mask]
            
        else:
            # Neither fit explains enough points — done
            break
    
    # Dedup: merge primitives that are stroke-width duplicates
    primitives = _dedup_stroke_width(primitives, stroke_width=12)
    
    return primitives, remaining


def _dedup_stroke_width(primitives, stroke_width=12):
    """Merge duplicate primitives caused by inner/outer edges of thick strokes.
    
    Two lines are duplicates if their midpoints are within stroke_width and 
    they have similar angles. Two arcs are duplicates if their centers are 
    within stroke_width and radii differ by less than stroke_width.
    """
    kept = []
    for p in primitives:
        is_dup = False
        params = p['params']
        
        for k in kept:
            kp = k['params']
            
            if p['type'] == 'line' and k['type'] == 'line':
                # Compare midpoints and angles
                mx1 = (params['x1'] + params['x2']) / 2
                my1 = (params['y1'] + params['y2']) / 2
                mx2 = (kp['x1'] + kp['x2']) / 2
                my2 = (kp['y1'] + kp['y2']) / 2
                mid_dist = math.sqrt((mx1-mx2)**2 + (my1-my2)**2)
                
                a1 = math.atan2(params['y2']-params['y1'], params['x2']-params['x1'])
                a2 = math.atan2(kp['y2']-kp['y1'], kp['x2']-kp['x1'])
                angle_diff = abs(a1 - a2) % math.pi
                if angle_diff > math.pi/2:
                    angle_diff = math.pi - angle_diff
                
                if mid_dist < stroke_width and angle_diff < 0.15:
                    is_dup = True
                    # Keep the one with more inliers
                    if p.get('n_inliers', 0) > k.get('n_inliers', 0):
                        kept.remove(k)
                        kept.append(p)
                    break
                    
            elif p['type'] == 'arc' and k['type'] == 'arc':
                center_dist = math.sqrt((params['cx']-kp['cx'])**2 + (params['cy']-kp['cy'])**2)
                r_diff = abs(params['radius'] - kp['radius'])
                
                if center_dist < stroke_width and r_diff < stroke_width:
                    is_dup = True
                    if p.get('n_inliers', 0) > k.get('n_inliers', 0):
                        kept.remove(k)
                        kept.append(p)
                    break
        
        if not is_dup:
            kept.append(p)
    
    if len(kept) < len(primitives):
        print(f"  Dedup: {len(primitives)} → {len(kept)} ({len(primitives)-len(kept)} stroke-width duplicates)")
    
    return kept


def main():
    parser = argparse.ArgumentParser(description="Aristotelian primitive detection")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="detected.json")
    args = parser.parse_args()
    
    print(f"Loading {args.image}...")
    edge_points, image_shape = find_edge_pixels(args.image)
    H, W = image_shape
    print(f"Image: {W}x{H}, {len(edge_points)} edge pixels")
    
    print("\nDetecting primitives (RANSAC)...")
    primitives, remaining = detect_primitives(edge_points, image_shape)
    
    n_lines = sum(1 for p in primitives if p['type'] == 'line')
    n_arcs = sum(1 for p in primitives if p['type'] == 'arc')
    print(f"\nFound {len(primitives)} primitives: {n_lines} lines + {n_arcs} arcs")
    print(f"Unexplained edge pixels: {len(remaining)} ({100*len(remaining)/len(edge_points):.1f}%)")
    
    for p in primitives:
        params = p['params']
        if p['type'] == 'line':
            length = math.sqrt((params['x2']-params['x1'])**2 + (params['y2']-params['y1'])**2)
            print(f"  LINE ({params['x1']:.0f},{params['y1']:.0f}) → ({params['x2']:.0f},{params['y2']:.0f}) len={length:.0f} pts={p['n_inliers']}")
        else:
            sweep_deg = abs(math.degrees(params['sweep']))
            label = 'circle' if sweep_deg > 350 else f'arc {sweep_deg:.0f}°'
            print(f"  ARC  ({params['cx']:.0f},{params['cy']:.0f}) r={params['radius']:.0f} {label} pts={p['n_inliers']}")
    
    output = {
        'canvas': {'w': W, 'h': H},
        'primitives': primitives,
        'source_primitives': primitives,  # for overlay
        'edge_count': len(edge_points),
        'unexplained': len(remaining),
    }
    
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
