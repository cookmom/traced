#!/usr/bin/env python3
"""
Ruler-and-Compass Optimizer

First principles: only two primitives exist.
  - LINE: (x1, y1, x2, y2) — a ruler stroke
  - ARC: (cx, cy, radius, start_angle, sweep) — a compass stroke

Everything is optimized by minimizing chamfer distance between
rendered primitives and detected edge pixels. No architectural
assumptions, no dome profiles, no arch families.

Shape labels (circle, square, arch) are emergent from the primitive
composition, never inputs to the optimizer.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import cv2
from PIL import Image


def extract_edges(image_path: str):
    """Extract edge pixels from image using Canny."""
    img = np.array(Image.open(image_path).convert('L'))
    edges = cv2.Canny(img, 50, 150)
    ey, ex = np.where(edges > 0)
    return np.column_stack([ex, ey]).astype(float), img.shape[:2]


def render_line(x1, y1, x2, y2, n_samples=50):
    """Render a line as sample points."""
    t = np.linspace(0, 1, n_samples)
    x = x1 + (x2 - x1) * t
    y = y1 + (y2 - y1) * t
    return np.column_stack([x, y])


def render_arc(cx, cy, radius, start_angle, sweep, n_samples=60):
    """Render an arc as sample points."""
    n = max(10, int(abs(sweep) / (2 * math.pi) * n_samples))
    t = np.linspace(0, 1, n)
    angles = start_angle + sweep * t
    x = cx + radius * np.cos(angles)
    y = cy + radius * np.sin(angles)
    return np.column_stack([x, y])


def chamfer_distance(pts_a, pts_b):
    """One-directional chamfer: for each point in A, distance to nearest in B."""
    if len(pts_a) == 0 or len(pts_b) == 0:
        return 1e6
    # Vectorized nearest-neighbor
    diffs = pts_a[:, np.newaxis, :] - pts_b[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    return np.mean(np.min(dists, axis=1))


def chamfer_distance_fast(pts_a, pts_b, edge_tree=None):
    """Fast chamfer using KDTree."""
    from scipy.spatial import KDTree
    if edge_tree is None:
        edge_tree = KDTree(pts_b)
    dists, _ = edge_tree.query(pts_a)
    return np.mean(dists)


def primitives_from_extraction(extraction: dict, image_shape: tuple):
    """Convert SAM2 extraction elements into raw line/arc primitives
    using the R&C classifier results."""
    from classify_rc import classify_shape_rc, decompose_contour, fit_circle
    
    H, W = image_shape
    primitives = []
    
    for el in extraction.get('elements', []):
        area_pct = el.get('area_pct', 0)
        # Skip full-image and tiny elements
        if area_pct > 0.4 or area_pct < 0.001:
            continue
        
        # Shape info comes from R&C classifier — it's a dict with 'type', 'bbox', etc.
        shape_info = el.get('shape', {})
        if isinstance(shape_info, str):
            shape = shape_info
            bb = None
        else:
            shape = shape_info.get('type', 'unknown')
            bb = shape_info.get('bbox', None)
        
        name = el.get('name', 'unknown')
        
        # Get bounding box — prefer shape bbox (from contour), fall back to element bbox
        if bb and isinstance(bb, (list, tuple)) and len(bb) == 4:
            bx, by, bw, bh = bb
        else:
            # Estimate from position
            pos = el.get('position', {})
            cx_norm = pos.get('cx', 0.5)
            cy_norm = pos.get('cy', 0.5)
            size = el.get('size', 0.01)
            side = math.sqrt(size) * max(W, H)
            bx = cx_norm * W - side/2
            by = cy_norm * H - side/2
            bw = bh = side
        
        cx = bx + bw/2
        cy = by + bh/2
        
        # Also grab circle center/radius if R&C classifier found it
        rc_center = shape_info.get('center', None) if isinstance(shape_info, dict) else None
        rc_radius = shape_info.get('radius', None) if isinstance(shape_info, dict) else None
        rc_arc = shape_info.get('arc', None) if isinstance(shape_info, dict) else None
        
        # Determine primitive type from shape classification
        if shape in ('circle', 'ellipse'):
            if rc_center and rc_radius:
                c_cx, c_cy = rc_center
                r = rc_radius
            else:
                c_cx, c_cy = cx, cy
                r = max(bw, bh) / 2
            primitives.append({
                'name': name,
                'type': 'arc',
                'params': {
                    'cx': float(c_cx), 'cy': float(c_cy),
                    'radius': float(r),
                    'start_angle': 0.0,
                    'sweep': 2 * math.pi
                },
                'source_shape': shape
            })
        
        elif shape in ('square', 'rectangle'):
            # 4 lines
            x0, y0 = bx, by
            x1, y1 = bx + bw, by + bh
            for i, (lx1, ly1, lx2, ly2) in enumerate([
                (x0, y0, x1, y0),  # top
                (x1, y0, x1, y1),  # right
                (x1, y1, x0, y1),  # bottom
                (x0, y1, x0, y0),  # left
            ]):
                primitives.append({
                    'name': f'{name}_line_{i}',
                    'type': 'line',
                    'params': {
                        'x1': float(lx1), 'y1': float(ly1),
                        'x2': float(lx2), 'y2': float(ly2)
                    },
                    'source_shape': shape,
                    'group': name
                })
        
        elif 'arch' in shape:
            # Use R&C arc data if available
            if rc_arc:
                a_cx = rc_arc['center'][0]
                a_cy = rc_arc['center'][1]
                a_r = rc_arc['radius']
                a_start = rc_arc['start_angle']
                a_sweep = rc_arc['sweep']
            else:
                hs = bw / 2
                a_cx = cx
                a_cy = by + bh * 0.7
                a_r = hs
                a_start = math.pi
                a_sweep = -math.pi
            
            primitives.append({
                'name': f'{name}_arc',
                'type': 'arc',
                'params': {
                    'cx': float(a_cx), 'cy': float(a_cy),
                    'radius': float(a_r),
                    'start_angle': float(a_start),
                    'sweep': float(a_sweep)
                },
                'source_shape': shape,
                'group': name
            })
            # Left leg — use arc radius as half-span
            leg_hs = a_r
            leg_sy = a_cy  # springing point = arc center y
            primitives.append({
                'name': f'{name}_leg_l',
                'type': 'line',
                'params': {
                    'x1': float(a_cx - leg_hs), 'y1': float(leg_sy),
                    'x2': float(a_cx - leg_hs), 'y2': float(by + bh)
                },
                'source_shape': shape,
                'group': name
            })
            # Right leg
            primitives.append({
                'name': f'{name}_leg_r',
                'type': 'line',
                'params': {
                    'x1': float(a_cx + leg_hs), 'y1': float(leg_sy),
                    'x2': float(a_cx + leg_hs), 'y2': float(by + bh)
                },
                'source_shape': shape,
                'group': name
            })
        
        elif shape == 'dome':
            # Arc (top half)
            r = max(bw, bh) / 2
            primitives.append({
                'name': name,
                'type': 'arc',
                'params': {
                    'cx': float(cx), 'cy': float(cy + r * 0.3),
                    'radius': float(r),
                    'start_angle': math.pi,
                    'sweep': -math.pi  # top semicircle
                },
                'source_shape': shape
            })
        
        elif shape == 'triangle':
            # 3 lines
            x0, y0 = cx, by  # top
            x1, y1 = bx, by + bh  # bottom-left
            x2, y2 = bx + bw, by + bh  # bottom-right
            for i, (lx1, ly1, lx2, ly2) in enumerate([
                (x0, y0, x1, y1), (x1, y1, x2, y2), (x2, y2, x0, y0)
            ]):
                primitives.append({
                    'name': f'{name}_line_{i}',
                    'type': 'line',
                    'params': {
                        'x1': float(lx1), 'y1': float(ly1),
                        'x2': float(lx2), 'y2': float(ly2)
                    },
                    'source_shape': shape,
                    'group': name
                })
        
        else:
            # Unknown — try as a generic arc or skip
            if bw > 20 and bh > 20:
                r = max(bw, bh) / 2
                primitives.append({
                    'name': name,
                    'type': 'arc',
                    'params': {
                        'cx': float(cx), 'cy': float(cy),
                        'radius': float(r),
                        'start_angle': 0.0,
                        'sweep': 2 * math.pi
                    },
                    'source_shape': shape
                })
    
    return primitives


def render_primitive(p):
    """Render a primitive to sample points."""
    params = p['params']
    if p['type'] == 'line':
        return render_line(params['x1'], params['y1'], params['x2'], params['y2'])
    elif p['type'] == 'arc':
        return render_arc(params['cx'], params['cy'], params['radius'],
                         params['start_angle'], params['sweep'])
    return np.array([])


def optimize_primitives(primitives, edge_pts, image_shape, max_iter=100, lr=1.0):
    """Optimize all primitive parameters to minimize chamfer distance to edges."""
    from scipy.spatial import KDTree
    
    if len(edge_pts) == 0:
        print("  No edge pixels found!")
        return primitives
    
    edge_tree = KDTree(edge_pts)
    H, W = image_shape
    
    # Initial chamfer
    all_pts = np.vstack([render_primitive(p) for p in primitives if len(render_primitive(p)) > 0])
    initial_chamfer = chamfer_distance_fast(all_pts, edge_pts, edge_tree)
    print(f"  Initial chamfer: {initial_chamfer:.2f}")
    
    # Optimize each primitive independently (simple but effective)
    for pi, prim in enumerate(primitives):
        params = prim['params']
        param_keys = list(params.keys())
        
        best_chamfer = 1e9
        best_params = dict(params)
        
        for iteration in range(max_iter):
            pts = render_primitive(prim)
            if len(pts) == 0:
                break
            
            current_chamfer = chamfer_distance_fast(pts, edge_pts, edge_tree)
            
            if current_chamfer < best_chamfer:
                best_chamfer = current_chamfer
                best_params = dict(params)
            
            # Numerical gradient
            grad = {}
            for key in param_keys:
                old_val = params[key]
                # Adaptive step size
                if 'angle' in key or 'sweep' in key:
                    delta = 0.02  # radians
                elif 'radius' in key:
                    delta = max(1.0, abs(old_val) * 0.01)
                else:
                    delta = max(1.0, abs(old_val) * 0.005)
                
                params[key] = old_val + delta
                pts_plus = render_primitive(prim)
                c_plus = chamfer_distance_fast(pts_plus, edge_pts, edge_tree) if len(pts_plus) > 0 else 1e6
                
                params[key] = old_val - delta
                pts_minus = render_primitive(prim)
                c_minus = chamfer_distance_fast(pts_minus, edge_pts, edge_tree) if len(pts_minus) > 0 else 1e6
                
                params[key] = old_val
                grad[key] = (c_plus - c_minus) / (2 * delta)
            
            # Gradient step
            moved = False
            for key in param_keys:
                step = -lr * grad[key]
                # Clamp step
                if 'angle' in key or 'sweep' in key:
                    step = max(-0.1, min(0.1, step))
                elif 'radius' in key:
                    step = max(-5.0, min(5.0, step))
                else:
                    step = max(-3.0, min(3.0, step))
                
                if abs(step) > 0.001:
                    params[key] += step
                    moved = True
            
            if not moved:
                break
            
            # Decay learning rate
            if iteration > 0 and iteration % 20 == 0:
                lr *= 0.8
        
        # Restore best
        prim['params'] = best_params
        pts = render_primitive(prim)
        final = chamfer_distance_fast(pts, edge_pts, edge_tree) if len(pts) > 0 else 1e6
        
        if pi < 20:  # don't spam for tons of primitives
            print(f"    {prim['name']}: {initial_chamfer:.1f} → {final:.1f}")
    
    # Remove primitives with poor fit (chamfer > 50px = not matching any edge)
    good = []
    for prim in primitives:
        pts = render_primitive(prim)
        if len(pts) > 0:
            c = chamfer_distance_fast(pts, edge_pts, edge_tree)
            if c < 50:
                good.append(prim)
            else:
                print(f"    DROPPED {prim['name']}: chamfer={c:.1f} (too far from edges)")
    
    if len(good) < len(primitives):
        print(f"  Filtered: {len(primitives)} → {len(good)} primitives")
    primitives = good
    
    # Final total chamfer
    if primitives:
        all_pts = np.vstack([render_primitive(p) for p in primitives if len(render_primitive(p)) > 0])
        final_chamfer = chamfer_distance_fast(all_pts, edge_pts, edge_tree)
        print(f"  Final chamfer: {final_chamfer:.2f} (improvement: {(1-final_chamfer/initial_chamfer)*100:.1f}%)")
    
    return primitives


def dedup_primitives(primitives, line_thresh=15, arc_thresh=20):
    """Remove duplicate primitives that are too close to each other.
    
    Lines within line_thresh px (midpoint distance) are duplicates.
    Arcs within arc_thresh px (center distance) AND similar radius are duplicates.
    Keep the first (largest-area source element) of each group.
    """
    kept = []
    
    for p in primitives:
        is_dup = False
        params = p['params']
        
        for k in kept:
            kp = k['params']
            
            if p['type'] == 'line' and k['type'] == 'line':
                # Compare midpoints
                mx1 = (params['x1'] + params['x2']) / 2
                my1 = (params['y1'] + params['y2']) / 2
                mx2 = (kp['x1'] + kp['x2']) / 2
                my2 = (kp['y1'] + kp['y2']) / 2
                dist = math.sqrt((mx1-mx2)**2 + (my1-my2)**2)
                if dist < line_thresh:
                    # Also check similar angle
                    a1 = math.atan2(params['y2']-params['y1'], params['x2']-params['x1'])
                    a2 = math.atan2(kp['y2']-kp['y1'], kp['x2']-kp['x1'])
                    angle_diff = abs(a1 - a2) % math.pi
                    if angle_diff < 0.2 or angle_diff > math.pi - 0.2:
                        is_dup = True
                        break
            
            elif p['type'] == 'arc' and k['type'] == 'arc':
                cx_dist = abs(params['cx'] - kp['cx'])
                cy_dist = abs(params['cy'] - kp['cy'])
                r_diff = abs(params['radius'] - kp['radius'])
                center_dist = math.sqrt(cx_dist**2 + cy_dist**2)
                if center_dist < arc_thresh and r_diff < params['radius'] * 0.2:
                    is_dup = True
                    break
        
        if not is_dup:
            kept.append(p)
    
    skipped = len(primitives) - len(kept)
    if skipped > 0:
        print(f"  Dedup: {len(primitives)} → {len(kept)} primitives ({skipped} duplicates removed)")
    
    return kept


def main():
    parser = argparse.ArgumentParser(description="Ruler-and-compass optimizer")
    parser.add_argument("--extraction", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="optimized_rc.json")
    args = parser.parse_args()
    
    extraction = json.loads(Path(args.extraction).read_text())
    edge_pts, image_shape = extract_edges(args.image)
    H, W = image_shape
    print(f"Image: {W}x{H}, {len(edge_pts)} edge pixels")
    
    # Convert extraction to primitives
    primitives = primitives_from_extraction(extraction, image_shape)
    print(f"Primitives: {len(primitives)} ({sum(1 for p in primitives if p['type']=='line')} lines, {sum(1 for p in primitives if p['type']=='arc')} arcs)")
    
    # Deduplicate overlapping primitives before optimization
    primitives = dedup_primitives(primitives)
    print(f"After dedup: {len(primitives)} ({sum(1 for p in primitives if p['type']=='line')} lines, {sum(1 for p in primitives if p['type']=='arc')} arcs)")
    
    # Optimize
    print("\nOptimizing...")
    primitives = optimize_primitives(primitives, edge_pts, image_shape)
    
    # Output
    output = {
        'canvas': {'w': W, 'h': H},
        'primitives': primitives,
        'edge_count': len(edge_pts),
    }
    
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {args.output}")
    
    # Summary
    print("\n=== Optimized Primitives ===")
    for p in primitives:
        params = p['params']
        if p['type'] == 'line':
            print(f"  LINE {p['name']}: ({params['x1']:.0f},{params['y1']:.0f}) → ({params['x2']:.0f},{params['y2']:.0f})")
        elif p['type'] == 'arc':
            sweep_deg = math.degrees(params['sweep'])
            print(f"  ARC  {p['name']}: center=({params['cx']:.0f},{params['cy']:.0f}) r={params['radius']:.1f} sweep={sweep_deg:.0f}°")


if __name__ == '__main__':
    main()
