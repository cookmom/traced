"""
Muqarnas — Stalactite vault 2D projection generator.
Generates the top-down plan view as drawing paths for p5.brush.

Mathematical basis:
- Concentric tiers radiating from center, each tier smaller
- n-fold rotational symmetry (typically 8 or 12)
- Unit cells: square, rhombus, almond, biped shapes
- Each tier steps inward by a ratio (often 1/φ or 1/√2)

Usage:
    paths = generate_muqarnas(cx, cy, radius, n_fold=8, n_tiers=4)
    # Returns list of [[x,y], [x,y], ...] polyline paths
"""

import math

PHI = (1 + math.sqrt(5)) / 2


def generate_muqarnas(cx: float, cy: float, radius: float,
                       n_fold: int = 8, n_tiers: int = 4,
                       step_ratio: float = None) -> list:
    """Generate muqarnas 2D projection as drawing paths.
    
    Args:
        cx, cy: center point
        radius: outer radius
        n_fold: rotational symmetry (8 or 12 typical)
        n_tiers: number of concentric tiers
        step_ratio: how much each tier shrinks (default 1/φ)
    
    Returns:
        List of polyline paths [[x,y], [x,y], ...]
    """
    if step_ratio is None:
        step_ratio = 1 / PHI
    
    paths = []
    sector_angle = 2 * math.pi / n_fold
    
    for tier in range(n_tiers):
        r_outer = radius * (step_ratio ** tier)
        r_inner = radius * (step_ratio ** (tier + 1))
        
        # Outer ring — polygon connecting tier boundary
        ring = []
        n_pts = n_fold * 4  # Smooth ring
        for i in range(n_pts + 1):
            angle = 2 * math.pi * i / n_pts
            ring.append([
                round(cx + r_outer * math.cos(angle), 1),
                round(cy + r_outer * math.sin(angle), 1)
            ])
        paths.append(ring)
        
        # Radial lines from outer to inner at each fold division
        for k in range(n_fold):
            angle = sector_angle * k
            paths.append([
                [round(cx + r_outer * math.cos(angle), 1),
                 round(cy + r_outer * math.sin(angle), 1)],
                [round(cx + r_inner * math.cos(angle), 1),
                 round(cy + r_inner * math.sin(angle), 1)]
            ])
        
        # Cell shapes within each sector
        for k in range(n_fold):
            a1 = sector_angle * k
            a2 = sector_angle * (k + 0.5)  # midpoint
            a3 = sector_angle * (k + 1)
            r_mid = (r_outer + r_inner) / 2
            
            # Almond shape — curved cell boundary
            almond = []
            for i in range(8):
                t = i / 7
                angle = a1 + (a3 - a1) * t
                # Bulge outward in middle of cell
                bulge = 0.15 * r_outer * math.sin(math.pi * t)
                r = r_mid + bulge
                almond.append([
                    round(cx + r * math.cos(angle), 1),
                    round(cy + r * math.sin(angle), 1)
                ])
            paths.append(almond)
            
            # Diamond/rhombus at sector midpoint
            if tier < n_tiers - 1:
                diamond_r = (r_outer - r_inner) * 0.3
                diamond = [
                    [round(cx + (r_mid + diamond_r) * math.cos(a2), 1),
                     round(cy + (r_mid + diamond_r) * math.sin(a2), 1)],
                    [round(cx + r_mid * math.cos(a2 + diamond_r / r_mid), 1),
                     round(cy + r_mid * math.sin(a2 + diamond_r / r_mid), 1)],
                    [round(cx + (r_mid - diamond_r) * math.cos(a2), 1),
                     round(cy + (r_mid - diamond_r) * math.sin(a2), 1)],
                    [round(cx + r_mid * math.cos(a2 - diamond_r / r_mid), 1),
                     round(cy + r_mid * math.sin(a2 - diamond_r / r_mid), 1)],
                    [round(cx + (r_mid + diamond_r) * math.cos(a2), 1),
                     round(cy + (r_mid + diamond_r) * math.sin(a2), 1)],
                ]
                paths.append(diamond)
    
    # Center star
    star_r = radius * (step_ratio ** n_tiers)
    star = []
    for i in range(n_fold * 2 + 1):
        angle = math.pi * i / n_fold
        r = star_r if i % 2 == 0 else star_r * 0.5
        star.append([
            round(cx + r * math.cos(angle), 1),
            round(cy + r * math.sin(angle), 1)
        ])
    paths.append(star)
    
    return paths


def generate_muqarnas_js(name: str, cx: float, cy: float, radius: float,
                          n_fold: int = 8, n_tiers: int = 4) -> str:
    """Generate p5.brush JavaScript for muqarnas drawing."""
    paths = generate_muqarnas(cx, cy, radius, n_fold, n_tiers)
    paths_json = str(paths).replace("'", "")
    
    return f"""
    // {name}: muqarnas ({n_fold}-fold, {n_tiers} tiers, R={radius:.0f})
    var muq_{name} = {paths_json};"""
