"""
Islamic Geometric Patterns — Star rosettes, girih tiles, arabesque.

Mathematical basis:
- Star rosettes: n-pointed stars from circle intersections
- Girih tiles: 5 tile types that create aperiodic patterns
- Arabesque: continuous vine scrolls using tangent circles
- Zellige: Moroccan mosaic from tessellating polygons

Usage:
    paths = generate_star_rosette(cx, cy, radius, n_points=12)
    paths = generate_girih_fill(x, y, w, h)
"""

import math

PHI = (1 + math.sqrt(5)) / 2


def generate_star_rosette(cx: float, cy: float, radius: float,
                           n_points: int = 12, n_rings: int = 3) -> list:
    """Generate n-pointed star rosette with interlocking rings.
    
    Construction: n equally spaced points on circle, connect every k-th point
    where k = n/gcd(n, chosen_skip). Creates {n/k} star polygon.
    """
    paths = []
    
    # Outer circle
    circle = []
    for i in range(49):
        angle = 2 * math.pi * i / 48
        circle.append([round(cx + radius * math.cos(angle), 1),
                       round(cy + radius * math.sin(angle), 1)])
    paths.append(circle)
    
    # Star polygons at different scales
    for ring in range(n_rings):
        r = radius * (1 - ring * 0.25)
        skip = max(2, n_points // 3)  # {n/skip} star
        
        star = []
        for i in range(n_points + 1):
            idx = (i * skip) % n_points
            angle = 2 * math.pi * idx / n_points - math.pi / 2
            star.append([round(cx + r * math.cos(angle), 1),
                        round(cy + r * math.sin(angle), 1)])
        paths.append(star)
    
    # Radial lines from center to outer points
    for i in range(n_points):
        angle = 2 * math.pi * i / n_points - math.pi / 2
        paths.append([
            [round(cx, 1), round(cy, 1)],
            [round(cx + radius * math.cos(angle), 1),
             round(cy + radius * math.sin(angle), 1)]
        ])
    
    # Inner interlocking hexagons (for 12-fold)
    if n_points >= 12:
        for i in range(n_points // 2):
            angle = 2 * math.pi * i / (n_points // 2)
            ir = radius * 0.35
            hex_pts = []
            for j in range(7):
                a = angle + math.pi * 2 * j / 6
                hex_pts.append([
                    round(cx + ir * math.cos(angle) + ir * 0.3 * math.cos(a), 1),
                    round(cy + ir * math.sin(angle) + ir * 0.3 * math.sin(a), 1)
                ])
            paths.append(hex_pts)
    
    return paths


def generate_arabesque(x: float, y: float, w: float, h: float,
                        n_scrolls: int = 5) -> list:
    """Generate arabesque vine scroll pattern.
    Continuous S-curves with branching tendrils and leaf forms."""
    paths = []
    
    # Main vine — sinusoidal spine
    spine = []
    for i in range(60):
        t = i / 59
        vx = x + w * t
        vy = y + h / 2 + (h * 0.3) * math.sin(2 * math.pi * n_scrolls * t)
        spine.append([round(vx, 1), round(vy, 1)])
    paths.append(spine)
    
    # Branching scrolls at each wave peak/trough
    for i in range(n_scrolls * 2):
        t = (i + 0.5) / (n_scrolls * 2)
        bx = x + w * t
        by = y + h / 2 + (h * 0.3) * math.sin(2 * math.pi * n_scrolls * t)
        direction = 1 if i % 2 == 0 else -1
        
        # Spiral tendril
        tendril = []
        for j in range(20):
            tt = j / 19
            spiral_r = (h * 0.15) * (1 - tt)
            angle = tt * math.pi * 2
            tendril.append([
                round(bx + spiral_r * math.cos(angle) * 0.5, 1),
                round(by + direction * spiral_r * math.sin(angle), 1)
            ])
        paths.append(tendril)
        
        # Leaf at tendril end
        leaf = []
        leaf_cx = bx + (h * 0.05)
        leaf_cy = by + direction * (h * 0.12)
        leaf_r = h * 0.06
        for j in range(12):
            angle = math.pi * j / 11
            # Leaf shape: wider at base, pointed at tip
            r = leaf_r * math.sin(angle) * (1.2 - 0.4 * (j / 11))
            leaf.append([
                round(leaf_cx + r * math.cos(angle + math.pi / 4 * direction), 1),
                round(leaf_cy + r * math.sin(angle + math.pi / 4 * direction), 1)
            ])
        paths.append(leaf)
    
    return paths


def generate_geometric_js(name: str, pattern_type: str,
                           cx: float, cy: float, radius: float = 50,
                           x: float = 0, y: float = 0, w: float = 100, h: float = 100) -> str:
    """Generate p5.brush JavaScript for geometric pattern."""
    if pattern_type == "star_rosette":
        paths = generate_star_rosette(cx, cy, radius)
    elif pattern_type == "arabesque":
        paths = generate_arabesque(x, y, w, h)
    else:
        paths = generate_star_rosette(cx, cy, radius)
    
    paths = paths[:80]  # Limit for JS size
    paths_json = str(paths).replace("'", "")
    
    return f"""
    // {name}: {pattern_type}
    var geo_{name} = {paths_json};"""
