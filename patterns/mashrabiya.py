"""
Mashrabiya — Geometric lattice screen generator.
Generates repeating geometric patterns as drawing paths for p5.brush.

Mathematical basis:
- Wallpaper symmetry groups (p4m for square, p6m for hexagonal)
- Unit cell motifs that tile seamlessly
- Traditional: turned wood balusters in grid
- Modern: geometric stars, hexagons, interlocking circles

Usage:
    paths = generate_mashrabiya(x, y, w, h, pattern="hexagonal", density=8)
    # Returns list of [[x,y], [x,y], ...] polyline paths
"""

import math

PHI = (1 + math.sqrt(5)) / 2


def generate_hexagonal(x: float, y: float, w: float, h: float,
                        density: int = 8) -> list:
    """Hexagonal lattice (p6m symmetry) — honeycomb pattern."""
    paths = []
    cell_w = w / density
    cell_h = cell_w * math.sqrt(3) / 2
    n_rows = int(h / cell_h) + 1
    n_cols = density + 1
    
    for row in range(n_rows):
        for col in range(n_cols):
            offset = cell_w / 2 if row % 2 else 0
            cx = x + col * cell_w + offset
            cy = y + row * cell_h
            
            if cx < x - cell_w or cx > x + w + cell_w:
                continue
            if cy < y - cell_h or cy > y + h + cell_h:
                continue
            
            # Hexagon
            hex_r = cell_w * 0.45
            hex_pts = []
            for i in range(7):
                angle = math.pi / 6 + math.pi * 2 * i / 6
                hex_pts.append([
                    round(cx + hex_r * math.cos(angle), 1),
                    round(cy + hex_r * math.sin(angle), 1)
                ])
            paths.append(hex_pts)
    
    return paths


def generate_star_lattice(x: float, y: float, w: float, h: float,
                           density: int = 6, n_points: int = 8) -> list:
    """Interlocking star pattern — n-pointed stars in grid."""
    paths = []
    cell_size = w / density
    n_rows = int(h / cell_size) + 1
    
    for row in range(n_rows):
        for col in range(density + 1):
            cx = x + col * cell_size + (cell_size / 2 if row % 2 else 0)
            cy = y + row * cell_size
            
            if cx < x or cx > x + w or cy < y or cy > y + h:
                continue
            
            r_outer = cell_size * 0.4
            r_inner = r_outer * 0.4
            
            star = []
            for i in range(n_points * 2 + 1):
                angle = math.pi * i / n_points - math.pi / 2
                r = r_outer if i % 2 == 0 else r_inner
                star.append([
                    round(cx + r * math.cos(angle), 1),
                    round(cy + r * math.sin(angle), 1)
                ])
            paths.append(star)
            
            # Connecting lines between stars
            if col < density:
                paths.append([
                    [round(cx + r_outer, 1), round(cy, 1)],
                    [round(cx + cell_size - r_outer, 1), round(cy, 1)]
                ])
    
    return paths


def generate_circles_lattice(x: float, y: float, w: float, h: float,
                              density: int = 8) -> list:
    """Interlocking circles pattern."""
    paths = []
    cell_size = w / density
    r = cell_size * 0.5
    n_rows = int(h / cell_size) + 1
    
    for row in range(n_rows):
        for col in range(density + 1):
            cx = x + col * cell_size
            cy = y + row * cell_size
            
            if cx < x or cx > x + w or cy < y or cy > y + h:
                continue
            
            circle = []
            for i in range(25):
                angle = 2 * math.pi * i / 24
                circle.append([
                    round(cx + r * math.cos(angle), 1),
                    round(cy + r * math.sin(angle), 1)
                ])
            paths.append(circle)
    
    return paths


def generate_turned_wood(x: float, y: float, w: float, h: float,
                          density: int = 10) -> list:
    """Traditional turned wood baluster grid."""
    paths = []
    spacing = w / density
    
    # Vertical balusters
    for i in range(density + 1):
        bx = x + i * spacing
        # Baluster profile — wider at top and bottom, thin in middle
        profile = []
        n_pts = 16
        for j in range(n_pts + 1):
            t = j / n_pts
            by = y + h * t
            # Sinusoidal profile
            bulge = spacing * 0.3 * (0.3 + 0.7 * abs(math.sin(math.pi * t * 2)))
            profile.append([round(bx - bulge / 2, 1), round(by, 1)])
        paths.append(profile)
        
        # Mirror side
        profile_r = []
        for j in range(n_pts + 1):
            t = j / n_pts
            by = y + h * t
            bulge = spacing * 0.3 * (0.3 + 0.7 * abs(math.sin(math.pi * t * 2)))
            profile_r.append([round(bx + bulge / 2, 1), round(by, 1)])
        paths.append(profile_r)
    
    # Horizontal rails
    for rail_y in [y, y + h * 0.33, y + h * 0.67, y + h]:
        paths.append([
            [round(x, 1), round(rail_y, 1)],
            [round(x + w, 1), round(rail_y, 1)]
        ])
    
    return paths


def generate_mashrabiya(x: float, y: float, w: float, h: float,
                         pattern: str = "hexagonal", density: int = 8) -> list:
    """Generate mashrabiya pattern within bounding box.
    
    Args:
        x, y: top-left corner
        w, h: dimensions
        pattern: "hexagonal", "star_lattice", "circles", "turned_wood"
        density: number of cells across width
    """
    generators = {
        "hexagonal": generate_hexagonal,
        "star_lattice": generate_star_lattice,
        "circles": generate_circles_lattice,
        "turned_wood": generate_turned_wood,
    }
    
    gen = generators.get(pattern, generate_hexagonal)
    return gen(x, y, w, h, density)


def generate_mashrabiya_js(name: str, x: float, y: float, w: float, h: float,
                            pattern: str = "hexagonal", density: int = 8) -> str:
    """Generate p5.brush JavaScript for mashrabiya drawing."""
    paths = generate_mashrabiya(x, y, w, h, pattern, density)
    # Limit for JS size
    paths = paths[:100]
    paths_json = str(paths).replace("'", "")
    
    return f"""
    // {name}: mashrabiya ({pattern}, density={density})
    var mash_{name} = {paths_json};"""
