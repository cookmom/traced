#!/usr/bin/env python3
"""
Traced Pipeline — Stage 5: Architectural Construction
Takes optimized parameters and builds a structurally coherent drawing
by inferring architectural connections between elements.

This is NOT just drawing isolated shapes — it constructs the building:
- Ground line connecting all bases
- Columns rising from bases
- Arches spanning between column tops
- Drums sitting on arch/wall tops
- Domes sitting on drums
- Finials on dome apexes
- Cornices, moldings, bands connecting elements horizontally
- Window/door openings within walls
- Ornamental detail within identified hotspots

Usage:
    python construct.py --optimized optimized.json --extraction extraction.json --output construction.json
"""

import argparse
import json
import math
from pathlib import Path

PHI = (1 + math.sqrt(5)) / 2


def infer_ground_line(params: dict) -> dict:
    """Find the lowest structural element and create a continuous ground line."""
    max_y = 0
    min_x = float("inf")
    max_x = 0
    
    for name, p in params.items():
        if "arch" in p.get("type", ""):
            base_y = p.get("spring_y", 0) + p.get("half_span", 0) * 0.3
            if base_y > max_y:
                max_y = base_y
            min_x = min(min_x, p["cx"] - p["half_span"] - 20)
            max_x = max(max_x, p["cx"] + p["half_span"] + 20)
        elif p.get("type") in ("wall", "rectangle", "square"):
            base_y = p.get("y", 0) + p.get("h", 0)
            if base_y > max_y:
                max_y = base_y
            min_x = min(min_x, p.get("x", 0))
            max_x = max(max_x, p.get("x", 0) + p.get("w", 0))
    
    if max_y == 0:
        max_y = 1800
        min_x, max_x = 100, 980
    
    return {
        "type": "ground_line",
        "y": max_y + 10,
        "x1": max(20, min_x - 30),
        "x2": min(1060, max_x + 30),
    }


def infer_columns(params: dict, ground: dict) -> list:
    """Infer column positions from arch springing points."""
    columns = []
    seen_x = set()
    
    for name, p in params.items():
        if "arch" not in p.get("type", ""):
            continue
        
        spring_y = p.get("spring_y", 0)
        half_span = p.get("half_span", 0)
        cx = p.get("cx", 0)
        weight = p.get("line_weight", 1.5)
        
        # Left column
        lx = round(cx - half_span)
        if lx not in seen_x and half_span > 20:
            columns.append({
                "type": "column",
                "x": lx,
                "top_y": spring_y,
                "bot_y": ground["y"],
                "width": max(4, half_span * 0.06),
                "line_weight": weight,
                "source_arch": name,
            })
            seen_x.add(lx)
        
        # Right column
        rx = round(cx + half_span)
        if rx not in seen_x and half_span > 20:
            columns.append({
                "type": "column",
                "x": rx,
                "top_y": spring_y,
                "bot_y": ground["y"],
                "width": max(4, half_span * 0.06),
                "line_weight": weight,
                "source_arch": name,
            })
            seen_x.add(rx)
    
    return columns


def infer_drums(params: dict) -> list:
    """Infer drum structures below domes."""
    drums = []
    
    for name, p in params.items():
        if "dome" not in p.get("type", ""):
            continue
        
        r = p.get("radius", 0)
        if r < 15:
            continue
        
        cx = p["cx"]
        cy = p["cy"]
        hr = p.get("h_ratio", 1.0)
        
        drum_h = r / (PHI * PHI)  # Golden ratio derived
        drum_top = cy  # Dome sits on top of drum
        drum_bot = drum_top + drum_h
        drum_w = r * 1.7  # Drum slightly wider than dome
        
        drums.append({
            "type": "drum",
            "cx": cx,
            "top_y": drum_top,
            "bot_y": drum_bot,
            "width": drum_w,
            "n_windows": max(3, int(drum_w / 30)),
            "line_weight": p.get("line_weight", 1.2),
            "source_dome": name,
        })
    
    return drums


def infer_finials(params: dict) -> list:
    """Infer crescent/spire finials on dome tops."""
    finials = []
    
    for name, p in params.items():
        if "dome" not in p.get("type", ""):
            continue
        
        r = p.get("radius", 0)
        if r < 25:  # Only on significant domes
            continue
        
        cx = p["cx"]
        cy = p["cy"]
        hr = p.get("h_ratio", 1.0)
        dome_top = cy - r * max(hr, 1.0)
        
        finials.append({
            "type": "finial",
            "cx": cx,
            "top_y": dome_top - r * 0.3,
            "bot_y": dome_top,
            "line_weight": max(0.8, p.get("line_weight", 1.0) - 0.3),
            "source_dome": name,
        })
    
    return finials


def infer_cornices(params: dict) -> list:
    """Infer horizontal cornice/band lines connecting elements at same height."""
    # Find clusters of elements at similar Y positions
    y_positions = []
    
    for name, p in params.items():
        if "arch" in p.get("type", ""):
            y_positions.append((p["spring_y"], p["cx"] - p["half_span"], p["cx"] + p["half_span"], p.get("line_weight", 1.5)))
        elif "dome" in p.get("type", ""):
            y_positions.append((p["cy"], p["cx"] - p["radius"], p["cx"] + p["radius"], p.get("line_weight", 1.2)))
    
    if not y_positions:
        return []
    
    # Cluster by Y (within 30px = same level)
    y_positions.sort(key=lambda yp: yp[0])
    cornices = []
    
    for i, (y, x1, x2, wt) in enumerate(y_positions):
        # Find other elements at same height
        connected = [(x1, x2)]
        for j, (y2, x1b, x2b, wt2) in enumerate(y_positions):
            if i != j and abs(y - y2) < 30:
                connected.append((x1b, x2b))
        
        if len(connected) > 1:
            all_x1 = min(c[0] for c in connected)
            all_x2 = max(c[1] for c in connected)
            cornices.append({
                "type": "cornice",
                "y": y,
                "x1": all_x1,
                "x2": all_x2,
                "line_weight": wt * 0.6,
            })
    
    # Deduplicate cornices at similar heights
    deduped = []
    for c in cornices:
        is_dup = False
        for d in deduped:
            if abs(c["y"] - d["y"]) < 20:
                is_dup = True
                d["x1"] = min(d["x1"], c["x1"])
                d["x2"] = max(d["x2"], c["x2"])
                break
        if not is_dup:
            deduped.append(c)
    
    return deduped


def construct(optimized: dict, extraction: dict) -> dict:
    """Build full architectural construction from optimized params."""
    params = optimized["params"]
    
    print("Constructing architectural elements...")
    
    # Infer structural connections
    ground = infer_ground_line(params)
    print(f"  Ground line: y={ground['y']:.0f}, x=[{ground['x1']:.0f}, {ground['x2']:.0f}]")
    
    columns = infer_columns(params, ground)
    print(f"  Columns: {len(columns)} inferred from arch springing points")
    
    drums = infer_drums(params)
    print(f"  Drums: {len(drums)} under domes")
    
    finials = infer_finials(params)
    print(f"  Finials: {len(finials)} on dome tops")
    
    cornices = infer_cornices(params)
    print(f"  Cornices: {len(cornices)} horizontal connections")
    
    # Combine all construction elements
    construction = {
        "ground": ground,
        "columns": columns,
        "drums": drums,
        "finials": finials,
        "cornices": cornices,
        "total_inferred": len(columns) + len(drums) + len(finials) + len(cornices) + 1,
    }
    
    print(f"  Total inferred elements: {construction['total_inferred']}")
    
    return construction


def main():
    parser = argparse.ArgumentParser(description="Traced: Architectural construction inference")
    parser.add_argument("--optimized", required=True)
    parser.add_argument("--extraction", required=True)
    parser.add_argument("--output", default="construction.json")
    args = parser.parse_args()
    
    optimized = json.loads(Path(args.optimized).read_text())
    extraction = json.loads(Path(args.extraction).read_text())
    
    construction = construct(optimized, extraction)
    
    # Merge into optimized output
    optimized["construction"] = construction
    Path(args.output).write_text(json.dumps(optimized, indent=2))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
