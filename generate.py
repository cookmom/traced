#!/usr/bin/env python3
"""
Traced Pipeline — Stage 4: FUSION Code Generator
Reads extraction JSON + knowledge JSON → generates p5.brush drawing HTML
using mathematically fitted curves + filtered edge detail.

The fusion logic:
- SAM masks → element positions and bounding geometry
- Knowledge → correct curve type (horseshoe, hemisphere, pointed)
- Math fitting → clean parametric curves for each element
- Canny edges → filtered for architectural detail only (grilles, ribs, ornament)
- Depth layers → drawing order (back to front) + line weights

Usage:
    python generate.py --extraction extraction.json --output drawing.html --name "Building"
    python generate.py --extraction extraction.json --knowledge knowledge.json --output drawing.html
"""

import argparse
import json
import math
from pathlib import Path

# Pattern generators
try:
    from patterns.muqarnas import generate_muqarnas
    from patterns.mashrabiya import generate_mashrabiya
    from patterns.geometric import generate_star_rosette, generate_arabesque
    HAS_PATTERNS = True
except ImportError:
    HAS_PATTERNS = False

CANVAS_W = 1080
CANVAS_H = 1920
PHI = (1 + math.sqrt(5)) / 2


def compute_aspect_mapping(src_w, src_h):
    """Compute mapping from source image to canvas.
    If source is already 1080x1920 (preprocessed), returns identity mapping."""
    if src_w == CANVAS_W and src_h == CANVAS_H:
        return 1.0, 0.0, 0.0  # Identity — no mapping needed
    
    src_aspect = src_w / src_h
    dst_aspect = CANVAS_W / CANVAS_H
    
    if src_aspect > dst_aspect:
        scale = CANVAS_W / src_w
        offset_x = 0
        offset_y = (CANVAS_H - src_h * scale) / 2
    else:
        scale = CANVAS_H / src_h
        offset_x = (CANVAS_W - src_w * scale) / 2
        offset_y = 0
    
    return scale, offset_x, offset_y


def map_point(px, py, scale, offset_x, offset_y):
    """Map source image coords to canvas coords."""
    return [round(px * scale + offset_x, 1), round(py * scale + offset_y, 1)]


def map_contour(contour, scale, ox, oy):
    """Map a full contour to canvas coords."""
    return [map_point(p[0], p[1], scale, ox, oy) for p in contour]


def filter_edges(edge_paths, shape_type, bbox, min_length_pct=0.05):
    """Filter edge paths keeping only architecturally relevant ones.
    
    - Keep edges longer than min_length_pct of element size
    - Keep edges aligned with structural axes
    - Reject isolated fragments, noise, shadows
    """
    if not edge_paths:
        return []
    
    bw = max(1, bbox.get("w", 100))
    bh = max(1, bbox.get("h", 100))
    min_pts = max(3, int(min_length_pct * max(bw, bh)))
    
    filtered = []
    for path in edge_paths:
        # Reject too-short paths (noise)
        if len(path) < 3:
            continue
        
        # Compute path length in normalized coords
        total_len = 0
        for i in range(1, len(path)):
            dx = (path[i][0] - path[i-1][0]) * bw
            dy = (path[i][1] - path[i-1][1]) * bh
            total_len += math.sqrt(dx*dx + dy*dy)
        
        # Reject short edges (< 5% of element diagonal)
        diag = math.sqrt(bw*bw + bh*bh)
        if total_len < diag * min_length_pct:
            continue
        
        # For domes: prefer horizontal edges (ribs, tile lines)
        # For arches: prefer edges that follow the curve
        # For rectangles/walls: prefer horizontal and vertical edges
        # For openwork: keep most edges (they ARE the detail)
        
        if shape_type in ("dome", "dome_like", "hemisphere"):
            # Compute dominant direction
            dx_total = abs(path[-1][0] - path[0][0]) * bw
            dy_total = abs(path[-1][1] - path[0][1]) * bh
            # Keep if mostly horizontal (dome ribs) or curved
            if dy_total > dx_total * 3:  # Reject mostly-vertical noise in domes
                continue
        
        elif shape_type in ("column", "minaret"):
            dx_total = abs(path[-1][0] - path[0][0]) * bw
            dy_total = abs(path[-1][1] - path[0][1]) * bh
            # Keep vertical edges (fluting) and horizontal (bands)
            if dx_total > dy_total * 2 and dx_total > bw * 0.3:
                continue  # Reject diagonal noise in columns
        
        elif shape_type in ("wall", "rectangle", "square"):
            # Keep grid-aligned edges
            pass  # Most edges in walls are structural — keep them
        
        # Path passed all filters
        filtered.append(path)
    
    # Cap at reasonable number to avoid drawing overload
    max_paths = 8 if shape_type in ("dome", "dome_like") else 12
    return filtered[:max_paths]


def generate_dome_js(name, cx, cy, radius, profile="hemisphere"):
    """Generate JS for a mathematically correct dome curve."""
    if profile == "onion":
        # Onion dome: wider at equator, pointed at top
        return f"""
    // {name}: onion dome (R={radius:.0f})
    var pts_{name} = [];
    for(var i=0;i<=30;i++){{var t=i/30;var a=Math.PI*(1-t);
      var r_mod = {radius} * (1 + 0.15*Math.sin(a*2)); // bulge
      pts_{name}.push([{cx}+r_mod*Math.cos(a), {cy}-{radius}*1.1*Math.sin(a), 0.3+0.7*Math.sin(Math.PI*t)]);}}"""
    elif profile == "pointed":
        return f"""
    // {name}: pointed dome (R={radius:.0f})
    var pts_{name} = [];
    for(var i=0;i<=30;i++){{var t=i/30;var a=Math.PI*(1-t);
      pts_{name}.push([{cx}+{radius}*Math.cos(a), {cy}-{radius}*1.2*Math.pow(Math.sin(a),0.85), 0.3+0.7*Math.sin(Math.PI*t)]);}}"""
    else:  # hemisphere (default)
        return f"""
    // {name}: hemisphere dome (R={radius:.0f})
    var pts_{name} = [];
    for(var i=0;i<=30;i++){{var t=i/30;var a=Math.PI*(1-t);
      pts_{name}.push([{cx}+{radius}*Math.cos(a), {cy}-{radius}*Math.sin(a), 0.3+0.7*Math.sin(Math.PI*t)]);}}"""


def generate_arch_js(name, cx, spring_y, half_span, profile="pointed", rise_span=0.866):
    """Generate JS for a mathematically correct arch curve."""
    rise = half_span * 2 * rise_span
    
    if profile == "horseshoe":
        # Horseshoe: arc continues past 180°, center above springing
        return f"""
    // {name}: horseshoe arch (span={half_span*2:.0f}, rise/span={rise_span:.3f})
    var pts_{name} = [];
    var archR_{name} = {half_span} * 1.05;
    var centerY_{name} = {spring_y} - {half_span} * 0.1;
    for(var i=0;i<=40;i++){{var t=i/40;
      var angle = Math.PI*0.15 + (Math.PI - Math.PI*0.3)*t; // > 180° sweep
      pts_{name}.push([{cx}+archR_{name}*Math.cos(Math.PI-angle), centerY_{name}-archR_{name}*Math.sin(angle), 0.3+0.7*Math.sin(Math.PI*t)]);}}"""
    
    elif profile == "semicircular":
        return f"""
    // {name}: semicircular arch (span={half_span*2:.0f})
    var pts_{name} = [];
    for(var i=0;i<=30;i++){{var t=i/30;var a=Math.PI*(1-t);
      pts_{name}.push([{cx}+{half_span}*Math.cos(a), {spring_y}-{half_span}*Math.sin(a), 0.3+0.7*Math.sin(Math.PI*t)]);}}"""
    
    elif profile == "ogee":
        # Four-center S-curve
        return f"""
    // {name}: ogee arch (span={half_span*2:.0f})
    var pts_{name} = [];
    for(var i=0;i<=40;i++){{var t=i/40;
      var x = {cx} - {half_span} + {half_span}*2*t;
      var nt = t <= 0.5 ? t*2 : (1-t)*2;
      var y = {spring_y} - {rise}*(0.5*Math.sin(Math.PI*nt) + 0.5*Math.pow(Math.sin(Math.PI*t),0.6));
      pts_{name}.push([x, y, 0.3+0.7*Math.sin(Math.PI*t)]);}}"""
    
    elif profile == "cusped":
        return f"""
    // {name}: cusped arch (span={half_span*2:.0f})
    var pts_{name} = [];
    for(var i=0;i<=40;i++){{var t=i/40;
      var x = {cx} - {half_span} + {half_span}*2*t;
      var base = Math.sin(Math.PI*t);
      var cusps = 0.08*Math.sin(Math.PI*t*5); // small scallops
      pts_{name}.push([x, {spring_y}-{rise}*(base+cusps), 0.3+0.7*base]);}}"""
    
    elif profile == "tudor":
        # Four-center depressed arch — flatter middle, steeper sides
        return f"""
    // {name}: tudor arch (span={half_span*2:.0f}, rise/span={rise_span:.3f})
    var pts_{name} = [];
    for(var i=0;i<=40;i++){{var t=i/40;
      var x={cx}-{half_span}+{half_span}*2*t;
      var y;
      if(t<0.2||t>0.8){{var nt=t<0.2?t/0.2:(1-t)/0.2;y={spring_y}-{rise}*0.4*Math.sin(Math.PI/2*nt);}}
      else{{var nt=(t-0.2)/0.6;y={spring_y}-{rise}*(0.4+0.6*Math.sin(Math.PI*nt));}}
      pts_{name}.push([x,y,0.3+0.7*Math.sin(Math.PI*t)]);}}"""
    
    elif profile == "elliptical":
        # Semi-ellipse
        return f"""
    // {name}: elliptical arch (span={half_span*2:.0f}, rise/span={rise_span:.3f})
    var pts_{name} = [];
    for(var i=0;i<=30;i++){{var t=i/30;var a=Math.PI*(1-t);
      pts_{name}.push([{cx}+{half_span}*Math.cos(a),{spring_y}-{rise}*Math.sin(a),0.3+0.7*Math.sin(Math.PI*t)]);}}"""
    
    elif profile == "catenary":
        # Inverted catenary — structurally optimal
        return f"""
    // {name}: catenary arch (span={half_span*2:.0f}, rise/span={rise_span:.3f})
    var pts_{name} = [];
    var _a_{name}={rise}/(Math.cosh(1.5)-1);
    for(var i=0;i<=30;i++){{var t=i/30;
      var x={cx}-{half_span}+{half_span}*2*t;
      var nx=(t-0.5)*3;
      var y={spring_y}-_a_{name}*(Math.cosh(1.5)-Math.cosh(nx));
      pts_{name}.push([x,y,0.3+0.7*Math.sin(Math.PI*t)]);}}"""
    
    elif profile == "parabolic":
        # Parabolic — y = a*x²
        return f"""
    // {name}: parabolic arch (span={half_span*2:.0f}, rise/span={rise_span:.3f})
    var pts_{name} = [];
    for(var i=0;i<=30;i++){{var t=i/30;
      var x={cx}-{half_span}+{half_span}*2*t;
      var nx=(x-{cx})/{half_span};
      pts_{name}.push([x,{spring_y}-{rise}*(1-nx*nx),0.3+0.7*Math.sin(Math.PI*t)]);}}"""
    
    else:  # pointed (default)
        return f"""
    // {name}: pointed arch (span={half_span*2:.0f}, rise/span={rise_span:.3f})
    var pts_{name} = [];
    for(var i=0;i<=30;i++){{var t=i/30;var x,y;
      if(t<=0.5){{var tt=t*2;x={cx}-{half_span}+{half_span}*tt;y={spring_y}-{rise}*Math.sin(Math.PI/2*tt);}}
      else{{var tt=(t-0.5)*2;x={cx}+{half_span}*tt;y={spring_y}-{rise}+{rise}*(1-Math.cos(Math.PI/2*tt));}}
      pts_{name}.push([x,y,0.3+0.7*Math.sin(Math.PI*t)]);}}"""


def generate_html(extraction: dict, knowledge: dict = None, building_name: str = "Building") -> str:
    """Generate p5.brush drawing with math-fitted curves + filtered edges."""
    
    elements = extraction["elements"]
    img_w = extraction["image_size"]["w"]
    img_h = extraction["image_size"]["h"]
    
    # Aspect-preserving mapping
    scale, ox, oy = compute_aspect_mapping(img_w, img_h)
    
    # Known arch/dome types from research
    known_arches = set()
    known_domes = set()
    if knowledge:
        known_arches = set(knowledge.get("arch_types", []))
        known_domes = set(knowledge.get("dome_types", []))
    
    # Sort by depth (back to front)
    layer_order = {"distant_sky": 0, "background_domes": 1, "mid_facade": 2, "foreground_frame": 3}
    sorted_els = sorted(elements, key=lambda e: layer_order.get(e.get("depth", {}).get("layer", "mid_facade"), 2))
    
    # Skip full-image elements (openwork_0 is usually the whole photo)
    drawable = [e for e in sorted_els if e.get("area_pct", 0) < 0.4 and e.get("area_pct", 0) > 0.003]
    
    # Build curve definitions and steps
    curve_defs = []
    steps_js = []
    edge_data_all = {}
    
    for idx, el in enumerate(drawable):
        name = el["name"]
        shape = el.get("shape", {}).get("type", "unknown")
        depth = el.get("depth", {})
        weight = depth.get("line_weight", 1.5)
        detail_weight = max(0.3, weight - 0.6)
        layer = depth.get("layer", "mid_facade")
        
        bbox = el.get("primitives", {}).get("bbox", {})
        center = el.get("primitives", {}).get("center", {})
        
        # Map center and bbox to canvas
        ccx, ccy = map_point(center.get("x", 0), center.get("y", 0), scale, ox, oy)
        bx, by = map_point(bbox.get("x", 0), bbox.get("y", 0), scale, ox, oy)
        bw = bbox.get("w", 50) * scale
        bh = bbox.get("h", 50) * scale
        
        # Determine formula annotation
        arch_geo = el.get("primitives", {}).get("arch_geometry", {})
        dome_geo = el.get("primitives", {}).get("dome_geometry", {})
        rise_span = arch_geo.get("rise_to_span", 0.866)
        
        formula = f"{shape} | {layer} (wt={weight})"
        if arch_geo:
            formula = f"{arch_geo.get('profile_type', shape)} rise/span={rise_span:.3f} | wt={weight}"
        if dome_geo:
            formula = f"{dome_geo.get('profile_type', 'dome')} h/d={dome_geo.get('height_to_diameter', 0):.3f} | wt={weight}"
        
        # === MATH-FITTED CURVES ===
        half_span = bw / 2
        spring_y = by + bh  # bottom of bbox
        radius = min(bw, bh) / 2
        
        if "dome" in shape:
            # Determine dome profile from knowledge
            dome_profile = "hemisphere"
            if dome_geo:
                hd = dome_geo.get("height_to_diameter", 0.5)
                if hd > 0.7:
                    dome_profile = "onion"
                elif hd > 0.55:
                    dome_profile = "pointed"
            if "hemisphere" in known_domes:
                dome_profile = "hemisphere"  # Research override
            
            curve_defs.append(generate_dome_js(name, ccx, ccy, radius, dome_profile))
            draw_fn = f"brush.set('2H','#5a5248',{weight});if(pts_{name}){{var n=Math.max(3,Math.round(p*pts_{name}.length));var sub=pts_{name}.slice(0,n);if(sub.length>=2)brush.spline(sub,0.3);}}"
        
        elif "arch" in shape:
            # Determine arch profile from knowledge + detected type
            arch_profile = "pointed"
            if "horseshoe" in shape or "horseshoe" in known_arches:
                arch_profile = "horseshoe"
            elif "ogee" in shape:
                arch_profile = "ogee"
            elif "cusped" in shape or "cusped" in known_arches:
                arch_profile = "cusped"
            elif "semicircular" in known_arches and rise_span < 0.55:
                arch_profile = "semicircular"
            
            curve_defs.append(generate_arch_js(name, ccx, spring_y, half_span, arch_profile, rise_span))
            draw_fn = f"brush.set('2H','#5a5248',{weight});if(pts_{name}){{var n=Math.max(3,Math.round(p*pts_{name}.length));var sub=pts_{name}.slice(0,n);if(sub.length>=2)brush.spline(sub,0.3);}}"
            # Add legs for arches
            draw_fn += f"if(p>0.8){{brush.line({ccx-half_span:.0f},{spring_y:.0f},{ccx-half_span:.0f},{spring_y+bh*0.2:.0f});brush.line({ccx+half_span:.0f},{spring_y:.0f},{ccx+half_span:.0f},{spring_y+bh*0.2:.0f});}}"
        
        elif shape in ("wall", "rectangle", "square"):
            # Draw as rectangle
            curve_defs.append(f"\n    // {name}: rectangle")
            draw_fn = f"""brush.set('2H','#5a5248',{weight});
      var rp = p;
      if(rp<=0.25){{brush.line({bx:.0f},{by:.0f},{bx+bw:.0f},{by:.0f});}}
      if(rp>0.25&&rp<=0.5){{brush.line({bx:.0f},{by:.0f},{bx+bw:.0f},{by:.0f});brush.line({bx+bw:.0f},{by:.0f},{bx+bw:.0f},{by+bh:.0f});}}
      if(rp>0.5&&rp<=0.75){{brush.line({bx:.0f},{by:.0f},{bx+bw:.0f},{by:.0f});brush.line({bx+bw:.0f},{by:.0f},{bx+bw:.0f},{by+bh:.0f});brush.line({bx+bw:.0f},{by+bh:.0f},{bx:.0f},{by+bh:.0f});}}
      if(rp>0.75){{brush.line({bx:.0f},{by:.0f},{bx+bw:.0f},{by:.0f});brush.line({bx+bw:.0f},{by:.0f},{bx+bw:.0f},{by+bh:.0f});brush.line({bx+bw:.0f},{by+bh:.0f},{bx:.0f},{by+bh:.0f});brush.line({bx:.0f},{by+bh:.0f},{bx:.0f},{by:.0f});}}"""
        
        else:
            # Fallback: draw contour from SAM mask (mapped with aspect correction)
            contour = el.get("primitives", {}).get("contour", [])
            if contour:
                mapped = map_contour(contour, scale, ox, oy)
                ckey = f"contour_{name}"
                edge_data_all[ckey] = mapped
                draw_fn = f"brush.set('2H','#5a5248',{weight});var c=_data['{ckey}'];if(c&&c.length>=3){{var n=Math.max(3,Math.round(p*c.length));var pts=[];for(var i=0;i<n;i++)pts.push([c[i][0],c[i][1],0.3+0.7*Math.sin(Math.PI*i/c.length)]);if(pts.length>=2)brush.spline(pts,0.3);}}"
        
        # === FILTERED EDGE DETAIL ===
        raw_edges = el.get("edge_paths", [])
        filtered = filter_edges(raw_edges, shape, bbox)
        edge_draw = ""
        if filtered:
            # Scale filtered edges to canvas
            scaled_edges = []
            for path in filtered:
                sp = [map_point(
                    bbox.get("x", 0) + p[0] * bbox.get("w", 1),
                    bbox.get("y", 0) + p[1] * bbox.get("h", 1),
                    scale, ox, oy
                ) for p in path]
                scaled_edges.append(sp)
            
            ekey = f"edges_{name}"
            edge_data_all[ekey] = scaled_edges
            edge_draw = f"""
      if(p>0.4){{var edges=_data['{ekey}'];if(edges){{brush.set('2H','#8a8278',{detail_weight:.1f});var ep=(p-0.4)/0.6;var nP=Math.max(1,Math.round(ep*edges.length));for(var ei=0;ei<nP;ei++){{var path=edges[ei];if(path&&path.length>=2){{var pts2=[];for(var pi=0;pi<path.length;pi++)pts2.push([path[pi][0],path[pi][1],0.3]);if(pts2.length>=2)brush.spline(pts2,0.3);}}}}}}}}"""
        
        # Duration
        dur = max(0.3, min(1.8, (len(filtered) + 5) / 12))
        
        # Build step
        bbox_js = f"[{int(bx)},{int(by)},{int(bx+bw)},{int(by+bh)}]"
        steps_js.append(f"""  {{name:'{name.upper().replace("_"," ")}',formula:'{formula}',dur:{dur:.2f},
   bbox:{bbox_js},
   draw:function(p){{
      {draw_fn}{edge_draw}
      return[{ccx:.0f},{ccy:.0f}];
   }}}},""")
    
    # Wash steps
    wash_start = len(steps_js)
    wash_colors = {"distant_sky": ("#87CEEB", 30), "background_domes": ("#D4C5B0", 20), "mid_facade": ("#4A5568", 15), "foreground_frame": ("#C4B89A", 25)}
    
    for idx, el in enumerate(drawable):
        if el.get("area_pct", 0) < 0.02:
            continue
        name = el["name"]
        layer = el.get("depth", {}).get("layer", "mid_facade")
        color, alpha = wash_colors.get(layer, ("#888888", 15))
        center = el.get("primitives", {}).get("center", {})
        ccx, ccy = map_point(center.get("x", 0), center.get("y", 0), scale, ox, oy)
        bbox = el.get("primitives", {}).get("bbox", {})
        bx, by = map_point(bbox.get("x", 0), bbox.get("y", 0), scale, ox, oy)
        bw = bbox.get("w", 50) * scale
        bh = bbox.get("h", 50) * scale
        
        steps_js.append(f"""  {{name:'{name.upper().replace("_"," ")} WASH',formula:'{color} α={alpha}',dur:0.05,
   bbox:[{int(bx)},{int(by)},{int(bx+bw)},{int(by+bh)}],
   draw:function(p){{if(p>=0){{randomSeed({6000+idx});brush.noStroke();brush.fill('{color}',{alpha});brush.fillBleed(0.001);brush.fillTexture(0,0);brush.rect({bx:.0f},{by:.0f},{bw:.0f},{bh:.0f});brush.noFill();}}return[{ccx:.0f},{ccy:.0f}];}}}},""")
    
    all_steps = "\n".join(steps_js)
    curve_defs_js = "\n".join(curve_defs)
    data_js = json.dumps(edge_data_all)
    
    html = f"""<!-- بسم الله الرحمن الرحيم -->
<!-- Generated by Traced Pipeline — {building_name} -->
<!-- Math-fitted curves + filtered edges + depth layers -->
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/p5@2.0.3/lib/p5.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/p5.brush@2.0.2-beta"></script>
<style>
*{{margin:0}}body{{background:#f2eada;overflow:hidden;margin:0;padding:0;display:flex;justify-content:center;align-items:flex-start}}canvas{{object-fit:contain!important;display:block;max-height:100vh;max-width:calc(100vh*9/16)}}
#hud{{position:fixed;top:0;left:0;width:100vw;height:100%;pointer-events:none;font-family:'SF Mono','Fira Code',monospace;overflow:hidden}}
.track-box{{position:absolute;transition:opacity 0.3s,left 0.4s,top 0.4s,width 0.4s,height 0.4s;min-width:60px;min-height:60px;overflow:visible}}
.track-box::before,.track-box::after,.track-box .bl::before,.track-box .br::before{{content:'';position:absolute;width:30px;height:30px;border-color:#5a5248;border-style:solid;border-width:0}}
.track-box::before{{top:-2px;left:-2px;border-top-width:1.5px;border-left-width:1.5px}}.track-box::after{{top:-2px;right:-2px;border-top-width:1.5px;border-right-width:1.5px}}
.track-box .bl::before{{bottom:-2px;left:-2px;border-bottom-width:1.5px;border-left-width:1.5px;position:absolute}}.track-box .br::before{{bottom:-2px;right:-2px;border-bottom-width:1.5px;border-right-width:1.5px;position:absolute}}
.corner-circ{{position:absolute;width:4px;height:4px;border:1px solid #5a5248;border-radius:50%}}.corner-circ.tl{{top:-2px;left:-2px}}.corner-circ.tr{{top:-2px;right:-2px}}.corner-circ.bl2{{bottom:-2px;left:-2px}}.corner-circ.br2{{bottom:-2px;right:-2px}}
.hud-info{{position:absolute;bottom:100%;left:0;margin-bottom:4px;background:rgba(242,234,218,0.88);border:1px solid rgba(90,82,72,0.3);border-radius:4px;padding:4px 8px;white-space:nowrap}}
.hud-label{{color:#3a3430;font-weight:500;letter-spacing:0.3px;text-transform:uppercase;margin-bottom:3px}}.hud-formula{{color:#5a5248;letter-spacing:0.3px;margin-bottom:2px}}.hud-coords{{color:#8a8278;letter-spacing:0.3px}}
#pencil-dot{{position:absolute;width:14px;height:14px;pointer-events:none;z-index:15}}#pencil-dot::before{{content:'';position:absolute;left:6px;top:0;width:1px;height:14px;background:#5a5248}}#pencil-dot::after{{content:'';position:absolute;top:6px;left:0;width:14px;height:1px;background:#5a5248}}
#pencil-dot .ch-circle{{position:absolute;top:4px;left:4px;width:6px;height:6px;border:1px solid #5a5248;border-radius:50%}}
</style>
</head>
<body>
<div id="hud"><div id="pencil-dot" style="display:none"><div class="ch-circle"></div></div></div>
<script>
var W={CANVAS_W},H={CANVAS_H},cx=W/2;
var _data={data_js};
{curve_defs_js}

var ANNOT_FONT=13,_dimLabels=[],_lastDimStep=-1;
function clearDimLabels(){{for(var i=0;i<_dimLabels.length;i++)_dimLabels[i].remove();_dimLabels=[];}}
function addDimText(x,y,t,a){{for(var i=0;i<_dimLabels.length;i++)if(_dimLabels[i]._t===t)return;var el=document.createElement('div');el._t=t;var s=window._hudScale||1,r=window._canvasEl?window._canvasEl.getBoundingClientRect():{{left:0,top:0}};el.style.cssText='position:absolute;color:#8a8278;font-family:SF Mono,Fira Code,monospace;font-size:'+Math.max(9,Math.round(ANNOT_FONT*s))+'px;white-space:nowrap;pointer-events:none;';el.style.left=(x*s+r.left)+'px';el.style.top=(y*s+r.top)+'px';if(a==='center')el.style.transform='translateX(-50%)';el.textContent=t;document.getElementById('hud').appendChild(el);_dimLabels.push(el);}}
function drawAnnotations(step){{if(step!==_lastDimStep){{clearDimLabels();_lastDimStep=step;}}if(step>=0)addDimText(cx,H-40,'Traced | {building_name} | math-fitted curves + filtered edges','center');}}

var FPS=30;
var STEPS=[
{all_steps}
];
var PAUSE_S=0.12,WASH_START={wash_start};
var frameStarts=[],frameEnds=[],f2=0;
for(var i=0;i<STEPS.length;i++){{frameStarts.push(Math.round(f2));f2+=STEPS[i].dur*FPS;frameEnds.push(Math.round(f2));if(i<STEPS.length-1&&i<WASH_START-1)f2+=PAUSE_S*FPS;}}
var totalAnimFrames=Math.round(f2);
var trackBox=null,pencilDot=null;
function createTrackBox(step){{var hud=document.getElementById('hud');if(trackBox)trackBox.remove();var box=document.createElement('div');box.className='track-box';var s=window._hudScale||1,r=window._canvasEl?window._canvasEl.getBoundingClientRect():{{left:0,top:0}};var b=step.bbox,pad=12*s;var cb0=Math.max(20,b[0]),cb1=Math.max(20,b[1]),cb2=Math.min(W-20,b[2]),cb3=Math.min(H-20,b[3]);box.style.left=(cb0*s+r.left-pad)+'px';box.style.top=(cb1*s+r.top-pad)+'px';box.style.width=((cb2-cb0)*s+pad*2)+'px';box.style.height=((cb3-cb1)*s+pad*2)+'px';['tl','tr','bl2','br2'].forEach(function(c){{var ci=document.createElement('div');ci.className='corner-circ '+c;box.appendChild(ci);}});var bl=document.createElement('div');bl.className='bl';bl.style.cssText='position:absolute;bottom:0;left:0;width:100%;height:100%;pointer-events:none;';box.appendChild(bl);var br=document.createElement('div');br.className='br';br.style.cssText='position:absolute;bottom:0;right:0;width:100%;height:100%;pointer-events:none;';box.appendChild(br);var _fs=Math.max(9,Math.round(ANNOT_FONT*(window._hudScale||1)))+'px';var info=document.createElement('div');info.className='hud-info';if((cb1*s+r.top-pad)<80){{info.style.bottom='auto';info.style.top='6px';info.style.marginBottom='0';}}var lbl=document.createElement('div');lbl.className='hud-label';lbl.style.fontSize=_fs;lbl.textContent=step.name;info.appendChild(lbl);var frm=document.createElement('div');frm.className='hud-formula';frm.style.fontSize=_fs;frm.textContent=step.formula;info.appendChild(frm);var crd=document.createElement('div');crd.className='hud-coords';crd.style.fontSize=_fs;crd.id='live-coords';crd.textContent='x:\\u2014 y:\\u2014';info.appendChild(crd);box.appendChild(info);hud.appendChild(box);trackBox=box;}}
function updatePencilDot(x,y){{if(!pencilDot)pencilDot=document.getElementById('pencil-dot');pencilDot.style.display='block';var s=window._hudScale||1,r=window._canvasEl?window._canvasEl.getBoundingClientRect():{{left:0,top:0}};pencilDot.style.left=(x*s+r.left-7)+'px';pencilDot.style.top=(y*s+r.top-7)+'px';var el=document.getElementById('live-coords');if(el)el.textContent='x:'+Math.round(x)+' y:'+Math.round(y);}}
var lastHudStep=-1,_inFillPhase=false;
function setup(){{var c=createCanvas(W,H,WEBGL);pixelDensity(1);var vh=window.innerHeight;var cw=Math.min(window.innerWidth,vh*9/16);c.elt.style.width=cw+'px';c.elt.style.height=(cw/(W/H))+'px';c.elt.style.display='block';c.elt.style.margin='0';brush.load();frameRate(FPS);window._hudScale=c.elt.getBoundingClientRect().width/W;window._canvasEl=c.elt;window.addEventListener('resize',function(){{window._hudScale=c.elt.getBoundingClientRect().width/W;_lastDimStep=-1;}});}}
function draw(){{
  translate(-width/2,-height/2);var f3=frameCount-1;if(f3>=totalAnimFrames)f3=totalAnimFrames-1;
  var activeStep=0;for(var i=STEPS.length-1;i>=0;i--){{if(f3>=frameStarts[i]){{activeStep=i;break;}}}}
  if(activeStep<WASH_START){{_inFillPhase=false;background(242,234,218);for(var i=0;i<activeStep;i++)STEPS[i].draw(1);}}
  else{{if(!_inFillPhase){{background(242,234,218);for(var i=0;i<WASH_START;i++)STEPS[i].draw(1);drawAnnotations(WASH_START-1);_inFillPhase=true;}}}}
  var sf=f3-frameStarts[activeStep],sd=Math.max(1,frameEnds[activeStep]-frameStarts[activeStep]),prog=Math.min(1,sf/sd);var tip=STEPS[activeStep].draw(prog);
  if(!_inFillPhase)drawAnnotations(activeStep-1);
  if(activeStep!==lastHudStep){{createTrackBox(STEPS[activeStep]);lastHudStep=activeStep;}}
  if(trackBox){{var _r3=window._canvasEl?window._canvasEl.getBoundingClientRect():{{left:0,top:0}},_s3=window._hudScale||1,_b3=STEPS[activeStep].bbox,_p3=12*_s3;trackBox.style.left=(Math.max(20,_b3[0])*_s3+_r3.left-_p3)+'px';trackBox.style.top=(Math.max(20,_b3[1])*_s3+_r3.top-_p3)+'px';}}
  if(tip)updatePencilDot(tip[0],tip[1]);
  if(f3>=totalAnimFrames-1){{noLoop();if(pencilDot)pencilDot.style.display='none';if(trackBox)trackBox.remove();trackBox=null;setTimeout(function(){{clearDimLabels();}},6500);setTimeout(function(){{window.location.reload();}},7000);}}
}}
</script>
</body>
</html>"""
    return html


def generate_from_optimized(optimized: dict, extraction: dict, knowledge: dict, building_name: str, ref_image_path: str = None, construction: dict = None) -> str:
    """Generate drawing using ONLY optimized mathematical parameters.
    No edge paths. No contour tracing. Pure compass-and-straightedge geometry."""
    
    params = optimized["params"]
    
    # Build lookup early — needed for dedup
    el_lookup = {e["name"]: e for e in extraction.get("elements", [])}
    
    # === DEDUPLICATION: Remove sub-elements that overlap >70% with parent ===
    def bbox_overlap(a, b):
        """Compute overlap ratio between two elements."""
        ax = a.get("cx", a.get("x", 0) + a.get("w", 0)/2)
        ay = a.get("cy", a.get("y", 0) + a.get("h", 0)/2)
        ar = a.get("radius", max(a.get("w", 50), a.get("h", 50)) / 2)
        bx = b.get("cx", b.get("x", 0) + b.get("w", 0)/2)
        by = b.get("cy", b.get("y", 0) + b.get("h", 0)/2)
        br = b.get("radius", max(b.get("w", 50), b.get("h", 50)) / 2)
        dist = math.sqrt((ax - bx)**2 + (ay - by)**2)
        max_r = max(ar, br)
        if max_r == 0:
            return 0
        return max(0, 1 - dist / max_r)
    
    # Sort by size (largest first) — keep larger, skip smaller duplicates
    sized_names = sorted(params.keys(), key=lambda n: 
        params[n].get("radius", 0) + params[n].get("w", 0) * params[n].get("h", 0) / 10000, reverse=True)
    
    kept = {}
    skipped = set()
    for name in sized_names:
        p = params[name]
        # Skip full-image elements
        el = el_lookup.get(name, {})
        if el.get("area_pct", 0) > 0.35:
            skipped.add(name)
            continue
        # Skip tiny elements
        if el.get("area_pct", 0) < 0.0008:
            skipped.add(name)
            continue
        # Skip wall/panel background elements (they're not architectural features to draw)
        if p.get("type") in ("wall", "panel") and el.get("area_pct", 0) > 0.1:
            skipped.add(name)
            continue
        # Check overlap with already-kept elements
        is_dup = False
        for kept_name, kept_p in kept.items():
            overlap = bbox_overlap(p, kept_p)
            if overlap > 0.5:
                is_dup = True
                break
        if is_dup:
            skipped.add(name)
            continue
        kept[name] = p
    
    print(f"  Deduplication: {len(params)} → {len(kept)} elements ({len(skipped)} skipped)")
    
    # Sort by depth layer (back to front)
    layer_order = {"distant_sky": 0, "background_domes": 1, "mid_facade": 2, "foreground_frame": 3}
    
    # Match params to extraction elements for depth info
    el_lookup = {e["name"]: e for e in extraction.get("elements", [])}
    
    sorted_names = sorted(kept.keys(), key=lambda n: 
        layer_order.get(el_lookup.get(n, {}).get("depth", {}).get("layer", "mid_facade"), 2))
    
    # Build curve definitions and steps
    curve_defs = []
    steps_js = []
    
    for idx, name in enumerate(sorted_names):
        p = kept[name]
        shape = p.get("type", "unknown")
        el = el_lookup.get(name, {})
        depth = el.get("depth", {})
        weight = depth.get("line_weight", 1.5)
        layer = depth.get("layer", "mid_facade")
        
        # Generate the mathematical curve
        if "dome" in shape:
            dome_profile = p.get("best_dome_profile", None)
            if not dome_profile:
                hr = p.get("h_ratio", 1.0)
                if hr > 1.4:
                    dome_profile = "onion"
                elif hr > 0.6:
                    dome_profile = "hemisphere"
                else:
                    dome_profile = "saucer"
            
            hr = p.get("h_ratio", 1.0)
            curve_defs.append(generate_dome_js(name, p["cx"], p["cy"], p["radius"], dome_profile))
            formula = f'{dome_profile} R={p["radius"]:.1f} | {layer} wt={weight}'
            
            bbox = f'[{int(p["cx"]-p["radius"]-10)},{int(p["cy"]-p["radius"]*max(hr,1)-10)},{int(p["cx"]+p["radius"]+10)},{int(p["cy"]+10)}]'
            dur = 1.2
            
            draw = f"""brush.set('2H','#5a5248',{weight});
      if(pts_{name}){{var n=Math.max(3,Math.round(p*pts_{name}.length));
      var sub=pts_{name}.slice(0,n);if(sub.length>=2)brush.spline(sub,0.3);
      var tip=sub[sub.length-1];
      if(p<0.9){{brush.set('2H','#a09888',0.5);brush.line({p["cx"]:.0f},{p["cy"]:.0f},tip[0],tip[1]);brush.circle({p["cx"]:.0f},{p["cy"]:.0f},4);}}}}"""
            
            # Add drum line below dome
            if p["radius"] > 50:
                draw += f"""
      if(p>0.85){{brush.set('2H','#5a5248',{weight*0.8:.1f});
      brush.line({p["cx"]-p["radius"]*0.85:.0f},{p["cy"]:.0f},{p["cx"]+p["radius"]*0.85:.0f},{p["cy"]:.0f});}}"""
        
        elif "arch" in shape:
            profile = "pointed"
            if "horseshoe" in shape:
                profile = "horseshoe"
            elif "ogee" in shape:
                profile = "ogee"
            elif "tudor" in shape:
                profile = "tudor"
            elif "elliptical" in shape:
                profile = "elliptical"
            elif "catenary" in shape:
                profile = "catenary"
            elif "cusped" in shape:
                profile = "cusped"
            elif "parabolic" in shape:
                profile = "parabolic"
            
            rr = p.get("rise_ratio", 0.866)
            curve_defs.append(generate_arch_js(name, p["cx"], p["spring_y"], p["half_span"], profile, rr))
            formula = f'{profile} rise/span={rr:.3f} | {layer} wt={weight}'
            
            rise = p["half_span"] * 2 * rr
            bbox = f'[{int(p["cx"]-p["half_span"]-10)},{int(p["spring_y"]-rise-10)},{int(p["cx"]+p["half_span"]+10)},{int(p["spring_y"]+20)}]'
            dur = 1.4 if p["half_span"] > 100 else 0.8
            
            draw = f"""brush.set('2H','#5a5248',{weight});
      if(pts_{name}){{var n=Math.max(3,Math.round(p*0.8*pts_{name}.length));
      var sub=pts_{name}.slice(0,n);if(sub.length>=2)brush.spline(sub,0.3);
      if(p<0.85){{brush.set('2H','#a09888',0.5);var tip=sub[sub.length-1];brush.line({p["cx"]:.0f},{p["spring_y"]:.0f},tip[0],tip[1]);brush.circle({p["cx"]:.0f},{p["spring_y"]:.0f},4);}}}}
      if(p>0.8){{brush.set('2H','#5a5248',{weight});
      brush.line({p["cx"]-p["half_span"]:.0f},{p["spring_y"]:.0f},{p["cx"]-p["half_span"]:.0f},{p["spring_y"]+p["half_span"]*0.3:.0f});
      brush.line({p["cx"]+p["half_span"]:.0f},{p["spring_y"]:.0f},{p["cx"]+p["half_span"]:.0f},{p["spring_y"]+p["half_span"]*0.3:.0f});}}"""
        
        elif shape in ("wall", "rectangle", "square"):
            formula = f'rectangle {p["w"]:.0f}×{p["h"]:.0f} | {layer} wt={weight}'
            bbox = f'[{int(p["x"]-5)},{int(p["y"]-5)},{int(p["x"]+p["w"]+5)},{int(p["y"]+p["h"]+5)}]'
            dur = 0.6
            curve_defs.append(f"\n    // {name}: rectangle")
            
            draw = f"""brush.set('2H','#5a5248',{weight});
      var rp=p;
      if(rp>0)brush.line({p["x"]:.0f},{p["y"]:.0f},{p["x"]+p["w"]:.0f},{p["y"]:.0f});
      if(rp>0.25)brush.line({p["x"]+p["w"]:.0f},{p["y"]:.0f},{p["x"]+p["w"]:.0f},{p["y"]+p["h"]:.0f});
      if(rp>0.5)brush.line({p["x"]+p["w"]:.0f},{p["y"]+p["h"]:.0f},{p["x"]:.0f},{p["y"]+p["h"]:.0f});
      if(rp>0.75)brush.line({p["x"]:.0f},{p["y"]+p["h"]:.0f},{p["x"]:.0f},{p["y"]:.0f});"""
        elif HAS_PATTERNS and shape in ("muqarnas", "star_polygon", "rosette"):
            # Fill with mathematical pattern
            r = min(bw, bh) / 2
            n_fold = 12 if "star" in shape else 8
            pattern_paths = generate_muqarnas(ccx, ccy, r, n_fold=n_fold, n_tiers=3) if shape == "muqarnas" else generate_star_rosette(ccx, ccy, r, n_points=n_fold)
            pattern_paths = pattern_paths[:60]
            pkey = f"pattern_{name}"
            edge_data_all[pkey] = pattern_paths
            formula = f'{shape} {n_fold}-fold | {layer} wt={weight}'
            bbox = f'[{int(ccx-r-5)},{int(ccy-r-5)},{int(ccx+r+5)},{int(ccy+r+5)}]'
            dur = 1.5
            draw = f"""brush.set('2H','#6a6258',{weight*0.7:.1f});
      var pat=_data['{pkey}'];if(pat){{var nP=Math.max(1,Math.round(p*pat.length));for(var pi=0;pi<nP;pi++){{var path=pat[pi];if(path&&path.length>=2){{var pts=[];for(var j=0;j<path.length;j++)pts.push([path[j][0],path[j][1],0.3]);if(pts.length>=2)brush.spline(pts,0.3);}}}}}}"""
        
        elif HAS_PATTERNS and shape == "mashrabiya":
            pattern_paths = generate_mashrabiya(bx, by, bw, bh, "hexagonal", max(4, int(bw / 30)))
            pattern_paths = pattern_paths[:80]
            pkey = f"pattern_{name}"
            edge_data_all[pkey] = pattern_paths
            formula = f'mashrabiya lattice | {layer} wt={weight}'
            bbox = f'[{int(bx-5)},{int(by-5)},{int(bx+bw+5)},{int(by+bh+5)}]'
            dur = 1.2
            draw = f"""brush.set('2H','#7a7268',{weight*0.6:.1f});
      var pat=_data['{pkey}'];if(pat){{var nP=Math.max(1,Math.round(p*pat.length));for(var pi=0;pi<nP;pi++){{var path=pat[pi];if(path&&path.length>=2){{var pts=[];for(var j=0;j<path.length;j++)pts.push([path[j][0],path[j][1],0.3]);if(pts.length>=2)brush.spline(pts,0.3);}}}}}}"""
        
        elif HAS_PATTERNS and shape == "openwork":
            # Openwork gets arabesque fill
            pattern_paths = generate_arabesque(bx, by, bw, bh, max(3, int(bw / 60)))
            pattern_paths = pattern_paths[:40]
            pkey = f"pattern_{name}"
            edge_data_all[pkey] = pattern_paths
            formula = f'openwork arabesque | {layer} wt={weight}'
            bbox = f'[{int(bx-5)},{int(by-5)},{int(bx+bw+5)},{int(by+bh+5)}]'
            dur = 1.0
            draw = f"""brush.set('2H','#8a8278',{weight*0.5:.1f});
      var pat=_data['{pkey}'];if(pat){{var nP=Math.max(1,Math.round(p*pat.length));for(var pi=0;pi<nP;pi++){{var path=pat[pi];if(path&&path.length>=2){{var pts=[];for(var j=0;j<path.length;j++)pts.push([path[j][0],path[j][1],0.3]);if(pts.length>=2)brush.spline(pts,0.3);}}}}}}"""
        
        else:
            continue
        
        ccx = p.get("cx", p.get("x", 540) + p.get("w", 100)/2)
        ccy = p.get("cy", p.get("y", 960) + p.get("h", 100)/2)
        
        steps_js.append(f"""  {{name:'{name.upper().replace("_"," ")}',formula:'{formula}',dur:{dur:.1f},
   bbox:{bbox},
   draw:function(p){{
      {draw}
      return[{ccx:.0f},{ccy:.0f}];
   }}}},""")
    
    # === INFERRED CONSTRUCTION (columns, drums, finials, cornices) ===
    cons = construction.get("construction") if construction else optimized.get("construction")
    if cons:
        g = cons.get("ground", {})
        if g:
            steps_js.append(f"""  {{name:'GROUND LINE',formula:'base',dur:0.3,bbox:[{int(g.get("x1",20))},{int(g.get("y",1800)-5)},{int(g.get("x2",1060))},{int(g.get("y",1800)+5)}],draw:function(p){{brush.set('2H','#5a5248',2.0);brush.line({g.get("x1",20):.0f},{g.get("y",1800):.0f},{g.get("x1",20):.0f}+({g.get("x2",1060):.0f}-{g.get("x1",20):.0f})*p,{g.get("y",1800):.0f});return[{g.get("x2",1060):.0f},{g.get("y",1800):.0f}];}}}},""")
        for ci, col in enumerate(cons.get("columns", [])):
            wt = col.get("line_weight", 1.5); cw = col.get("width", 6)
            steps_js.append(f"""  {{name:'COL {ci+1}',formula:'column',dur:0.3,bbox:[{int(col["x"]-cw-2)},{int(col["top_y"])},{int(col["x"]+cw+2)},{int(col["bot_y"])}],draw:function(p){{brush.set('2H','#5a5248',{wt});var ey={col["bot_y"]:.0f}-({col["bot_y"]:.0f}-{col["top_y"]:.0f})*p;brush.line({col["x"]-cw/2:.0f},{col["bot_y"]:.0f},{col["x"]-cw/2:.0f},ey);brush.line({col["x"]+cw/2:.0f},{col["bot_y"]:.0f},{col["x"]+cw/2:.0f},ey);return[{col["x"]:.0f},ey];}}}},""")
        for di, drum in enumerate(cons.get("drums", [])):
            wt=drum.get("line_weight",1.2);dw=drum.get("width",100);nw=drum.get("n_windows",3)
            steps_js.append(f"""  {{name:'DRUM {di+1}',formula:'{nw} windows',dur:0.5,bbox:[{int(drum["cx"]-dw/2)},{int(drum["top_y"])},{int(drum["cx"]+dw/2)},{int(drum["bot_y"])}],draw:function(p){{brush.set('2H','#5a5248',{wt});var dl={drum["cx"]-dw/2:.0f},dr={drum["cx"]+dw/2:.0f},dt={drum["top_y"]:.0f},db={drum["bot_y"]:.0f};if(p>0)brush.line(dl,dt,dr,dt);if(p>0.25)brush.line(dr,dt,dr,db);if(p>0.5)brush.line(dr,db,dl,db);if(p>0.75)brush.line(dl,db,dl,dt);return[{drum["cx"]:.0f},{drum["bot_y"]:.0f}];}}}},""")
        for fi, fin in enumerate(cons.get("finials", [])):
            wt=fin.get("line_weight",0.8)
            fin_range = max(1, fin["bot_y"] - fin["top_y"])
            fin_top_safe = max(20, fin["top_y"])
            steps_js.append(f"""  {{name:'FINIAL {fi+1}',formula:'spire+crescent',dur:0.3,bbox:[{int(fin["cx"]-12)},{int(max(5,fin["top_y"]-5))},{int(fin["cx"]+12)},{int(fin["bot_y"]+5)}],draw:function(p){{brush.set('2H','#5a5248',{wt});var fRange={fin_range:.0f};brush.line({fin["cx"]:.0f},{fin["bot_y"]:.0f},{fin["cx"]:.0f},{fin["bot_y"]:.0f}-fRange*Math.min(1,p*2));if(p>0.5){{var cR=7,cPts=[];for(var i=0;i<=12;i++){{var a=-Math.PI/2+Math.PI*(i/12);cPts.push([{fin["cx"]:.0f}+3+cR*Math.cos(a),{fin_top_safe:.0f}+10+cR*Math.sin(a),0.4]);}}if(cPts.length>=2)brush.spline(cPts,0.3);}}return[{fin["cx"]:.0f},{fin_top_safe:.0f}];}}}},""")
        for ki, cor in enumerate(cons.get("cornices", [])):
            wt=cor.get("line_weight",0.8)
            steps_js.append(f"""  {{name:'CORNICE',formula:'y={cor["y"]:.0f}',dur:0.2,bbox:[{int(cor["x1"])},{int(cor["y"]-3)},{int(cor["x2"])},{int(cor["y"]+3)}],draw:function(p){{brush.set('2H','#8a8278',{wt});brush.line({cor["x1"]:.0f},{cor["y"]:.0f},{cor["x1"]:.0f}+({cor["x2"]:.0f}-{cor["x1"]:.0f})*p,{cor["y"]:.0f});return[{cor["x2"]:.0f},{cor["y"]:.0f}];}}}},""")
    
    # Wash steps
    wash_start = len(steps_js)
    wash_colors = {"distant_sky": ("#87CEEB", 30), "background_domes": ("#D4C5B0", 20), 
                   "mid_facade": ("#4A5568", 15), "foreground_frame": ("#C4B89A", 25)}
    
    for idx, name in enumerate(sorted_names):
        p = params[name]
        el = el_lookup.get(name, {})
        layer = el.get("depth", {}).get("layer", "mid_facade")
        color, alpha = wash_colors.get(layer, ("#888888", 15))
        
        # Only wash elements with significant area
        if "dome" in p.get("type", "") and p.get("radius", 0) > 40:
            ccx, ccy = p["cx"], p["cy"]
            r = p["radius"]
            steps_js.append(f"""  {{name:'{name.upper().replace("_"," ")} WASH',formula:'{color} α={alpha}',dur:0.05,
   bbox:[{int(ccx-r)},{int(ccy-r)},{int(ccx+r)},{int(ccy+r)}],
   draw:function(p){{if(p>=0){{randomSeed({7000+idx});brush.noStroke();brush.fill('{color}',{alpha});brush.fillBleed(0.001);brush.fillTexture(0,0);
   if(pts_{name})brush.polygon(pts_{name}.map(function(pt){{return[pt[0],pt[1]]}}));brush.noFill();}}return[{ccx:.0f},{ccy:.0f}];}}}},""")
        
        elif p.get("type") in ("wall", "rectangle", "square"):
            steps_js.append(f"""  {{name:'{name.upper().replace("_"," ")} WASH',formula:'{color} α={alpha}',dur:0.05,
   bbox:[{int(p["x"])},{int(p["y"])},{int(p["x"]+p["w"])},{int(p["y"]+p["h"])}],
   draw:function(p){{if(p>=0){{randomSeed({7000+idx});brush.noStroke();brush.fill('{color}',{alpha});brush.fillBleed(0.001);brush.fillTexture(0,0);
   brush.rect({p["x"]:.0f},{p["y"]:.0f},{p["w"]:.0f},{p["h"]:.0f});brush.noFill();}}return[{p["x"]+p["w"]/2:.0f},{p["y"]+p["h"]/2:.0f}];}}}},""")
    
    all_steps = "\n".join(steps_js)
    curve_defs_js = "\n".join(curve_defs)
    
    # Encode reference image if provided
    ref_img_data = ""
    ref_img_css = ""
    ref_img_div = ""
    if ref_image_path and Path(ref_image_path).exists():
        import base64
        with open(ref_image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = Path(ref_image_path).suffix.lower().replace(".", "")
        mime = "jpeg" if ext in ("jpg", "jpeg") else ext
        ref_img_css = f"""#ref-overlay{{position:fixed;top:0;left:0;width:100%;height:100%;display:flex;justify-content:center;align-items:flex-start;pointer-events:none;z-index:10}}
#ref-overlay img{{max-height:100vh;max-width:calc(100vh*9/16);opacity:0.40;object-fit:contain;mix-blend-mode:multiply}}"""
        ref_img_div = f'<div id="ref-overlay"><img src="data:image/{mime};base64,{b64}"></div>'
        print(f"  Reference overlay: {ref_image_path} (25% opacity)")
    
    html = f"""<!-- بسم الله الرحمن الرحيم -->
<!-- Generated by Traced Pipeline — {building_name} -->
<!-- OPTIMIZED: Pure mathematical curves from gradient descent -->
<!-- No edge tracing. Compass-and-straightedge geometry only. -->
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/p5@2.0.3/lib/p5.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/p5.brush@2.0.2-beta"></script>
<style>
*{{margin:0}}body{{background:#f2eada;overflow:hidden;margin:0;padding:0;display:flex;justify-content:center;align-items:flex-start}}canvas{{object-fit:contain!important;display:block;max-height:100vh;max-width:calc(100vh*9/16)}}
{ref_img_css}
#hud{{position:fixed;top:0;left:0;width:100vw;height:100%;pointer-events:none;font-family:'SF Mono','Fira Code',monospace;overflow:hidden}}
.track-box{{position:absolute;transition:opacity 0.3s,left 0.4s,top 0.4s,width 0.4s,height 0.4s;min-width:60px;min-height:60px;overflow:visible}}
.track-box::before,.track-box::after,.track-box .bl::before,.track-box .br::before{{content:'';position:absolute;width:30px;height:30px;border-color:#5a5248;border-style:solid;border-width:0}}
.track-box::before{{top:-2px;left:-2px;border-top-width:1.5px;border-left-width:1.5px}}.track-box::after{{top:-2px;right:-2px;border-top-width:1.5px;border-right-width:1.5px}}
.track-box .bl::before{{bottom:-2px;left:-2px;border-bottom-width:1.5px;border-left-width:1.5px;position:absolute}}.track-box .br::before{{bottom:-2px;right:-2px;border-bottom-width:1.5px;border-right-width:1.5px;position:absolute}}
.corner-circ{{position:absolute;width:4px;height:4px;border:1px solid #5a5248;border-radius:50%}}.corner-circ.tl{{top:-2px;left:-2px}}.corner-circ.tr{{top:-2px;right:-2px}}.corner-circ.bl2{{bottom:-2px;left:-2px}}.corner-circ.br2{{bottom:-2px;right:-2px}}
.hud-info{{position:absolute;bottom:100%;left:0;margin-bottom:4px;background:rgba(242,234,218,0.88);border:1px solid rgba(90,82,72,0.3);border-radius:4px;padding:4px 8px;white-space:nowrap}}
.hud-label{{color:#3a3430;font-weight:500;letter-spacing:0.3px;text-transform:uppercase;margin-bottom:3px}}.hud-formula{{color:#5a5248;letter-spacing:0.3px;margin-bottom:2px}}.hud-coords{{color:#8a8278;letter-spacing:0.3px}}
#pencil-dot{{position:absolute;width:14px;height:14px;pointer-events:none;z-index:15}}#pencil-dot::before{{content:'';position:absolute;left:6px;top:0;width:1px;height:14px;background:#5a5248}}#pencil-dot::after{{content:'';position:absolute;top:6px;left:0;width:14px;height:1px;background:#5a5248}}
#pencil-dot .ch-circle{{position:absolute;top:4px;left:4px;width:6px;height:6px;border:1px solid #5a5248;border-radius:50%}}
</style>
</head>
<body>
{ref_img_div}
<div id="hud"><div id="pencil-dot" style="display:none"><div class="ch-circle"></div></div></div>
<script>
var W={CANVAS_W},H={CANVAS_H},cx=W/2;
{curve_defs_js}

var ANNOT_FONT=13,_dimLabels=[],_lastDimStep=-1;
function clearDimLabels(){{for(var i=0;i<_dimLabels.length;i++)_dimLabels[i].remove();_dimLabels=[];}}
function addDimText(x,y,t,a){{for(var i=0;i<_dimLabels.length;i++)if(_dimLabels[i]._t===t)return;var el=document.createElement('div');el._t=t;var s=window._hudScale||1,r=window._canvasEl?window._canvasEl.getBoundingClientRect():{{left:0,top:0}};el.style.cssText='position:absolute;color:#8a8278;font-family:SF Mono,Fira Code,monospace;font-size:'+Math.max(9,Math.round(ANNOT_FONT*s))+'px;white-space:nowrap;pointer-events:none;';el.style.left=(x*s+r.left)+'px';el.style.top=(y*s+r.top)+'px';if(a==='center')el.style.transform='translateX(-50%)';el.textContent=t;document.getElementById('hud').appendChild(el);_dimLabels.push(el);}}
function drawAnnotations(step){{if(step!==_lastDimStep){{clearDimLabels();_lastDimStep=step;}}if(step>=0)addDimText(cx,H-40,'Traced | {building_name} | optimized mathematical geometry','center');}}

var FPS=30;
var STEPS=[
{all_steps}
];
var PAUSE_S=0.14,WASH_START={wash_start};
var frameStarts=[],frameEnds=[],f2=0;
for(var i=0;i<STEPS.length;i++){{frameStarts.push(Math.round(f2));f2+=STEPS[i].dur*FPS;frameEnds.push(Math.round(f2));if(i<STEPS.length-1&&i<WASH_START-1)f2+=PAUSE_S*FPS;}}
var totalAnimFrames=Math.round(f2);
var trackBox=null,pencilDot=null;
function createTrackBox(step){{var hud=document.getElementById('hud');if(trackBox)trackBox.remove();var box=document.createElement('div');box.className='track-box';var s=window._hudScale||1,r=window._canvasEl?window._canvasEl.getBoundingClientRect():{{left:0,top:0}};var b=step.bbox,pad=12*s;var cb0=Math.max(20,b[0]),cb1=Math.max(20,b[1]),cb2=Math.min(W-20,b[2]),cb3=Math.min(H-20,b[3]);box.style.left=(cb0*s+r.left-pad)+'px';box.style.top=(cb1*s+r.top-pad)+'px';box.style.width=((cb2-cb0)*s+pad*2)+'px';box.style.height=((cb3-cb1)*s+pad*2)+'px';['tl','tr','bl2','br2'].forEach(function(c){{var ci=document.createElement('div');ci.className='corner-circ '+c;box.appendChild(ci);}});var bl=document.createElement('div');bl.className='bl';bl.style.cssText='position:absolute;bottom:0;left:0;width:100%;height:100%;pointer-events:none;';box.appendChild(bl);var br=document.createElement('div');br.className='br';br.style.cssText='position:absolute;bottom:0;right:0;width:100%;height:100%;pointer-events:none;';box.appendChild(br);var _fs=Math.max(9,Math.round(ANNOT_FONT*(window._hudScale||1)))+'px';var info=document.createElement('div');info.className='hud-info';if((cb1*s+r.top-pad)<80){{info.style.bottom='auto';info.style.top='6px';info.style.marginBottom='0';}}var lbl=document.createElement('div');lbl.className='hud-label';lbl.style.fontSize=_fs;lbl.textContent=step.name;info.appendChild(lbl);var frm=document.createElement('div');frm.className='hud-formula';frm.style.fontSize=_fs;frm.textContent=step.formula;info.appendChild(frm);var crd=document.createElement('div');crd.className='hud-coords';crd.style.fontSize=_fs;crd.id='live-coords';crd.textContent='x:\\u2014 y:\\u2014';info.appendChild(crd);box.appendChild(info);hud.appendChild(box);trackBox=box;}}
function updatePencilDot(x,y){{if(!pencilDot)pencilDot=document.getElementById('pencil-dot');pencilDot.style.display='block';var s=window._hudScale||1,r=window._canvasEl?window._canvasEl.getBoundingClientRect():{{left:0,top:0}};pencilDot.style.left=(x*s+r.left-7)+'px';pencilDot.style.top=(y*s+r.top-7)+'px';var el=document.getElementById('live-coords');if(el)el.textContent='x:'+Math.round(x)+' y:'+Math.round(y);}}
var lastHudStep=-1,_inFillPhase=false;
function setup(){{var c=createCanvas(W,H,WEBGL);pixelDensity(1);var vh=window.innerHeight;var cw=Math.min(window.innerWidth,vh*9/16);c.elt.style.width=cw+'px';c.elt.style.height=(cw/(W/H))+'px';c.elt.style.display='block';c.elt.style.margin='0';brush.load();frameRate(FPS);window._hudScale=c.elt.getBoundingClientRect().width/W;window._canvasEl=c.elt;window.addEventListener('resize',function(){{window._hudScale=c.elt.getBoundingClientRect().width/W;_lastDimStep=-1;}});}}
function draw(){{
  translate(-width/2,-height/2);var f3=frameCount-1;if(f3>=totalAnimFrames)f3=totalAnimFrames-1;
  var activeStep=0;for(var i=STEPS.length-1;i>=0;i--){{if(f3>=frameStarts[i]){{activeStep=i;break;}}}}
  if(activeStep<WASH_START){{_inFillPhase=false;background(242,234,218);for(var i=0;i<activeStep;i++)STEPS[i].draw(1);}}
  else{{if(!_inFillPhase){{background(242,234,218);for(var i=0;i<WASH_START;i++)STEPS[i].draw(1);drawAnnotations(WASH_START-1);_inFillPhase=true;}}}}
  var sf=f3-frameStarts[activeStep],sd=Math.max(1,frameEnds[activeStep]-frameStarts[activeStep]),prog=Math.min(1,sf/sd);var tip=STEPS[activeStep].draw(prog);
  if(!_inFillPhase)drawAnnotations(activeStep-1);
  if(activeStep!==lastHudStep){{createTrackBox(STEPS[activeStep]);lastHudStep=activeStep;}}
  if(trackBox){{var _r3=window._canvasEl?window._canvasEl.getBoundingClientRect():{{left:0,top:0}},_s3=window._hudScale||1,_b3=STEPS[activeStep].bbox,_p3=12*_s3;trackBox.style.left=(Math.max(20,_b3[0])*_s3+_r3.left-_p3)+'px';trackBox.style.top=(Math.max(20,_b3[1])*_s3+_r3.top-_p3)+'px';}}
  if(tip)updatePencilDot(tip[0],tip[1]);
  if(f3>=totalAnimFrames-1){{noLoop();if(pencilDot)pencilDot.style.display='none';if(trackBox)trackBox.remove();trackBox=null;setTimeout(function(){{clearDimLabels();}},6500);setTimeout(function(){{window.location.reload();}},7000);}}
}}
</script>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description="Traced: Generate drawing from extraction + knowledge")
    parser.add_argument("--extraction", required=True)
    parser.add_argument("--optimized", default=None, help="Optimized params from optimize.py")
    parser.add_argument("--knowledge", default=None)
    parser.add_argument("--output", default="drawing.html")
    parser.add_argument("--name", default="Building")
    parser.add_argument("--ref-image", default=None, help="Reference image for onion skin overlay (dev mode)")
    parser.add_argument("--construction", default=None, help="Construction JSON from construct.py")
    args = parser.parse_args()
    
    extraction = json.loads(Path(args.extraction).read_text())
    knowledge = None
    if args.knowledge and Path(args.knowledge).exists():
        knowledge = json.loads(Path(args.knowledge).read_text())
    elif extraction.get("knowledge"):
        knowledge = extraction["knowledge"]
    
    optimized = None
    if args.optimized and Path(args.optimized).exists():
        optimized = json.loads(Path(args.optimized).read_text())
        print(f"Using optimized parameters from {args.optimized}")
    
    construction = None
    if args.construction and Path(args.construction).exists():
        construction = json.loads(Path(args.construction).read_text())
        print(f"Using construction data: {construction.get('construction', {}).get('total_inferred', 0)} inferred elements")
    
    if optimized:
        html = generate_from_optimized(optimized, extraction, knowledge, args.name, args.ref_image, construction)
    else:
        html = generate_html(extraction, knowledge, args.name)
    
    Path(args.output).write_text(html)
    
    elements = [e for e in extraction["elements"] if 0.003 < e.get("area_pct", 0) < 0.4]
    
    print(f"Generated {args.output}")
    print(f"  Mode: {'OPTIMIZED (pure math curves)' if optimized else 'standard (contours + edges)'}")
    print(f"  Elements: {len(elements)}")
    print(f"  Aspect ratio: preserved ({extraction['image_size']['w']}×{extraction['image_size']['h']} → {CANVAS_W}×{CANVAS_H})")
    if knowledge:
        print(f"  Knowledge applied: {knowledge.get('style_influences', [])}")


if __name__ == "__main__":
    main()
