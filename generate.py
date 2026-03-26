#!/usr/bin/env python3
"""
Traced Pipeline — Stage 4: Code Generator
Reads SAM 2 extraction JSON and generates a p5.brush drawing HTML
using ACTUAL contour data from masks + edge paths.

Usage:
    python generate.py --extraction szm-v5.json --output szm.html
    python generate.py --extraction szm-v5.json --output szm.html --name "Sheikh Zayed Grand Mosque"
"""

import argparse
import json
import math
from pathlib import Path

# Canvas dimensions (9:16 portrait, matching dome.html template)
CANVAS_W = 1080
CANVAS_H = 1920


def scale_contour(contour_pts, src_w, src_h, dst_w, dst_h):
    """Scale contour points from source image coords to canvas coords."""
    sx = dst_w / src_w
    sy = dst_h / src_h
    return [[round(p[0] * sx, 1), round(p[1] * sy, 1)] for p in contour_pts]


def scale_edge_paths(edge_paths, bbox, src_w, src_h, dst_w, dst_h):
    """Scale normalized edge paths (0-1 within bbox) to canvas coords."""
    sx = dst_w / src_w
    sy = dst_h / src_h
    bx = bbox["x"] * sx
    by = bbox["y"] * sy
    bw = bbox["w"] * sx
    bh = bbox["h"] * sy
    
    scaled = []
    for path in edge_paths:
        scaled_path = []
        for p in path:
            scaled_path.append([
                round(bx + p[0] * bw, 1),
                round(by + p[1] * bh, 1)
            ])
        scaled.append(scaled_path)
    return scaled


def generate_html(extraction: dict, building_name: str = "Building") -> str:
    """Generate complete p5.brush drawing HTML from extraction data."""
    
    elements = extraction["elements"]
    img_w = extraction["image_size"]["w"]
    img_h = extraction["image_size"]["h"]
    
    # Sort elements by depth (back to front)
    layer_order = {"distant_sky": 0, "background_domes": 1, "mid_facade": 2, "foreground_frame": 3}
    sorted_elements = sorted(elements, key=lambda e: layer_order.get(e.get("depth", {}).get("layer", "mid_facade"), 2))
    
    # Separate pencil steps from wash steps
    pencil_elements = [e for e in sorted_elements if e.get("area_pct", 0) > 0.003]
    
    # Build contour data as JS arrays
    contour_data = {}
    edge_data = {}
    
    for el in pencil_elements:
        name = el["name"]
        primitives = el.get("primitives", {})
        
        # Scale mask contour to canvas
        contour = primitives.get("contour", [])
        if contour:
            scaled = scale_contour(contour, img_w, img_h, CANVAS_W, CANVAS_H)
            contour_data[name] = scaled
        
        # Scale edge paths to canvas
        edges = el.get("edge_paths", [])
        bbox = primitives.get("bbox", {})
        if edges and bbox:
            scaled_edges = scale_edge_paths(edges, bbox, img_w, img_h, CANVAS_W, CANVAS_H)
            edge_data[name] = scaled_edges
    
    # Generate STEPS array entries
    steps_js = []
    for idx, el in enumerate(pencil_elements):
        name = el["name"]
        shape_type = el.get("shape", {}).get("type", "unknown")
        depth_info = el.get("depth", {})
        layer = depth_info.get("layer", "mid_facade")
        weight = depth_info.get("line_weight", 1.5)
        detail_weight = max(0.4, weight - 0.8)
        
        # Position and size on canvas
        pos = el.get("position_pct", {})
        size = el.get("size_pct", {})
        
        bbox_js = f"[{int(pos.get('x',0.5)*CANVAS_W - size.get('w',0.1)*CANVAS_W/2)}, {int(pos.get('y',0.5)*CANVAS_H - size.get('h',0.1)*CANVAS_H/2)}, {int(pos.get('x',0.5)*CANVAS_W + size.get('w',0.1)*CANVAS_W/2)}, {int(pos.get('y',0.5)*CANVAS_H + size.get('h',0.1)*CANVAS_H/2)}]"
        
        # Duration based on contour complexity
        n_contour = len(contour_data.get(name, []))
        n_edges = sum(len(p) for p in edge_data.get(name, []))
        dur = max(0.3, min(2.0, (n_contour + n_edges) / 80))
        
        # Arch geometry info for formula
        arch_geo = el.get("primitives", {}).get("arch_geometry", {})
        dome_geo = el.get("primitives", {}).get("dome_geometry", {})
        formula = f"{shape_type} | {layer} (wt={weight})"
        if arch_geo:
            formula = f"rise/span={arch_geo.get('rise_to_span', 0):.3f} ({arch_geo.get('profile_type', '')}) | wt={weight}"
        if dome_geo:
            formula = f"h/d={dome_geo.get('height_to_diameter', 0):.3f} ({dome_geo.get('profile_type', '')}) | wt={weight}"
        
        step_js = f"""  // {idx}: {name} — {layer}
  {{name:'{name.upper().replace('_',' ')}', formula:'{formula}', dur:{dur:.2f},
   bbox:{bbox_js},
   draw:function(p){{
     // Outer contour from SAM mask
     var contour = _contours['{name}'];
     if(contour && contour.length >= 3) {{
       brush.set('2H','#5a5248',{weight});
       var n = Math.max(3, Math.round(p * contour.length));
       var pts = [];
       for(var i=0; i<n; i++) pts.push([contour[i][0], contour[i][1], 0.3+0.7*Math.sin(Math.PI*i/contour.length)]);
       if(pts.length >= 2) brush.spline(pts, 0.3);
     }}
     // Inner detail from edge paths
     var edges = _edges['{name}'];
     if(edges && p > 0.3) {{
       var ep = (p - 0.3) / 0.7;
       var nPaths = Math.max(1, Math.round(ep * edges.length));
       brush.set('2H','#8a8278',{detail_weight});
       for(var ei=0; ei<nPaths; ei++) {{
         var path = edges[ei];
         if(path.length >= 2) {{
           var nPts = Math.max(2, Math.round(ep * path.length));
           var epts = [];
           for(var pi=0; pi<nPts && pi<path.length; pi++) epts.push([path[pi][0], path[pi][1], 0.3]);
           if(epts.length >= 2) brush.spline(epts, 0.3);
         }}
       }}
     }}
     var last = contour ? contour[Math.min(Math.round(p*(contour.length-1)), contour.length-1)] : [{int(pos.get('x',0.5)*CANVAS_W)},{int(pos.get('y',0.5)*CANVAS_H)}];
     return last;
   }}}},
"""
        steps_js.append(step_js)
    
    # Build wash steps for elements with area > 5%
    wash_js = []
    wash_colors = {
        "distant_sky": ("#87CEEB", 30),
        "background_domes": ("#D4C5B0", 20),
        "mid_facade": ("#4A5568", 15),
        "foreground_frame": ("#C4B89A", 25),
    }
    
    for idx, el in enumerate(pencil_elements):
        if el.get("area_pct", 0) < 0.02:
            continue
        name = el["name"]
        layer = el.get("depth", {}).get("layer", "mid_facade")
        color, alpha = wash_colors.get(layer, ("#888888", 15))
        
        wash_js.append(f"""  {{name:'{name.upper().replace('_',' ')} WASH', formula:'{color} α={alpha}', dur:0.05,
   bbox:{bbox_js},
   draw:function(p){{
     if(p>=0){{
       var contour = _contours['{name}'];
       if(contour && contour.length >= 3) {{
         randomSeed({5000 + idx});
         brush.noStroke();
         brush.fill('{color}',{alpha});
         brush.fillBleed(0.001);brush.fillTexture(0,0);
         brush.polygon(contour);
         brush.noFill();
       }}
     }}
     return[cx,{int(pos.get('y',0.5)*CANVAS_H)}];
   }}}},
""")
    
    wash_start = len(steps_js)
    all_steps = "\n".join(steps_js) + "\n  // === WASHES ===\n" + "\n".join(wash_js)
    
    # Serialize contour and edge data as JS
    contour_js = json.dumps(contour_data)
    edge_js = json.dumps(edge_data)
    
    # Build full HTML
    html = f"""<!-- بسم الله الرحمن الرحيم -->
<!-- Generated by Traced Pipeline — {building_name} -->
<!-- SAM 2 contours + Canny edges + Depth Anything V2 layers -->
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/p5@2.0.3/lib/p5.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/p5.brush@2.0.2-beta"></script>
<style>
* {{ margin:0; }}
body {{ background:#f2eada; overflow:hidden; margin:0; padding:0; }}
canvas {{ object-fit:contain !important; display:block; }}
#hud {{ position:fixed; top:0; left:0; width:100vw; height:100%; pointer-events:none; font-family:'SF Mono','Fira Code',monospace; overflow:hidden; }}
.track-box {{ position:absolute; transition: opacity 0.3s ease, left 0.4s ease, top 0.4s ease, width 0.4s ease, height 0.4s ease; min-width:60px; min-height:60px; overflow:visible; }}
.track-box::before, .track-box::after, .track-box .bl::before, .track-box .br::before {{ content:''; position:absolute; width:30px; height:30px; border-color:#5a5248; border-style:solid; border-width:0; }}
.track-box::before {{ top:-2px; left:-2px; border-top-width:1.5px; border-left-width:1.5px; }}
.track-box::after {{ top:-2px; right:-2px; border-top-width:1.5px; border-right-width:1.5px; }}
.track-box .bl::before {{ bottom:-2px; left:-2px; border-bottom-width:1.5px; border-left-width:1.5px; position:absolute; }}
.track-box .br::before {{ bottom:-2px; right:-2px; border-bottom-width:1.5px; border-right-width:1.5px; position:absolute; }}
.corner-circ {{ position:absolute; width:4px; height:4px; border:1px solid #5a5248; border-radius:50%; }}
.corner-circ.tl {{ top:-2px; left:-2px; }} .corner-circ.tr {{ top:-2px; right:-2px; }}
.corner-circ.bl2 {{ bottom:-2px; left:-2px; }} .corner-circ.br2 {{ bottom:-2px; right:-2px; }}
.hud-info {{ position:absolute; bottom:100%; left:0; margin-bottom:4px; background:rgba(242,234,218,0.88); border:1px solid rgba(90,82,72,0.3); border-radius:4px; padding:4px 8px; white-space:nowrap; }}
.hud-label {{ color:#3a3430; font-weight:500; letter-spacing:0.3px; text-transform:uppercase; margin-bottom:3px; }}
.hud-formula {{ color:#5a5248; font-weight:normal; letter-spacing:0.3px; margin-bottom:2px; }}
.hud-coords {{ color:#8a8278; font-weight:normal; letter-spacing:0.3px; }}
.track-box.complete {{ animation: pulse-complete 0.5s ease; }}
@keyframes pulse-complete {{ 0%{{filter:drop-shadow(0 0 4px rgba(90,82,72,0.6))}} 50%{{filter:drop-shadow(0 0 12px rgba(90,82,72,0.8))}} 100%{{filter:drop-shadow(0 0 2px rgba(90,82,72,0.1))}} }}
#pencil-dot {{ position:absolute; width:14px; height:14px; pointer-events:none; z-index:15; }}
#pencil-dot::before {{ content:''; position:absolute; left:6px; top:0; width:1px; height:14px; background:#5a5248; }}
#pencil-dot::after {{ content:''; position:absolute; top:6px; left:0; width:14px; height:1px; background:#5a5248; }}
#pencil-dot .ch-circle {{ position:absolute; top:4px; left:4px; width:6px; height:6px; border:1px solid #5a5248; border-radius:50%; }}
</style>
</head>
<body>
<div id="hud"><div id="pencil-dot" style="display:none"><div class="ch-circle"></div></div></div>
<script>
var W = {CANVAS_W}, H = {CANVAS_H}, cx = W/2;
var PHI = (1 + Math.sqrt(5)) / 2;

// SAM 2 mask contours (scaled to canvas)
var _contours = {contour_js};

// Canny edge paths per element (scaled to canvas)
var _edges = {edge_js};

var ANNOT_FONT = 13;
var _dimLabels=[],_lastDimStep=-1;
function clearDimLabels(){{for(var i=0;i<_dimLabels.length;i++)_dimLabels[i].remove();_dimLabels=[];}}
function addDimText(x,y,t,a){{for(var i=0;i<_dimLabels.length;i++)if(_dimLabels[i]._t===t)return;var el=document.createElement('div');el._t=t;var s=window._hudScale||1,r=window._canvasEl?window._canvasEl.getBoundingClientRect():{{left:0,top:0}};el.style.cssText='position:absolute;color:#8a8278;font-family:SF Mono,Fira Code,monospace;font-size:'+Math.max(9,Math.round(ANNOT_FONT*s))+'px;white-space:nowrap;pointer-events:none;letter-spacing:0.3px;';el.style.left=(x*s+r.left)+'px';el.style.top=(y*s+r.top)+'px';if(a==='center')el.style.transform='translateX(-50%)';el.textContent=t;document.getElementById('hud').appendChild(el);_dimLabels.push(el);}}
function drawAnnotations(step){{if(step!==_lastDimStep){{clearDimLabels();_lastDimStep=step;}}if(step>=0)addDimText(cx,H-40,'Traced | {building_name} | SAM\\u2082 contours + edges','center');}}

var FPS = 30;
var STEPS = [
{all_steps}
];

var PAUSE_S=0.12, WASH_START={wash_start};
var frameStarts=[],frameEnds=[],f2=0;
for(var i=0;i<STEPS.length;i++){{frameStarts.push(Math.round(f2));f2+=STEPS[i].dur*FPS;frameEnds.push(Math.round(f2));if(i<STEPS.length-1&&i<WASH_START-1)f2+=PAUSE_S*FPS;}}
var totalAnimFrames=Math.round(f2);

var trackBox=null,pencilDot=null;
function createTrackBox(step){{var hud=document.getElementById('hud');if(trackBox)trackBox.remove();var box=document.createElement('div');box.className='track-box';var s=window._hudScale||1,r=window._canvasEl?window._canvasEl.getBoundingClientRect():{{left:0,top:0}};var b=step.bbox,pad=12*s;var cb0=Math.max(20,b[0]),cb1=Math.max(20,b[1]),cb2=Math.min(W-20,b[2]),cb3=Math.min(H-20,b[3]);box.style.left=(cb0*s+r.left-pad)+'px';box.style.top=(cb1*s+r.top-pad)+'px';box.style.width=((cb2-cb0)*s+pad*2)+'px';box.style.height=((cb3-cb1)*s+pad*2)+'px';['tl','tr','bl2','br2'].forEach(function(c){{var ci=document.createElement('div');ci.className='corner-circ '+c;box.appendChild(ci);}});var bl=document.createElement('div');bl.className='bl';bl.style.cssText='position:absolute;bottom:0;left:0;width:100%;height:100%;pointer-events:none;';box.appendChild(bl);var br=document.createElement('div');br.className='br';br.style.cssText='position:absolute;bottom:0;right:0;width:100%;height:100%;pointer-events:none;';box.appendChild(br);var _fs=Math.max(9,Math.round(ANNOT_FONT*(window._hudScale||1)))+'px';var info=document.createElement('div');info.className='hud-info';if((cb1*s+r.top-pad)<80){{info.style.bottom='auto';info.style.top='6px';info.style.marginBottom='0';}}var lbl=document.createElement('div');lbl.className='hud-label';lbl.style.fontSize=_fs;lbl.textContent=step.name;info.appendChild(lbl);var frm=document.createElement('div');frm.className='hud-formula';frm.style.fontSize=_fs;frm.textContent=step.formula;info.appendChild(frm);var crd=document.createElement('div');crd.className='hud-coords';crd.style.fontSize=_fs;crd.id='live-coords';crd.textContent='x:\\u2014 y:\\u2014';info.appendChild(crd);box.appendChild(info);hud.appendChild(box);trackBox=box;}}
function updatePencilDot(x,y){{if(!pencilDot)pencilDot=document.getElementById('pencil-dot');pencilDot.style.display='block';var s=window._hudScale||1,r=window._canvasEl?window._canvasEl.getBoundingClientRect():{{left:0,top:0}};pencilDot.style.left=(x*s+r.left-7)+'px';pencilDot.style.top=(y*s+r.top-7)+'px';var el=document.getElementById('live-coords');if(el)el.textContent='x:'+Math.round(x)+' y:'+Math.round(y);}}

var lastHudStep=-1,_inFillPhase=false;
function setup(){{var c=createCanvas(W,H,WEBGL);pixelDensity(1);c.elt.style.width='100vw';c.elt.style.height=(window.innerWidth/(W/H))+'px';c.elt.style.display='block';c.elt.style.margin='0';brush.load();frameRate(FPS);window._hudScale=c.elt.getBoundingClientRect().width/W;window._canvasEl=c.elt;window.addEventListener('resize',function(){{window._hudScale=c.elt.getBoundingClientRect().width/W;_lastDimStep=-1;}});}}
function draw(){{
  translate(-width/2,-height/2);var f3=frameCount-1;if(f3>=totalAnimFrames)f3=totalAnimFrames-1;
  var activeStep=0;for(var i=STEPS.length-1;i>=0;i--){{if(f3>=frameStarts[i]){{activeStep=i;break;}}}}
  if(activeStep<WASH_START){{_inFillPhase=false;background(242,234,218);for(var i=0;i<activeStep;i++)STEPS[i].draw(1);}}
  else{{if(!_inFillPhase){{background(242,234,218);for(var i=0;i<WASH_START;i++)STEPS[i].draw(1);drawAnnotations(WASH_START-1);_inFillPhase=true;}}}}
  var sf=f3-frameStarts[activeStep],sd=Math.max(1,frameEnds[activeStep]-frameStarts[activeStep]),prog=Math.min(1,sf/sd);var tip=STEPS[activeStep].draw(prog);
  if(!_inFillPhase)drawAnnotations(activeStep-1);
  if(activeStep!==lastHudStep){{
    if(lastHudStep>=0&&trackBox&&activeStep<WASH_START){{var _s2=window._hudScale||1,_r2=window._canvasEl?window._canvasEl.getBoundingClientRect():{{left:0,top:0}};var oB=STEPS[lastHudStep].bbox,nB=STEPS[activeStep].bbox,_p2=12*_s2;var ox=oB[0]*_s2+_r2.left-_p2,oy=oB[1]*_s2+_r2.top-_p2,nx=nB[0]*_s2+_r2.left-_p2,ny=nB[1]*_s2+_r2.top-_p2;var ln=document.createElement('div');ln.style.cssText='position:absolute;pointer-events:none;z-index:5;';var dx3=nx-ox,dy3=ny-oy,len=Math.sqrt(dx3*dx3+dy3*dy3),ang=Math.atan2(dy3,dx3)*180/Math.PI;ln.style.left=ox+'px';ln.style.top=oy+'px';ln.style.width=len+'px';ln.style.height='1.5px';ln.style.background='#5a5248';ln.style.transformOrigin='0 0';ln.style.transform='rotate('+ang+'deg)';ln.style.opacity='0.7';ln.style.transition='opacity 0.55s ease';document.getElementById('hud').appendChild(ln);setTimeout(function(){{ln.style.opacity='0';}},300);setTimeout(function(){{ln.remove();}},900);}}
    createTrackBox(STEPS[activeStep]);lastHudStep=activeStep;}}
  if(trackBox){{var _r3=window._canvasEl?window._canvasEl.getBoundingClientRect():{{left:0,top:0}},_s3=window._hudScale||1,_b3=STEPS[activeStep].bbox,_p3=12*_s3;trackBox.style.left=(Math.max(20,_b3[0])*_s3+_r3.left-_p3)+'px';trackBox.style.top=(Math.max(20,_b3[1])*_s3+_r3.top-_p3)+'px';}}
  if(tip)updatePencilDot(tip[0],tip[1]);
  if(f3>=totalAnimFrames-1){{noLoop();if(pencilDot)pencilDot.style.display='none';if(trackBox)trackBox.remove();trackBox=null;setTimeout(function(){{clearDimLabels();_lastDimStep=-1;}},6500);setTimeout(function(){{window.location.reload();}},7000);}}
}}
</script>
</body>
</html>"""
    
    return html


def main():
    parser = argparse.ArgumentParser(description="Traced: Generate p5.brush drawing from extraction data")
    parser.add_argument("--extraction", required=True, help="Path to extraction JSON")
    parser.add_argument("--output", default="drawing.html", help="Output HTML path")
    parser.add_argument("--name", default="Building", help="Building name")
    args = parser.parse_args()
    
    data = json.loads(Path(args.extraction).read_text())
    html = generate_html(data, args.name)
    Path(args.output).write_text(html)
    
    n_contours = sum(1 for e in data["elements"] if e.get("primitives", {}).get("contour"))
    n_edges = sum(len(e.get("edge_paths", [])) for e in data["elements"])
    print(f"Generated {args.output}")
    print(f"  Elements: {len(data['elements'])}")
    print(f"  Contours: {n_contours} elements with mask outlines")
    print(f"  Edge paths: {n_edges} detail paths")
    print(f"  Drawing steps: {len([e for e in data['elements'] if e.get('area_pct', 0) > 0.003])}")


if __name__ == "__main__":
    main()
