#!/usr/bin/env python3
"""
Ruler-and-Compass Generator

Draws optimized primitives (lines + arcs) using p5.brush.
No architectural assumptions. Just ruler strokes and compass strokes.
"""

import argparse
import json
import math
import base64
from pathlib import Path


def generate_html(optimized: dict, name: str = "Building", ref_image_path: str = None) -> str:
    W = optimized['canvas']['w']
    H = optimized['canvas']['h']
    primitives = optimized['primitives']
    
    # Build source shape drawing code (grey, drawn before redraw)
    # Uses the SAME primitives but with original (pre-optimized) params from extraction
    source_prims = optimized.get('source_primitives', [])
    source_draw_lines = []
    for sp in source_prims:
        params = sp['params']
        if sp['type'] == 'arc':
            sweep = params['sweep']
            n_pts = max(30, int(abs(math.degrees(sweep)) / 3))
            is_circ = abs(math.degrees(sweep)) > 350
            extra = 3 if is_circ else 0
            actual_sweep = 2 * math.pi * (1 if sweep > 0 else -1) if is_circ else sweep
            source_draw_lines.append(f"var _sp=[];for(var j=0;j<={n_pts+extra};j++){{var t=j/{n_pts};var a={params['start_angle']:.6f}+{actual_sweep:.6f}*t;_sp.push([{params['cx']:.1f}+{params['radius']:.1f}*Math.cos(a),{params['cy']:.1f}+{params['radius']:.1f}*Math.sin(a),1.0]);}}")
            source_draw_lines.append("brush.set('2H','#aaaaaa',3.0);if(_sp.length>=2)brush.spline(_sp,0.3);")
        elif sp['type'] == 'line':
            source_draw_lines.append(f"brush.set('2H','#aaaaaa',3.0);brush.line({params['x1']:.0f},{params['y1']:.0f},{params['x2']:.0f},{params['y2']:.0f});")
    source_draw_js = "\n      ".join(source_draw_lines) if source_draw_lines else "// no source shapes"
    
    # Source overlay: draw ref image as textured plane in WEBGL
    if ref_image_path and Path(ref_image_path).exists():
        source_native_js = "if(_refImg){push();translate(W/2,H/2);noStroke();texture(_refImg);plane(W,H);pop();}"
    else:
        source_native_js = "// no source"
    
    # Build drawing steps
    steps_js = []
    curve_defs = []
    
    for i, p in enumerate(primitives):
        params = p['params']
        pname = p['name'].upper().replace('_', ' ')
        
        if p['type'] == 'line':
            x1, y1 = params['x1'], params['y1']
            x2, y2 = params['x2'], params['y2']
            length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            bbox = f'[{int(min(x1,x2)-5)},{int(min(y1,y2)-5)},{int(max(x1,x2)+5)},{int(max(y1,y2)+5)}]'
            
            dx = x2 - x1
            dy = y2 - y1
            draw = f"""brush.set('2H','#c0392b',2.0);
      brush.line({x1:.0f},{y1:.0f},{x1:.0f}+{dx:.0f}*p,{y1:.0f}+{dy:.0f}*p);
      return [{x1:.0f}+{dx:.0f}*p,{y1:.0f}+{dy:.0f}*p];"""
            
            steps_js.append(f"""  {{name:'{pname}',formula:'line L={length:.0f}',dur:{max(0.2, length/500):.2f},
   bbox:{bbox},
   draw:function(p){{
      {draw}
   }}}}""")
        
        elif p['type'] == 'arc':
            cx, cy = params['cx'], params['cy']
            r = params['radius']
            start = params['start_angle']
            sweep = params['sweep']
            sweep_deg = abs(math.degrees(sweep))
            n_pts = max(20, int(sweep_deg / 3))
            
            # Generate arc points
            var_name = f'pts_{p["name"].replace("-","_")}'
            is_circle = sweep_deg > 350
            if is_circle:
                # For full circles: clamp to exactly 360°, add overlap to close
                actual_sweep = 2 * math.pi * (1 if sweep > 0 else -1)
                # Add 3 extra points past 360° so spline closes smoothly
                extra = 3
                pts_js = f"""var {var_name} = [];
    for(var j=0;j<={n_pts + extra};j++){{var t=j/{n_pts};
      var a={start:.6f}+{actual_sweep:.6f}*t;
      {var_name}.push([{cx:.1f}+{r:.1f}*Math.cos(a),{cy:.1f}+{r:.1f}*Math.sin(a),1.0]);}}"""
            else:
                pts_js = f"""var {var_name} = [];
    for(var j=0;j<={n_pts};j++){{var t=j/{n_pts};
      var a={start:.6f}+{sweep:.6f}*t;
      {var_name}.push([{cx:.1f}+{r:.1f}*Math.cos(a),{cy:.1f}+{r:.1f}*Math.sin(a),0.3+0.7*Math.sin(Math.PI*t)]);}}"""
            curve_defs.append(pts_js)
            
            bbox = f'[{int(cx-r-10)},{int(cy-r-10)},{int(cx+r+10)},{int(cy+r+10)}]'
            
            if sweep_deg > 350:
                formula = f'circle R={r:.0f}'
            elif sweep_deg > 170:
                formula = f'arc {sweep_deg:.0f}° R={r:.0f}'
            else:
                formula = f'arc {sweep_deg:.0f}° R={r:.0f}'
            
            draw = f"""brush.set('2H','#c0392b',2.0);
      if({var_name}){{var n=Math.max(3,Math.round(p*{var_name}.length));
      var sub={var_name}.slice(0,n);if(sub.length>=2)brush.spline(sub,0.3);
      var tip=sub[sub.length-1];return [tip[0],tip[1]];}}"""
            
            dur = max(0.3, sweep_deg / 300)
            steps_js.append(f"""  {{name:'{pname}',formula:'{formula}',dur:{dur:.2f},
   bbox:{bbox},
   draw:function(p){{
      {draw}
   }}}}""")
    
    all_curves = "\n".join(curve_defs)
    all_steps = ",\n".join(steps_js)
    n_draw_steps = len(steps_js)
    
    # Reference image (loaded in preload, drawn as WEBGL textured plane)
    ref_preload = "// no ref image"
    ref_draw = "// no ref overlay"
    
    if ref_image_path and Path(ref_image_path).exists():
        with open(ref_image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = Path(ref_image_path).suffix.lower().replace(".", "")
        mime = "jpeg" if ext in ("jpg", "jpeg") else ext
        ref_preload = f"_refImg=loadImage('data:image/{mime};base64,{b64}');"
        ref_draw = "// ref drawn via texture plane below"
    
    html = f"""<!-- بسم الله الرحمن الرحيم -->
<!-- Generated by Traced R&C Pipeline — {name} -->
<!-- Ruler-and-compass geometry only. Lines + arcs. -->
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/p5@2.0.3/lib/p5.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/p5.brush@2.0.2-beta"></script>
<style>
*{{margin:0}}body{{background:#f2eada;overflow:hidden;margin:0;padding:0;display:flex;justify-content:center;align-items:flex-start}}canvas{{object-fit:contain!important;display:block;max-height:100vh;max-width:calc(100vh*{W}/{H})}}
#hud{{position:fixed;top:0;left:0;width:100vw;height:100%;pointer-events:none;font-family:'SF Mono','Fira Code',monospace;overflow:hidden}}
.track-box{{position:absolute;transition:opacity 0.3s,left 0.4s,top 0.4s,width 0.4s,height 0.4s;min-width:60px;min-height:60px;overflow:visible}}
.track-box::before,.track-box::after,.track-box .bl::before,.track-box .br::before{{content:'';position:absolute;width:30px;height:30px;border-color:#5a5248;border-style:solid;border-width:0}}
.track-box::before{{top:-2px;left:-2px;border-top-width:1.5px;border-left-width:1.5px}}.track-box::after{{top:-2px;right:-2px;border-top-width:1.5px;border-right-width:1.5px}}
.track-box .bl::before{{bottom:-2px;left:-2px;border-bottom-width:1.5px;border-left-width:1.5px;position:absolute}}.track-box .br::before{{bottom:-2px;right:-2px;border-bottom-width:1.5px;border-right-width:1.5px;position:absolute}}
.hud-info{{position:absolute;bottom:100%;left:0;margin-bottom:16px;background:rgba(242,234,218,0.85);padding:6px 10px;border-radius:4px;white-space:nowrap}}
.hud-label{{color:#5a5248;font-weight:600;letter-spacing:0.5px}}.hud-formula{{color:#8a8278;margin-top:2px}}.hud-coords{{color:#a09888;margin-top:2px}}
#pencil-dot{{position:fixed;width:14px;height:14px;pointer-events:none;z-index:100}}
#pencil-dot .ch-circle{{position:absolute;top:4px;left:4px;width:6px;height:6px;border:1px solid #5a5248;border-radius:50%}}
</style>
</head>
<body>
<div id="hud"><div id="pencil-dot" style="display:none"><div class="ch-circle"></div></div></div>
<script>
var W={W},H={H},cx=W/2;

    {all_curves}

var ANNOT_FONT=13,_dimLabels=[],_lastDimStep=-1;
function clearDimLabels(){{for(var i=0;i<_dimLabels.length;i++)_dimLabels[i].remove();_dimLabels=[];}}
function addDimText(x,y,t,a){{for(var i=0;i<_dimLabels.length;i++)if(_dimLabels[i]._t===t)return;var el=document.createElement('div');el._t=t;var s=window._hudScale||1,r=window._canvasEl?window._canvasEl.getBoundingClientRect():{{left:0,top:0}};el.style.cssText='position:absolute;color:#8a8278;font-family:SF Mono,Fira Code,monospace;font-size:'+Math.max(9,Math.round(ANNOT_FONT*s))+'px;white-space:nowrap;pointer-events:none;';el.style.left=(x*s+r.left)+'px';el.style.top=(y*s+r.top)+'px';if(a==='center')el.style.transform='translateX(-50%)';el.textContent=t;document.getElementById('hud').appendChild(el);_dimLabels.push(el);}}
function drawAnnotations(step){{if(step!==_lastDimStep){{clearDimLabels();_lastDimStep=step;}}if(step>=0)addDimText(cx,H-40,'Traced R&C | {name} | lines + arcs','center');}}

var FPS=30;
var STEPS=[
{all_steps}
];
var PAUSE_S=0.14;
var frameStarts=[],frameEnds=[],f2=0;
for(var i=0;i<STEPS.length;i++){{frameStarts.push(Math.round(f2));f2+=STEPS[i].dur*FPS;frameEnds.push(Math.round(f2));if(i<STEPS.length-1)f2+=PAUSE_S*FPS;}}
var totalAnimFrames=Math.round(f2);
var trackBox=null,pencilDot=null;
function createTrackBox(step){{var hud=document.getElementById('hud');if(trackBox)trackBox.remove();var box=document.createElement('div');box.className='track-box';var s=window._hudScale||1,r=window._canvasEl?window._canvasEl.getBoundingClientRect():{{left:0,top:0}};var b=step.bbox,pad=12*s;var cb0=Math.max(20,b[0]),cb1=Math.max(20,b[1]),cb2=Math.min(W-20,b[2]),cb3=Math.min(H-20,b[3]);box.style.left=(cb0*s+r.left-pad)+'px';box.style.top=(cb1*s+r.top-pad)+'px';box.style.width=((cb2-cb0)*s+pad*2)+'px';box.style.height=((cb3-cb1)*s+pad*2)+'px';['tl','tr','bl2','br2'].forEach(function(c){{var ci=document.createElement('div');ci.className='corner-circ '+c;box.appendChild(ci);}});var bl=document.createElement('div');bl.className='bl';bl.style.cssText='position:absolute;bottom:0;left:0;width:100%;height:100%;pointer-events:none;';box.appendChild(bl);var br=document.createElement('div');br.className='br';br.style.cssText='position:absolute;bottom:0;right:0;width:100%;height:100%;pointer-events:none;';box.appendChild(br);var _fs=Math.max(9,Math.round(ANNOT_FONT*(window._hudScale||1)))+'px';var info=document.createElement('div');info.className='hud-info';if((cb1*s+r.top-pad)<80){{info.style.bottom='auto';info.style.top='6px';info.style.marginBottom='0';}}var lbl=document.createElement('div');lbl.className='hud-label';lbl.style.fontSize=_fs;lbl.textContent=step.name;info.appendChild(lbl);var frm=document.createElement('div');frm.className='hud-formula';frm.style.fontSize=_fs;frm.textContent=step.formula;info.appendChild(frm);var crd=document.createElement('div');crd.className='hud-coords';crd.style.fontSize=_fs;crd.id='live-coords';crd.textContent='x:\\u2014 y:\\u2014';info.appendChild(crd);box.appendChild(info);hud.appendChild(box);trackBox=box;}}
function updatePencilDot(x,y){{if(!pencilDot)pencilDot=document.getElementById('pencil-dot');pencilDot.style.display='block';var s=window._hudScale||1,r=window._canvasEl?window._canvasEl.getBoundingClientRect():{{left:0,top:0}};pencilDot.style.left=(x*s+r.left-7)+'px';pencilDot.style.top=(y*s+r.top-7)+'px';var el=document.getElementById('live-coords');if(el)el.textContent='x:'+Math.round(x)+' y:'+Math.round(y);}}
var lastHudStep=-1,_refImg=null;
function preload(){{
  {ref_preload}
}}
function setup(){{var c=createCanvas(W,H,WEBGL);pixelDensity(1);var vh=window.innerHeight;var cw=Math.min(window.innerWidth,vh*{W}/{H});c.elt.style.width=cw+'px';c.elt.style.height=(cw/({W}/{H}))+'px';c.elt.style.display='block';c.elt.style.margin='0';brush.load();frameRate(FPS);window._hudScale=c.elt.getBoundingClientRect().width/W;window._canvasEl=c.elt;window.addEventListener('resize',function(){{window._hudScale=c.elt.getBoundingClientRect().width/W;_lastDimStep=-1;}});}}
function draw(){{
  translate(-width/2,-height/2);var f3=frameCount-1;if(f3>=totalAnimFrames)f3=totalAnimFrames-1;
  var activeStep=0;for(var i=STEPS.length-1;i>=0;i--){{if(f3>=frameStarts[i]){{activeStep=i;break;}}}}
  background(242,234,218);
  // Draw source image as WEBGL textured plane
  if(_refImg){{push();translate(W/2,H/2);noStroke();texture(_refImg);plane(W,H);pop();}}
  for(var i=0;i<activeStep;i++)STEPS[i].draw(1);
  var sf=f3-frameStarts[activeStep],sd=Math.max(1,frameEnds[activeStep]-frameStarts[activeStep]),prog=Math.min(1,sf/sd);var tip=STEPS[activeStep].draw(prog);
  drawAnnotations(activeStep);
  if(activeStep!==lastHudStep){{createTrackBox(STEPS[activeStep]);lastHudStep=activeStep;}}
  if(trackBox){{var _r3=window._canvasEl?window._canvasEl.getBoundingClientRect():{{left:0,top:0}},_s3=window._hudScale||1,_b3=STEPS[activeStep].bbox,_p3=12*_s3;trackBox.style.left=(Math.max(20,_b3[0])*_s3+_r3.left-_p3)+'px';trackBox.style.top=(Math.max(20,_b3[1])*_s3+_r3.top-_p3)+'px';}}
  if(tip)updatePencilDot(tip[0],tip[1]);
  if(f3>=totalAnimFrames-1){{noLoop();if(pencilDot)pencilDot.style.display='none';if(trackBox)trackBox.remove();trackBox=null;setTimeout(function(){{clearDimLabels();}},6500);}}
}}
</script>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimized", required=True, help="Optimized R&C JSON")
    parser.add_argument("--output", default="drawing_rc.html")
    parser.add_argument("--name", default="Building")
    parser.add_argument("--ref-image", default=None)
    args = parser.parse_args()
    
    optimized = json.loads(Path(args.optimized).read_text())
    html = generate_html(optimized, args.name, args.ref_image)
    Path(args.output).write_text(html)
    
    prims = optimized['primitives']
    n_lines = sum(1 for p in prims if p['type'] == 'line')
    n_arcs = sum(1 for p in prims if p['type'] == 'arc')
    print(f"Generated {args.output}: {n_lines} lines + {n_arcs} arcs = {len(prims)} primitives")


if __name__ == '__main__':
    main()
