#!/usr/bin/env python3
"""Ruler-and-Compass Generator — draws lines + arcs with p5.brush."""
import argparse, json, math, base64
from pathlib import Path

def generate_html(optimized, name="Building", ref_image_path=None):
    W = optimized['canvas']['w']
    H = optimized['canvas']['h']
    primitives = optimized['primitives']
    
    # Sort: bottom→top, right→left (like a right-handed draftsman)
    def sort_key(p):
        params = p['params']
        if p['type'] == 'line':
            cy = max(params['y1'], params['y2'])  # bottom-most point
            cx = max(params['x1'], params['x2'])
        else:
            cy = params['cy'] + params['radius']  # bottom of arc
            cx = params['cx'] + params['radius']
        return (-cy, -cx)  # negative = bottom first, right first
    
    primitives = sorted(primitives, key=sort_key)
    
    # Build drawing steps
    curve_defs = []
    steps_js = []
    for p in primitives:
        params = p['params']
        pname = p['name'].upper().replace('_', ' ')
        if p['type'] == 'line':
            x1,y1,x2,y2 = params['x1'],params['y1'],params['x2'],params['y2']
            dx, dy = x2-x1, y2-y1
            length = math.sqrt(dx*dx+dy*dy)
            bbox = f'[{int(min(x1,x2)-5)},{int(min(y1,y2)-5)},{int(max(x1,x2)+5)},{int(max(y1,y2)+5)}]'
            steps_js.append(f"""  {{name:'{pname}',formula:'line L={length:.0f}',dur:{max(0.2,length/500):.2f},bbox:{bbox},draw:function(p){{brush.set('2H','#c0392b',2.0);brush.line({x1:.0f},{y1:.0f},{x1:.0f}+{dx:.0f}*p,{y1:.0f}+{dy:.0f}*p);return [{x1:.0f}+{dx:.0f}*p,{y1:.0f}+{dy:.0f}*p];}}}}""")
        elif p['type'] == 'arc':
            cx,cy,r = params['cx'],params['cy'],params['radius']
            start,sweep = params['start_angle'],params['sweep']
            sweep_deg = abs(math.degrees(sweep))
            is_circ = sweep_deg > 350
            bbox = f'[{int(cx-r-10)},{int(cy-r-10)},{int(cx+r+10)},{int(cy+r+10)}]'
            formula = f'circle R={r:.0f}' if is_circ else f'arc {sweep_deg:.0f}° R={r:.0f}'
            dur = max(0.5, sweep_deg/250)
            d = r * 2
            # Compass drawing: plant at center, sweep the arc progressively
            # p5.js arc(x, y, w, h, start, stop) in WEBGL mode
            if is_circ:
                actual_sweep = 2 * math.pi
                steps_js.append(f"""  {{name:'{pname}',formula:'{formula}',dur:{dur:.2f},bbox:{bbox},draw:function(p){{
      push();noFill();
      stroke(192,57,43);strokeWeight(2);
      arc({cx:.1f},{cy:.1f},{d:.1f},{d:.1f},{start:.4f},{start:.4f}+{actual_sweep:.4f}*p);
      if(p<1){{var a={start:.4f}+{actual_sweep:.4f}*p;
        stroke(192,57,43,60);strokeWeight(0.8);
        line({cx:.1f},{cy:.1f},{cx:.1f}+{r:.1f}*Math.cos(a),{cy:.1f}+{r:.1f}*Math.sin(a));
        fill(192,57,43);noStroke();ellipse({cx:.1f},{cy:.1f},5,5);}}
      pop();
      var a2={start:.4f}+{actual_sweep:.4f}*p;
      return [{cx:.1f}+{r:.1f}*Math.cos(a2),{cy:.1f}+{r:.1f}*Math.sin(a2)];
    }}}}""")
            else:
                steps_js.append(f"""  {{name:'{pname}',formula:'{formula}',dur:{dur:.2f},bbox:{bbox},draw:function(p){{
      push();noFill();
      stroke(192,57,43);strokeWeight(2);
      var s1={start:.4f}+{sweep:.4f}*p;
      arc({cx:.1f},{cy:.1f},{d:.1f},{d:.1f},{start:.4f},s1);
      if(p<1){{stroke(192,57,43,60);strokeWeight(0.8);
        line({cx:.1f},{cy:.1f},{cx:.1f}+{r:.1f}*Math.cos(s1),{cy:.1f}+{r:.1f}*Math.sin(s1));
        fill(192,57,43);noStroke();ellipse({cx:.1f},{cy:.1f},5,5);}}
      pop();
      return [{cx:.1f}+{r:.1f}*Math.cos(s1),{cy:.1f}+{r:.1f}*Math.sin(s1)];
    }}}}""")
    
    all_curves = "\n    ".join(curve_defs) if curve_defs else "// no curve arrays needed"
    all_steps = ",\n".join(steps_js)
    
    # Source image as HTML img (grey, behind canvas)
    ref_img_tag = ""
    if ref_image_path and Path(ref_image_path).exists():
        with open(ref_image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = Path(ref_image_path).suffix.lower().replace(".", "")
        mime = "jpeg" if ext in ("jpg","jpeg") else ext
        ref_img_tag = f'<img id="ref-src" src="data:image/{mime};base64,{b64}">'
    
    html = f"""<!-- بسم الله الرحمن الرحيم -->
<!DOCTYPE html><html><head><meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/p5@2.0.3/lib/p5.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/p5.brush@2.0.2-beta"></script>
<style>
*{{margin:0;padding:0}}
body{{background:#f2eada;overflow:hidden;display:flex;justify-content:center;align-items:flex-start}}
#wrap{{position:relative;height:100vh;width:calc(100vh*{W}/{H})}}
#ref-src{{position:absolute;top:0;left:0;width:100%;height:100%;opacity:0.5;filter:grayscale(100%);z-index:1}}
canvas{{position:absolute;top:0;left:0;width:100%!important;height:100%!important;z-index:2}}
#hud{{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:10;font-family:'SF Mono','Fira Code',monospace;overflow:hidden}}
.track-box{{position:absolute;transition:opacity 0.3s,left 0.4s,top 0.4s,width 0.4s,height 0.4s;min-width:60px;min-height:60px}}
.track-box::before,.track-box::after,.track-box .bl::before,.track-box .br::before{{content:'';position:absolute;width:24px;height:24px;border-color:#c0392b;border-style:solid;border-width:0}}
.track-box::before{{top:-2px;left:-2px;border-top-width:2px;border-left-width:2px}}
.track-box::after{{top:-2px;right:-2px;border-top-width:2px;border-right-width:2px}}
.track-box .bl::before{{bottom:-2px;left:-2px;border-bottom-width:2px;border-left-width:2px;position:absolute}}
.track-box .br::before{{bottom:-2px;right:-2px;border-bottom-width:2px;border-right-width:2px;position:absolute}}
.hud-info{{position:absolute;bottom:100%;left:0;margin-bottom:12px;background:rgba(242,234,218,0.9);padding:6px 10px;border-radius:4px;white-space:nowrap}}
.hud-label{{color:#c0392b;font-weight:700;font-size:11px;letter-spacing:0.5px}}
.hud-formula{{color:#8a8278;font-size:10px;margin-top:2px}}
.hud-coords{{color:#a09888;font-size:10px;margin-top:2px}}
#pencil-dot{{position:absolute;width:14px;height:14px;pointer-events:none;z-index:20;display:none}}
#pencil-dot .ch{{width:8px;height:8px;border:2px solid #c0392b;border-radius:50%;margin:1px}}
#footer{{position:absolute;bottom:8px;left:0;width:100%;text-align:center;color:#a09888;font-family:'SF Mono',monospace;font-size:10px;z-index:10;pointer-events:none}}
</style>
</head><body>
<div id="wrap">
{ref_img_tag}
<div id="hud"><div id="pencil-dot"><div class="ch"></div></div></div>
<div id="footer">Traced R&C | {name}</div>
</div>
<script>
var W={W},H={H},cx=W/2;
    {all_curves}
var FPS=30,_hs=1,trackBox=null,pencilDot=null;
var STEPS=[
{all_steps}
];
var PAUSE_S=0.2;
var frameStarts=[],frameEnds=[],f2=0;
for(var i=0;i<STEPS.length;i++){{frameStarts.push(Math.round(f2));f2+=STEPS[i].dur*FPS;frameEnds.push(Math.round(f2));if(i<STEPS.length-1)f2+=PAUSE_S*FPS;}}
var totalAnimFrames=Math.round(f2),lastHudStep=-1;

function createTrackBox(step){{
  var hud=document.getElementById('hud');if(trackBox)trackBox.remove();
  var box=document.createElement('div');box.className='track-box';
  var s=_hs,b=step.bbox,pad=10*s;
  box.style.left=(b[0]*s-pad)+'px';box.style.top=(b[1]*s-pad)+'px';
  box.style.width=((b[2]-b[0])*s+pad*2)+'px';box.style.height=((b[3]-b[1])*s+pad*2)+'px';
  var bl=document.createElement('div');bl.className='bl';bl.style.cssText='position:absolute;bottom:0;left:0;width:100%;height:100%;pointer-events:none';box.appendChild(bl);
  var br=document.createElement('div');br.className='br';br.style.cssText='position:absolute;bottom:0;right:0;width:100%;height:100%;pointer-events:none';box.appendChild(br);
  var info=document.createElement('div');info.className='hud-info';
  if(b[1]*s-pad<60){{info.style.bottom='auto';info.style.top='6px';}}
  var lbl=document.createElement('div');lbl.className='hud-label';lbl.textContent=step.name;info.appendChild(lbl);
  var frm=document.createElement('div');frm.className='hud-formula';frm.textContent=step.formula;info.appendChild(frm);
  var crd=document.createElement('div');crd.className='hud-coords';crd.id='live-coords';crd.textContent='x:\u2014 y:\u2014';info.appendChild(crd);
  box.appendChild(info);hud.appendChild(box);trackBox=box;
}}
function updateDot(x,y){{
  if(!pencilDot)pencilDot=document.getElementById('pencil-dot');
  pencilDot.style.display='block';pencilDot.style.left=(x*_hs-7)+'px';pencilDot.style.top=(y*_hs-7)+'px';
  var el=document.getElementById('live-coords');if(el)el.textContent='x:'+Math.round(x)+' y:'+Math.round(y);
}}

function setup(){{
  var c=createCanvas(W,H,WEBGL);pixelDensity(1);c.parent('wrap');
  _hs=document.getElementById('wrap').offsetWidth/W;
  brush.load();frameRate(FPS);
  window.addEventListener('resize',function(){{_hs=document.getElementById('wrap').offsetWidth/W;}});
}}
function draw(){{
  translate(-width/2,-height/2);
  clear();// Clear each frame — redraw everything (needed for compass arm animation)
  var f3=Math.min(frameCount-1,totalAnimFrames-1);
  var activeStep=0;for(var i=STEPS.length-1;i>=0;i--){{if(f3>=frameStarts[i]){{activeStep=i;break;}}}}
  // Redraw all completed steps
  for(var i=0;i<activeStep;i++)STEPS[i].draw(1);
  var sf=f3-frameStarts[activeStep],sd=Math.max(1,frameEnds[activeStep]-frameStarts[activeStep]),prog=Math.min(1,sf/sd);
  var tip=STEPS[activeStep].draw(prog);
  if(activeStep!==lastHudStep){{createTrackBox(STEPS[activeStep]);lastHudStep=activeStep;}}
  if(trackBox){{var b=STEPS[activeStep].bbox,s=_hs,pad=10*s;trackBox.style.left=(b[0]*s-pad)+'px';trackBox.style.top=(b[1]*s-pad)+'px';}}
  if(tip)updateDot(tip[0],tip[1]);
  if(f3>=totalAnimFrames-1){{noLoop();if(pencilDot)pencilDot.style.display='none';if(trackBox)setTimeout(function(){{trackBox.remove();trackBox=null;}},3000);}};
}}
</script>
</body></html>"""
    return html

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimized", required=True)
    parser.add_argument("--output", default="drawing_rc.html")
    parser.add_argument("--name", default="Building")
    parser.add_argument("--ref-image", default=None)
    args = parser.parse_args()
    optimized = json.loads(Path(args.optimized).read_text())
    html = generate_html(optimized, args.name, args.ref_image)
    Path(args.output).write_text(html)
    prims = optimized['primitives']
    print(f"Generated {args.output}: {sum(1 for p in prims if p['type']=='line')} lines + {sum(1 for p in prims if p['type']=='arc')} arcs")

if __name__ == '__main__':
    main()
