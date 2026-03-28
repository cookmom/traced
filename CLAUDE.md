# CLAUDE.md — Traced Pipeline

## Philosophy
Aristotelian first principles. Two primitives only: **lines** and **arcs**.
Everything is detected from edge pixels, never from labels or classifications.

## Architecture
```
Image → Threshold → Skeletonize → Edge pixels
Edge pixels → Blob clustering (connected components)
Each blob → RANSAC fit lines + circles
Primitives → Optimize (chamfer distance) → Generate (p5.brush)
```

## Rules

### Detection (detect.py)
- **Lines before circles.** A line is a simpler truth (fewer parameters). Only accept a circle if it's genuinely circular (>50% of blob coverage), not just a large-radius approximation of straight edges.
- **Blob isolation.** RANSAC runs within each connected component separately. Never fit across disconnected regions — that creates phantom shapes.
- **Skeleton centerlines.** Use Otsu threshold + skeletonize, NOT Canny. Canny produces double edges (inner/outer of stroke). Skeleton gives one centerline per stroke.
- **Endpoint trimming.** Line endpoints come from the longest continuous dense run of inlier points. Gaps at vertex junctions don't extend the line.

### Lessons (update when mistakes happen)
1. **Never classify before detecting.** Labels like "dome", "arch", "triangle" are output descriptions, not input to the detector. The detector finds lines and arcs. Period.
2. **SAM masks are filled regions, not stroke contours.** A filled triangle mask looks circular. Don't decompose SAM masks — decompose edges.
3. **Large-radius arcs approximate lines.** Any 4 points from a square's edges can fit a circle with r=500. The parsimony check (line wins unless circle genuinely dominates with >50% blob coverage) prevents this.
4. **RANSAC samples from the global pool find phantom shapes.** Edge pixels from different shapes that happen to be collinear create false lines. Blob clustering prevents this.
5. **Canny edge detection creates stroke-width duplicates.** A 6px stroke has 2 Canny edges. Threshold + skeletonize gives 1 centerline.
6. **p5.js WEBGL `clear()` is required for compass arm animation.** Without clear, accumulated strokes prevent the temp compass arm from disappearing. Redraw all completed strokes each frame.
7. **`tint()` + `image()` don't work in p5.js WEBGL mode.** Use HTML `<img>` behind transparent canvas for source overlay.

### Verification
- Always run `detect.py` and check the primitive count + unexplained pixel percentage before generating HTML.
- GPU Chrome screenshot after every deploy to verify visual correctness.
- Compare detected primitive parameters against known ground truth (for calibration tests).

### Conventions
- Source image: grey (CSS `filter:grayscale(100%); opacity:0.5`)
- Redraw strokes: red (#c0392b)
- Compass arm during arc drawing: faint red line from center + center dot
- Draw order: bottom to top, right to left
- All HTML starts with `<!-- بسم الله الرحمن الرحيم -->`

### Lesson 8: VERIFY YOUR OWN OUTPUT
Before sending ANY result to Tawfeeq, actually look at the screenshot. Check pixel counts. If >5% of pixels are red on a real image, something is wrong. Don't send garbage and wait to be told it's garbage.

### Lesson 9: SAM contour edges ≠ architectural lines
SAM region boundaries are organic blobs, not geometric shapes. Tracing SAM contour vertices and fitting arcs between them produces noise, not architecture. SAM tells you WHERE shapes are, not WHAT shapes they are. The shapes still need to come from the actual image edges within each region.

### Lesson 10: Don't keep iterating without rethinking
When the approach fundamentally isn't working (1483 primitives, then 574, then 105, then 38 — and still garbage), STOP and rethink the approach instead of tuning parameters. Three failed attempts on the same idea means the idea is wrong.

## Real Image Pipeline (Claude Vision + Hough Refinement)

### The Approach That Works
1. **Claude vision** analyzes the image and identifies major architectural features as a JSON spec of lines + arcs
2. **Hough lines** from the actual image refine the x-coordinates of vertical/horizontal edges
3. **HoughCircles** in masked regions find arcs (with heavy filtering)
4. Coordinates converge through validation loops (render → compare → adjust)

### Why This Works
- Claude UNDERSTANDS architecture — it knows what a horseshoe arch is, where door frames are, how panels divide
- Edge detection doesn't understand — it finds every texture edge equally
- SAM finds regions but can't tell you the geometry within them
- The human-like approach: look at the image, understand the structure, THEN measure

### Skill Rules for Architectural Spec Generation
- Start with the largest feature (the arch) — draw it first
- Frame lines come from Hough with threshold>200, minLength>200 — only the strongest edges survive
- Arcs: run HoughCircles only in the region where the arch lives (mask the top 40%)
- Typical Moroccan door: horseshoe arch r ≈ 0.36×W, center at (W/2, ~0.25×H)
- Frame verticals cluster at consistent x-positions — use Hough to find exact x, extend full height
- 15-25 primitives is the right number for major architectural skeleton
- Always largest-first draw order

### Lesson 11: Claude vision > edge detection for architectural understanding
Edge detection finds every pixel intensity change equally. Claude understands WHAT the architecture is and WHERE the structural lines should be. Use Claude for the initial spec, edge analysis for refinement.

## Real Image Strategy (validated 2026-03-27)

### The Winning Approach: Claude Vision + Hough Refinement
1. **Claude vision** identifies the major architectural features and estimates coordinates
2. **HoughLinesP** (high threshold, long minLength) finds exact structural line positions
3. **HoughCircles** (on masked arch region only) finds the arch radius/center
4. Merge Claude's structural understanding with Hough's precise measurements
5. Generate primitives JSON → feed to existing pipeline

### Why Other Approaches Failed
- **Pure edge detection** (RANSAC on all edges): 119K edge pixels, texture overwhelms
- **SAM + edge detect per region**: SAM boundaries are organic, not geometric
- **SAM boundary tracing**: SAM contours ≠ architectural lines
- **Hough on full image**: too many small lines from texture

### What Works
- **Bilateral filter (d=15)** + **Canny (80/200)** + **HoughLinesP (threshold=250, minLen=250)** finds structural verticals
- **HoughCircles on arch region only** (top 45%, middle 80% width) finds the arch
- **Claude's architectural knowledge** provides the semantic structure (which lines form the frame, where the arch is, door panels, etc.)

### Skill Pattern for New Images
1. Run HoughLinesP with high thresholds → strongest lines
2. Identify which lines are frame, walls, dividers (from position/length)
3. Mask the arch region → HoughCircles → average top candidates
4. Add door features (knocker, panels) from Claude understanding
5. Sort by size (biggest first) → generate

### Lesson 11: Identify arch TYPE before computing geometry
- Round arch = one arc, semicircular (180°)
- Horseshoe = one arc, >180° sweep
- POINTED = TWO arcs from offset centers meeting at a sharp apex
- Pointed horseshoe = TWO arcs, each >90°, from offset centers, with legs continuing past vertical
- Always check: does the arch meet at a POINT or a CURVE at the top?

### Feature Markers for Moroccan/Islamic Architecture
- **Apex**: the topmost point of the arch (may be pointed or curved)
- **Springing points**: where arch curves meet the vertical legs
- **Band width**: the decorative border width (outer arch - inner arch)
- **Horseshoe pinch**: where the arch curves INWARD past vertical before meeting legs
- The arch band should be consistent width throughout — inner arch tracks outer arch at constant offset

### Lesson 12: Point-Based Architecture > Parameter-Based
Don't compute arches from center/radius/sweep. Instead:
1. Mark key POINTS on the image (apex, springing L/R, leg bottoms, frame corners)
2. Derive arcs FROM the points: "arc from P1 to P2, centered at P3"
3. Derive lines FROM the points: "line from P1 to P2"
4. Iterate: compare render to source, adjust POINT POSITIONS, regenerate

### Lesson 13: Zellige/Islamic tile images are ALL dark
The arch isn't a light opening with dark border. The entire surface is decorated tile.
The arch shape is defined by pattern changes and carved borders, not brightness boundaries.
Edge detection approaches fail because there IS no clear edge — it's texture on texture.
Use visual understanding (Claude vision) or manual point marking, not pixel analysis.

### Pointed Horseshoe Arch Geometry
- Each half is an arc centered at the OPPOSITE springing point
- r = distance from opposite springing to apex (both sides equal for symmetric arch)
- The horseshoe pinch: the arc continues past the widest point, curving inward before meeting vertical legs
- Band width: inner arch is offset ~30-40px from outer, tracking the same curve shape
- Legs start at springing points and go straight down

### Lesson 14: p5.js arc() in WEBGL mode is unreliable
arc() with translate(-w/2,-h/2) draws unpredictably. Use beginShape()/vertex()/endShape() 
to draw arcs as polylines from computed cos/sin points. This is deterministic and works correctly.

### Lesson 15: Pointed arch geometry — the arc MUST narrow at the top
A pointed arch from springing at (100,480)/(675,480) with apex at (387,136) has r=448.
At y=200, the right arc is at x≈450 — this is GEOMETRICALLY CORRECT.
The arch is wide at the bottom and narrow at the top. Don't expect x=650 at y=200.
If the source image shows a wider arch at mid-height, the springing points or apex are wrong,
not the arc rendering.

### Lesson 16: My point estimates were ALL wrong — let the human place them
Moroccan pointed horseshoe arch — my estimates vs Tawfeeq's corrections:
- Apex: y=136 → y=77 (59px off — arch is MUCH higher than I thought)
- Horseshoe: y=480 → y=621 (141px off — pinch is MUCH lower)
- Wide points: y=380 → y=476 (96px off)
- Outer width: 605px → 684px (arch is wider)
- Legs: x=100→675 → x=70→704 (legs are further out)

**Lesson: BUILD AN INTERACTIVE EDITOR and let the human place points.**
Don't guess coordinates from pixel analysis on complex textured images.
The point editor with live arc preview is the correct tool.

### Islamic Arch Feature Point Template
For a pointed horseshoe arch, mark these 14+ points:
- OA/IA: Outer/Inner Apex (the point where two arcs meet)
- OWL/OWR, IWL/IWR: Widest points of the arch (horseshoe bulge)
- OHL/OHR, IHL/IHR: Horseshoe points (where curve reverses to meet legs)
- OTLL/OTLR, ITLL/ITLR: Leg tops (may differ from horseshoe points)
- OLL/OLR, ILL/ILR: Leg bottoms

Arc construction: fit circle through 3 points (HS → Wide → Apex) for each half.

### Lesson 17: ALWAYS visually verify your own output before sending
Use GPU Chrome screenshot + pixel analysis. Check for:
- Are all expected colors present? (red outer, blue inner)  
- Use appropriate detection thresholds (blue at 136 won't pass >200 check)
- Compare overlay against source at key feature points

### Lesson 18: Arc direction = middle point position
An arc through 3 points bows toward the middle point. If the middle point is:
- Outward from center → arc bows outward ✓
- Inward from center → arc bows inward ✗ (for horseshoe)
No flip flags needed. Just put the middle point on the correct side.

### Horseshoe Arch Construction (Final)
1. Main arcs: HS → wide → apex (4 arcs, 2 per border)
2. Pinch arcs: wide → HS → pinch (4 arcs, curving outward)
3. Lines: pinch → legTop (short horizontal, 4 lines)
4. Lines: legTop → leg bottom (vertical legs, 4 lines)
5. Line: leg bottom L → leg bottom R (frame bottom)
Total: 8 arcs + 9 lines = 17 primitives for double-border horseshoe arch

### Lesson 19: CHECK YOUR OWN UI FOR OCCLUSION
The UI panel was 220px wide and covered the right 28% of the 774px image.
Every "broken right arc" for the last 6 hours was actually correct — just hidden.
ALWAYS close/hide UI overlays before evaluating the render.
Use dev-browser to programmatically hide panels before screenshots.

### dev-browser Superpowers (Unlocked 2026-03-28)
1. **See any URL instantly** — screenshot + DOM read, no deploy wait
2. **Interact with pages** — click, fill, read values. Use the point editor to place points myself
3. **Test locally** — `python3 -m http.server 7777` + dev-browser, no GitHub push needed
4. **Playwright scripts** — full browser automation, evaluate JS in page context
5. **Iterate fast** — generate → serve → screenshot → check → fix → repeat

**Key workflow**: Always hide UI panels before screenshots. Use `page.evaluate()` to read computed values (arc coords, point positions) directly from the page.

**Usage pattern**:
```
cat > /tmp/script.js << 'JS'
const page = await browser.newPage();
await page.setViewportSize({width:W, height:H});
await page.goto('http://localhost:7777/file.html');
await new Promise(r => setTimeout(r, 3000));
// Hide panels
await page.evaluate(() => document.getElementById('panel').style.display='none');
const buf = await page.screenshot();
await saveScreenshot(buf, 'check.png');
// Read values
const data = await page.evaluate(() => someGlobal);
console.log(JSON.stringify(data));
JS
GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA dev-browser --headless run /tmp/script.js
```
Screenshots save to `~/.dev-browser/tmp/` — copy to workspace for messaging.

### Karpathy Method for Arc Optimization
1. Load source edges (Canny on bilateral-filtered grayscale)
2. Build KDTree of edge pixel positions
3. For each arc: render as polyline, compute mean distance to nearest edge pixel (chamfer)
4. Gradient descent: nudge each anchor point ±1px in 8 directions, keep the move that reduces chamfer
5. For pinch arcs with auto-midpoint: also sweep the offset parameter
6. Run 50-100 iterations until convergence

**Results**: main arcs 0.8-1.2px, pinch arcs 0.5-0.8px chamfer. Sub-pixel alignment.

### Standard Workflow (ALWAYS follow this)
1. **Aristotle**: identify the simplest true geometric primitives
2. **Point editor**: human places anchor points for features AI can't accurately detect
3. **Karpathy**: gradient descent optimizes points against source edges
4. **dev-browser**: verify every change visually before sending
5. **Skill update**: document every lesson, every tool usage pattern, every architecture decision

### Lesson 20: Gradient descent needs SIDE constraints
Minimizing chamfer to nearest edge can find the WRONG edge (inside vs outside of an arch band).
When optimizing arc positions, constrain the search:
- Pin the arc's bow direction (midpoint offset sign must not flip)
- Or: only match edges that are on the expected side of the arc
- Or: use the signed distance (positive = correct side, negative = wrong side)
Never let the optimizer flip an arc's curvature direction just because a closer edge exists on the other side.

### Lesson 21: ALWAYS enforce symmetry after optimization
Gradient descent treats each point independently — it will break L/R symmetry to find local minima.
After ANY optimization pass, re-enforce symmetry:
1. For each L/R pair, average the distances from center
2. Average the Y positions
3. Apply symmetric positions
Run optimization WITH symmetry constraint (move L/R pairs together), not independently.

### Lesson 22: THINK FIRST — study the source before coding
Before ANY arc/point change:
1. Study the actual source pixel data at the feature location
2. Understand what the REAL curve looks like (is it dramatic or subtle?)
3. THEN decide the parameter values
The horseshoe pinch in this image is nearly vertical — a 25px offset was way too much.
5-6px matches the actual gentle inward lean.
Always measure the source before guessing parameters.

## Pretext — Text as a Primitive (Added 2026-03-28)

### What
Pure JS text measurement & layout. No DOM, no reflow. Canvas/SVG/WebGL/server-side.
`npm install @chenglou/pretext`

### Why in TRACED
Text is the third primitive alongside lines and arcs:
- **Dimension labels**: "R=401", "span=575px" positioned precisely on canvas
- **Architectural annotation**: room names, material callouts, scale bars
- **Mihrab calligraphy**: Arabic text flowing along arch curves using `layoutNextLine()` with variable width from arch geometry
- **Calligraphic patterns**: text shaped to fill geometric zones (spandrels, concentric bands, tile panels)

### Core API
```js
import { prepare, layout, prepareWithSegments, layoutNextLine } from '@chenglou/pretext'

// Simple measurement
const prepared = prepare(text, '16px Inter')
const { height } = layout(prepared, maxWidth, lineHeight)

// Line-by-line for curved paths
const prepared = prepareWithSegments(text, '18px "Noto Naskh Arabic"')
let cursor = { segmentIndex: 0, graphemeIndex: 0 }
while (true) {
    const width = getArchWidthAtY(y) // from TRACED arch geometry
    const line = layoutNextLine(prepared, cursor, width)
    if (!line) break
    // Render line along arc path using p5.brush
    cursor = line.end
    y += lineHeight
}
```

### Integration with TRACED pipeline
TRACED geometry (arcs/lines) → Pretext text layout along geometry → p5.brush calligraphic rendering

### Connection to fatiha.app
Voice recitation → Pretext layouts text along TRACED geometric paths → p5.brush renders calligraphic strokes progressively

## Islamic Pattern Fill — Hasba Method (Added 2026-03-28)

### The Hasba Method (Moroccan Zellige Construction)
1. **Central rosette** — n-fold star (8, 12, 16-fold). The seed shape.
2. **Regulation grid** — determines how rosettes tile/connect. Grid spacing = rosette diameter.
3. **Interstice shapes** — fill gaps between stars: crosses, diamonds, smaller stars, kite shapes.
4. **Line hierarchy** — structural lines heaviest, pattern lines lightest.

### Pattern Vocabulary
- `zelligeStar16(cx, cy, r)` — 16-fold star with inner regulation circle + spokes
- `rosette8(cx, cy, r)` — 8-fold star with petal arcs + inner circle
- `star6(cx, cy, r)` — 6-fold (two overlapping triangles, Star of David form)
- `intersticeX(cx, cy, s)` — cross-shaped gap filler (4 arms)
- `intersticeDiamond(cx, cy, s)` — diamond/lozenge gap filler
- `interlaceBand(x1,y1,x2,y2,amp,periods)` — serpentine wave between two points

### Pretext-Style Fill (pretextFill)
The key composition function. Fills ANY shape by scanning row-by-row:
```js
pretextFill(getWidth, yStart, yEnd, unitSize, drawUnit)
// getWidth(y) → [xLeft, xRight] or null
// Fits floor(rowWidth/unitSize) units per row, centered
```
This is the Pretext concept applied to geometry: just as text flows into variable-width lines,
pattern units flow into variable-width shape regions.

### Golden Ratio Sizing (v2+)
- Primary tile: `size`
- Secondary: `size / φ` (0.618×)
- Tertiary: `size / φ²` (0.382×)
- Quaternary: `size / φ³` (0.236×)
Creates natural visual hierarchy matching Islamic mathematical tradition.

### Composition Principles (Islamic Art)
- **Horror vacui**: fill all space — no empty zones
- **Self-similarity**: same patterns at multiple scales
- **Symmetry**: bilateral (doors) + rotational (rosettes)
- **Hierarchy**: largest patterns dominate, smaller fill gaps
- **Rhythm**: alternating pattern types across zones
- **Graded density**: denser patterns in higher-focus areas

### p5.js WEBGL Constraints (HARD-WON)
- **p5.brush requires WEBGL** — cannot use 2D mode
- **No `quadraticVertex()`** — manual bezier: `bx=(1-t)²*sx + 2(1-t)*t*mx + t²*ex`
- **No raw canvas context** — `drawingContext.beginPath()` draws nothing in WEBGL
- **`TAU` is predefined** — do NOT declare `var TAU`
- **Use `beginShape()/vertex()/endShape()`** for all curves and polygons

### Reference
- **Zouaq Pattern Generator**: github.com/volumique/zouaq-pattern-generator
  - Hasba method implementation, 17 wallpaper groups
  - Pattern types: zellige, zouaq strapwork, rosettes, fractals, mandalas, shamsa, tawriq, tastir, amazigh
  - Architecture: PatternConfig → normalizeConfig → generateTileShapes → generatePattern → SVG

### Lesson 23: Pixel counts alone aren't verification
Counting "reddish" pixels with loose thresholds (R>G) catches warm-toned backgrounds, not actual drawing.
Use strict threshold: `(R > G+15) & (R > B+15)` for distinctly red content.
Also check: zone distribution, component count, transitions per row, JS error count, frameCount.
"41K reddish pixels" was 100% background. "64K distinctly red pixels" was real.

### Lesson 24: Rewrite > patch when accumulated edits break a file
Multiple sed/python patches create cascading syntax errors (stripped comments, broken tokens).
When >3 patches fail, use Write tool to rewrite the entire file clean from scratch.

### TRACED Verification Checklist (pattern fills)
1. `node --check` — zero syntax errors
2. dev-browser errors array — zero JS errors
3. `frameCount > 0` — draw() actually ran
4. Strict red pixel count > 20K
5. Per-zone distribution (spandrels, arch, panels, borders)
6. Component count per zone (patterns = many small components)
7. Transitions per row (patterns = high, plain lines = low)

## Girih Tiles — Islamic Quasicrystal Pattern System (Added 2026-03-28)

### The Discovery
Lu & Steinhardt (2007, Science) showed that Islamic artisans at the Darb-i Imam shrine in Isfahan (1453)
created quasicrystal patterns matching Penrose tilings — 500 years before Western mathematics.

### The 5 Girih Tiles
All edges equal length. All angles = n×36° (π/5). This is the "girih quantum."

| Tile | Persian Name | Sides | Angles |
|------|-------------|-------|--------|
| Regular decagon | Tabl | 10 | 144° |
| Elongated hexagon | Shesh Band | 6 | [72°, 144°, 144°] ×2 |
| Bowtie (non-convex) | Sormeh Dan | 6 | [72°, 72°, 216°] ×2 |
| Golden rhombus | Torange | 4 | [72°, 108°] ×2 |
| Regular pentagon | Pange | 5 | 108° |

### Strapwork (Girih Lines) — What You Actually See
- Lines cross each tile edge at **54° (3π/10)** from the edge
- Two lines cross each edge
- Lines are straight, may have one sharp bend (at multiples of 36°)
- The tile boundaries are HIDDEN — only the strapwork is visible
- This is pure ruler-and-compass: consistent with TRACED philosophy

### The Golden Ratio Connection
- Pentagon diagonal/side = φ = (1+√5)/2
- Decagon inherits φ from its pentagonal subdivision
- Rhombus angles (72°/108°) are the golden gnomon angles
- φ-scaling between tile levels creates self-similar quasicrystal

### Mathematical Constants (girih.js)
```
A36 = π/5       // The quantum angle
A72 = 2π/5      // Pentagon interior supplement
A54 = 3π/10     // Strapwork crossing angle
PHI = (1+√5)/2  // Golden ratio (embedded in all 5 tiles)
SIN36, COS36, SIN72, COS72  // Precomputed
```

### Strapwork Construction Algorithm
For each tile:
1. Find midpoints of all edges
2. Connect midpoints that are N edges apart (N depends on tile type):
   - Decagon: connect midpoints 3 apart → {10/3} star polygon
   - Pentagon: connect midpoints 2 apart → pentagram
   - Rhombus: connect opposite edge midpoints → X pattern
   - Hexagon/Bowtie: connect midpoints 2 apart → zigzag
3. The resulting line network IS the visible pattern

### Multi-Scale Composition
Self-similar at exactly 2 levels (traditional):
- Level 1: Large decagons at grid intersections
- Level 2: Pentagons filling gaps (φ⁻¹ scale)
- Level 3: Rhombi + bowties in remaining gaps (φ⁻² scale)

For 3+ levels, each large tile subdivides into smaller tiles of the same 5 types
(the Darb-i Imam substitution rules).

### Integration with pretextFill
Girih tiles adapt to variable-width containers naturally:
- `pretextFill(archWidth, yStart, yEnd, tileSize, girihPentagon)` — pentagons flow along arch
- Staggered rows create quasicrystal feel (offset every other row by tileSize/2)

### References
- Lu & Steinhardt, "Decagonal and Quasi-Crystalline Tilings in Medieval Islamic Architecture" (Science, 2007)
- Bonner, "Islamic Geometric Patterns" (Springer, 2017)
- Eriksson, "Extended Girih Tiles" (Bridges 2020)
- Darb-i Imam shrine, Isfahan, Iran (1453)
- Topkapi Scroll (15th century) — pattern template scroll
