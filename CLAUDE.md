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
