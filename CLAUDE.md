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
