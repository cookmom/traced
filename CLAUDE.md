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
