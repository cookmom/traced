# TRACED — Photo → Parametric Architectural Drawing

**The gap nobody's filled:** Photo → proportional skeleton → animated procedural drawing.

Academia stops at facade segmentation or 3D reconstruction. Nobody extracts the *architectural grammar* (golden ratio relationships, Islamic geometric ratios) and converts it into a drawing recipe.

## The Problem
When I eyeball proportions from a reference photo ("dome is at H×0.26"), the drawing doesn't match. Translation loss compounds across 15+ elements. Manual guessing = unrepeatable, inaccurate, slow.

## The Solution: Traced Pipeline (5 Stages)

### Stage 1: STRUCTURE EXTRACTION
- Canny/Sobel edge detection on photo
- Hough line transform → dominant vertical/horizontal lines
- Circle/ellipse detection → domes, arches
- **Output:** Raw geometric primitives with pixel coordinates

### Stage 2: SEMANTIC LABELING
- LLM vision or manual keypoint placement labels each primitive
- "This arc = main dome", "these lines = portal frame"
- Establish hierarchy: foreground frame → background elements
- **Output:** Labeled element tree with bounding boxes

### Stage 3: PROPORTIONAL ANALYSIS (THE KEY)
- For each element pair, compute ratios
- Test against known proportional systems:
  - **Golden:** φ (1.618), 1/φ (0.618), φ² (2.618), 1/φ² (0.382)
  - **Geometric:** √2 (1.414), 1/√2, √3 (1.732), √5 (2.236)
  - **Simple:** 1/2, 1/3, 2/3, 1/4, 3/4
  - **Angular:** π/2, π/3, π/4
- Flag closest match + error %
- Green (<2% error) = strong match, Yellow (2-5%) = possible, Grey (>5%) = none
- **Output:** Proportional skeleton — `mainDomeR = frameW × 0.38 ≈ frameW/φ²`

### Stage 4: CODE GENERATION
- Convert proportional skeleton → p5.brush STEPS array
- Each element: name, formula, bbox, draw(progress) function
- Drawing order from depth or hierarchy
- **Output:** Complete drawing HTML with exact proportions from photo

### Stage 5: VISUAL VALIDATION
- Render drawing → screenshot at same resolution
- validate.html → edge overlay + SSIM score
- If SSIM < threshold → feed deviations back to Stage 3 → adjust → re-render
- Loop until convergence

## Tools Built

### `traced.html` — Keypoint Extraction + Proportional Analysis
- Load reference photo, click keypoints on architectural landmarks
- Name each keypoint (dome_apex, arch_spring_L, portal_top_L...)
- Group keypoints into elements (main_dome = {apex, base_L, base_R})
- Auto-computes all pairwise ratios, finds nearest mathematical constant
- Generates JavaScript proportional constants ready to paste
- Export/import keypoints as JSON for iteration
- Mobile-friendly (touch events, pinch-to-zoom)

### `validate.html` — Visual Comparison Tool
- Split wipe, opacity blend, onion skin, edge overlay
- Sobel edge detection on both images — RED = ref only, CYAN = drawing only, WHITE = aligned
- SSIM + MAD + centroid offset metrics
- 9:16 crop for reference images
- Golden ratio alignment grid

### `szm.html` — Sheikh Zayed Mosque (first test case)
- 21-step animated pencil drawing with p5.brush
- Keyhole arch frame, domes, portal, courtyard
- HUD tracking boxes, annotations, crosshair cursor
- Based on dome.html v34 template

## Research Context

### What Exists (State of the Art)
- **DeepFacade / SOLOv2 / Mask R-CNN** — facade semantic segmentation (wall, window, door). Gives *what*, not precise *where*.
- **Reflect3D (CVPR 2025)** — zero-shot 3D symmetry detection from single image. Useful for center axes.
- **SI3FP (2025)** — Scalable Image-to-3D Facade Parser. Extracts LoD3 geometry from photos.
- **DAN-PSPNet-Lsym** — deep learning framework integrating prior knowledge for facade element detection with proportionate shapes.

### What Doesn't Exist (Our Gap)
Nobody has built **photo → parametric proportional skeleton → animated procedural drawing**.
- Papers stop at segmentation or 3D reconstruction
- Nobody extracts golden ratio relationships or architectural grammar
- Nobody generates a drawing recipe from proportional analysis
- This is the novel contribution

## Architecture Principle
**Keypoints > Segmentation.** Architects don't segment — they locate landmark points and derive proportions. Our tool works the same way. Click 20 keypoints → get the full proportional system → generate the drawing.

## Future: Browser-Side Inference
- SAM 2/3 have ONNX export paths — investigate running in-browser via ONNX Runtime Web / WebGPU
- Transformers.js supports SAM 2 (check model size feasibility)
- WebGPU on Tawfeeq's A6000s would give GPU inference in Chrome
- Goal: entire Traced pipeline runs client-side, no server, no Python — just a webpage
- This would make Traced a shareable tool anyone can use

## Future Stages (Not Yet Built)
- **Auto-keypoint detection** — CV finds keypoints automatically (dome peaks, arch springings, symmetry intersections)
- **Template library** — save proportional skeletons for reuse (Ottoman mosque template, Gothic cathedral template, etc.)
- **Style transfer** — apply one building's proportional system to another building's elements
- **Animation grammar** — auto-determine drawing order from architectural hierarchy (foundation → structure → ornament)
- **p5.brush code generation** — full automatic STEPS array from keypoint data

## File Structure
```
fatiha-proto/
├── traced.html          # Keypoint extraction tool
├── validate.html        # Visual comparison tool  
├── szm.html            # Sheikh Zayed Mosque drawing
├── szm-reference.jpg   # Reference photo
├── SZM-SPEC.md         # Drawing specification
├── TRACED.md           # This file
└── dome.html           # Ottoman mihrab (v34-banked template)
```

## Origin
- Conceived 2026-03-26 by chef + Tawfeeq
- Born from the frustration of eyeballing SZM proportions and getting them wrong
- "I want to come up with a method that is repeatable" — Tawfeeq
- The goal: give any reference image → get an architectural clone
