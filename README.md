# TRACED — Photo → Parametric Architectural Drawing

_"I want to come up with a method that is repeatable — give you an image, get an architectural clone."_ — Tawfeeq Martin, 2026-03-26

**The gap nobody's filled:** Photo → proportional skeleton → animated procedural drawing.

Academia stops at facade segmentation or 3D reconstruction. Nobody extracts the *architectural grammar* (golden ratio relationships, Islamic geometric ratios) and converts it into a drawing recipe. Until now.

---

## Table of Contents
1. [The Problem](#1-the-problem)
2. [The Solution: Traced Pipeline](#2-the-solution-traced-pipeline)
3. [Quick Start](#3-quick-start)
4. [Tools Reference](#4-tools-reference)
5. [Architectural Foundations](#5-architectural-foundations)
6. [The Three Canonical Proportional Systems](#6-the-three-canonical-proportional-systems)
7. [Historical Proportional Methods](#7-historical-proportional-methods)
8. [Segmentation Models (SAM 2 / SAM 3)](#8-segmentation-models)
9. [First Test Case: Sheikh Zayed Grand Mosque](#9-first-test-case-sheikh-zayed-grand-mosque)
10. [What Makes Traced Novel](#10-what-makes-traced-novel)
11. [How to Present This](#11-how-to-present-this)
12. [Roadmap](#12-roadmap)
13. [References & Vocabulary](#13-references--vocabulary)

---

## 1. The Problem

When extracting proportions from a reference photo by eye ("dome is at H×0.26"), the drawing doesn't match. Translation loss compounds across 15+ elements. Manual guessing = unrepeatable, inaccurate, slow.

## 2. The Solution: Traced Pipeline

Five stages, from photo to verified drawing:

### Stage 1: STRUCTURE EXTRACTION
- **SAM 2/3** automatic mask generation segments every element
- Fallback: Canny/Sobel edge detection + Hough line transform
- Circle/ellipse detection for domes and arches
- **Output:** Raw geometric primitives with pixel coordinates

### Stage 2: SEMANTIC LABELING
- Shape classification: dome, arch, column, wall, panel, openwork (grille)
- Based on circularity, solidity, convexity, aspect ratio
- SAM 3 adds text-prompt labeling ("segment all domes")
- **Output:** Labeled element tree with bounding boxes

### Stage 3: PROPORTIONAL ANALYSIS (THE KEY)
- For each element pair, compute ratios (width, height, distance, position)
- Test against known proportional systems:
  - **Golden:** φ (1.618), 1/φ (0.618), φ² (2.618), 1/φ² (0.382)
  - **Geometric:** √2 (1.414), 1/√2, √3 (1.732), √5 (2.236)
  - **Simple:** 1/2, 1/3, 2/3, 1/4, 3/4, 2, 3
  - **Angular:** π/2, π/3, π/4
- Flag closest match + error %
- Green (<2% error) = strong match, Yellow (2-5%) = possible, Grey (>5%) = weak
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

---

## 3. Quick Start

### Setup (one-time, ~10 min)
```bash
cd /var/lib/cookmom-workspace
git clone https://github.com/cookmom/traced.git
cd traced
bash setup.sh
```

### Run Extraction
```bash
conda activate traced
cd /var/lib/cookmom-workspace/traced
python extract-sam2.py --image examples/szm-reference.jpg --output szm-extraction.json
```

### What Happens
1. SAM 2 segments every element in the photo automatically
2. Each segment classified by shape (dome, arch, column, wall, panel)
3. Geometric primitives fitted (circles, ellipses, rectangles)
4. All pairwise ratios computed and matched against φ, √2, √3, etc.
5. JavaScript code generated — paste into drawing template

### Output
- `szm-extraction.json` — full extraction with elements, primitives, ratios
- Console prints generated JavaScript constants
- Use `--js-output constants.js` to save JS to file

### File Structure
```
traced/
├── setup.sh              # One-time setup (conda + SAM 2 + checkpoints)
├── extract-sam2.py        # SAM 2 auto-extraction (primary)
├── extract.py             # SAM 3 extraction (when access approved)
├── traced.html            # Browser: manual keypoint tool
├── validate.html          # Browser: visual comparison/wipe tool
├── README.md              # This file (everything in one place)
├── examples/
│   ├── szm-reference.jpg  # Sheikh Zayed Mosque reference
│   └── szm-spec.md        # SZM drawing specification
└── checkpoints/           # SAM 2 model weights (auto-downloaded by setup.sh)
    ├── sam2.1_hiera_large.pt
    └── sam2.1_hiera_base_plus.pt
```

---

## 4. Tools Reference

### `extract-sam2.py` — SAM 2 Automatic Extraction
```bash
python extract-sam2.py --image photo.jpg --output extraction.json
python extract-sam2.py --image photo.jpg --output extraction.json --js-output constants.js
python extract-sam2.py --image photo.jpg --checkpoint /path/to/checkpoint.pt
```
- Automatic mask generation — no prompts needed
- Shape classification: dome, arch, column, wall, panel, openwork, horizontal_band
- Geometric primitive fitting: circles (domes), ellipses (arches), rectangles (panels)
- Falls back to OpenCV if SAM 2 not installed

### `extract.py` — SAM 3 Text-Prompted Extraction
```bash
python extract.py --image photo.jpg --output extraction.json
python extract.py --image photo.jpg --prompts "dome,arch,column,minaret"
```
- Requires SAM 3 access (HuggingFace gated, request at https://huggingface.co/facebook/sam3)
- Text prompts: "dome", "arch", "column" → segments matching elements
- Falls back to OpenCV if SAM 3 not installed

### `traced.html` — Browser Keypoint Tool
- Load reference photo, click keypoints on architectural landmarks
- Name each keypoint (dome_apex, arch_spring_L, portal_top_L...)
- Group keypoints into elements (main_dome = {apex, base_L, base_R})
- Auto-computes all pairwise ratios, finds nearest mathematical constant
- Generates JavaScript proportional constants ready to paste
- Export/import keypoints as JSON for iteration
- Mobile-friendly (touch events, pinch-to-zoom)
- **URL:** `https://cookmom.github.io/fatiha/fatiha-proto/traced.html`

### `validate.html` — Visual Comparison Tool
- Split wipe, opacity blend, onion skin, edge overlay
- Sobel edge detection: RED = ref only, CYAN = drawing only, WHITE = aligned
- SSIM + MAD + centroid offset metrics
- 9:16 crop for reference images (1080×1920 target)
- Golden ratio alignment grid + measurement tool
- **URL:** `https://cookmom.github.io/fatiha/fatiha-proto/validate.html`

---

## 5. Architectural Foundations

_Why this tool is grounded in 2,500 years of architectural practice, not just pixel matching._

Architectural proportion is the **oldest continuous design discipline in human civilization**. It predates written architectural theory — builders transmitted proportional knowledge through geometric construction techniques, not measurements.

### The Core Insight
Great architecture isn't designed in absolute dimensions (millimeters, feet). It's designed in **relational ratios** — the width of a column relative to its height, the height of a dome relative to its drum, the span of an arch relative to its rise. These ratios are what the eye perceives as harmony.

> "Proportional systems are arithmetic or geometric methods of organizing architectural form that determine the mutual proportional relationships of the parts and the whole." — Januszewski, *Architectus* (2023)

### Why This Matters for Traced
When we extract keypoints from a photograph and compute ratios, we're doing **exactly what an architect does when analyzing a building** — finding the relational system that makes it coherent. We're not tracing pixels. We're recovering the *proportional grammar* the architect used.

---

## 6. The Three Canonical Proportional Systems

Across all traditions (Classical, Islamic, Gothic, Renaissance), architecture uses three primary geometric ratios. Keith Critchlow (1976) identified these as the foundations of Islamic sacred geometry:

### √2 System — *Ad Quadratum* ("From the Square")
- **Ratio:** 1.414...
- **Source geometry:** Diagonal of the unit square
- **Where it appears:**
  - Roman house plans (Pompeii, Herculaneum)
  - Gothic cathedral cross-sections
  - Islamic octagonal patterns (the rotated square)
  - Floor tiling at Spoleto's Duomo
- **Construction:** Draw a square → its diagonal = side × √2
- **Architectural meaning:** Stability, earthly order, the material world

### √3 System — *Ad Triangulum* ("From the Triangle")
- **Ratio:** 1.732...
- **Source geometry:** Height of the equilateral triangle (h = side × √3/2)
- **Where it appears:**
  - Gothic cathedral elevations (Milan Cathedral debate of 1391)
  - Pointed arch proportions (equilateral arch: rise = span × √3/2)
  - Hexagonal Islamic patterns
  - Minaret shaft proportions
- **Construction:** Draw equilateral triangle → height = base × √3/2
- **Architectural meaning:** Aspiration, vertical thrust, spiritual ascent

### φ System — Golden Ratio / *Sectio Divina*
- **Ratio:** 1.618... (and its reciprocal 0.618...)
- **Source geometry:** Self-similar division of a line
- **Where it appears:**
  - Parthenon facade proportions
  - Le Corbusier's Modulor
  - Fibonacci spirals in Islamic rosettes
  - Renaissance facade divisions (Palladio, Alberti)
  - Dome-to-drum height ratios in Ottoman mosques
- **Construction:** Divide line at point where whole:large = large:small
- **Architectural meaning:** Organic harmony, divine proportion, natural growth

### The Islamic Synthesis
Islamic architecture is unique in using **all three systems simultaneously**, often in the same building:
- √2 in plan geometry (octagonal courtyards, rotated square minarets)
- √3 in pointed arches and vertical elevation
- φ in dome proportions and ornamental subdivision

This is why Traced tests against all three systems — any great building may use different ratios for different relationships.

---

## 7. Historical Proportional Methods

### Vitruvius (1st century BC)
First codified proportional theory in *De Architectura*. Established the **modular system**: choose a base unit (column diameter), derive everything else as integer multiples or fractions.

**Traced equivalent:** Our `U = W/(2φ+2)` base unit. Everything derived from U.

### Medieval Master Builders (12th–15th century)
Used **compass-and-straightedge geometry** rather than numerical calculation. Proportions were *constructed*, not computed — you didn't need to know √2 = 1.414, you just drew the diagonal of a square.

The great Gothic cathedral debates: **Ad Quadratum vs Ad Triangulum**. Milan Cathedral (1391) famously resolved a dispute between builders who wanted to use √2 proportions and those who wanted √3. The archives record this as *"aut secundum quadratum aut secundum triangulum"* — they literally voted on which geometric system to use.

**Traced equivalent:** Our ratio matching doesn't impose a system — it *discovers* which system the building uses.

### Palladio (16th century)
Published specific room proportions in *Quattro Libri*. Recommended seven ideal room shapes, all based on simple ratios: 1:1, √2:1, 4:3, 3:2, 5:3, φ:1, 2:1.

**Traced equivalent:** Our ratio library includes all of these.

### Le Corbusier (20th century)
Created the **Modulor** — a proportional system based on human body dimensions and φ. Two interlocking Fibonacci-like sequences (red and blue series) that architects use to derive all dimensions.

**Traced equivalent:** Our generated JavaScript code produces a similar coherent series of derived constants.

---

## 8. Segmentation Models

### Current: SAM 2 (Meta, 2024)
- **No gating** — fully open, download directly
- Automatic mask generation — finds every segment without prompts
- Point/click prompting for refinement
- ~300M parameters, runs easily on RTX A6000
- Checkpoint: `sam2.1_hiera_large.pt` (~900MB)

### Next: SAM 3 (Meta, Nov 2025)
- **Gated** — request access at https://huggingface.co/facebook/sam3
- Text-prompt segmentation: "segment all domes" → masks for every dome
- 848M parameters, 4M+ unique concepts
- Presence Token for discriminating similar prompts
- ONNX + web runtime support
- Waiting for access approval

### Segmentation Model Comparison

| Model | Prompting | Gating | Params | Our Use |
|-------|-----------|--------|--------|---------|
| **SAM 2** | Point/click, auto | None | ~300M | Primary — auto-extract all elements |
| **SAM 3** | Text, visual, auto | HF gated | 848M | Future — "segment all domes" |
| **DINOv2** | Features only | None | ~300M | Future — backbone for custom classifier |
| **YOLO11-seg** | Trained classes | None | Varies | Production speed, needs training data |

### Future: Browser-Side Inference
- SAM 2/3 have ONNX export paths — investigate running in-browser via ONNX Runtime Web / WebGPU
- Transformers.js supports SAM 2 (check model size feasibility)
- WebGPU on Tawfeeq's A6000s would give GPU inference in Chrome
- Goal: entire Traced pipeline runs client-side, no server, no Python — just a webpage
- This would make Traced a shareable tool anyone can use

---

## 9. First Test Case: Sheikh Zayed Grand Mosque

**Architect:** Yusef Abdelki (Syrian)
**Design team:** Basem Barghouti, Moataz Al-Halabi, Imad Malas
**Period:** 1996–2007
**Style:** Synthesis of Mughal, Ottoman, Fatimid, and Moorish traditions

### Key References Cited by Architect:
- **Abu al-Abbas al-Mursi Mosque** (Alexandria) — designed by Mario Rossi, 1920s
- **Badshahi Mosque** (Lahore, Pakistan) — Mughal proportional system
- Persian, Mughal, and Indo-Islamic architectural traditions

### Physical Dimensions:
- **82 domes** of varying sizes
- **4 minarets**, each 106m tall
- Minarets use **three geometric transitions**: square base → octagonal middle → circular top
  - Square → octagonal = **Ad Quadratum** (rotated 45° square inscribed in circle)
  - Octagonal → circular = progressive polygon smoothing
- Main prayer hall dome: one of the largest mosque domes in the world

### Observable Proportional System:
- **Minaret transitions:** Square → Octagon → Circle = Mamluk era influence
- **Dome hierarchy:** Main → secondary → colonnade domes follow reducing scale, likely φ-based
- **Pointed arches:** Ogee and horseshoe profiles = √3 or greater-than-equilateral
- **Courtyard grid:** 8-fold and 12-fold symmetry (Ad Quadratum family)

### Drawing (szm.html)
- 21-step animated pencil drawing with p5.brush
- Keyhole arch frame, domes, portal, courtyard
- HUD tracking boxes, annotations, crosshair cursor
- Based on dome.html v34 template (Ottoman mihrab — banked as `v34-banked` tag)

---

## 10. What Makes Traced Novel

| Existing Approach | What It Does | What It Doesn't Do |
|---|---|---|
| Facade Segmentation (DeepFacade) | Labels pixels as "wall/window/door" | Doesn't extract proportional relationships |
| 3D Reconstruction (SfM/NeRF) | Builds 3D model from photos | Doesn't identify the geometric grammar |
| BIM Generation | Creates parametric model | Requires manual modeling, no photo input |
| Style Transfer (neural) | Applies visual style | Destroys structural relationships |
| **Traced** | **Extracts proportional skeleton → tests against canonical ratios → generates parametric drawing** | — |

### Traced's Contribution
1. **Bridges the analysis-to-creation gap** — existing tools analyze OR create, never both
2. **Recovers architectural intent** — not just "where are things" but "what ratio governs their placement"
3. **Speaks the language of architects** — outputs in terms architects recognize (φ, √2, ad quadratum)
4. **Generates executable code** — not a report, but a working animated drawing
5. **Validates against source** — closed-loop verification via edge overlay + SSIM

---

## 11. How to Present This

### To Laymen:
"We built a tool that looks at a photo of a building and figures out the mathematical recipe the architect used — the same hidden ratios that make the Parthenon and the Taj Mahal feel 'right.' Then it draws the building from scratch using those exact proportions."

### To Architects:
"Traced performs proportional analysis on architectural photography — extracting ad quadratum, ad triangulum, and golden section relationships from keypoint measurements. It outputs a parametric proportional skeleton that can be verified against the source using structural similarity metrics. Think of it as computational regulating lines."

### To Engineers/Developers:
"Photo → keypoint extraction → pairwise ratio computation → nearest-constant matching (φ, √2, √3, simple fractions) → JavaScript code generation → animated procedural drawing → visual validation loop. Client-side, no ML inference needed for the manual path; SAM 2/3 for automatic extraction."

---

## 12. Roadmap

### Built ✅
- [x] Manual keypoint extraction tool (traced.html)
- [x] Visual comparison/wipe tool (validate.html)
- [x] SAM 2 automatic extraction pipeline (extract-sam2.py)
- [x] SAM 3 extraction pipeline (extract.py — pending access)
- [x] Shape classification (dome, arch, column, wall, panel, openwork)
- [x] Proportional analysis engine (17 canonical ratios)
- [x] JavaScript code generation
- [x] OpenCV fallback extraction
- [x] Sheikh Zayed Mosque first test case

### Next 🔜
- [ ] SAM 3 access + integration
- [ ] End-to-end: extraction → code generation → p5.brush HTML output
- [ ] Validate loop: auto-render → edge compare → iterate
- [ ] Template library (save proportional skeletons for reuse)

### Future 🔮
- [ ] Browser-side SAM inference (ONNX + WebGPU)
- [ ] Animation grammar (auto drawing order from architectural hierarchy)
- [ ] Style transfer (apply one building's proportions to another's elements)
- [ ] Auto-keypoint detection (CV finds dome peaks, arch springings automatically)
- [ ] Multi-view reconstruction (extract from multiple photos of same building)

---

## 13. References & Vocabulary

### Academic References
- Critchlow, K. (1976). *Islamic Patterns: An Analytical and Cosmological Approach*. Thames & Hudson.
- Dabbour, L. (2012). "Geometric proportions: The underlying structure of design process for Islamic geometric patterns." *Frontiers of Architectural Research*, 1(4).
- Fletcher, R. (2020). *Infinite Measure: Learning to Design in Geometric Harmony with Art, Architecture, and Nature*.
- Januszewski, W. (2023). "An outline of the geometric proportion systems in architecture." *Architectus*.
- Pacioli, L. (1509). *De Divina Proportione*. (With 60 drawings by Leonardo da Vinci.)
- Vitruvius. (1st century BC). *De Architectura*.
- Bork, R. (2003). "Ad Quadratum: The Practical Application of Geometry in Medieval Architecture."
- Bridges Mathematical Art Conference (2022). "Revisiting Ad Quadratum and Ad Triangulum to Generate Hyperbolic Tessellations."

### Key Vocabulary

| Term | Meaning |
|---|---|
| **Ad Quadratum** | Proportions derived from the square and its diagonal (√2) |
| **Ad Triangulum** | Proportions derived from the equilateral triangle (√3) |
| **Sectio Divina** | Divine section / golden ratio (φ = 1.618...) |
| **Regulating Lines** | Construction lines that establish proportional relationships in a facade |
| **Modular System** | Proportions as multiples/fractions of a chosen base unit |
| **Dynamic Symmetry** | Jay Hambidge's system using root rectangles (√2, √3, √5) |
| **Modulor** | Le Corbusier's human-scale proportional system based on φ |
| **Muqarnas** | Islamic stalactite vaulting — fractal subdivision following geometric ratios |
| **Girih** | Islamic geometric patterns using 5 tile types — encode φ relationships |
| **SSIM** | Structural Similarity Index — measures structural match between two images |

---

## Origin
- Conceived 2026-03-26 by chef + Tawfeeq Martin
- Born from the frustration of eyeballing SZM proportions and getting them wrong
- The goal: give any reference image → get an architectural clone
- Repository: https://github.com/cookmom/traced
