# Traced — Architectural Foundations

_Why this tool is grounded in 2,500 years of architectural practice, not just pixel matching._

## 1. The Discipline of Architectural Proportion

Architectural proportion is the **oldest continuous design discipline in human civilization**. It predates written architectural theory — builders transmitted proportional knowledge through geometric construction techniques, not measurements.

### The Core Insight
Great architecture isn't designed in absolute dimensions (millimeters, feet). It's designed in **relational ratios** — the width of a column relative to its height, the height of a dome relative to its drum, the span of an arch relative to its rise. These ratios are what the eye perceives as harmony.

> "Proportional systems are arithmetic or geometric methods of organizing architectural form that determine the mutual proportional relationships of the parts and the whole." — Januszewski, *Architectus* (2023)

### Why This Matters for Traced
When we extract keypoints from a photograph and compute ratios, we're doing **exactly what an architect does when analyzing a building** — finding the relational system that makes it coherent. We're not tracing pixels. We're recovering the *proportional grammar* the architect used.

## 2. The Three Canonical Proportional Systems

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
  - Gothic cathedral elevations (Milan Cathedral debate)
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

## 3. Historical Proportional Methods

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

## 4. Sheikh Zayed Grand Mosque — Design Context

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
- Courtyard: one of the largest marble mosaic designs

### Proportional System (Observable):
- **Minaret transitions:** Square → Octagon → Circle = Mamluk era influence (geometric metamorphosis)
- **Dome hierarchy:** Main dome → secondary domes → colonnade domes follow a **reducing scale**, likely φ-based
- **Pointed arches:** Ogee and horseshoe profiles = √3 or greater-than-equilateral construction
- **Courtyard grid:** Geometric floral inlay follows **8-fold and 12-fold** symmetry (Ad Quadratum family)

## 5. What Makes Traced Novel

| Existing Approach | What It Does | What It Doesn't Do |
|---|---|---|
| Facade Segmentation (DeepFacade) | Labels pixels as "wall/window/door" | Doesn't extract proportional relationships |
| 3D Reconstruction (SfM/NeRF) | Builds 3D model from photos | Doesn't identify the geometric grammar |
| BIM Generation | Creates parametric model | Requires manual modeling, no photo input |
| Style Transfer (neural) | Applies visual style | Destroys structural relationships |
| **Traced** | **Extracts proportional skeleton from photo → tests against canonical ratios → generates parametric drawing code** | — |

### Traced's Contribution
1. **Bridges the analysis-to-creation gap** — existing tools analyze OR create, never both
2. **Recovers architectural intent** — not just "where are things" but "what ratio governs their placement"
3. **Speaks the language of architects** — outputs in terms architects recognize (φ, √2, ad quadratum)
4. **Generates executable code** — not a report, but a working animated drawing
5. **Validates against source** — closed-loop verification via edge overlay + SSIM

## 6. How to Present This

### To Laymen:
"We built a tool that looks at a photo of a building and figures out the mathematical recipe the architect used — the same hidden ratios that make the Parthenon and the Taj Mahal feel 'right.' Then it draws the building from scratch using those exact proportions."

### To Architects:
"Traced performs proportional analysis on architectural photography — extracting ad quadratum, ad triangulum, and golden section relationships from keypoint measurements. It outputs a parametric proportional skeleton that can be verified against the source using structural similarity metrics. Think of it as computational regulating lines."

### To Engineers/Developers:
"Photo → keypoint extraction → pairwise ratio computation → nearest-constant matching (φ, √2, √3, simple fractions) → JavaScript code generation → animated procedural drawing → visual validation loop. All client-side, no ML inference needed."

## References

- Critchlow, K. (1976). *Islamic Patterns: An Analytical and Cosmological Approach*. Thames & Hudson.
- Dabbour, L. (2012). "Geometric proportions: The underlying structure of design process for Islamic geometric patterns." *Frontiers of Architectural Research*, 1(4).
- Fletcher, R. (2020). *Infinite Measure: Learning to Design in Geometric Harmony with Art, Architecture, and Nature*.
- Januszewski, W. (2023). "An outline of the geometric proportion systems in architecture." *Architectus*.
- Pacioli, L. (1509). *De Divina Proportione*. (With 60 drawings by Leonardo da Vinci.)
- Vitruvius. (1st century BC). *De Architectura*.
- Bork, R. (2003). "Ad Quadratum: The Practical Application of Geometry in Medieval Architecture."
- Bridges Mathematical Art Conference (2022). "Revisiting Ad Quadratum and Ad Triangulum to Generate Hyperbolic Tessellations."

## Key Vocabulary

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
