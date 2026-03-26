# Sheikh Zayed Grand Mosque — Pencil Drawing Spec

## Reference
Photo looking through a horseshoe/ogee arch (foreground frame) toward the main facade of Sheikh Zayed Grand Mosque, Abu Dhabi.

## Vertical Hierarchy (top → bottom, approximate % of canvas height)

| Element | % Height | Description |
|---------|----------|-------------|
| Sky / negative space | ~5% | Above crescent finial |
| Crescent finial | ~3% | Gold crescent on spire atop main dome |
| Main dome | ~15% | Large hemisphere, white marble with subtle ribs |
| Main dome drum | ~5% | Cylindrical base with blind arched windows |
| Secondary dome | ~8% | Smaller dome in front/below main dome |
| Secondary drum | ~4% | Colonnade with pointed arch windows |
| Flanking minor domes | ~3% | Row of small domes on colonnades, each side |
| Facade / portal zone | ~15% | Central entrance: nested pointed arches, geometric panels, gold window grilles |
| Courtyard floor | ~5% | White marble with geometric inlay pattern |
| Framing arch — top | ~25% | The horseshoe/ogee arch we're looking through |
| Framing arch — sides | ~12% | Thick masonry walls with decorative panels |

## Proportional System (φ-based)

```
U = W / (2φ + 2)                    // base unit (~206px at 1080w)
mainDomeR = U × φ                   // main dome radius
secDomeR = mainDomeR / φ            // secondary dome radius  
drumH = mainDomeR / φ²              // drum heights
archFrameW = U × φ²                 // framing arch wall thickness
minorDomeR = secDomeR / φ           // flanking dome radius
portalH = U × φ                     // portal zone height
finialH = U / φ²                    // crescent finial height
```

## Drawing Sequence (18 steps)

### Phase 1: Framing Arch (the viewport we look through)
0. Ground line — base of the framing arch
1. Left arch pier — thick masonry wall, draws upward
2. Right arch pier — mirror
3. Horseshoe arch — ogee/horseshoe curve connecting piers overhead

### Phase 2: Main Dome Complex (background, seen through arch)
4. Main dome — large hemisphere, compass sweep
5. Main dome drum — cylindrical base with arched blind windows
6. Crescent finial — spire + crescent at apex

### Phase 3: Secondary Dome
7. Secondary dome — smaller hemisphere in front
8. Secondary drum — with pointed arch colonnade windows

### Phase 4: Flanking Elements
9. Left minor domes — row of small domes on colonnade
10. Right minor domes — mirror

### Phase 5: Facade Portal
11. Portal frame — outer pointed arch of entrance
12. Inner nested arches — 2-3 nested receding arches
13. Geometric panels — decorative rectangles flanking portal
14. Window grilles — gold geometric lattice patterns

### Phase 6: Floor + Details
15. Courtyard floor — geometric inlay pattern visible through arch

### Phase 7: Watercolor Washes
16. Sky wash — gradient blue through the arch opening
17. Dome wash — very pale warm grey/white tint on domes
18. Gold accents — crescent, window grilles, portal details
19. Shadow wash — cool grey for depth/perspective layers
20. Floor wash — warm white marble tint

## Framing Arch Geometry
- **Type**: Horseshoe arch (slightly wider than semicircle)
- **Construction**: Center raised above springing line by ~15% of radius
- **The arch IS the canvas frame** — everything else is seen through it
- **Arch wall thickness**: ~15-20% of canvas width on each side

## Color Palette (watercolor washes)
- Sky: `#87CEEB` α=40 (soft blue)
- White marble: `#F5F0E8` α=15 (warm off-white)  
- Gold accents: `#C4A35A` α=60 (muted gold)
- Shadow depth: `#4A5568` α=25 (cool blue-grey)
- Stone wall: `#D4C5B0` α=30 (warm sandstone)

## Key Differences from Dome Prototype
1. **Perspective depth** — foreground arch frames a distant building
2. **Horseshoe arch** vs pointed arch — different construction geometry
3. **Multiple domes** at different scales and depths
4. **Horizontal spread** — wider composition with flanking colonnades
5. **More ornamental detail** — geometric grilles, nested arches
6. **White/gold palette** vs blue/magenta
