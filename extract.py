#!/usr/bin/env python3
"""
Traced Pipeline — Stage 1+2+3: SAM 3 Architectural Element Extraction
Extracts architectural elements from a reference photo using SAM 3,
fits geometric primitives, and computes proportional analysis.

Usage:
    python extract.py --image photo.jpg --output extraction.json
    python extract.py --image photo.jpg --output extraction.json --prompts "dome,arch,column"
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

# ============================================================
# CONSTANTS — Canonical architectural proportions
# ============================================================
PHI = (1 + math.sqrt(5)) / 2  # 1.618...
CANONICAL_RATIOS = {
    "φ": PHI,
    "1/φ": 1 / PHI,
    "φ²": PHI ** 2,
    "1/φ²": 1 / PHI ** 2,
    "√2": math.sqrt(2),
    "1/√2": 1 / math.sqrt(2),
    "√3": math.sqrt(3),
    "√3/2": math.sqrt(3) / 2,
    "√5": math.sqrt(5),
    "2": 2.0,
    "3": 3.0,
    "1/2": 0.5,
    "1/3": 1 / 3,
    "2/3": 2 / 3,
    "3/4": 0.75,
    "1/4": 0.25,
    "π/2": math.pi / 2,
    "π/3": math.pi / 3,
    "π/4": math.pi / 4,
}

# Architectural prompts for SAM 3 — ordered by typical drawing sequence
ARCH_PROMPTS = [
    # Framing elements (foreground)
    "stone arch frame",
    "archway",
    "masonry wall",
    # Major forms (background)
    "large dome",
    "main dome",
    "small dome",
    "dome drum",
    # Vertical elements
    "minaret",
    "crescent finial",
    "spire",
    # Portal / entrance
    "entrance portal",
    "pointed arch doorway",
    "nested arches",
    # Decorative
    "geometric window grille",
    "column",
    "colonnade",
    "inscription band",
    # Ground
    "courtyard floor",
    "marble floor pattern",
]


def find_closest_ratio(value: float) -> dict:
    """Find the closest canonical ratio to a given value."""
    if value <= 0:
        return {"match": "none", "error_pct": 100.0}
    
    best_name = "none"
    best_error = float("inf")
    
    for name, constant in CANONICAL_RATIOS.items():
        error = abs(value / constant - 1) * 100
        if error < best_error:
            best_error = error
            best_name = name
    
    # Also check the reciprocal
    reciprocal = 1 / value if value > 0 else 0
    for name, constant in CANONICAL_RATIOS.items():
        error = abs(reciprocal / constant - 1) * 100
        if error < best_error:
            best_error = error
            best_name = f"1/({name})" if not name.startswith("1/") else name.replace("1/", "")
    
    return {
        "match": best_name,
        "constant": CANONICAL_RATIOS.get(best_name, 0),
        "error_pct": round(best_error, 2),
        "quality": "strong" if best_error < 2 else ("possible" if best_error < 5 else "weak"),
    }


def fit_ellipse_to_mask(mask: np.ndarray) -> dict | None:
    """Fit an ellipse (dome/arch) to a binary mask's contour."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return None
    
    ellipse = cv2.fitEllipse(largest)
    center, axes, angle = ellipse
    return {
        "center": {"x": float(center[0]), "y": float(center[1])},
        "axes": {"w": float(axes[0]), "h": float(axes[1])},
        "angle": float(angle),
        "area": float(cv2.contourArea(largest)),
    }


def fit_rect_to_mask(mask: np.ndarray) -> dict | None:
    """Fit a bounding rectangle to a binary mask."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return {
        "x": int(x), "y": int(y), "w": int(w), "h": int(h),
        "center": {"x": x + w / 2, "y": y + h / 2},
        "aspect": round(w / h, 4) if h > 0 else 0,
        "area": float(cv2.contourArea(largest)),
    }


def extract_contour_points(mask: np.ndarray, simplify_epsilon: float = 2.0) -> list:
    """Extract simplified contour points from a mask."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    
    largest = max(contours, key=cv2.contourArea)
    simplified = cv2.approxPolyDP(largest, simplify_epsilon, True)
    return [[int(pt[0][0]), int(pt[0][1])] for pt in simplified]


def run_sam3_extraction(image_path: str, prompts: list[str] | None = None) -> dict:
    """Run SAM 3 on an image with architectural prompts."""
    
    # Import SAM 3
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError:
        print("ERROR: SAM 3 not installed. Run:")
        print("  pip install -e . (from sam3 repo)")
        print("")
        print("Falling back to OpenCV edge detection...")
        return run_opencv_fallback(image_path, prompts)
    
    # Load model
    print("Loading SAM 3 model...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    
    # Load image
    image = Image.open(image_path)
    img_w, img_h = image.size
    print(f"Image: {img_w}×{img_h}")
    
    inference_state = processor.set_image(image)
    
    if prompts is None:
        prompts = ARCH_PROMPTS
    
    elements = []
    
    for prompt in prompts:
        print(f"  Segmenting: '{prompt}'...")
        try:
            output = processor.set_text_prompt(state=inference_state, prompt=prompt)
            masks = output["masks"]
            boxes = output["boxes"]
            scores = output["scores"]
            
            if masks is None or len(masks) == 0:
                print(f"    → no detections")
                continue
            
            # Process each detection
            for i in range(min(len(masks), 3)):  # Max 3 per prompt
                score = float(scores[i]) if scores is not None else 0
                if score < 0.3:
                    continue
                
                mask = masks[i].cpu().numpy() if hasattr(masks[i], 'cpu') else np.array(masks[i])
                if mask.ndim == 3:
                    mask = mask[0]
                mask = (mask > 0.5).astype(np.uint8)
                
                box = boxes[i].cpu().numpy().tolist() if hasattr(boxes[i], 'cpu') else list(boxes[i])
                
                # Fit geometric primitives
                ellipse = fit_ellipse_to_mask(mask)
                rect = fit_rect_to_mask(mask)
                contour = extract_contour_points(mask)
                
                element = {
                    "name": f"{prompt.replace(' ', '_')}_{i}",
                    "prompt": prompt,
                    "score": round(score, 4),
                    "box": box,
                    "rect": rect,
                    "ellipse": ellipse,
                    "contour_points": contour[:50],  # Limit for JSON size
                    "position_pct": {
                        "x": round(rect["center"]["x"] / img_w, 4) if rect else 0,
                        "y": round(rect["center"]["y"] / img_h, 4) if rect else 0,
                    },
                    "size_pct": {
                        "w": round(rect["w"] / img_w, 4) if rect else 0,
                        "h": round(rect["h"] / img_h, 4) if rect else 0,
                    },
                }
                
                # Position analysis — test against canonical fractions
                if rect:
                    element["position_analysis"] = {
                        "x_ratio": find_closest_ratio(rect["center"]["x"] / img_w),
                        "y_ratio": find_closest_ratio(rect["center"]["y"] / img_h),
                    }
                
                elements.append(element)
                print(f"    → detection {i}: score={score:.3f}, "
                      f"pos=({element['position_pct']['x']:.3f}, {element['position_pct']['y']:.3f}), "
                      f"size=({element['size_pct']['w']:.3f}×{element['size_pct']['h']:.3f})")
        
        except Exception as e:
            print(f"    → ERROR: {e}")
    
    return {
        "image": str(image_path),
        "image_size": {"w": img_w, "h": img_h},
        "elements": elements,
    }


def run_opencv_fallback(image_path: str, prompts: list[str] | None = None) -> dict:
    """Fallback when SAM 3 isn't available — use OpenCV edge/contour detection."""
    print("Running OpenCV fallback extraction...")
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Cannot read image {image_path}")
        sys.exit(1)
    
    img_h, img_w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by area (>1% of image)
    min_area = img_w * img_h * 0.01
    significant = [c for c in contours if cv2.contourArea(c) > min_area]
    significant.sort(key=cv2.contourArea, reverse=True)
    
    elements = []
    for i, contour in enumerate(significant[:20]):  # Top 20
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # Try ellipse fit
        ellipse = None
        if len(contour) >= 5:
            el = cv2.fitEllipse(contour)
            ellipse = {
                "center": {"x": float(el[0][0]), "y": float(el[0][1])},
                "axes": {"w": float(el[1][0]), "h": float(el[1][1])},
                "angle": float(el[2]),
            }
        
        # Classify by shape
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        shape_guess = "unknown"
        if circularity > 0.7:
            shape_guess = "dome_or_circle"
        elif w / h > 2:
            shape_guess = "horizontal_band"
        elif h / w > 2:
            shape_guess = "vertical_element"
        elif 0.8 < w / h < 1.2:
            shape_guess = "square_element"
        else:
            shape_guess = "rectangle"
        
        simplified = cv2.approxPolyDP(contour, 2.0, True)
        
        element = {
            "name": f"element_{i}_{shape_guess}",
            "shape_guess": shape_guess,
            "circularity": round(circularity, 4),
            "score": round(area / (img_w * img_h), 4),
            "rect": {
                "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                "center": {"x": x + w / 2, "y": y + h / 2},
                "aspect": round(w / h, 4) if h > 0 else 0,
                "area": float(area),
            },
            "ellipse": ellipse,
            "contour_points": [[int(pt[0][0]), int(pt[0][1])] for pt in simplified[:50]],
            "position_pct": {
                "x": round((x + w / 2) / img_w, 4),
                "y": round((y + h / 2) / img_h, 4),
            },
            "size_pct": {
                "w": round(w / img_w, 4),
                "h": round(h / img_h, 4),
            },
        }
        
        element["position_analysis"] = {
            "x_ratio": find_closest_ratio((x + w / 2) / img_w),
            "y_ratio": find_closest_ratio((y + h / 2) / img_h),
        }
        
        elements.append(element)
    
    return {
        "image": str(image_path),
        "image_size": {"w": img_w, "h": img_h},
        "elements": elements,
        "method": "opencv_fallback",
    }


def compute_proportional_analysis(extraction: dict) -> dict:
    """Compute pairwise proportional relationships between all elements."""
    elements = extraction["elements"]
    img_w = extraction["image_size"]["w"]
    img_h = extraction["image_size"]["h"]
    
    relationships = []
    
    for i, a in enumerate(elements):
        for j, b in enumerate(elements):
            if j <= i:
                continue
            
            a_rect = a.get("rect")
            b_rect = b.get("rect")
            if not a_rect or not b_rect:
                continue
            
            # Width ratio
            if a_rect["w"] > 0 and b_rect["w"] > 0:
                ratio = max(a_rect["w"], b_rect["w"]) / min(a_rect["w"], b_rect["w"])
                match = find_closest_ratio(ratio)
                if match["quality"] != "weak":
                    relationships.append({
                        "a": a["name"], "b": b["name"],
                        "type": "width_ratio",
                        "value": round(ratio, 4),
                        **match,
                    })
            
            # Height ratio
            if a_rect["h"] > 0 and b_rect["h"] > 0:
                ratio = max(a_rect["h"], b_rect["h"]) / min(a_rect["h"], b_rect["h"])
                match = find_closest_ratio(ratio)
                if match["quality"] != "weak":
                    relationships.append({
                        "a": a["name"], "b": b["name"],
                        "type": "height_ratio",
                        "value": round(ratio, 4),
                        **match,
                    })
            
            # Distance between centers
            dx = a_rect["center"]["x"] - b_rect["center"]["x"]
            dy = a_rect["center"]["y"] - b_rect["center"]["y"]
            dist = math.sqrt(dx * dx + dy * dy)
            
            # Distance as fraction of image diagonal
            diag = math.sqrt(img_w ** 2 + img_h ** 2)
            dist_frac = dist / diag
            match = find_closest_ratio(dist_frac)
            if match["quality"] != "weak":
                relationships.append({
                    "a": a["name"], "b": b["name"],
                    "type": "center_distance_frac",
                    "value": round(dist_frac, 4),
                    **match,
                })
    
    # Sort by quality (strong first)
    quality_order = {"strong": 0, "possible": 1, "weak": 2}
    relationships.sort(key=lambda r: (quality_order.get(r["quality"], 3), r["error_pct"]))
    
    extraction["relationships"] = relationships
    return extraction


def generate_js_constants(extraction: dict) -> str:
    """Generate JavaScript proportional constants from extraction."""
    img_w = extraction["image_size"]["w"]
    img_h = extraction["image_size"]["h"]
    
    lines = [
        "// Auto-generated proportional system from Traced pipeline",
        "// Source: " + extraction.get("image", "unknown"),
        f"// Image: {img_w}×{img_h}",
        "",
        "var W = 1080, H = 1920;",
        "var cx = W / 2;",
        "var PHI = (1 + Math.sqrt(5)) / 2;",
        "",
    ]
    
    for el in extraction["elements"]:
        name = el["name"].replace(" ", "_").replace("-", "_")
        pos = el.get("position_pct", {})
        size = el.get("size_pct", {})
        pos_analysis = el.get("position_analysis", {})
        
        x_match = pos_analysis.get("x_ratio", {})
        y_match = pos_analysis.get("y_ratio", {})
        
        x_comment = f"  // x={pos.get('x', 0):.3f}"
        if x_match.get("quality") in ("strong", "possible"):
            x_comment += f" ≈ {x_match['match']} (err {x_match['error_pct']}%)"
        
        y_comment = f"  // y={pos.get('y', 0):.3f}"
        if y_match.get("quality") in ("strong", "possible"):
            y_comment += f" ≈ {y_match['match']} (err {y_match['error_pct']}%)"
        
        lines.append(f"// {name} — score: {el.get('score', 0):.3f}")
        lines.append(f"var {name}_cx = W * {pos.get('x', 0):.4f};{x_comment}")
        lines.append(f"var {name}_cy = H * {pos.get('y', 0):.4f};{y_comment}")
        lines.append(f"var {name}_w = W * {size.get('w', 0):.4f};")
        lines.append(f"var {name}_h = H * {size.get('h', 0):.4f};")
        lines.append("")
    
    # Add relationship comments
    strong = [r for r in extraction.get("relationships", []) if r["quality"] == "strong"]
    if strong:
        lines.append("// === Strong proportional relationships (< 2% error) ===")
        for r in strong[:10]:
            lines.append(f"// {r['a']} ↔ {r['b']}: {r['type']} = {r['value']} ≈ {r['match']} (err {r['error_pct']}%)")
        lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Traced: Architectural element extraction with SAM 3")
    parser.add_argument("--image", required=True, help="Path to reference image")
    parser.add_argument("--output", default="extraction.json", help="Output JSON path")
    parser.add_argument("--prompts", default=None, help="Comma-separated custom prompts")
    parser.add_argument("--js-output", default=None, help="Output JS constants file")
    args = parser.parse_args()
    
    prompts = args.prompts.split(",") if args.prompts else None
    
    # Run extraction
    extraction = run_sam3_extraction(args.image, prompts)
    
    # Compute proportional analysis
    extraction = compute_proportional_analysis(extraction)
    
    # Generate JS constants
    js_code = generate_js_constants(extraction)
    extraction["generated_js"] = js_code
    
    # Save
    output_path = Path(args.output)
    output_path.write_text(json.dumps(extraction, indent=2))
    print(f"\nSaved extraction to {output_path}")
    print(f"Found {len(extraction['elements'])} elements")
    print(f"Found {len(extraction.get('relationships', []))} proportional relationships")
    
    strong = [r for r in extraction.get("relationships", []) if r["quality"] == "strong"]
    print(f"  Strong matches (< 2% error): {len(strong)}")
    
    # Save JS if requested
    if args.js_output:
        Path(args.js_output).write_text(js_code)
        print(f"Saved JS constants to {args.js_output}")
    else:
        print("\n" + "=" * 60)
        print("GENERATED JAVASCRIPT:")
        print("=" * 60)
        print(js_code)


if __name__ == "__main__":
    main()
