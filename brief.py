#!/usr/bin/env python3
"""
Traced Pipeline — Stage 0: LLM Vision Brief
Sends reference photo to an LLM with vision to identify every architectural element.
Outputs a structured brief that guides the entire downstream pipeline.

The brief includes:
- Element inventory (what's in the photo)
- Spatial layout (where elements are relative to each other)
- Style identification (architectural tradition)
- Detail hotspots (areas of highest visual interest)
- Drawing checklist (every element that must appear in the final drawing)

Supports: Anthropic Claude, OpenAI GPT, Ollama local models

Usage:
    python brief.py --image photo.jpg --output brief.json
    python brief.py --image photo.jpg --output brief.json --provider ollama --model llava
    python brief.py --image photo.jpg --research knowledge.json --output brief.json
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path

VISION_PROMPT = """You are an expert architectural analyst. Examine this photograph of a building and provide a comprehensive architectural brief.

For EVERY visible architectural element, identify:
1. Element name (use proper architectural terminology)
2. Element type (dome, arch, column, minaret, muqarnas, mashrabiya, calligraphy_band, geometric_panel, etc.)
3. Approximate position (top/middle/bottom, left/center/right)
4. Approximate size relative to the whole image (large/medium/small/tiny)
5. Any sub-elements within it
6. The mathematical curve type if applicable (pointed_arch, horseshoe_arch, semicircular, ogee, catenary, elliptical, etc.)

Also identify:
- The architectural style/tradition (Ottoman, Mughal, Moorish, Persian, Gothic, Classical, Modern, etc.)
- Bilateral symmetry (yes/no, and the symmetry axis)
- Proportional system you can observe (golden ratio, √2, √3 relationships)
- The most visually striking/detailed areas (where an artist would spend the most time drawing)
- Any calligraphy, inscriptions, or text zones
- Material types visible (marble, stone, tile, stucco, wood, glass, metal)

Respond in this exact JSON format:
{
  "building_name_guess": "string — your best guess of the building name",
  "architectural_style": ["list of styles — e.g., 'Mughal', 'Moorish', 'Ottoman'"],
  "period_guess": "approximate era/century",
  "symmetry": {
    "bilateral": true/false,
    "axis": "vertical_center" or description
  },
  "elements": [
    {
      "name": "descriptive name",
      "type": "architectural type",
      "position": {"vertical": "top/upper/middle/lower/bottom", "horizontal": "left/center/right"},
      "size": "large/medium/small/tiny",
      "curve_type": "pointed/horseshoe/semicircular/ogee/catenary/elliptical/null",
      "rise_span_estimate": 0.0 to 2.0 or null,
      "contains": ["list of sub-element names"],
      "detail_level": "high/medium/low — how much ornamental detail",
      "material": "marble/stone/tile/stucco/wood/glass/metal/unknown",
      "notes": "any special observations"
    }
  ],
  "spatial_hierarchy": {
    "foreground": ["element names in front"],
    "midground": ["element names in middle"],
    "background": ["element names behind"],
    "framing": ["elements that frame/border the view"]
  },
  "detail_hotspots": [
    {"name": "element name", "reason": "why this area has high visual interest"}
  ],
  "proportional_observations": [
    "any golden ratio, √2, √3, or other proportional relationships you can observe"
  ],
  "drawing_checklist": [
    "every element that MUST appear in an accurate architectural drawing of this building"
  ],
  "drawing_order_suggestion": [
    "suggested order for drawing elements (typically background to foreground, or ground up)"
  ]
}

Be thorough — miss nothing. An architectural drawing will be generated from your analysis."""


def encode_image(image_path: str) -> tuple:
    """Encode image as base64 with mime type."""
    ext = Path(image_path).suffix.lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else f"image/{ext.replace('.', '')}"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return b64, mime


def call_anthropic(image_path: str, model: str = "claude-sonnet-4-20250514") -> dict:
    """Call Anthropic Claude with vision."""
    try:
        import anthropic
    except ImportError:
        # Try raw HTTP
        return call_anthropic_raw(image_path, model)
    
    client = anthropic.Anthropic()
    b64, mime = encode_image(image_path)
    
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
                {"type": "text", "text": VISION_PROMPT}
            ]
        }]
    )
    
    text = response.content[0].text
    # Extract JSON from response
    return extract_json(text)


def call_anthropic_raw(image_path: str, model: str = "claude-sonnet-4-20250514") -> dict:
    """Call Anthropic API directly with requests."""
    import requests
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  ERROR: ANTHROPIC_API_KEY not set")
        return {}
    
    b64, mime = encode_image(image_path)
    
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
        },
        json={
            "model": model,
            "max_tokens": 4096,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
                    {"type": "text", "text": VISION_PROMPT}
                ]
            }]
        },
        timeout=60,
    )
    
    data = response.json()
    text = data.get("content", [{}])[0].get("text", "")
    return extract_json(text)


def call_openai(image_path: str, model: str = "gpt-4o") -> dict:
    """Call OpenAI GPT with vision."""
    import requests
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("  ERROR: OPENAI_API_KEY not set")
        return {}
    
    b64, mime = encode_image(image_path)
    
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 4096,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": VISION_PROMPT}
                ]
            }]
        },
        timeout=60,
    )
    
    data = response.json()
    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return extract_json(text)


def call_ollama(image_path: str, model: str = "llava") -> dict:
    """Call local Ollama with vision model."""
    import requests
    
    b64, _ = encode_image(image_path)
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": VISION_PROMPT,
            "images": [b64],
            "stream": False,
        },
        timeout=120,
    )
    
    data = response.json()
    text = data.get("response", "")
    return extract_json(text)


def extract_json(text: str) -> dict:
    """Extract JSON from LLM response text."""
    # Try to find JSON block
    import re
    
    # Look for ```json ... ``` blocks
    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find raw JSON object
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Return raw text as fallback
    return {"raw_response": text, "parse_error": "Could not extract JSON"}


def merge_with_research(brief: dict, research: dict) -> dict:
    """Merge LLM brief with research data for enriched brief."""
    if not research:
        return brief
    
    # Add research dimensions
    if research.get("dimensions"):
        brief["verified_dimensions"] = research["dimensions"]
    
    # Cross-reference styles
    research_styles = set(research.get("style_influences", []))
    brief_styles = set(brief.get("architectural_style", []))
    brief["architectural_style_merged"] = list(brief_styles | research_styles)
    
    # Add known arch/dome types from research
    if research.get("arch_types"):
        brief["verified_arch_types"] = research["arch_types"]
    if research.get("dome_types"):
        brief["verified_dome_types"] = research["dome_types"]
    
    # Add traditions
    if research.get("traditions"):
        brief["traditions"] = research["traditions"]
    
    # Mark brief as research-enriched
    brief["research_enriched"] = True
    brief["research_sources"] = len(research.get("sources", []))
    
    return brief


def generate_checklist(brief: dict) -> list:
    """Generate a drawing completion checklist from the brief."""
    checklist = []
    
    for el in brief.get("elements", []):
        checklist.append({
            "element": el.get("name", "unknown"),
            "type": el.get("type", "unknown"),
            "required": True,
            "satisfied": False,
            "detail_level": el.get("detail_level", "medium"),
        })
    
    # Add from drawing_checklist if present
    for item in brief.get("drawing_checklist", []):
        if not any(c["element"].lower() == item.lower() for c in checklist):
            checklist.append({
                "element": item,
                "type": "from_checklist",
                "required": True,
                "satisfied": False,
                "detail_level": "medium",
            })
    
    return checklist


def main():
    parser = argparse.ArgumentParser(description="Traced: LLM Vision Brief")
    parser.add_argument("--image", required=True, help="Reference image path")
    parser.add_argument("--output", default="brief.json", help="Output brief JSON")
    parser.add_argument("--research", default=None, help="Research knowledge.json to merge")
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai", "ollama"])
    parser.add_argument("--model", default=None, help="Model name override")
    args = parser.parse_args()
    
    print(f"Generating architectural brief from {args.image}...")
    print(f"  Provider: {args.provider}")
    
    # Call vision LLM
    if args.provider == "anthropic":
        model = args.model or "claude-sonnet-4-20250514"
        print(f"  Model: {model}")
        brief = call_anthropic(args.image, model)
    elif args.provider == "openai":
        model = args.model or "gpt-4o"
        print(f"  Model: {model}")
        brief = call_openai(args.image, model)
    elif args.provider == "ollama":
        model = args.model or "llava"
        print(f"  Model: {model}")
        brief = call_ollama(args.image, model)
    
    if not brief or "raw_response" in brief:
        print(f"  WARNING: Could not parse LLM response as JSON")
        if "raw_response" in brief:
            print(f"  Raw response (first 500 chars): {brief['raw_response'][:500]}")
    
    # Merge with research if provided
    if args.research and Path(args.research).exists():
        research = json.loads(Path(args.research).read_text())
        brief = merge_with_research(brief, research)
        print(f"  Merged with research data ({research.get('building_name', 'unknown')})")
    
    # Generate checklist
    brief["checklist"] = generate_checklist(brief)
    
    # Save
    Path(args.output).write_text(json.dumps(brief, indent=2))
    
    # Print summary
    print(f"\n  Building: {brief.get('building_name_guess', 'Unknown')}")
    print(f"  Style: {brief.get('architectural_style', [])}")
    print(f"  Elements found: {len(brief.get('elements', []))}")
    print(f"  Detail hotspots: {len(brief.get('detail_hotspots', []))}")
    print(f"  Checklist items: {len(brief.get('checklist', []))}")
    print(f"  Drawing order: {len(brief.get('drawing_order_suggestion', []))} steps")
    
    if brief.get("elements"):
        print(f"\n  Elements:")
        for el in brief["elements"]:
            detail = el.get("detail_level", "?")
            curve = el.get("curve_type", "")
            curve_str = f" ({curve})" if curve and curve != "null" else ""
            print(f"    {el.get('name', '?'):30s} {el.get('type', '?'):20s} {el.get('size', '?'):8s} detail={detail}{curve_str}")
    
    if brief.get("drawing_checklist"):
        print(f"\n  Drawing Checklist:")
        for item in brief["drawing_checklist"]:
            print(f"    □ {item}")
    
    print(f"\n  Saved to {args.output}")


if __name__ == "__main__":
    main()
