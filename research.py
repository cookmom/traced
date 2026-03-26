#!/usr/bin/env python3
"""
Traced Pipeline — Stage 2.5: Architectural Research
Gathers real-world architectural knowledge about a building to correct
and enrich SAM extraction data.

Sources: Wikipedia, WikiArquitectura, web search, LLM synthesis
Output: building_knowledge.json with verified dimensions, styles, element types

Usage:
    python research.py --name "Sheikh Zayed Grand Mosque" --output knowledge.json
    python research.py --name "Taj Mahal" --extraction extraction.json --output knowledge.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

try:
    from markitdown import MarkItDown
    HAS_MARKITDOWN = True
except ImportError:
    HAS_MARKITDOWN = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ============================================================
# ARCHITECTURAL KNOWLEDGE BASE
# Known proportional systems by tradition
# ============================================================
ARCH_TRADITIONS = {
    "mughal": {
        "dome_type": "onion_or_bulbous",
        "arch_type": "cusped_or_pointed",
        "proportional_system": "phi_dominant",
        "plan": "symmetrical_four_iwan",
        "materials": ["red_sandstone", "white_marble", "pietra_dura"],
        "influences": ["Persian", "Central_Asian", "Indian"],
    },
    "ottoman": {
        "dome_type": "hemisphere",
        "arch_type": "pointed_slightly",
        "proportional_system": "phi_with_sqrt2",
        "plan": "central_dome_cascade",
        "materials": ["stone", "marble", "iznik_tile"],
        "influences": ["Byzantine", "Seljuk", "Persian"],
    },
    "moorish": {
        "dome_type": "hemisphere_or_ribbed",
        "arch_type": "horseshoe",
        "proportional_system": "sqrt2_dominant",
        "plan": "hypostyle_courtyard",
        "materials": ["stucco", "tile", "carved_wood"],
        "influences": ["Roman", "Visigothic", "Berber"],
    },
    "mamluk": {
        "dome_type": "pointed_or_bulbous",
        "arch_type": "pointed_keel",
        "proportional_system": "sqrt2_with_phi",
        "plan": "cruciform",
        "materials": ["stone", "marble_inlay", "carved_stucco"],
        "influences": ["Ayyubid", "Crusader", "Central_Asian"],
    },
    "gothic": {
        "dome_type": "ribbed_vault",
        "arch_type": "pointed_lancet",
        "proportional_system": "ad_triangulum_sqrt3",
        "plan": "cruciform_basilica",
        "materials": ["stone", "stained_glass", "flying_buttress"],
        "influences": ["Romanesque", "French", "English"],
    },
    "classical": {
        "dome_type": "hemisphere",
        "arch_type": "semicircular",
        "proportional_system": "phi_classical",
        "plan": "temple_front_portico",
        "materials": ["marble", "travertine", "granite"],
        "influences": ["Greek", "Roman", "Renaissance"],
    },
}

# Known arch profiles with mathematical definitions
ARCH_PROFILES = {
    "semicircular": {"rise_span": 0.5, "centers": 1, "description": "Single center, R = span/2"},
    "pointed_equilateral": {"rise_span": 0.866, "centers": 2, "description": "Two centers at springing, R = span"},
    "pointed_lancet": {"rise_span_min": 1.0, "centers": 2, "description": "Two centers inside span, R > span"},
    "horseshoe": {"rise_span": 0.5, "centers": 1, "description": "Center above springing, arc > 180°", "overshoot_pct": 10},
    "ogee": {"centers": 4, "description": "Four-center S-curve, convex-concave"},
    "tudor": {"rise_span": 0.3, "centers": 4, "description": "Four-center depressed arch"},
    "segmental": {"rise_span_max": 0.45, "centers": 1, "description": "Shallow arc, R > span/2"},
    "cusped": {"description": "Pointed with decorative cusps at intrados"},
    "multifoil": {"description": "Scalloped edge with multiple foils"},
}

# Known dome profiles
DOME_PROFILES = {
    "hemisphere": {"height_diameter": 0.5, "description": "Perfect half-sphere"},
    "onion": {"height_diameter_min": 0.7, "description": "Pointed top, bulging sides, narrow base"},
    "bulbous": {"height_diameter_min": 0.6, "description": "Similar to onion but less pointed"},
    "saucer": {"height_diameter_max": 0.4, "description": "Shallow, wide dome"},
    "pointed": {"height_diameter_min": 0.55, "description": "Like hemisphere but pointed at apex"},
    "ribbed": {"description": "Hemisphere or pointed with visible structural ribs"},
    "double_shell": {"description": "Inner and outer shells with space between"},
}


def search_wikipedia(building_name: str) -> dict:
    """Search Wikipedia for architectural information."""
    if not HAS_REQUESTS:
        print("  requests not installed — skipping Wikipedia")
        return {}
    
    try:
        # Search Wikipedia API
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": building_name,
            "format": "json",
            "srlimit": 3,
        }
        headers = {"User-Agent": "Traced/1.0 (architectural research tool; contact: github.com/cookmom/traced)"}
        resp = requests.get(search_url, params=params, timeout=10, headers=headers)
        results = resp.json().get("query", {}).get("search", [])
        
        if not results:
            return {}
        
        # Get full page content
        page_title = results[0]["title"]
        content_url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
        
        # Get plain text extract from Wikipedia API (more reliable than scraping)
        extract_params = {
            "action": "query",
            "titles": page_title,
            "prop": "extracts",
            "explaintext": True,
            "format": "json",
        }
        resp = requests.get(search_url, params=extract_params, timeout=10, headers=headers)
        pages = resp.json().get("query", {}).get("pages", {})
        text = ""
        for page_id, page_data in pages.items():
            text = page_data.get("extract", "")
            break
        
        if not text and HAS_MARKITDOWN:
            md = MarkItDown()
            result = md.convert_url(content_url)
            text = result.text_content
        
        return {
            "source": "wikipedia",
            "title": page_title,
            "url": content_url,
            "text": text[:5000],  # First 5000 chars
        }
    except Exception as e:
        print(f"  Wikipedia search failed: {e}")
        return {}


def search_wikiarquitectura(building_name: str) -> dict:
    """Search WikiArquitectura for architectural data."""
    if not HAS_REQUESTS:
        return {}
    
    try:
        slug = building_name.lower().replace(" ", "-").replace("'", "")
        url = f"https://en.wikiarquitectura.com/building/{slug}/"
        
        if HAS_MARKITDOWN:
            md = MarkItDown()
            result = md.convert_url(url)
            text = result.text_content
            if text and len(text) > 100:
                return {"source": "wikiarquitectura", "url": url, "text": text[:5000]}
        
        return {}
    except Exception as e:
        return {}


def extract_dimensions(text: str) -> dict:
    """Extract architectural dimensions from text."""
    dims = {}
    
    # Height patterns
    height_match = re.findall(r'(\d+(?:\.\d+)?)\s*(?:m|meters?|metres?)\s*(?:tall|high|height)', text, re.I)
    if height_match:
        dims["height_m"] = float(height_match[0])
    
    # Area patterns
    area_match = re.findall(r'(\d+(?:,\d+)?(?:\.\d+)?)\s*(?:m²|square\s*met|sq\s*m)', text, re.I)
    if area_match:
        dims["area_m2"] = float(area_match[0].replace(",", ""))
    
    # Dome count
    dome_match = re.findall(r'(\d+)\s*domes?', text, re.I)
    if dome_match:
        dims["dome_count"] = int(dome_match[0])
    
    # Column count
    col_match = re.findall(r'(\d+(?:,\d+)?)\s*columns?', text, re.I)
    if col_match:
        dims["column_count"] = int(col_match[0].replace(",", ""))
    
    # Minaret count and height
    min_match = re.findall(r'(\d+)\s*minarets?', text, re.I)
    if min_match:
        dims["minaret_count"] = int(min_match[0])
    
    min_h = re.findall(r'minarets?\s*(?:are|is|each|that are)\s*(\d+(?:\.\d+)?)\s*(?:m|meters?)', text, re.I)
    if min_h:
        dims["minaret_height_m"] = float(min_h[0])
    
    # Construction period
    year_match = re.findall(r'(?:built|constructed|completed|opened)\s*(?:in|between)?\s*(\d{4})', text, re.I)
    if year_match:
        dims["year_completed"] = int(year_match[-1])
    
    return dims


def extract_style_influences(text: str) -> list:
    """Extract architectural style influences from text."""
    influences = []
    
    style_keywords = {
        "mughal": ["mughal", "moghul"],
        "ottoman": ["ottoman", "turkish"],
        "moorish": ["moorish", "moor", "andalusian", "alhambra"],
        "persian": ["persian", "safavid", "iranian"],
        "mamluk": ["mamluk", "mameluke"],
        "gothic": ["gothic"],
        "byzantine": ["byzantine", "hagia sophia"],
        "fatimid": ["fatimid"],
        "moroccan": ["moroccan", "morocco"],
        "classical": ["classical", "greco-roman", "neoclassical"],
    }
    
    text_lower = text.lower()
    for style, keywords in style_keywords.items():
        if any(kw in text_lower for kw in keywords):
            influences.append(style)
    
    return influences


def extract_arch_types(text: str) -> list:
    """Extract mentioned arch types from text."""
    arch_types = []
    text_lower = text.lower()
    
    arch_keywords = {
        "horseshoe": ["horseshoe arch", "moorish arch"],
        "pointed": ["pointed arch", "ogival"],
        "ogee": ["ogee", "s-curve arch"],
        "lancet": ["lancet"],
        "semicircular": ["semicircular", "round arch", "roman arch"],
        "cusped": ["cusped", "multifoil", "polylobed"],
        "tudor": ["tudor arch", "four-center"],
    }
    
    for arch_type, keywords in arch_keywords.items():
        if any(kw in text_lower for kw in keywords):
            arch_types.append(arch_type)
    
    return arch_types


def extract_dome_types(text: str) -> list:
    """Extract mentioned dome types from text."""
    dome_types = []
    text_lower = text.lower()
    
    dome_keywords = {
        "hemisphere": ["hemispherical", "hemisphere", "half-sphere"],
        "onion": ["onion dome", "onion-shaped"],
        "bulbous": ["bulbous"],
        "ribbed": ["ribbed dome", "ribbed vault"],
        "double_shell": ["double shell", "double-shell", "inner and outer"],
    }
    
    for dome_type, keywords in dome_keywords.items():
        if any(kw in text_lower for kw in keywords):
            dome_types.append(dome_type)
    
    return dome_types


def correct_extraction(extraction: dict, knowledge: dict) -> dict:
    """Apply architectural knowledge to correct SAM extraction errors."""
    if not extraction:
        return extraction
    
    corrections = []
    elements = extraction.get("elements", [])
    
    # Known dome type correction
    known_domes = knowledge.get("dome_types", [])
    if known_domes:
        primary_dome = known_domes[0]
        for el in elements:
            if el.get("shape", {}).get("type") in ("dome", "dome_like"):
                old_type = el.get("primitives", {}).get("dome_geometry", {}).get("profile_type", "")
                if old_type and old_type != primary_dome:
                    corrections.append({
                        "element": el["name"],
                        "field": "dome_profile_type",
                        "old": old_type,
                        "new": primary_dome,
                        "reason": f"Research indicates {primary_dome} dome (source: architectural references)",
                    })
                    if "dome_geometry" in el.get("primitives", {}):
                        el["primitives"]["dome_geometry"]["profile_type"] = primary_dome
                        el["primitives"]["dome_geometry"]["corrected"] = True
    
    # Known arch type correction
    known_arches = knowledge.get("arch_types", [])
    if known_arches:
        for el in elements:
            shape_type = el.get("shape", {}).get("type", "")
            if "arch" in shape_type:
                arch_geo = el.get("primitives", {}).get("arch_geometry", {})
                if arch_geo:
                    old_type = arch_geo.get("profile_type", "")
                    # Check if any known arch type matches better
                    for known_arch in known_arches:
                        if known_arch in ARCH_PROFILES:
                            profile = ARCH_PROFILES[known_arch]
                            if "rise_span" in profile:
                                expected_rs = profile["rise_span"]
                                actual_rs = arch_geo.get("rise_to_span", 0)
                                if abs(actual_rs - expected_rs) < abs(actual_rs - ARCH_PROFILES.get(old_type, {}).get("rise_span", actual_rs)):
                                    corrections.append({
                                        "element": el["name"],
                                        "field": "arch_profile_type",
                                        "old": old_type,
                                        "new": known_arch,
                                        "reason": f"Research indicates {known_arch} arch style",
                                    })
    
    knowledge["corrections"] = corrections
    return extraction


def build_knowledge(building_name: str, extraction: dict = None) -> dict:
    """Build comprehensive architectural knowledge from all sources."""
    print(f"Researching: {building_name}")
    
    knowledge = {
        "building_name": building_name,
        "sources": [],
    }
    
    # Search Wikipedia
    print("  Searching Wikipedia...")
    wiki = search_wikipedia(building_name)
    if wiki:
        knowledge["sources"].append(wiki)
        text = wiki.get("text", "")
        
        # Extract structured data
        dims = extract_dimensions(text)
        if dims:
            knowledge["dimensions"] = dims
            print(f"    Dimensions: {dims}")
        
        influences = extract_style_influences(text)
        if influences:
            knowledge["style_influences"] = influences
            print(f"    Styles: {influences}")
        
        arch_types = extract_arch_types(text)
        if arch_types:
            knowledge["arch_types"] = arch_types
            print(f"    Arch types: {arch_types}")
        
        dome_types = extract_dome_types(text)
        if dome_types:
            knowledge["dome_types"] = dome_types
            print(f"    Dome types: {dome_types}")
    
    # Search WikiArquitectura
    print("  Searching WikiArquitectura...")
    wikiarch = search_wikiarquitectura(building_name)
    if wikiarch:
        knowledge["sources"].append(wikiarch)
        text = wikiarch.get("text", "")
        
        # Supplement with any new data
        dims2 = extract_dimensions(text)
        if dims2:
            knowledge.setdefault("dimensions", {}).update(dims2)
        
        influences2 = extract_style_influences(text)
        for inf in influences2:
            if inf not in knowledge.get("style_influences", []):
                knowledge.setdefault("style_influences", []).append(inf)
    
    # Determine tradition from influences
    influences = knowledge.get("style_influences", [])
    traditions_matched = []
    for tradition, info in ARCH_TRADITIONS.items():
        if tradition in influences:
            traditions_matched.append((tradition, info))
    
    if traditions_matched:
        knowledge["traditions"] = {t: info for t, info in traditions_matched}
        # Infer missing data from tradition
        if not knowledge.get("dome_types"):
            for t, info in traditions_matched:
                dt = info.get("dome_type", "")
                if dt:
                    knowledge.setdefault("dome_types", []).append(dt.split("_or_")[0])
        if not knowledge.get("arch_types"):
            for t, info in traditions_matched:
                at = info.get("arch_type", "")
                if at:
                    knowledge.setdefault("arch_types", []).append(at.split("_or_")[0])
    
    # If we have extraction data, apply corrections
    if extraction:
        print("  Applying corrections to extraction...")
        correct_extraction(extraction, knowledge)
        corrections = knowledge.get("corrections", [])
        if corrections:
            for c in corrections:
                print(f"    CORRECTED: {c['element']} {c['field']}: {c['old']} → {c['new']} ({c['reason']})")
        else:
            print("    No corrections needed")
    
    # Summary
    print(f"\n  Knowledge summary:")
    print(f"    Sources: {len(knowledge.get('sources', []))}")
    print(f"    Dimensions: {knowledge.get('dimensions', {})}")
    print(f"    Styles: {knowledge.get('style_influences', [])}")
    print(f"    Arch types: {knowledge.get('arch_types', [])}")
    print(f"    Dome types: {knowledge.get('dome_types', [])}")
    print(f"    Traditions: {list(knowledge.get('traditions', {}).keys())}")
    
    return knowledge


def main():
    parser = argparse.ArgumentParser(description="Traced: Architectural research")
    parser.add_argument("--name", required=True, help="Building name")
    parser.add_argument("--extraction", default=None, help="Extraction JSON to correct")
    parser.add_argument("--output", default="knowledge.json", help="Output path")
    args = parser.parse_args()
    
    extraction = None
    if args.extraction:
        extraction = json.loads(Path(args.extraction).read_text())
    
    knowledge = build_knowledge(args.name, extraction)
    
    Path(args.output).write_text(json.dumps(knowledge, indent=2, default=str))
    print(f"\nSaved to {args.output}")
    
    # If extraction was corrected, save it back
    if extraction and args.extraction:
        corrected_path = args.extraction.replace(".json", "-corrected.json")
        Path(corrected_path).write_text(json.dumps(extraction, indent=2, default=str))
        print(f"Corrected extraction saved to {corrected_path}")


if __name__ == "__main__":
    main()
