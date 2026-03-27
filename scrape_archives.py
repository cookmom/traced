#!/usr/bin/env python3
"""
Traced — Archive-Grade Training Data Downloader
Downloads curated, high-quality architectural photos from scholarly/museum sources.

Sources:
1. Manar al-Athar (Oxford) — 150K+ Islamic architecture photos
2. Europeana — Europe's digital heritage, API
3. Smithsonian Open Access — 4.5M+ images, free API
4. Digital Public Library of America — open API

Usage:
    python scrape_archives.py --source all --query "mosque" --limit 100 --output-dir training_data/archives
    python scrape_archives.py --source manar --query "muqarnas" --limit 50
    python scrape_archives.py --source smithsonian --query "Islamic dome" --limit 50
    python scrape_archives.py --source europeana --query "mosque architecture" --limit 50
"""

import argparse
import json
import os
import time
from pathlib import Path

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


def download_image(url: str, filepath: str, headers: dict = None) -> bool:
    """Download a single image with error handling."""
    if not HAS_REQUESTS:
        return False
    try:
        h = headers or {"User-Agent": "Traced/1.0 (architectural research; github.com/cookmom/traced)"}
        resp = requests.get(url, headers=h, timeout=15, stream=True)
        if resp.status_code == 200 and len(resp.content) > 5000:
            Path(filepath).write_bytes(resp.content)
            return True
    except Exception:
        pass
    return False


def scrape_manar_al_athar(query: str, limit: int, output_dir: str) -> int:
    """Manar al-Athar — Oxford's Islamic architecture photo archive.
    150K+ scholarly photos of Islamic buildings."""
    os.makedirs(output_dir, exist_ok=True)
    headers = {"User-Agent": "Traced/1.0 (architectural research)"}
    
    # Manar al-Athar search API
    base_url = "https://www.manar-al-athar.ox.ac.uk"
    search_url = f"{base_url}/search?q={query.replace(' ', '+')}&format=json"
    
    downloaded = 0
    try:
        resp = requests.get(search_url, headers=headers, timeout=15)
        if resp.status_code != 200:
            # Try scraping the HTML search results
            search_html = f"{base_url}/search?q={query.replace(' ', '+')}"
            resp = requests.get(search_html, headers=headers, timeout=15)
            
            import re
            img_urls = re.findall(r'src="([^"]*(?:jpg|jpeg|png)[^"]*)"', resp.text, re.I)
            img_urls = [u if u.startswith('http') else base_url + u for u in img_urls]
            img_urls = [u for u in img_urls if 'thumbnail' not in u.lower() and 'icon' not in u.lower()]
            
            for url in img_urls[:limit]:
                filename = f"manar_{query.replace(' ', '_')}_{downloaded:04d}.jpg"
                if download_image(url, os.path.join(output_dir, filename), headers):
                    downloaded += 1
                    if downloaded % 10 == 0:
                        print(f"    Downloaded {downloaded}/{limit}")
                time.sleep(0.3)
    except Exception as e:
        print(f"    Manar al-Athar error: {e}")
    
    print(f"  Manar al-Athar: {downloaded} images")
    return downloaded


def scrape_europeana(query: str, limit: int, output_dir: str, api_key: str = None) -> int:
    """Europeana — Europe's digital heritage library.
    Free API, millions of cultural heritage images."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Europeana API (free key at https://pro.europeana.eu/page/get-api)
    key = api_key or os.environ.get("EUROPEANA_API_KEY", "")
    if not key:
        print("  Europeana: No API key. Get one free at https://pro.europeana.eu/page/get-api")
        print("  Set EUROPEANA_API_KEY env var or pass --europeana-key")
        # Try without key (limited)
        key = "api2demo"
    
    search_url = "https://api.europeana.eu/record/v2/search.json"
    params = {
        "wskey": key,
        "query": query,
        "qf": "TYPE:IMAGE",
        "rows": min(limit, 100),
        "profile": "rich",
    }
    
    downloaded = 0
    try:
        resp = requests.get(search_url, params=params, timeout=15)
        data = resp.json()
        items = data.get("items", [])
        
        for item in items:
            if downloaded >= limit:
                break
            
            # Get image URL
            img_url = None
            edmIsShownBy = item.get("edmIsShownBy", [])
            if edmIsShownBy:
                img_url = edmIsShownBy[0]
            elif item.get("edmPreview"):
                img_url = item["edmPreview"][0]
            
            if not img_url:
                continue
            
            title = item.get("title", ["unknown"])[0][:50].replace("/", "_").replace(" ", "_")
            filename = f"europeana_{downloaded:04d}_{title}.jpg"
            
            if download_image(img_url, os.path.join(output_dir, filename)):
                downloaded += 1
                if downloaded % 10 == 0:
                    print(f"    Downloaded {downloaded}/{limit}")
            time.sleep(0.3)
    except Exception as e:
        print(f"    Europeana error: {e}")
    
    print(f"  Europeana: {downloaded} images")
    return downloaded


def scrape_smithsonian(query: str, limit: int, output_dir: str) -> int:
    """Smithsonian Open Access — 4.5M+ images, free API, no key needed."""
    os.makedirs(output_dir, exist_ok=True)
    
    search_url = "https://api.si.edu/openaccess/api/v1.0/search"
    params = {
        "q": query,
        "rows": min(limit, 100),
        "api_key": os.environ.get("SMITHSONIAN_API_KEY", ""),
    }
    
    # Smithsonian requires API key — get free at https://api.data.gov/signup/
    if not params["api_key"]:
        print("  Smithsonian: No API key. Get one free at https://api.data.gov/signup/")
        print("  Set SMITHSONIAN_API_KEY env var")
        # Try demo endpoint
        params["api_key"] = "DEMO_KEY"
    
    downloaded = 0
    try:
        resp = requests.get(search_url, params=params, timeout=15)
        data = resp.json()
        rows = data.get("response", {}).get("rows", [])
        
        for row in rows:
            if downloaded >= limit:
                break
            
            content = row.get("content", {})
            desc = content.get("descriptiveNonRepeating", {})
            
            # Get image URL
            online_media = desc.get("online_media", {}).get("media", [])
            for media in online_media:
                img_url = media.get("content", "")
                if img_url and any(ext in img_url.lower() for ext in ['.jpg', '.jpeg', '.png']):
                    title = desc.get("title", {}).get("content", "unknown")[:40].replace("/", "_").replace(" ", "_")
                    filename = f"smithsonian_{downloaded:04d}_{title}.jpg"
                    
                    if download_image(img_url, os.path.join(output_dir, filename)):
                        downloaded += 1
                        if downloaded % 10 == 0:
                            print(f"    Downloaded {downloaded}/{limit}")
                    time.sleep(0.3)
                    break
    except Exception as e:
        print(f"    Smithsonian error: {e}")
    
    print(f"  Smithsonian: {downloaded} images")
    return downloaded


def scrape_dpla(query: str, limit: int, output_dir: str) -> int:
    """Digital Public Library of America — open API, no key needed for basic access."""
    os.makedirs(output_dir, exist_ok=True)
    
    search_url = "https://api.dp.la/v2/items"
    params = {
        "q": query,
        "page_size": min(limit, 100),
        "api_key": os.environ.get("DPLA_API_KEY", ""),
    }
    
    if not params["api_key"]:
        print("  DPLA: No API key. Get one free at https://dp.la/info/developers/codex/api-basics/")
        print("  Set DPLA_API_KEY env var")
        return 0
    
    downloaded = 0
    try:
        resp = requests.get(search_url, params=params, timeout=15)
        data = resp.json()
        docs = data.get("docs", [])
        
        for doc in docs:
            if downloaded >= limit:
                break
            
            img_url = doc.get("object", "")
            if not img_url:
                continue
            
            title = doc.get("sourceResource", {}).get("title", ["unknown"])
            if isinstance(title, list):
                title = title[0] if title else "unknown"
            title = title[:40].replace("/", "_").replace(" ", "_")
            filename = f"dpla_{downloaded:04d}_{title}.jpg"
            
            if download_image(img_url, os.path.join(output_dir, filename)):
                downloaded += 1
                if downloaded % 10 == 0:
                    print(f"    Downloaded {downloaded}/{limit}")
            time.sleep(0.3)
    except Exception as e:
        print(f"    DPLA error: {e}")
    
    print(f"  DPLA: {downloaded} images")
    return downloaded


QUERIES = [
    "mosque architecture",
    "Islamic dome",
    "muqarnas vault",
    "mashrabiya screen",
    "minaret",
    "mihrab",
    "pointed arch Islamic",
    "Islamic geometric pattern",
    "iwan portal",
    "calligraphy mosque",
]


def main():
    parser = argparse.ArgumentParser(description="Traced: Archive-grade training data")
    parser.add_argument("--source", choices=["manar", "europeana", "smithsonian", "dpla", "all"], default="all")
    parser.add_argument("--query", default=None, help="Single query (overrides default list)")
    parser.add_argument("--limit", type=int, default=50, help="Images per query")
    parser.add_argument("--output-dir", default="training_data/archives")
    parser.add_argument("--europeana-key", default=None)
    args = parser.parse_args()
    
    queries = [args.query] if args.query else QUERIES
    total = 0
    
    for query in queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        
        per_source = max(5, args.limit // len(queries))
        
        if args.source in ("manar", "all"):
            print("  Searching Manar al-Athar (Oxford)...")
            total += scrape_manar_al_athar(query, per_source, f"{args.output_dir}/manar")
        
        if args.source in ("europeana", "all"):
            print("  Searching Europeana...")
            total += scrape_europeana(query, per_source, f"{args.output_dir}/europeana", args.europeana_key)
        
        if args.source in ("smithsonian", "all"):
            print("  Searching Smithsonian Open Access...")
            total += scrape_smithsonian(query, per_source, f"{args.output_dir}/smithsonian")
        
        if args.source in ("dpla", "all"):
            print("  Searching DPLA...")
            total += scrape_dpla(query, per_source, f"{args.output_dir}/dpla")
    
    print(f"\n{'='*50}")
    print(f"Total archive images downloaded: {total}")
    print(f"Output: {args.output_dir}")
    
    # Count existing training images
    raw_count = sum(1 for _ in Path("training_data/raw").rglob("*") if _.is_file()) if Path("training_data/raw").exists() else 0
    archive_count = sum(1 for _ in Path(args.output_dir).rglob("*") if _.is_file()) if Path(args.output_dir).exists() else 0
    print(f"\nTraining data total: {raw_count} raw + {archive_count} archive = {raw_count + archive_count}")


if __name__ == "__main__":
    main()
