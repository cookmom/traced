#!/usr/bin/env python3
"""
Traced — Bulk Training Data Scraper
Downloads architectural images from public sources for LoRA training.

Sources:
1. Google Open Images — labeled dome/minaret/mosque segments
2. Wikimedia Commons — CC-licensed Islamic architecture photos
3. Archnet — Islamic architecture database (Aga Khan Trust)

Usage:
    python scrape_training_data.py --source wikimedia --query "Islamic architecture mosque" --limit 100 --output-dir training_data/raw
    python scrape_training_data.py --source openimages --classes "Dome,Minaret,Mosque" --limit 200 --output-dir training_data/raw
    python scrape_training_data.py --source archnet --limit 50 --output-dir training_data/raw
    python scrape_training_data.py --all --limit 50 --output-dir training_data/raw
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


def scrape_wikimedia(query: str, limit: int, output_dir: str) -> int:
    """Download CC-licensed images from Wikimedia Commons."""
    if not HAS_REQUESTS:
        print("  requests not installed")
        return 0
    
    os.makedirs(output_dir, exist_ok=True)
    headers = {"User-Agent": "Traced/1.0 (architectural research; github.com/cookmom/traced)"}
    
    # Search Wikimedia Commons API
    url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "generator": "search",
        "gsrnamespace": 6,  # File namespace
        "gsrsearch": query,
        "gsrlimit": min(limit, 50),
        "prop": "imageinfo",
        "iiprop": "url|size|mime",
        "iiurlwidth": 800,  # Thumbnail at 800px
        "format": "json",
    }
    
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
    except Exception as e:
        print(f"  Wikimedia API error: {e}")
        return 0
    
    downloaded = 0
    for page_id, page in pages.items():
        if downloaded >= limit:
            break
        
        imageinfo = page.get("imageinfo", [{}])[0]
        thumb_url = imageinfo.get("thumburl", imageinfo.get("url"))
        mime = imageinfo.get("mime", "")
        
        if not thumb_url or "image" not in mime:
            continue
        
        # Download
        ext = ".jpg" if "jpeg" in mime else ".png"
        filename = f"wikimedia_{downloaded:04d}{ext}"
        filepath = Path(output_dir) / filename
        
        try:
            img_resp = requests.get(thumb_url, headers=headers, timeout=10)
            if img_resp.status_code == 200:
                filepath.write_bytes(img_resp.content)
                downloaded += 1
                if downloaded % 10 == 0:
                    print(f"    Downloaded {downloaded}/{limit}")
                time.sleep(0.5)  # Be polite
        except Exception:
            continue
    
    print(f"  Wikimedia: {downloaded} images downloaded")
    return downloaded


def scrape_archnet(limit: int, output_dir: str) -> int:
    """Scrape images from Archnet (Aga Khan architecture database)."""
    if not HAS_REQUESTS:
        return 0
    
    os.makedirs(output_dir, exist_ok=True)
    headers = {"User-Agent": "Traced/1.0 (architectural research)"}
    
    # Archnet search for mosque/Islamic architecture
    queries = [
        "mosque", "minaret", "dome", "muqarnas", "mihrab",
        "mashrabiya", "iwan", "minbar", "geometric pattern",
    ]
    
    downloaded = 0
    per_query = max(1, limit // len(queries))
    
    for query in queries:
        if downloaded >= limit:
            break
        
        try:
            url = f"https://www.archnet.org/search?q={query.replace(' ', '+')}"
            resp = requests.get(url, headers=headers, timeout=15)
            
            # Extract image URLs from page (simple regex)
            import re
            img_urls = re.findall(r'src="(https://[^"]*archnet[^"]*\.(?:jpg|jpeg|png))"', resp.text, re.I)
            
            for img_url in img_urls[:per_query]:
                if downloaded >= limit:
                    break
                
                filename = f"archnet_{query.replace(' ', '_')}_{downloaded:04d}.jpg"
                filepath = Path(output_dir) / filename
                
                try:
                    img_resp = requests.get(img_url, headers=headers, timeout=10)
                    if img_resp.status_code == 200 and len(img_resp.content) > 5000:
                        filepath.write_bytes(img_resp.content)
                        downloaded += 1
                        time.sleep(0.5)
                except Exception:
                    continue
        except Exception as e:
            print(f"    Archnet query '{query}' failed: {e}")
    
    print(f"  Archnet: {downloaded} images downloaded")
    return downloaded


def scrape_openimages_metadata(classes: list, limit: int, output_dir: str) -> int:
    """Download Open Images metadata for architectural classes.
    Actual images need the OIDV7 downloader tool."""
    if not HAS_REQUESTS:
        return 0
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Open Images class IDs for architecture
    class_map = {
        "Dome": "/m/02_n6",
        "Minaret": "/m/04_r5c", 
        "Mosque": "/m/054_l",
        "Tower": "/m/01fdzj",
        "Arch": "/m/025nd",
        "Column": "/m/02dgv",
        "Window": "/m/0d4v4",
        "Door": "/m/02dgv",
    }
    
    print(f"  Open Images class IDs:")
    for cls in classes:
        cid = class_map.get(cls, "not found")
        print(f"    {cls}: {cid}")
    
    # Save class mapping for later bulk download
    meta_file = Path(output_dir) / "openimages_classes.json"
    meta_file.write_text(json.dumps({
        "classes": {cls: class_map.get(cls, "") for cls in classes},
        "limit_per_class": limit // len(classes),
        "download_instructions": [
            "pip install openimages",
            "oi_download_dataset --base_dir training_data/openimages --labels Dome Minaret Mosque --format darknet --limit " + str(limit),
        ],
    }, indent=2))
    
    print(f"  Saved Open Images metadata to {meta_file}")
    print(f"  To download actual images, run:")
    print(f"    pip install openimages")
    print(f"    oi_download_dataset --base_dir {output_dir}/openimages --labels {' '.join(classes)} --limit {limit}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(description="Traced: Scrape training data")
    parser.add_argument("--source", choices=["wikimedia", "archnet", "openimages", "all"], default="all")
    parser.add_argument("--query", default="Islamic architecture mosque dome minaret")
    parser.add_argument("--classes", default="Dome,Minaret,Mosque")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output-dir", default="training_data/raw")
    args = parser.parse_args()
    
    total = 0
    
    if args.source in ("wikimedia", "all"):
        print("Scraping Wikimedia Commons...")
        total += scrape_wikimedia(args.query, args.limit, args.output_dir + "/wikimedia")
    
    if args.source in ("archnet", "all"):
        print("Scraping Archnet (Aga Khan)...")
        total += scrape_archnet(args.limit, args.output_dir + "/archnet")
    
    if args.source in ("openimages", "all"):
        print("Setting up Google Open Images...")
        classes = [c.strip() for c in args.classes.split(",")]
        scrape_openimages_metadata(classes, args.limit, args.output_dir + "/openimages")
    
    print(f"\nTotal images downloaded: {total}")
    print(f"Output directory: {args.output_dir}")
    print(f"\nNext: Run the pipeline on each image to auto-label:")
    print(f"  for img in {args.output_dir}/*/*.jpg; do")
    print(f"    python extract-sam2.py --image $img --output /tmp/ext.json --name auto")
    print(f"    python collect_training.py --extraction /tmp/ext.json --image $img")
    print(f"  done")


if __name__ == "__main__":
    main()
