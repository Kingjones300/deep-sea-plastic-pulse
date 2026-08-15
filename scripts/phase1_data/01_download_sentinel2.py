#!/usr/bin/env python3
"""
Phase 1 - Script 01
Sentinel-2 Smart Seasonal Download for Region R3
Downloads one best scene per season per tile (2022-2025)
Target: ~96 scenes (~67 GB)
"""

import os
import sys
import json
import time
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils import setup_logging, ensure_dir, load_config, REGIONS

setup_logging("01_download_sentinel2")
from loguru import logger

TOKEN_URL  = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
SEARCH_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
CHUNK_SIZE = 8192

SEASONS = {
    "Winter": ("01-01", "03-15"),
    "Spring": ("03-16", "06-14"),
    "Summer": ("06-15", "09-14"),
    "Autumn": ("09-15", "12-31"),
}

YEARS = [2022, 2023, 2024, 2025]

TARGET_TILES = ["31TGK", "31TGJ", "31TGL", "32TMP", "32TMQ", "33TUG"]


def get_token(username, password):
    try:
        resp = requests.post(TOKEN_URL, data={
            "grant_type": "password",
            "username": username,
            "password": password,
            "client_id": "cdse-public",
        }, timeout=30)
        if resp.status_code == 200:
            logger.info("Token obtained successfully")
            return resp.json()["access_token"]
        else:
            logger.error(f"Token failed {resp.status_code}: {resp.text[:200]}")
            return None
    except Exception as e:
        logger.error(f"Token error: {e}")
        return None


def search_best_scene(tile, date_from, date_to):
    filter_str = (
        f"Collection/Name eq 'SENTINEL-2' and "
        f"Attributes/OData.CSC.StringAttribute/any("
        f"att:att/Name eq 'productType' and "
        f"att/OData.CSC.StringAttribute/Value eq 'S2MSI2A') and "
        f"Attributes/OData.CSC.StringAttribute/any("
        f"att:att/Name eq 'tileId' and "
        f"att/OData.CSC.StringAttribute/Value eq '{tile}') and "
        f"Attributes/OData.CSC.DoubleAttribute/any("
        f"att:att/Name eq 'cloudCover' and "
        f"att/OData.CSC.DoubleAttribute/Value le 15) and "
        f"ContentDate/Start gt {date_from}T00:00:00.000Z and "
        f"ContentDate/Start lt {date_to}T23:59:59.999Z"
    )

    for attempt in range(1, 4):
        try:
            resp = requests.get(
                SEARCH_URL,
                params={
                    "$filter": filter_str,
                    "$orderby": "ContentDate/Start asc",
                    "$top": 1,
                },
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json().get("value", [])
                return data[0] if data else None
            else:
                logger.warning(f"Search error {resp.status_code}")
                return None
        except Exception as e:
            logger.warning(f"Attempt {attempt}/3 failed: {e}")
            time.sleep(10)

    return None


def download_scene(product, token, out_dir):
    pid = product["Id"]
    name = product["Name"]
    out_path = out_dir / f"{name}.zip"

    if out_path.exists():
        logger.info(f"  SKIP (exists): {name[:60]}")
        return True

    url = (
        f"https://zipper.dataspace.copernicus.eu"
        f"/odata/v1/Products({pid})/$value"
    )
    headers = {"Authorization": f"Bearer {token}"}

    try:
        with requests.get(
            url, headers=headers, stream=True, timeout=600
        ) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            mb = total / 1024 / 1024
            logger.info(f"  Downloading: {name[:55]} ({mb:.0f} MB)")
            with open(out_path, "wb") as f:
                downloaded = 0
                for chunk in r.iter_content(CHUNK_SIZE):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded / total * 100
                        print(f"\r    {pct:.1f}%", end="", flush=True)
            print()
        logger.success(f"  Done: {name[:60]}")
        return True
    except Exception as e:
        logger.error(f"  Failed: {e}")
        if out_path.exists():
            out_path.unlink()
        return False


def main():
    config = load_config()
    out_dir = ensure_dir(
        Path(config.get("SENTINEL2_DIR", "outputs/sentinel2")) / "R3"
    )
    username = config.get("SCIHUB_USERNAME", "")
    password = config.get("SCIHUB_PASSWORD", "")

    logger.info("=" * 60)
    logger.info("Phase 1 - Sentinel-2 SMART Seasonal Download")
    logger.info("Region:  R3 Western Mediterranean")
    logger.info(f"Tiles:   {TARGET_TILES}")
    logger.info(f"Years:   {YEARS}")
    logger.info(f"Seasons: {list(SEASONS.keys())}")
    logger.info(f"Target:  ~{len(TARGET_TILES) * len(SEASONS) * len(YEARS)} scenes")
    logger.info(f"Output:  {out_dir}")
    logger.info("=" * 60)

    token = get_token(username, password)
    if not token:
        sys.exit(1)

    plan = []
    for year in YEARS:
        for season, (d_start, d_end) in SEASONS.items():
            for tile in TARGET_TILES:
                plan.append({
                    "year":      year,
                    "season":    season,
                    "tile":      tile,
                    "date_from": f"{year}-{d_start}",
                    "date_to":   f"{year}-{d_end}",
                })

    logger.info(f"\nTotal slots to search: {len(plan)}")
    logger.info("Starting search and download...\n")

    ok = fail = not_found = 0
    downloaded_log = []

    for i, slot in enumerate(plan, 1):
        logger.info(
            f"[{i}/{len(plan)}] "
            f"{slot['year']} {slot['season']} "
            f"Tile {slot['tile']}"
        )

        if i % 10 == 0:
            logger.info("Refreshing token...")
            token = get_token(username, password)

        product = search_best_scene(
            slot["tile"],
            slot["date_from"],
            slot["date_to"],
        )

        if product is None:
            logger.warning(
                f"  No scene found for {slot['tile']} "
                f"{slot['year']} {slot['season']}"
            )
            not_found += 1
            continue

        logger.info(f"  Found: {product['Name'][:55]}")

        if download_scene(product, token, out_dir):
            ok += 1
            downloaded_log.append(product["Name"])
        else:
            fail += 1

        time.sleep(2)

    log_path = out_dir / "downloaded_scenes.json"
    with open(log_path, "w") as f:
        json.dump(downloaded_log, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info(f"COMPLETE:")
    logger.info(f"  Downloaded:  {ok}")
    logger.info(f"  Failed:      {fail}")
    logger.info(f"  Not found:   {not_found}")
    logger.info(f"  Log saved:   {log_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()