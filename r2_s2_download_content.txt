import os
import subprocess
import sys

# R2 North Pacific Gyre bounds
# 140-160W, 25-35N
TILES = ["01QDA","01QEA","02QMK","02QNK","03QWR","03QXR"]

OUT_DIR = os.path.join("outputs","sentinel2","R2")
os.makedirs(OUT_DIR, exist_ok=True)

USER = "adegaking1@gmail.com"
PASS = "Kingjones400$"

# Date range 2022-01-01 to 2025-12-31, cloud cover < 20%
import sentinelsat
from sentinelsat import SentinelAPI, make_path_filter
from datetime import date

api = SentinelAPI(USER, PASS, "https://apihub.copernicus.eu/apihub")

for tile in TILES:
    print(f"Searching tile {tile}...")
    products = api.query(
        platformname="Sentinel-2",
        producttype="S2MSI1C",
        date=("20220101","20251231"),
        cloudcoverpercentage=(0,20),
        filename=f"*_T{tile}_*"
    )
    print(f"  Found {len(products)} products for {tile}")
    if len(products) > 0:
        # Download up to 8 per tile
        import itertools
        subset = dict(itertools.islice(products.items(), 8))
        api.download_all(subset, directory_path=OUT_DIR)
        print(f"  Downloaded {len(subset)} scenes for {tile}")

print("R2 Sentinel-2 download complete.")
