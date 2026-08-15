import os
import requests
import time

USER = "adegaking1@gmail.com"
PASS = "Kingjones400$"
OUT_DIR = os.path.join("outputs", "sentinel2", "R2")
os.makedirs(OUT_DIR, exist_ok=True)

TILES = ["04QFJ", "04QGJ", "05QKB", "05QKC", "05QLB", "05QLC"]
MAX_PER_TILE = 8

def get_token(user, password):
    r = requests.post(
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        data={"client_id": "cdse-public", "username": user, "password": password, "grant_type": "password"},
        timeout=30
    )
    r.raise_for_status()
    return r.json()["access_token"]

def download_product(product_id, product_name, token, out_dir):
    out_path = os.path.join(out_dir, product_name + ".zip")
    if os.path.exists(out_path):
        print(f"    Already exists: {product_name}")
        return True
    url = f"https://download.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        r = requests.get(url, headers=headers, stream=True, timeout=120)
        if r.status_code == 200:
            total = 0
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    total += len(chunk)
            size_mb = total / 1024 / 1024
            print(f"    Downloaded: {product_name} ({size_mb:.1f} MB)")
            return True
        else:
            print(f"    FAILED {r.status_code}: {product_name}")
            return False
    except Exception as e:
        print(f"    ERROR: {e}")
        return False

print("Getting token...")
token = get_token(USER, PASS)
print("Token OK")

base = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
all_products = []

for tile in TILES:
    filt = (
        "Collection/Name eq 'SENTINEL-2'"
        f" and contains(Name,'_T{tile}_')"
        " and contains(Name,'MSIL1C')"
        " and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le 20.00)"
        " and ContentDate/Start gt 2022-01-01T00:00:00.000Z"
        " and ContentDate/Start lt 2025-12-31T00:00:00.000Z"
    )
    params = {"$filter": filt, "$orderby": "ContentDate/Start asc", "$top": MAX_PER_TILE}
    r = requests.get(base, headers={"Authorization": f"Bearer {token}"}, params=params, timeout=60)
    products = r.json().get("value", [])
    print(f"Tile {tile}: {len(products)} products queued")
    all_products.extend(products)

print(f"\nTotal products to download: {len(all_products)}")
print("Starting downloads...\n")

success = 0
fail = 0
for i, p in enumerate(all_products):
    print(f"[{i+1}/{len(all_products)}] {p['Name']}")
    # Refresh token every 20 downloads
    if i > 0 and i % 20 == 0:
        token = get_token(USER, PASS)
        print("  Token refreshed")
    ok = download_product(p["Id"], p["Name"], token, OUT_DIR)
    if ok:
        success += 1
    else:
        fail += 1
    time.sleep(1)

print(f"\nDone. Success: {success}, Failed: {fail}")
print(f"Files in {OUT_DIR}:")
files = os.listdir(OUT_DIR)
for f in files:
    size = os.path.getsize(os.path.join(OUT_DIR, f)) / 1024 / 1024
    print(f"  {f} ({size:.1f} MB)")