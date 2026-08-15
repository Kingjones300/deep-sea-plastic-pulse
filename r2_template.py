import requests

USER = "adegaking1@gmail.com"
PASS = "Kingjones400$"

def get_token(user, password):
    r = requests.post(
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        data={"client_id": "cdse-public", "username": user, "password": password, "grant_type": "password"},
        timeout=30
    )
    r.raise_for_status()
    return r.json()["access_token"]

token = get_token(USER, PASS)
headers = {"Authorization": f"Bearer {token}"}
base = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

# Correct MGRS tiles for 140-160W, 25-35N (North Pacific Gyre)
# Zone 4-6, latitude bands Q-R (25-35N)
# 04QFJ, 04QGJ, 05QKB, 05QKC, 05QLB, 05QLC, 06QUH, 06QVH
# Also try: 03QWR, 03QXR (near 160W), 04QFJ, 05QKB, 06QUH
TILES = ["04QFJ", "04QGJ", "05QKB", "05QKC", "05QLB", "05QLC", "06QUH", "06QVH", "03QWR", "03QXR"]

for tile in TILES:
    filt = (
        "Collection/Name eq 'SENTINEL-2'"
        f" and contains(Name,'_T{tile}_')"
        " and contains(Name,'MSIL1C')"
        " and ContentDate/Start gt 2022-01-01T00:00:00.000Z"
        " and ContentDate/Start lt 2022-06-01T00:00:00.000Z"
    )
    params = {"$filter": filt, "$top": 3}
    r = requests.get(base, headers=headers, params=params, timeout=60)
    products = r.json().get("value", [])
    if len(products) > 0:
        print(f"Tile {tile}: {len(products)} found")
        for p in products:
            print(f"  {p['Name']}")
    else:
        print(f"Tile {tile}: 0 found")