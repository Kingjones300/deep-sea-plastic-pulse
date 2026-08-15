import os
import requests
import time

OUT_DIR = os.path.join("outputs", "modis", "R2")
os.makedirs(OUT_DIR, exist_ok=True)

USERNAME = "kingjones"
PASSWORD = "Kingjones300$"

BASE_URL = "https://oceandata.sci.gsfc.nasa.gov/ob/getfile/"

FILES = [
    "AQUA_MODIS.20220101_20221231.L3m.YR.CHL.chlor_a.4km.nc",
    "AQUA_MODIS.20230101_20231231.L3m.YR.CHL.chlor_a.4km.nc",
    "AQUA_MODIS.20240101_20241231.L3m.YR.CHL.chlor_a.4km.nc",
    "AQUA_MODIS.20220101_20221231.L3m.YR.SST.sst.4km.nc",
    "AQUA_MODIS.20230101_20231231.L3m.YR.SST.sst.4km.nc",
    "AQUA_MODIS.20240101_20241231.L3m.YR.SST.sst.4km.nc",
]

session = requests.Session()
session.auth = (USERNAME, PASSWORD)

for fname in FILES:
    out_path = os.path.join(OUT_DIR, fname)
    if os.path.exists(out_path):
        size = os.path.getsize(out_path) / 1024 / 1024
        print(f"Already exists: {fname} ({size:.1f} MB)")
        continue
    url = BASE_URL + fname
    print(f"Downloading: {fname}")
    try:
        r = session.get(url, stream=True, timeout=120)
        if r.status_code == 200:
            total = 0
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    total += len(chunk)
            print(f"  Done: {total/1024/1024:.1f} MB")
        else:
            print(f"  FAILED: {r.status_code} for {fname}")
    except Exception as e:
        print(f"  ERROR: {e}")
    time.sleep(2)

print("MODIS download complete.")