import os
import copernicusmarine

OUT_DIR = os.path.join("outputs", "cmems", "R2")
os.makedirs(OUT_DIR, exist_ok=True)

print("Downloading R2 CMEMS ocean physics (GLORYS12v1)...")
print("Region: 140-160W, 25-35N, 2022-2025")

copernicusmarine.subset(
    dataset_id="cmems_mod_glo_phy_my_0.083deg_P1M-m",
    variables=["uo", "vo", "thetao", "so", "zos"],
    minimum_longitude=-160.0,
    maximum_longitude=-140.0,
    minimum_latitude=25.0,
    maximum_latitude=35.0,
    minimum_depth=0.494,
    maximum_depth=2000.0,
    start_datetime="2022-01-01T00:00:00",
    end_datetime="2025-12-31T00:00:00",
    output_filename="R2_GLORYS12_2022_2025.nc",
    output_directory=OUT_DIR,
    username="kadega",
    password="Kingjones300$",
)

print("CMEMS download complete.")
print("File saved to:", os.path.join(OUT_DIR, "R2_GLORYS12_2022_2025.nc"))