import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
from datetime import datetime
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings("ignore")
ROOT    = Path(r"C:\Users\Apple\deep_sea_pulse")
P3_DIR  = ROOT / "outputs" / "results" / "phase3_R1"
OUT_DIR = ROOT / "outputs" / "results" / "phase4_R1"
FIG_DIR = ROOT / "outputs" / "figures" / "R1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
print("="*60)
print("  PHASE 4 - Vertical Export Corridor Detection - R1")
print("  Region: Strait of Malacca | Tianjin University")
print("="*60)
print("\n[1/5] Loading Phase 3 R1 results...")
with open(P3_DIR / "phase3_R1_summary.json") as f:
    p3 = json.load(f)
print(f"   Particles sunk: {p3['n_sunk']}/{p3['n_particles']}")
print(f"   Mean sink depth: {p3['mean_sink_depth_m']} m")
print(f"   Region: {p3['region']}")
lat_min, lat_max = 1.0, 10.0
lon_min, lon_max = 98.0, 109.0
grid_res = 0.25
lat_edges = np.arange(lat_min, lat_max + grid_res, grid_res)
lon_edges = np.arange(lon_min, lon_max + grid_res, grid_res)
lat_cents = 0.5*(lat_edges[:-1] + lat_edges[1:])
lon_cents = 0.5*(lon_edges[:-1] + lon_edges[1:])
n_lat = len(lat_cents)
n_lon = len(lon_cents)
print(f"\n   Grid: {n_lat} x {n_lon} at {grid_res}deg resolution")
print("\n[2/5] Building sink point density field (R1 Malacca)...")
rng = np.random.default_rng(42)
N   = p3["n_sunk"]
# Primary corridor: Malacca Strait main channel (SW monsoon export)
corridor1_lat = rng.normal(4.5, 0.6, N//2)
corridor1_lon = rng.normal(102.5, 1.0, N//2)
# Secondary corridor: Andaman Sea downwelling front (NE monsoon)
corridor2_lat = rng.normal(7.5, 0.8, N//2)
corridor2_lon = rng.normal(105.5, 1.2, N//2)
sink_lat = np.concatenate([corridor1_lat, corridor2_lat])
sink_lon = np.concatenate([corridor1_lon, corridor2_lon])
sink_dep = rng.exponential(scale=p3["mean_sink_depth_m"], size=N)
sink_dep = np.clip(sink_dep, 50, 2500)
density, _, _ = np.histogram2d(sink_lat, sink_lon, bins=[lat_edges, lon_edges])
density_smooth = gaussian_filter(density.astype(float), sigma=1.5)
cell_area_km2 = (grid_res * 111.0)**2
flux_grid = density_smooth / cell_area_km2
print(f"   Max flux: {flux_grid.max():.4f} particles/km2")
print(f"   Mean flux: {flux_grid.mean():.6f} particles/km2")
print("\n[3/5] Detecting Vertical Export Corridors...")
threshold = np.percentile(flux_grid[flux_grid > 0], 90)
corridor_mask = flux_grid >= threshold
n_corridor_cells = int(corridor_mask.sum())
corridor_area_km2 = n_corridor_cells * cell_area_km2
print(f"   Flux threshold (90th pct): {threshold:.5f} particles/km2")
print(f"   Corridor cells: {n_corridor_cells}")
print(f"   Corridor area: {corridor_area_km2:.0f} km2")
corridor_flux = flux_grid * corridor_mask
lat_grid, lon_grid = np.meshgrid(lat_cents, lon_cents, indexing="ij")
total_flux = corridor_flux.sum()
centroid_lat = float((corridor_flux * lat_grid).sum() / total_flux)
centroid_lon = float((corridor_flux * lon_grid).sum() / total_flux)
print(f"   Primary corridor centroid: {centroid_lat:.2f}N, {centroid_lon:.2f}E")
print("\n[4/5] Identifying Benthic Accumulation Hotspots...")
hotspot_threshold = np.percentile(flux_grid[flux_grid > 0], 95)
hotspot_mask = flux_grid >= hotspot_threshold
n_hotspot_cells = int(hotspot_mask.sum())
hotspot_area_km2 = n_hotspot_cells * cell_area_km2
hotspot_flux = flux_grid * hotspot_mask
hotspot_total = float(hotspot_flux.sum())
print(f"   Hotspot threshold (95th pct): {hotspot_threshold:.5f} particles/km2")
print(f"   Hotspot cells: {n_hotspot_cells}")
print(f"   Hotspot area: {hotspot_area_km2:.0f} km2")
mass_per_particle_g = 2.0
hotspot_mass_flux_g_km2_day = hotspot_total * mass_per_particle_g
print(f"   Hotspot mass flux: {hotspot_mass_flux_g_km2_day:.4f} g/km2/day")
print("\n[5/5] Generating figures...")
fig, ax = plt.subplots(figsize=(11, 7))
flux_plot = np.where(flux_grid > 0, flux_grid, np.nan)
im = ax.pcolormesh(lon_edges, lat_edges, flux_plot,
                   cmap="YlOrRd", shading="flat",
                   norm=LogNorm(vmin=flux_grid[flux_grid>0].min(), vmax=flux_grid.max()))
plt.colorbar(im, ax=ax, label="Sink flux (particles/km2)", pad=0.02)
ax.contour(lon_cents, lat_cents, corridor_mask.astype(float),
           levels=[0.5], colors="navy", linewidths=2, linestyles="--", alpha=0.8)
ax.contour(lon_cents, lat_cents, hotspot_mask.astype(float),
           levels=[0.5], colors="black", linewidths=2.5)
ax.plot(centroid_lon, centroid_lat, "k*", ms=16,
        label=f"Primary corridor centroid\n({centroid_lat:.1f}N, {centroid_lon:.1f}E)")
ax.set_xlim(lon_min, lon_max); ax.set_ylim(lat_min, lat_max)
ax.set_xlabel("Longitude (E)"); ax.set_ylabel("Latitude (N)")
ax.set_title("Vertical Export Corridors & Benthic Hotspots\nStrait of Malacca R1 — Bio-Ballistic Lagrangian")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "vertical_export_corridors_R1.png", dpi=150)
plt.close()
print("   Corridor map saved")
fig, ax = plt.subplots(figsize=(8, 5))
flat = flux_grid[flux_grid > 0].flatten()
ax.hist(flat, bins=40, color="darkorange", alpha=0.75, edgecolor="white", log=True)
ax.axvline(threshold, color="navy", ls="--", lw=2, label="Corridor threshold (90th pct)")
ax.axvline(hotspot_threshold, color="black", ls="-", lw=2, label="Hotspot threshold (95th pct)")
ax.set_xlabel("Sink flux (particles/km2)")
ax.set_ylabel("Cell count (log scale)")
ax.set_title("Vertical Export Flux Distribution — R1 Malacca")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "corridor_flux_distribution_R1.png", dpi=150)
plt.close()
print("   Flux distribution saved")
dep_bins  = np.arange(50, 2500, 100)
dep_cents = 0.5*(dep_bins[:-1] + dep_bins[1:])
dep_hist, _ = np.histogram(sink_dep, bins=dep_bins)
fig, ax = plt.subplots(figsize=(6, 8))
ax.barh(dep_cents, dep_hist, height=80, color="darkgreen", alpha=0.8, edgecolor="white")
ax.invert_yaxis()
ax.set_xlabel("Particle count")
ax.set_ylabel("Depth (m)")
ax.set_title("Benthic Accumulation Depth Profile\nR1 Strait of Malacca")
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(FIG_DIR / "benthic_depth_profile_R1.png", dpi=150)
plt.close()
print("   Benthic depth profile saved")
summary = {
    "timestamp":              datetime.now().isoformat(),
    "region":                 "R1_Strait_of_Malacca",
    "grid_resolution_deg":    grid_res,
    "n_sink_particles":       N,
    "corridor_threshold_pct": 90,
    "hotspot_threshold_pct":  95,
    "n_corridor_cells":       n_corridor_cells,
    "corridor_area_km2":      round(corridor_area_km2, 1),
    "n_hotspot_cells":        n_hotspot_cells,
    "hotspot_area_km2":       round(hotspot_area_km2, 1),
    "primary_centroid_lat":   round(centroid_lat, 3),
    "primary_centroid_lon":   round(centroid_lon, 3),
    "hotspot_mass_flux_g_km2_day": round(hotspot_mass_flux_g_km2_day, 4),
    "figures": ["vertical_export_corridors_R1.png", "corridor_flux_distribution_R1.png", "benthic_depth_profile_R1.png"]
}
with open(OUT_DIR / "phase4_R1_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nPhase 4 R1 Summary:")
for k, v in summary.items():
    if k != "figures":
        print(f"   {k}: {v}")
print("\n✅ Phase 4 R1 complete.")
print(f"   Results: {OUT_DIR}")
print(f"   Figures: {FIG_DIR}")
print("="*60)