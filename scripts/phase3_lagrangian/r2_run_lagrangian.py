import numpy as np
import netCDF4 as nc
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

ROOT    = Path(r"C:\Users\Apple\deep_sea_pulse")
PHY_DIR = ROOT / "outputs" / "cmems" / "R2"
OUT_DIR = ROOT / "outputs" / "results" / "phase3" / "R2"
FIG_DIR = ROOT / "outputs" / "figures" / "R2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("  PHASE 3 - Bio-Ballistic Lagrangian Simulation")
print("  Region R2: North Pacific Subtropical Gyre")
print("  Deep Sea Plastic Pulse | Tianjin University")
print("="*60)

# R2 physics constants — oligotrophic North Pacific
RHO_HDPE    = 952.0      # kg/m3 virgin HDPE
RHO_BIOFILM = 1388.0     # kg/m3 wet biofilm (Kooi 2017)
RHO_SW      = 1026.0     # kg/m3 North Pacific subtropical (less saline)
G           = 9.81
MU_WATER    = 1.02e-3    # Pa.s (warmer water = lower viscosity)
# Biofilm growth — oligotrophic: lower MU_MAX, lower K_N
MU_MAX      = 0.10       # /day (reduced — oligotrophic, Chl-a=0.08)
K_N         = 0.008      # mg/L Michaelis-Menten (low nutrient)
K_D         = 0.04       # /day detachment
ALPHA       = 0.30       # max biofilm fraction (less productive)
W_MAX_MDAY  = 25.0       # m/day cap (slower in gyre)
DT          = 3600.0     # s
N_PARTICLES = 500
SIM_DAYS    = 180        # longer — slow biofouling needs more time
MIN_SINK_DEPTH = 50.0

print(f"\n   Particles: {N_PARTICLES} | Duration: {SIM_DAYS} days | dt: 1 hr")
print(f"   RHO_SW: {RHO_SW} kg/m3 | Chl-a ref: 0.08 mg/m3 (oligotrophic)")
print(f"   Terminal velocity capped at {W_MAX_MDAY} m/day")

# Load CMEMS
print("\n[1/6] Loading CMEMS physics data (R2 North Pacific)...")
ds   = nc.Dataset(PHY_DIR / "R2_GLORYS12_2022_2025.nc")
lats = ds.variables["latitude"][:]
lons = ds.variables["longitude"][:]
deps = ds.variables["depth"][:]
uo   = ds.variables["uo"][:]
vo   = ds.variables["vo"][:]
sst  = ds.variables["thetao"][:, 0, :, :]
print(f"   Grid: {len(lats)}lat x {len(lons)}lon x {len(deps)}dep")
print(f"   SST mean: {float(np.nanmean(sst)):.1f} C")
print(f"   Time steps: {uo.shape[0]} (monthly 2022-2025)")

def bio_ballistic_density(t_days, sst_c, chl_mg=0.08):
    """
    Bio-Ballistic Density Function — R2 oligotrophic parameterisation
    Chl-a default 0.08 mg/m3 (North Pacific Subtropical Gyre)
    """
    mu_eff = MU_MAX * (chl_mg / (chl_mg + K_N)) * np.exp(0.063*(sst_c - 20.0))
    mu_net = mu_eff - K_D
    if mu_net <= 0:
        f_bio = 0.001
    else:
        f_bio = ALPHA * (1.0 - np.exp(-mu_net * t_days))
    f_bio = float(np.clip(f_bio, 0.0, ALPHA))
    rho_p = RHO_HDPE*(1.0 - f_bio) + RHO_BIOFILM*f_bio
    return rho_p, f_bio

def terminal_velocity_mday(rho_p, rho_sw=RHO_SW):
    delta_rho = rho_p - rho_sw
    w = np.tanh(delta_rho / 10.0) * W_MAX_MDAY
    return float(w)

# Verify physics
print("\n[2/6] Verifying Bio-Ballistic physics (SST=22C, Chl-a=0.08)...")
for t_check in [10, 30, 60, 90, 120, 150, 180]:
    rho_t, f_t = bio_ballistic_density(t_check, 22.0)
    w_t = terminal_velocity_mday(rho_t)
    status = "SINKING" if rho_t > RHO_SW else "floating"
    print(f"   t={t_check:3d}d: rho={rho_t:.2f} kg/m3 | f={f_t:.3f} | w={w_t:+6.2f} m/day | {status}")

t_sink = None
for t in range(1, 1000):
    rho_t, _ = bio_ballistic_density(t, 22.0)
    if rho_t > RHO_SW:
        t_sink = t
        break
if t_sink:
    rho_ts, _ = bio_ballistic_density(t_sink, 22.0)
    w_ts = terminal_velocity_mday(rho_ts)
    print(f"\n   >>> Sinking threshold: day {t_sink} | w = {w_ts:.2f} m/day <<<")
else:
    print("\n   WARNING: no sinking detected — adjusting to day 90")
    t_sink = 90

# Initialise particles — North Pacific Gyre bounds
print("\n[3/6] Initialising particles (North Pacific Gyre)...")
rng = np.random.default_rng(42)
lat_min, lat_max = 25.0, 35.0
lon_min, lon_max = -160.0, -140.0

p_lat    = rng.uniform(lat_min, lat_max, N_PARTICLES)
p_lon    = rng.uniform(lon_min, lon_max, N_PARTICLES)
p_dep    = np.zeros(N_PARTICLES)
# Gyre plastic is aged — initialise with 20-60 days age (accumulated)
p_age    = rng.uniform(20, 60, N_PARTICLES)
p_active = np.ones(N_PARTICLES, dtype=bool)
p_sink_day = np.full(N_PARTICLES, np.nan)
p_sink_dep = np.full(N_PARTICLES, np.nan)
p_sink_lat = np.full(N_PARTICLES, np.nan)
p_sink_lon = np.full(N_PARTICLES, np.nan)
print(f"   {N_PARTICLES} particles | Age: 20-60 days (gyre-accumulated)")
print(f"   Release zone: {lat_min}-{lat_max}N, {lon_min}-{lon_max}W")

def nearest_idx(arr, val):
    return int(np.argmin(np.abs(np.array(arr) - val)))

# Lagrangian integration
print(f"\n[4/6] Running Lagrangian integration ({SIM_DAYS} days)...")
n_steps = SIM_DAYS * 24
kz      = 5e-5   # m2/s vertical diffusion (oligotrophic — less turbulent)
max_dep = 2500.0
n_time  = uo.shape[0]

for step in range(n_steps):
    t_idx   = min(step // (24*30), n_time-1)  # monthly index
    active  = np.where(p_active)[0]
    if len(active) == 0:
        break

    for i in active:
        ilat = nearest_idx(lats, p_lat[i])
        ilon = nearest_idx(lons, p_lon[i])
        idep = nearest_idx(deps, max(p_dep[i], 0.0))

        u = float(uo[t_idx, idep, ilat, ilon])
        v = float(vo[t_idx, idep, ilat, ilon])
        if np.isnan(u): u = 0.0
        if np.isnan(v): v = 0.0

        sst_local = float(sst[t_idx, ilat, ilon])
        if np.isnan(sst_local): sst_local = 22.0

        rho_p, _ = bio_ballistic_density(p_age[i], sst_local, chl_mg=0.08)
        w_mday   = terminal_velocity_mday(rho_p)
        w_ms     = w_mday / 86400.0

        dz_diff = np.sqrt(2.0 * kz * DT) * rng.standard_normal()
        ddep    = w_ms * DT + dz_diff

        dlat = (v * DT) / 111320.0
        dlon = (u * DT) / (111320.0 * np.cos(np.radians(p_lat[i])))

        p_lat[i] = float(np.clip(p_lat[i]+dlat, lat_min-5, lat_max+5))
        p_lon[i] = float(np.clip(p_lon[i]+dlon, lon_min-10, lon_max+10))
        p_dep[i] = float(np.clip(p_dep[i]+ddep, 0.0, max_dep))
        p_age[i] += DT / 86400.0

        if rho_p > RHO_SW and p_dep[i] >= MIN_SINK_DEPTH:
            p_sink_day[i] = p_age[i]
            p_sink_dep[i] = p_dep[i]
            p_sink_lat[i] = p_lat[i]
            p_sink_lon[i] = p_lon[i]
            p_active[i]   = False

    if step % (n_steps // 10) == 0:
        sunk = int((~p_active).sum())
        print(f"   {100*step/n_steps:4.0f}% | Day {step//24:3d} | "
              f"Sunk: {sunk}/{N_PARTICLES} ({100*sunk/N_PARTICLES:.0f}%)")

print("   Simulation complete.")

# Results
print("\n[5/6] Computing results...")
sunk_mask     = ~p_active
n_sunk        = int(sunk_mask.sum())
pct_sunk      = 100.0 * n_sunk / N_PARTICLES
mean_sink_day = float(np.nanmean(p_sink_day)) if n_sunk > 0 else 0.0
mean_sink_dep = float(np.nanmean(p_sink_dep)) if n_sunk > 0 else 0.0
print(f"   Particles sunk:    {n_sunk}/{N_PARTICLES} ({pct_sunk:.1f}%)")
print(f"   Mean days to sink: {mean_sink_day:.1f}")
print(f"   Mean sink depth:   {mean_sink_dep:.1f} m")

# Figures
print("\n[6/6] Generating figures...")

fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(p_lon[p_active], p_lat[p_active],
           c="steelblue", s=8, alpha=0.5, label="Still floating")
if n_sunk > 0:
    sc = ax.scatter(p_sink_lon[sunk_mask], p_sink_lat[sunk_mask],
                    c=p_sink_dep[sunk_mask], cmap="plasma_r",
                    s=18, alpha=0.9, label="Sink points")
    plt.colorbar(sc, ax=ax, label="Sink depth (m)")
ax.set_xlim(lon_min-10, lon_max+10)
ax.set_ylim(lat_min-5, lat_max+5)
ax.set_xlabel("Longitude (W)"); ax.set_ylabel("Latitude (N)")
ax.set_title("Bio-Ballistic Sink Points — North Pacific Subtropical Gyre R2")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "lagrangian_sink_points.png", dpi=150)
plt.close()
print("   Sink point map saved")

fig, ax = plt.subplots(figsize=(8, 5))
valid = p_sink_dep[~np.isnan(p_sink_dep)]
if len(valid) > 0:
    ax.hist(valid, bins=30, color="navy", alpha=0.75, edgecolor="white")
ax.set_xlabel("Sink Depth (m)"); ax.set_ylabel("Count")
ax.set_title("Plastic Sink Depth Distribution — R2 North Pacific Gyre")
plt.tight_layout()
plt.savefig(FIG_DIR / "sink_depth_distribution.png", dpi=150)
plt.close()
print("   Sink depth histogram saved")

t_arr   = np.linspace(0, 180, 300)
rho_arr = [bio_ballistic_density(t, 22.0, chl_mg=0.08)[0] for t in t_arr]
w_arr   = [terminal_velocity_mday(r) for r in rho_arr]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
ax1.plot(t_arr, rho_arr, "b-", lw=2.5, label=r"$\rho_p(t)$")
ax1.axhline(RHO_SW, color="red", ls="--", lw=1.8,
            label=f"Seawater {RHO_SW} kg/m³ (N. Pacific)")
ax1.fill_between(t_arr, rho_arr, RHO_SW,
                 where=[r > RHO_SW for r in rho_arr],
                 alpha=0.15, color="red", label="Sinking zone")
ax1.set_ylabel("Density (kg/m³)")
ax1.set_title(r"Bio-Ballistic $\rho_p(t)$ — R2 Oligotrophic Gyre")
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(t_arr, w_arr, "g-", lw=2.5)
ax2.axhline(0, color="black", lw=1)
ax2.fill_between(t_arr, w_arr, 0,
                 where=[w > 0 for w in w_arr],
                 alpha=0.2, color="green", label="Sinking")
ax2.set_xlabel("Days at sea")
ax2.set_ylabel("Terminal velocity (m/day)")
ax2.legend(); ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "bioBallistic_density_curve.png", dpi=150)
plt.close()
print("   Density + velocity curves saved")

summary = {
    "timestamp":         datetime.now().isoformat(),
    "region":            "R2_North_Pacific_Gyre",
    "n_particles":       N_PARTICLES,
    "sim_days":          SIM_DAYS,
    "n_sunk":            n_sunk,
    "pct_sunk":          round(pct_sunk, 2),
    "mean_days_to_sink": round(mean_sink_day, 1),
    "mean_sink_depth_m": round(mean_sink_dep, 1),
    "days_to_threshold": t_sink,
    "w_max_cap_mday":    W_MAX_MDAY,
    "min_sink_depth_m":  MIN_SINK_DEPTH,
    "rho_sw_kg_m3":      RHO_SW,
    "chl_a_ref_mg_m3":   0.08,
    "sst_ref_c":         22.0,
    "physics_ref":       "Kooi_2017_oligotrophic + Cozar_2014"
}
with open(OUT_DIR / "phase3_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nPhase 3 Summary:")
for k, v in summary.items():
    print(f"   {k}: {v}")
print("\n✅ Phase 3 R2 complete.")
print(f"   Results → {OUT_DIR}")
print(f"   Figures → {FIG_DIR}")
print("="*60)