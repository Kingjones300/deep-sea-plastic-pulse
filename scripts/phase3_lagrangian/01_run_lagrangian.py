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
PHY_DIR = ROOT / "outputs" / "cmems" / "physics" / "R3"
OUT_DIR = ROOT / "outputs" / "results" / "phase3"
FIG_DIR = ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("  PHASE 3 - Bio-Ballistic Lagrangian Simulation")
print("  Deep Sea Plastic Pulse | Tianjin University")
print("="*60)

# ── Physics constants ─────────────────────────────────────────
# Fragment geometry: thin film/sheet — high drag, low net velocity
# This matches real ocean plastic observations (1-30 m/day sink rate)
RHO_HDPE    = 952.0      # kg/m3 virgin HDPE
RHO_BIOFILM = 1388.0     # kg/m3 wet biofilm (Kooi 2017)
RHO_SW      = 1029.0     # kg/m3 Mediterranean seawater
G           = 9.81       # m/s2
MU_WATER    = 1.07e-3    # Pa.s dynamic viscosity
# Biofilm growth parameters — Kooi et al. 2017
MU_MAX      = 0.18       # /day
K_N         = 0.012      # mg/L
K_D         = 0.05       # /day detachment
ALPHA       = 0.35       # max biofilm volume fraction
# Terminal velocity cap (m/day) — empirical from field observations
# Cózar et al. 2014: plastics sink at 0.5-50 m/day in real ocean
W_MAX_MDAY  = 30.0       # hard cap on terminal velocity
# Integration
DT          = 3600.0     # s (1 hour)
N_PARTICLES = 500
SIM_DAYS    = 120
MIN_SINK_DEPTH = 50.0    # m

print(f"\n   Particles: {N_PARTICLES} | Duration: {SIM_DAYS} days | dt: 1 hr")
print(f"   Terminal velocity capped at {W_MAX_MDAY} m/day (empirical)")

# ── Load CMEMS ────────────────────────────────────────────────
print("\n[1/6] Loading CMEMS physics data (2022)...")
ds   = nc.Dataset(PHY_DIR / "R3_physics_2022.nc")
lats = ds.variables["latitude"][:]
lons = ds.variables["longitude"][:]
deps = ds.variables["depth"][:]
uo   = ds.variables["uo"][:]
vo   = ds.variables["vo"][:]
sst  = ds.variables["thetao"][:, 0, :, :]
print(f"   Grid: {len(lats)}lat x {len(lons)}lon x {len(deps)}dep")
print(f"   SST mean: {float(np.nanmean(sst)):.1f} C")

# ── Physics functions ─────────────────────────────────────────
def bio_ballistic_density(t_days, sst_c, chl_mg=0.5):
    """
    Eq.2-4: Bio-Ballistic particle density (Kooi et al. 2017)
    rho_p(t) = rho_hdpe*(1-f) + rho_biofilm*f
    f(t) = ALPHA * (1 - exp(-mu_net * t))
    mu_net = MU_MAX * chl/(chl+K_N) * exp(0.063*(SST-20)) - K_D
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
    """
    Eq.5: Empirically-capped terminal velocity (m/day)
    Stokes gives theoretical max; field observations cap at W_MAX_MDAY.
    Sign: positive = sinking, negative = buoyant.
    Ref: Cózar et al. 2014; Kooi et al. 2017
    """
    delta_rho = rho_p - rho_sw
    # Normalise to [-1, 1] range then scale to W_MAX
    # This preserves the sign and relative magnitude
    w = np.tanh(delta_rho / 10.0) * W_MAX_MDAY
    return float(w)

# ── Verify physics ────────────────────────────────────────────
print("\n[2/6] Verifying Bio-Ballistic physics...")
for t_check in [5, 15, 30, 45, 60, 90]:
    rho_t, f_t = bio_ballistic_density(t_check, 20.3)
    w_t = terminal_velocity_mday(rho_t)
    status = "SINKING" if rho_t > RHO_SW else "floating"
    print(f"   t={t_check:3d}d: rho={rho_t:.2f} kg/m3 | "
          f"f={f_t:.3f} | w={w_t:+6.2f} m/day | {status}")

t_sink = None
for t in range(1, 500):
    rho_t, _ = bio_ballistic_density(t, 20.3)
    if rho_t > RHO_SW:
        t_sink = t
        break
if t_sink:
    rho_ts, _ = bio_ballistic_density(t_sink, 20.3)
    w_ts = terminal_velocity_mday(rho_ts)
    print(f"\n   >>> Threshold: day {t_sink} | w = {w_ts:.2f} m/day <<<")
else:
    print("\n   WARNING: no sinking — parameters need adjustment")
    t_sink = 999

# ── Initialise particles ──────────────────────────────────────
print("\n[3/6] Initialising particles...")
rng = np.random.default_rng(42)
lat_min, lat_max = 38.0, 42.0
lon_min, lon_max =  5.0, 12.0

p_lat    = rng.uniform(lat_min, lat_max, N_PARTICLES)
p_lon    = rng.uniform(lon_min, lon_max, N_PARTICLES)
p_dep    = np.zeros(N_PARTICLES)
p_age    = rng.uniform(0, 8, N_PARTICLES)
p_active = np.ones(N_PARTICLES, dtype=bool)
p_sink_day = np.full(N_PARTICLES, np.nan)
p_sink_dep = np.full(N_PARTICLES, np.nan)
p_sink_lat = np.full(N_PARTICLES, np.nan)
p_sink_lon = np.full(N_PARTICLES, np.nan)
print(f"   {N_PARTICLES} particles | Age: 0-8 days | Min sink depth: {MIN_SINK_DEPTH}m")

def nearest_idx(arr, val):
    return int(np.argmin(np.abs(np.array(arr) - val)))

# ── Lagrangian integration ────────────────────────────────────
print(f"\n[4/6] Running Lagrangian integration ({SIM_DAYS} days)...")
n_steps = SIM_DAYS * 24
kz      = 1e-4   # m2/s vertical diffusion (surface mixed layer)
max_dep = 2500.0

for step in range(n_steps):
    day_idx = min(step // 24, 364)
    active  = np.where(p_active)[0]
    if len(active) == 0:
        break

    for i in active:
        ilat = nearest_idx(lats, p_lat[i])
        ilon = nearest_idx(lons, p_lon[i])
        idep = nearest_idx(deps, max(p_dep[i], 0.0))

        u = float(uo[day_idx, idep, ilat, ilon])
        v = float(vo[day_idx, idep, ilat, ilon])
        if np.isnan(u): u = 0.0
        if np.isnan(v): v = 0.0

        sst_local = float(sst[day_idx, ilat, ilon])
        if np.isnan(sst_local): sst_local = 20.3

        rho_p, _ = bio_ballistic_density(p_age[i], sst_local)
        w_mday   = terminal_velocity_mday(rho_p)
        w_ms     = w_mday / 86400.0

        # Stochastic vertical diffusion (Eq.7)
        dz_diff = np.sqrt(2.0 * kz * DT) * rng.standard_normal()
        ddep    = w_ms * DT + dz_diff

        dlat = (v * DT) / 111320.0
        dlon = (u * DT) / (111320.0 * np.cos(np.radians(p_lat[i])))

        p_lat[i] = float(np.clip(p_lat[i]+dlat, lat_min-3, lat_max+3))
        p_lon[i] = float(np.clip(p_lon[i]+dlon, lon_min-3, lon_max+3))
        p_dep[i] = float(np.clip(p_dep[i]+ddep, 0.0, max_dep))
        p_age[i] += DT / 86400.0

        # Sinking event
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

# ── Results ───────────────────────────────────────────────────
print("\n[5/6] Computing results...")
sunk_mask     = ~p_active
n_sunk        = int(sunk_mask.sum())
pct_sunk      = 100.0 * n_sunk / N_PARTICLES
mean_sink_day = float(np.nanmean(p_sink_day)) if n_sunk > 0 else 0.0
mean_sink_dep = float(np.nanmean(p_sink_dep)) if n_sunk > 0 else 0.0
print(f"   Particles sunk:    {n_sunk}/{N_PARTICLES} ({pct_sunk:.1f}%)")
print(f"   Mean days to sink: {mean_sink_day:.1f}")
print(f"   Mean sink depth:   {mean_sink_dep:.1f} m")

# ── Figures ───────────────────────────────────────────────────
print("\n[6/6] Generating figures...")

# Fig A: Sink point map
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(p_lon[p_active], p_lat[p_active],
           c="steelblue", s=8, alpha=0.5, label="Still floating")
if n_sunk > 0:
    sc = ax.scatter(p_sink_lon[sunk_mask], p_sink_lat[sunk_mask],
                    c=p_sink_dep[sunk_mask], cmap="plasma_r",
                    s=18, alpha=0.9, label="Sink points")
    plt.colorbar(sc, ax=ax, label="Sink depth (m)")
ax.set_xlim(lon_min-3, lon_max+3); ax.set_ylim(lat_min-3, lat_max+3)
ax.set_xlabel("Longitude (E)"); ax.set_ylabel("Latitude (N)")
ax.set_title("Bio-Ballistic Sink Points — Western Mediterranean R3")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "lagrangian_sink_points.png", dpi=150)
plt.close()
print("   Sink point map saved")

# Fig B: Sink depth histogram
fig, ax = plt.subplots(figsize=(8, 5))
valid = p_sink_dep[~np.isnan(p_sink_dep)]
if len(valid) > 0:
    ax.hist(valid, bins=30, color="navy", alpha=0.75, edgecolor="white")
ax.set_xlabel("Sink Depth (m)"); ax.set_ylabel("Count")
ax.set_title("Plastic Sink Depth Distribution — R3 Mediterranean")
plt.tight_layout()
plt.savefig(FIG_DIR / "sink_depth_distribution.png", dpi=150)
plt.close()
print("   Sink depth histogram saved")

# Fig C: Density + velocity curves
t_arr   = np.linspace(0, 120, 300)
rho_arr = [bio_ballistic_density(t, 20.3)[0] for t in t_arr]
w_arr   = [terminal_velocity_mday(r) for r in rho_arr]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
ax1.plot(t_arr, rho_arr, "b-", lw=2.5, label=r"$\rho_p(t)$")
ax1.axhline(RHO_SW, color="red", ls="--", lw=1.8,
            label=f"Seawater {RHO_SW} kg/m³")
ax1.fill_between(t_arr, rho_arr, RHO_SW,
                 where=[r > RHO_SW for r in rho_arr],
                 alpha=0.15, color="red", label="Sinking zone")
ax1.set_ylabel("Density (kg/m³)")
ax1.set_title(r"Bio-Ballistic $\rho_p(t)$ — Eq.2 Manuscript")
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

# ── Save summary ──────────────────────────────────────────────
summary = {
    "timestamp":         datetime.now().isoformat(),
    "n_particles":       N_PARTICLES,
    "sim_days":          SIM_DAYS,
    "n_sunk":            n_sunk,
    "pct_sunk":          round(pct_sunk, 2),
    "mean_days_to_sink": round(mean_sink_day, 1),
    "mean_sink_depth_m": round(mean_sink_dep, 1),
    "days_to_threshold": t_sink,
    "w_max_cap_mday":    W_MAX_MDAY,
    "min_sink_depth_m":  MIN_SINK_DEPTH,
    "region":            "R3_Western_Mediterranean",
    "physics_ref":       "Kooi_2017 + Cozar_2014"
}
with open(OUT_DIR / "phase3_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nPhase 3 Summary:")
for k, v in summary.items():
    print(f"   {k}: {v}")
print("\n✅ Phase 3 complete.")
print(f"   Results → {OUT_DIR}")
print(f"   Figures → {FIG_DIR}")
print("="*60)