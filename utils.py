"""
utils.py — Coastal Sentinel Shared Utilities
Physics constants, region definitions, Bio-Ballistic equations.
"""

import os
import sys
from pathlib import Path
import numpy as np
from loguru import logger
from dotenv import load_dotenv

# ── Load config ──────────────────────────────────────────────────────────────
def load_config():
    env_path = Path(__file__).parent / "config" / "settings.env"
    if not env_path.exists():
        logger.error(f"settings.env not found at {env_path}")
        sys.exit(1)
    load_dotenv(env_path, override=True)
    return dict(os.environ)

# ── Logging setup ────────────────────────────────────────────────────────────
def setup_logging(script_name):
    Path("logs").mkdir(exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        level="INFO")
    logger.add(f"logs/{script_name}.log", rotation="50 MB", level="DEBUG")

# ── Directory helper ─────────────────────────────────────────────────────────
def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

# ── Physical constants ───────────────────────────────────────────────────────
G_GRAVITY           = 9.81
RHO_SW_REF          = 1025.0
WADELL_PSI          = 0.6
RHO_BIOFILM         = 1388.0
BIOFILM_THICKNESS_INIT = 1e-7
BBM_MU_MAX          = 0.59
BBM_K_S             = 0.17
BBM_K_RESP          = 0.10
BBM_K_DET           = 0.05
BBM_EPPLEY_COEF     = 0.0633# ── Study regions ────────────────────────────────────────────────────────────
REGIONS = {
    "R1": {
        "code": "R1",
        "name": "Malacca_Strait",
        "lon_min": 98.0,  "lon_max": 109.0,
        "lat_min": 1.0,   "lat_max": 10.0,
        "description": "Strait of Malacca - high flux tropical source",
    },
    "R2": {
        "code": "R2",
        "name": "NP_Gyre",
        "lon_min": -160.0, "lon_max": -140.0,
        "lat_min": 25.0,   "lat_max": 35.0,
        "description": "North Pacific Gyre - slow sink accumulation zone",
    },
    "R3": {
        "code": "R3",
        "name": "W_Mediterranean",
        "lon_min": 0.0,  "lon_max": 15.0,
        "lat_min": 36.0, "lat_max": 43.0,
        "description": "Western Mediterranean - validation gold standard",
    },
}

# ── Polymer densities ────────────────────────────────────────────────────────
POLYMERS = {
    "LDPE": {"name": "Low-Density Polyethylene", "density": 917.0},
    "HDPE": {"name": "High-Density Polyethylene", "density": 952.0},
    "PP":   {"name": "Polypropylene",             "density": 905.0},
    "PET":  {"name": "Polyethylene Terephthalate","density": 1380.0},
    "PS":   {"name": "Polystyrene",               "density": 1050.0},
}

# ── Bio-Ballistic density function (Equation 2-4) ───────────────────────────
def bio_ballistic_density(rho_polymer, diameter_m, t_days, sst_C, chl_a_mgm3):
    """
    Compute composite particle density rho_p(t).
    Returns effective density in kg/m3.
    """
    mu = (BBM_MU_MAX
          * np.exp(BBM_EPPLEY_COEF * sst_C)
          * chl_a_mgm3 / (BBM_K_S + chl_a_mgm3))
    mu_net = mu - BBM_K_RESP - BBM_K_DET
    r_poly = diameter_m / 2.0
    delta_bio = BIOFILM_THICKNESS_INIT * np.exp(mu_net * t_days)
    delta_bio = min(delta_bio, r_poly * 0.30)
    r_comp = r_poly + delta_bio
    V_poly = (4/3) * np.pi * r_poly**3
    V_bio  = (4/3) * np.pi * r_comp**3 - V_poly
    m_poly = rho_polymer * V_poly
    m_bio  = RHO_BIOFILM * V_bio
    return (m_poly + m_bio) / (V_poly + V_bio)

# ── Dynamic viscosity of seawater ────────────────────────────────────────────
def dynamic_viscosity_seawater(T_celsius):
    T_K = T_celsius + 273.15
    return 2.414e-5 * 10 ** (247.8 / (T_K - 140))

# ── Terminal settling velocity (Equation 5) ──────────────────────────────────
def terminal_velocity(rho_particle, rho_seawater, diameter_m, T_celsius):
    """
    Modified Stokes settling velocity (m/s).
    Negative = sinking.
    """
    mu = dynamic_viscosity_seawater(T_celsius)
    delta_rho = rho_particle - rho_seawater
    w = (delta_rho * G_GRAVITY * diameter_m**2) / (18 * mu * WADELL_PSI)
    return -w

# ── PtS score to weathering class ────────────────────────────────────────────
def pts_score_to_class(pts):
    if pts < 0.15: return "W1"
    if pts < 0.45: return "W2"
    if pts < 0.80: return "W3"
    return "W4"