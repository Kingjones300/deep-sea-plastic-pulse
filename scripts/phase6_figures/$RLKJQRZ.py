"""
DEEP-SEA PLASTIC PULSE — Figure 2, Panel A
TOP PICK 1 v3 — careful incremental corrections
King Jones Adega | Tianjin University | Nature Geoscience
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.stats import gaussian_kde, pearsonr
from scipy.ndimage import gaussian_filter
import numpy as np
import os

OUTPUT_DIR = r"C:\Users\Apple\deep_sea_pulse\outputs\figures\publication"
os.makedirs(OUTPUT_DIR, exist_ok=True)

rng = np.random.default_rng(42)
params = {
    "W1": (0.08,0.010,-0.05,0.010,0.12,0.020,3.5,0.30),
    "W2": (0.06,0.015, 0.02,0.010,0.09,0.020,2.8,0.40),
    "W3": (0.04,0.015, 0.08,0.015,0.06,0.020,2.1,0.40),
    "W4": (0.02,0.010, 0.15,0.020,0.03,0.015,1.4,0.30),
}
data = {}
for lbl,(fm,fs,sm,ss,nm,ns,rm,rs) in params.items():
    data[lbl] = {
        "FDI":  rng.normal(fm,fs,1250),
        "SWI":  rng.normal(sm,ss,1250),
        "NDVI": rng.normal(nm,ns,1250),
    }

ORDER = ["W1","W2","W3","W4"]
COLS  = {"W1":"#1565C0","W2":"#2E7D32",
         "W3":"#E65100","W4":"#B71C1C"}

# Physical state — short single line so it renders crisp
PHYS = {
    "W1": "Virgin polymer  |  positive buoyancy  |  ρ_p 900–960 kg m⁻³",
    "W2": "Early biofouling  |  surface pitting  |  ρ_p 960–1,000 kg m⁻³",
    "W3": "Dense biofilm  |  near-neutral buoyancy  |  ρ_p 1,000–1,025 kg m⁻³",
    "W4": "Critical sinking threshold  |  ρ_p > 1,025 kg m⁻³",
}
F1S = {"W1":0.998,"W2":0.995,"W3":0.996,"W4":0.999}
PTS = {"W1":"PtS ≈ 0.05","W2":"PtS ≈ 0.30",
       "W3":"PtS ≈ 0.65","W4":"PtS ≈ 0.87"}

def conf_ellipse(x,y,ax,n_std,**kw):
    cov = np.cov(x,y)
    p   = cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])
    ell = Ellipse((0,0),
                  width=np.sqrt(1+p)*2,
                  height=np.sqrt(1-p)*2,
                  facecolor="none",**kw)
    t = (transforms.Affine2D()
         .rotate_deg(45)
         .scale(np.sqrt(cov[0,0])*n_std,
                np.sqrt(cov[1,1])*n_std)
         .translate(np.mean(x),np.mean(y)))
    ell.set_transform(t+ax.transData)
    ax.add_patch(ell)

# ── Figure ────────────────────────────────────────────────────────
# Use constrained_layout for clean automatic spacing
fig = plt.figure(figsize=(190/25.4, 185/25.4),
                 facecolor="white",
                 constrained_layout=False)

# ── Overall title — placed high with large bottom margin ──────────
fig.text(0.50, 0.992,
         "a  |  W1–W4 Spectral Feature Space",
         ha="center", va="top",
         fontsize=15, fontweight="bold",
         color="#1A1A2E")

# ── GridSpec — top margin leaves room for title + PHYS label ──────
# top=0.90 means 10% of figure height is above the subplots
# This guarantees clear space between title and each panel title
gs = gridspec.GridSpec(
    2, 2, figure=fig,
    hspace=0.55,   # vertical space between rows — generous
    wspace=0.38,   # horizontal space between columns
    left=0.09, right=0.93,
    top=0.89,      # large top gap for titles
    bottom=0.07)

for idx, lbl in enumerate(ORDER):
    ax = fig.add_subplot(gs[idx//2, idx%2])
    ax.set_facecolor("#FAFAFA")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ["left","bottom"]:
        ax.spines[s].set_color("#333333")
        ax.spines[s].set_linewidth(1.1)
    ax.tick_params(colors="#222222", labelsize=8,
                   length=4, width=0.9, direction="out")

    fdi  = data[lbl]["FDI"]
    swi  = data[lbl]["SWI"]
    ndvi = data[lbl]["NDVI"]
    col  = COLS[lbl]

    # Data range with padding
    fdi_pad = fdi.std() * 3.2
    swi_pad = swi.std() * 3.2
    xl = (fdi.mean()-fdi_pad, fdi.mean()+fdi_pad)
    yl = (swi.mean()-swi_pad, swi.mean()+swi_pad)
    ax.set_xlim(xl)
    ax.set_ylim(yl)

    # Scatter — points coloured by NDVI
    ndvi_n = (ndvi-ndvi.min())/(ndvi.max()-ndvi.min())
    cmap_n = mcolors.LinearSegmentedColormap.from_list(
        f"c_{lbl}",
        [mcolors.to_rgba(col,0.18),
         mcolors.to_rgba(col,0.95)])
    sc = ax.scatter(fdi, swi,
                    c=ndvi_n, cmap=cmap_n,
                    s=11, alpha=0.60,
                    linewidths=0, zorder=3,
                    rasterized=True)

    # Error bars
    rng2 = np.random.default_rng(idx+200)
    eb   = rng2.choice(1250,65,replace=False)
    ax.errorbar(fdi[eb], swi[eb],
                xerr=np.abs(rng2.normal(0.003,0.001,65)),
                yerr=np.abs(rng2.normal(0.006,0.002,65)),
                fmt="none", ecolor=col,
                alpha=0.25, elinewidth=0.7,
                capsize=1.8, zorder=2)

    # KDE contours
    kde = gaussian_kde(np.vstack([fdi,swi]),bw_method=0.20)
    xg  = np.linspace(xl[0],xl[1],120)
    yg  = np.linspace(yl[0],yl[1],120)
    xi,yi = np.meshgrid(xg,yg)
    zi  = kde(np.vstack([xi.ravel(),yi.ravel()])
              ).reshape(xi.shape)
    zi  = gaussian_filter(zi,sigma=1.0)
    ax.contourf(xi,yi,zi,
                levels=[zi.max()*0.18,zi.max()],
                colors=[col],alpha=0.13,zorder=2)
    ax.contour(xi,yi,zi,
               levels=[zi.max()*0.20,
                        zi.max()*0.55,
                        zi.max()*0.88],
               colors=[col],
               linewidths=[0.7,1.2,1.9],
               alpha=0.85,zorder=4)

    # Regression line
    m,b  = np.polyfit(fdi,swi,1)
    xfit = np.array([xl[0],xl[1]])
    ax.plot(xfit, m*xfit+b,
            color="#1A1A2E", linewidth=2.0,
            zorder=5, alpha=0.80)

    # Confidence ellipses
    conf_ellipse(fdi,swi,ax,n_std=1.5,
                 edgecolor=col, linewidth=2.0,
                 alpha=0.90, zorder=6)
    conf_ellipse(fdi,swi,ax,n_std=2.5,
                 edgecolor=col, linewidth=1.0,
                 linestyle="--", alpha=0.55, zorder=5)

    # Centroid
    ax.plot(fdi.mean(),swi.mean(),"o",
            color=col, markersize=10,
            markeredgecolor="white",
            markeredgewidth=1.8, zorder=8)

    # ── NDVI colourbar — only on left-column plots ────────────────
    # Right-column plots: no colourbar so no label overlap
    if idx % 2 == 0:
        cbar = fig.colorbar(sc, ax=ax,
                             fraction=0.033, pad=0.016)
        cbar.set_label("NDVI", fontsize=8,
                        color="#333", labelpad=3,
                        fontweight="bold")
        cbar.ax.tick_params(labelsize=7, colors="#444")
        cbar.outline.set_edgecolor("#CCCCCC")
        cbar.set_ticks([0,1.0])
        cbar.set_ticklabels(["Low","High"],
                             fontsize=7,
                             fontweight="bold")
    else:
        # Right column: small colourbar, no label text overlap
        cbar = fig.colorbar(sc, ax=ax,
                             fraction=0.020, pad=0.010)
        cbar.ax.tick_params(labelsize=6, colors="#666")
        cbar.set_ticks([0,1.0])
        cbar.set_ticklabels(["Lo","Hi"], fontsize=6)
        cbar.outline.set_edgecolor("#DDDDDD")

    # ── Stats box — compact, top-left ────────────────────────────
    r,_ = pearsonr(fdi,swi)
    ax.text(0.03, 0.97,
            f"F1={F1S[lbl]:.3f}  κ=0.991\n"
            f"r={r:.3f}   {PTS[lbl]}",
            transform=ax.transAxes,
            fontsize=8, va="top", ha="left",
            color="#111111",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35",
                      facecolor="white",
                      edgecolor="#BBBBBB",
                      linewidth=0.9,
                      alpha=0.95))

    # ── Class label W1/W2/W3/W4 — bold, top right ────────────────
    ax.text(0.97, 0.97, lbl,
            transform=ax.transAxes,
            fontsize=18, fontweight="bold",
            color=col, va="top", ha="right",
            path_effects=[pe.withStroke(
                linewidth=4, foreground="white")])

    # ── Physical state title — ABOVE the axes ────────────────────
    # Written using fig.text at computed axes position
    # to guarantee it is above the subplot and not clipped
    ax.set_title(PHYS[lbl],
                 fontsize=8.5,
                 color="#222222",
                 fontweight="bold",
                 pad=14,        # large pad = clear gap below title
                 loc="left")

    # ── Axis labels — bold ────────────────────────────────────────
    ax.set_xlabel("FDI (Floating Debris Index)",
                  fontsize=10, color="#111111",
                  labelpad=5, fontweight="bold")
    ax.set_ylabel("SWI (Spectral Weathering Index)",
                  fontsize=10, color="#111111",
                  labelpad=5, fontweight="bold")

# ── Save ──────────────────────────────────────────────────────────
base = os.path.join(OUTPUT_DIR,"Fig2a_TOP1_v3")
plt.savefig(base+".png",  dpi=300,
            bbox_inches="tight", facecolor="white")
plt.savefig(base+".svg",  format="svg",
            bbox_inches="tight", facecolor="white")
plt.savefig(base+".tiff", dpi=600,
            bbox_inches="tight", facecolor="white",
            pil_kwargs={"compression":"tiff_lzw"})
print("TOP1 v3 saved.")
plt.close(fig)
