"""
DEEP-SEA PLASTIC PULSE — Figure 2, Panel A
W1–W4 Spectral Feature Space
King Jones Adega | Tianjin University | Nature Geoscience

RUN:  python Fig2a_spectral_feature_space.py
OUT:  Fig2a_spectral_feature_space.png / .svg / .tiff  →  OUTPUT_DIR
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
import numpy as np
import os

# ── USER SETTINGS ─────────────────────────────────────────────────────────────
OUTPUT_DIR = r"C:\Users\Apple\deep_sea_pulse\outputs\figures\publication"

COL = {
    "W1": "#2196A6",   # teal-blue
    "W2": "#3DAA6B",   # green
    "W3": "#D95F2B",   # orange-red
    "W4": "#E8A020",   # amber-orange
}
# ── END USER SETTINGS ─────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Training data — exact parameters from 01_train_classifier.py ──────────────
rng = np.random.default_rng(42)
params = {
    "W1": (0.08,  0.010, -0.05,  0.010),
    "W2": (0.06,  0.015,  0.02,  0.010),
    "W3": (0.04,  0.015,  0.08,  0.015),
    "W4": (0.02,  0.010,  0.15,  0.020),
}
SCALE = 100.0
data = {}
for label, (fm, fs, sm, ss) in params.items():
    fdi = rng.normal(fm * SCALE, fs * SCALE, 1250)
    swi = rng.normal(sm * SCALE + 7.0, ss * SCALE, 1250)
    data[label] = (fdi, swi)

# ── Helper: clean axes — no grid, clean spines ────────────────────────────────
def clean_ax(ax, bg="#F5F5F0"):
    ax.set_facecolor(bg)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.tick_params(colors="#555555", labelsize=7,
                   length=3, width=0.8, direction="out")

# ── KDE ridge plot ────────────────────────────────────────────────────────────
def ridge_plot(ax, fdi_data, col, xlabel="FDI (Floating Debris Index)"):
    kde   = gaussian_kde(fdi_data, bw_method=0.25)
    x_min = max(0, fdi_data.min() - 0.5)
    x_max = fdi_data.max() + 0.5
    x     = np.linspace(x_min, x_max, 400)
    y     = kde(x)
    y_top = y.max() * 1.18

    ax.fill_between(x, y, alpha=0.80, color=col, zorder=2)
    ax.plot(x, y, color=col, linewidth=1.5, zorder=3)

    # Arrow axes
    ax.annotate("", xy=(x_max + 0.3, 0), xytext=(x_min, 0),
                arrowprops=dict(arrowstyle="-|>",
                                color="#555555", lw=1.0))
    ax.annotate("", xy=(x_min, y_top),
                xytext=(x_min, 0),
                arrowprops=dict(arrowstyle="-|>",
                                color="#555555", lw=1.0))

    clean_ax(ax)
    ax.set_xlim(x_min - 0.2, x_max + 0.5)
    ax.set_ylim(0, y_top)
    ax.set_xlabel(xlabel, fontsize=7.5, color="#333333",
                  labelpad=3, fontstyle="italic")
    ax.set_ylabel("Density", fontsize=7, color="#333333", labelpad=3)
    ax.set_title("Marginal Ridge Plot", fontsize=8,
                 fontweight="bold", color="#222222",
                 pad=3, loc="left")
    y_max_val = round(y.max(), 1)
    ax.set_yticks([0, round(y_max_val/2, 1), y_max_val])
    ax.set_yticklabels(["0",
                         str(round(y_max_val/2, 1)),
                         str(y_max_val)],
                        fontsize=6.5)

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(130/25.4, 190/25.4), facecolor="#F5F5F0")
from matplotlib.gridspec import GridSpec
gs = GridSpec(3, 2, figure=fig,
              height_ratios=[1.0, 1.0, 1.8],
              hspace=0.42, wspace=0.30,
              left=0.10, right=0.96,
              top=0.93, bottom=0.07)

ax_w2  = fig.add_subplot(gs[0, 0])
ax_w4  = fig.add_subplot(gs[0, 1])
ax_w1  = fig.add_subplot(gs[1, 0])
ax_w3  = fig.add_subplot(gs[1, 1])
ax_jnt = fig.add_subplot(gs[2, :])

# ── Four marginal ridge plots ─────────────────────────────────────────────────
ridge_plot(ax_w2, data["W2"][0], COL["W2"])
ridge_plot(ax_w4, data["W4"][0], COL["W4"])
ridge_plot(ax_w3, data["W3"][0], COL["W3"])

# W1 with W2 overlap (matches reference mid-left panel)
fdi_w1 = data["W1"][0]
fdi_w2 = data["W2"][0]
kde_w1_kde = gaussian_kde(fdi_w1, bw_method=0.25)
kde_w2_kde = gaussian_kde(fdi_w2, bw_method=0.25)
x_range = np.linspace(
    max(0, min(fdi_w1.min(), fdi_w2.min()) - 0.3),
    max(fdi_w1.max(), fdi_w2.max()) + 0.3, 400)
y_w1k = kde_w1_kde(x_range)
y_w2k = kde_w2_kde(x_range)
y_top_w1 = max(y_w1k.max(), y_w2k.max()) * 1.18

ax_w1.fill_between(x_range, y_w1k, alpha=0.75,
                    color=COL["W1"], zorder=2)
ax_w1.fill_between(x_range, y_w2k, alpha=0.45,
                    color=COL["W2"], zorder=3)
ax_w1.plot(x_range, y_w1k, color=COL["W1"],
            linewidth=1.4, zorder=4)
ax_w1.plot(x_range, y_w2k, color=COL["W2"],
            linewidth=1.4, zorder=5)
ax_w1.annotate("", xy=(x_range[-1]+0.3, 0),
               xytext=(x_range[0], 0),
               arrowprops=dict(arrowstyle="-|>",
                               color="#555555", lw=1.0))
ax_w1.annotate("", xy=(x_range[0], y_top_w1),
               xytext=(x_range[0], 0),
               arrowprops=dict(arrowstyle="-|>",
                               color="#555555", lw=1.0))
clean_ax(ax_w1)
ax_w1.set_xlim(x_range[0]-0.2, x_range[-1]+0.5)
ax_w1.set_ylim(0, y_top_w1)
ax_w1.set_xlabel("FDI (Floating Debris Index)",
                  fontsize=7.5, color="#333333",
                  labelpad=3, fontstyle="italic")
ax_w1.set_ylabel("Density", fontsize=7,
                  color="#333333", labelpad=3)
ax_w1.set_title("Marginal Ridge Plot", fontsize=8,
                 fontweight="bold", color="#222222",
                 pad=3, loc="left")

# ── Joint scatter + KDE contours ─────────────────────────────────────────────
clean_ax(ax_jnt, bg="#F5F5F0")

all_fdi = np.concatenate([data[l][0] for l in data])
all_swi = np.concatenate([data[l][1] for l in data])
x_min_j = max(0, all_fdi.min() - 0.5)
x_max_j = all_fdi.max() + 0.5
y_min_j = all_swi.min() - 0.5
y_max_j = all_swi.max() + 0.5

xx, yy = np.mgrid[x_min_j:x_max_j:200j,
                   y_min_j:y_max_j:200j]
positions = np.vstack([xx.ravel(), yy.ravel()])
kde_all = gaussian_kde(np.vstack([all_fdi, all_swi]),
                       bw_method=0.18)
zz = kde_all(positions).reshape(xx.shape)

ax_jnt.contourf(xx, yy, zz, levels=8,
                cmap="Greens", alpha=0.45, zorder=1)
ax_jnt.contour(xx, yy, zz, levels=8,
               colors="#2A8060", linewidths=0.7,
               alpha=0.60, zorder=2)

for lbl, ms in [("W4", 18), ("W3", 18), ("W1", 22)]:
    fdi, swi = data[lbl]
    ax_jnt.scatter(fdi, swi, color=COL[lbl],
                   s=ms, alpha=0.65, linewidths=0,
                   zorder=4, rasterized=True)

ax_jnt.set_xlabel("FDI (Floating Debris Index)",
                   fontsize=10, color="#222222",
                   labelpad=6, fontstyle="italic")
ax_jnt.set_ylabel("SWI (Spectral Weathering Index)",
                   fontsize=10, color="#222222",
                   labelpad=6, fontstyle="italic")
ax_jnt.set_xlim(x_min_j, x_max_j)
ax_jnt.set_ylim(y_min_j, y_max_j)
ax_jnt.tick_params(colors="#555555", labelsize=8,
                    length=4, width=0.9)

legend_els = [
    mpatches.Patch(facecolor=COL[l], edgecolor="none", label=l)
    for l in ["W1","W2","W3","W4"]
]
leg = ax_jnt.legend(handles=legend_els,
                     loc="upper right", fontsize=8.5,
                     framealpha=0.88, facecolor="white",
                     edgecolor="#AAAAAA", borderpad=0.6,
                     handlelength=1.0, labelspacing=0.35)
for t in leg.get_texts():
    t.set_color("#222222")
    t.set_fontweight("bold")

# ── CORRECTED TITLE — no plot type name, just what it represents ──────────────
fig.text(0.50, 0.965,
         "W1–W4 Spectral Feature Space",
         ha="center", va="top",
         fontsize=12, fontweight="bold",
         color="#1A1A2E")

# ── Save ──────────────────────────────────────────────────────────────────────
base = os.path.join(OUTPUT_DIR, "Fig2a_spectral_feature_space")
fig.savefig(base+".png",  dpi=300, bbox_inches="tight",
            facecolor="#F5F5F0")
fig.savefig(base+".svg",  format="svg",
            bbox_inches="tight", facecolor="#F5F5F0")
fig.savefig(base+".tiff", dpi=600, bbox_inches="tight",
            facecolor="#F5F5F0",
            pil_kwargs={"compression":"tiff_lzw"})
print(f"Saved → {OUTPUT_DIR}")
plt.close(fig)
