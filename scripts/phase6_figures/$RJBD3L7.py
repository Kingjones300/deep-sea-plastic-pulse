"""
DEEP-SEA PLASTIC PULSE — Figure 2, Panel A
TOP PICK 2 v3 — legend shrunk and repositioned only
King Jones Adega | Tianjin University | Nature Geoscience
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.stats import gaussian_kde
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
    }

ORDER = ["W1","W2","W3","W4"]
COLS  = {"W1":"#1565C0","W2":"#2E7D32",
         "W3":"#E65100","W4":"#B71C1C"}
PHYS  = {"W1":"Virgin polymer",
         "W2":"Early biofouling",
         "W3":"Dense biofilm",
         "W4":"Sinking threshold"}
F1S   = {"W1":0.998,"W2":0.995,"W3":0.996,"W4":0.999}

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

fig, ax = plt.subplots(figsize=(148/25.4,142/25.4),
                        facecolor="white")
ax.set_facecolor("#F8F9FA")
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for s in ["left","bottom"]:
    ax.spines[s].set_color("#333333")
    ax.spines[s].set_linewidth(1.0)
ax.tick_params(colors="#222",labelsize=9,
               length=4,width=0.9,direction="out")

# Physics background
x_g = np.linspace(-0.038,0.138,500)
y_g = np.linspace(-0.118,0.228,500)
xx,yy = np.meshgrid(x_g,y_g)
rho_proxy = 900 + yy*5800 - xx*3800
rho_norm  = np.clip((rho_proxy-880)/(1100-880),0,1)
bg_cmap   = mcolors.LinearSegmentedColormap.from_list(
    "rho_bg",
    ["#EBF5FB","#AED6F1","#FDEBD0",
     "#FAD7A0","#E74C3C","#922B21"],N=256)
im = ax.imshow(rho_norm,
               extent=[-0.038,0.138,-0.118,0.228],
               origin="lower",aspect="auto",
               cmap=bg_cmap,alpha=0.30,zorder=1)

# Isolines
iso_levels = [930,960,990,1010,1020,1025]
cs = ax.contour(xx,yy,rho_proxy,
                levels=iso_levels,
                colors="#888888",
                linewidths=0.55,alpha=0.35,
                zorder=2,linestyles=":")
ax.clabel(cs,fmt={v:str(v) for v in iso_levels},
          fontsize=5.5,colors="#888888",
          inline=True,inline_spacing=2)

# Phase boundaries
for xb in [0.063,0.043,0.022]:
    ax.axvline(xb,color="#666666",linewidth=1.0,
               linestyle="--",zorder=3,alpha=0.55)

# PtS threshold
ax.axhline(0.078,color="#B71C1C",linewidth=2.8,
           linestyle="-",zorder=7,alpha=0.92)
ax.text(0.001,0.085,"PtS = 0.70  (sinking threshold)",
        fontsize=8,color="#B71C1C",
        ha="left",va="bottom",fontweight="bold",
        path_effects=[pe.withStroke(
            linewidth=2.5,foreground="white")])

# Class contours + ellipses
for lbl in ORDER:
    fdi = data[lbl]["FDI"]
    swi = data[lbl]["SWI"]
    col = COLS[lbl]
    kde  = gaussian_kde(np.vstack([fdi,swi]),bw_method=0.18)
    zz2  = kde(np.vstack([xx.ravel(),yy.ravel()])
               ).reshape(xx.shape)
    zz2  = gaussian_filter(zz2,sigma=1.2)
    ax.contourf(xx,yy,zz2,
                levels=[zz2.max()*0.10,zz2.max()*0.40],
                colors=[col],alpha=0.15,zorder=3)
    ax.contourf(xx,yy,zz2,
                levels=[zz2.max()*0.40,zz2.max()],
                colors=[col],alpha=0.52,zorder=4)
    ax.contour(xx,yy,zz2,
               levels=[zz2.max()*0.12,
                        zz2.max()*0.45,
                        zz2.max()*0.82],
               colors=[col],
               linewidths=[0.6,1.1,2.0],
               alpha=0.90,zorder=5)
    rng2 = np.random.default_rng(ORDER.index(lbl)+30)
    eb   = rng2.choice(1250,55,replace=False)
    ax.errorbar(fdi[eb],swi[eb],
                xerr=np.abs(rng2.normal(0.003,0.001,55)),
                yerr=np.abs(rng2.normal(0.006,0.002,55)),
                fmt="none",ecolor=col,alpha=0.22,
                elinewidth=0.65,capsize=1.5,zorder=3)
    conf_ellipse(fdi,swi,ax,n_std=1.5,
                 edgecolor=col,linewidth=2.0,
                 alpha=0.90,zorder=6)
    conf_ellipse(fdi,swi,ax,n_std=2.5,
                 edgecolor=col,linewidth=0.9,
                 linestyle="--",alpha=0.50,zorder=5)
    cx,cy = fdi.mean(),swi.mean()
    ax.plot(cx,cy,"o",color=col,markersize=12,
            markeredgecolor="white",
            markeredgewidth=2.0,zorder=9)
    ax.text(cx,cy,f"{lbl}\nF1={F1S[lbl]:.3f}",
            fontsize=7,fontweight="bold",
            color="white",ha="center",va="center",
            zorder=10,
            path_effects=[pe.withStroke(
                linewidth=4.0,foreground=col)])

# Trajectory arrows
c_all = {l:(data[l]["FDI"].mean(),
             data[l]["SWI"].mean()) for l in ORDER}
for i in range(len(ORDER)-1):
    p0 = c_all[ORDER[i]]
    p1 = c_all[ORDER[i+1]]
    ax.annotate("",xy=p1,xytext=p0,
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#1A1A2E",lw=2.8,
                    mutation_scale=16,
                    connectionstyle="arc3,rad=0.20"),
                zorder=11)
ax.text(0.052,0.060,
        "← Weathering progression →",
        fontsize=8,color="#1A1A2E",
        fontweight="bold",rotation=50,
        path_effects=[pe.withStroke(
            linewidth=3.5,foreground="white")],
        zorder=12)

# Zone labels — top of each zone, clear of data
zones = [
    (0.095, 0.210, "ZONE I",  "Virgin polymer",      "#1565C0"),
    (0.053, 0.210, "ZONE II", "Early biofouling",    "#2E7D32"),
    (0.032, 0.210, "ZONE III","Near-neutral",         "#E65100"),
    (0.010, 0.210, "ZONE IV", "Sinking",             "#B71C1C"),
]
for xz,yz,zn,desc,col in zones:
    ax.text(xz,yz+0.005,zn,
            fontsize=7.5,color=col,ha="center",
            va="bottom",fontweight="bold",
            path_effects=[pe.withStroke(
                linewidth=2.5,foreground="white")])
    ax.text(xz,yz-0.003,desc,
            fontsize=6.5,color=col,ha="center",
            va="top",fontstyle="italic",
            path_effects=[pe.withStroke(
                linewidth=2,foreground="white")])

# Colourbar
cbar = fig.colorbar(im,ax=ax,fraction=0.030,pad=0.020)
cbar.set_label("Modelled ρ_p proxy (kg m⁻³)",
               fontsize=7,color="#333",labelpad=4)
cbar.ax.tick_params(labelsize=7,colors="#555")
cbar.set_ticks([0,0.5,1.0])
cbar.set_ticklabels(["880","990","≥1,100"],fontsize=7)
cbar.outline.set_edgecolor("#BBBBBB")

ax.set_xlabel("FDI (Floating Debris Index)",
              fontsize=11,color="#111",labelpad=7,
              fontweight="bold")
ax.set_ylabel("SWI (Spectral Weathering Index)",
              fontsize=11,color="#111",labelpad=7,
              fontweight="bold")
ax.set_xlim(-0.038,0.138)
ax.set_ylim(-0.118,0.228)

# ── LEGEND — small, compact, bottom-right ────────────────────────
# Positioned bottom-right where there is clear empty space
# Short label text, small font, no border box that is too large
leg = [mpatches.Patch(facecolor=COLS[l],
                       edgecolor="none",
                       label=f"{l} — {PHYS[l]}")
       for l in ORDER]
ax.legend(handles=leg,
          fontsize=7.5,
          loc="lower right",
          bbox_to_anchor=(0.99, 0.01),
          framealpha=0.93,
          facecolor="white",
          edgecolor="#CCCCCC",
          borderpad=0.45,
          handlelength=0.9,
          handleheight=0.7,
          handletextpad=0.4,
          labelspacing=0.28,
          borderaxespad=0.3)

ax.set_title("a  |  W1–W4 Spectral Feature Space",
             fontsize=13,fontweight="bold",
             color="#1A1A2E",pad=12,loc="left")

ax.text(0.02,0.02,
        "Background: ρ_p(FDI,SWI)  ·  "
        "Red line: PtS=0.70  ·  "
        "Arrows: W1→W4 trajectory",
        transform=ax.transAxes,
        fontsize=6,ha="left",va="bottom",
        color="#666666",fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.30",
                  facecolor="white",
                  edgecolor="#DDDDDD",
                  alpha=0.90))

plt.tight_layout(pad=0.9)
base = os.path.join(OUTPUT_DIR,"Fig2a_TOP2_v3")
plt.savefig(base+".png",dpi=300,
            bbox_inches="tight",facecolor="white")
plt.savefig(base+".svg",format="svg",
            bbox_inches="tight",facecolor="white")
plt.savefig(base+".tiff",dpi=600,
            bbox_inches="tight",facecolor="white",
            pil_kwargs={"compression":"tiff_lzw"})
print("TOP2 v3 saved.")
plt.close(fig)
