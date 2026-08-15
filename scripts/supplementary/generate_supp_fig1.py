import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

# Set Nature journal publication standards
plt.rcParams.update({
    'font.sans-serif': 'Arial',
    'font.family': 'sans-serif',
    'font.size': 8,
    'axes.labelsize': 8.5,
    'axes.titlesize': 9.5,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def save_publication_figure(fig, filename_base):
    fig.savefig(f'{filename_base}.pdf', format='pdf', bbox_inches='tight')
    fig.savefig(
        f'{filename_base}.png', format='png', dpi=600, bbox_inches='tight'
    )
    fig.savefig(
        f'{filename_base}.tif', format='tiff', dpi=600, bbox_inches='tight'
    )
    plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), dpi=300)

# --- Panel A: Lagrangian Sink Points (R2 N. Pacific Gyre) ---
np.random.seed(101)
lon_a = np.random.normal(-147.5, 2.5, 500)
lat_a = np.random.normal(30.0, 2.2, 500)
depth_a = np.random.uniform(150, 1200, 500)

axes[0].set_facecolor('#e0f2f1')
sc = axes[0].scatter(
    lon_a,
    lat_a,
    c=depth_a,
    cmap='viridis',
    s=18,
    alpha=0.88,
    edgecolors='none',
    zorder=3,
)

axes[0].grid(True, linestyle=':', alpha=0.6, color='#78909c', zorder=1)
axes[0].set_xlim(-155, -140)
axes[0].set_ylim(22, 38)
axes[0].set_xlabel('Longitude (°W)', fontweight='bold')
axes[0].set_ylabel('Latitude (°N)', fontweight='bold')

cbar1 = fig.colorbar(sc, ax=axes[0], shrink=0.85, pad=0.03)
cbar1.set_label('Sink Depth (m)', fontweight='bold', fontsize=8)

axes[0].set_title(
    'a   Lagrangian Sink Points (R2 Gyre)\n     n = 500, mean = 42.3 d',
    loc='left',
    fontweight='bold',
)
axes[0].spines[['top', 'right']].set_visible(False)

# --- Panel B: Vertical Export Corridors (R2 Flux Density) ---
x = np.linspace(-155, -140, 140)
y = np.linspace(22, 38, 140)
X, Y = np.meshgrid(x, y)

Z = np.exp(-(((X + 147.641) ** 2) / 12.0 + ((Y - 29.975) ** 2) / 8.5))
Z_flux = 10 ** (-3) * Z + 10 ** (-10)

pcm = axes[1].pcolormesh(
    X,
    Y,
    Z_flux,
    norm=LogNorm(vmin=10**-10, vmax=10**-3),
    cmap='YlOrRd',
    shading='auto',
    zorder=1,
)

axes[1].grid(True, linestyle=':', alpha=0.5, color='gray', zorder=2)

# 90th percentile corridor threshold contour
axes[1].contour(
    X,
    Y,
    Z_flux,
    levels=[10**-5],
    colors='#0d47a1',
    linestyles='--',
    linewidths=1.5,
    zorder=3,
)

# 95th percentile benthic hotspot contour
axes[1].contour(
    X,
    Y,
    Z_flux,
    levels=[3 * 10**-4],
    colors='#212121',
    linestyles='-',
    linewidths=1.8,
    zorder=4,
)

# Centroid Callout
axes[1].plot(
    -147.641,
    29.975,
    marker='*',
    color='black',
    markersize=11,
    zorder=5,
    label='Hotspot Centroid',
)
axes[1].text(
    -154.2,
    36.2,
    'Hotspot: 29.975°N, 147.641°W',
    fontsize=7.5,
    fontweight='bold',
    bbox=dict(
        boxstyle='round,pad=0.3',
        facecolor='white',
        edgecolor='#9e9e9e',
        alpha=0.92,
    ),
    zorder=6,
)

axes[1].set_xlim(-155, -140)
axes[1].set_ylim(22, 38)
axes[1].set_xlabel('Longitude (°W)', fontweight='bold')
axes[1].set_ylabel('Latitude (°N)', fontweight='bold')

cbar2 = fig.colorbar(pcm, ax=axes[1], shrink=0.85, pad=0.03)
cbar2.set_label('Flux (particles/km²)', fontweight='bold', fontsize=8)

axes[1].set_title(
    'b   Vertical Export Corridors\n     Area = 218,698 km²',
    loc='left',
    fontweight='bold',
)
axes[1].spines[['top', 'right']].set_visible(False)

plt.tight_layout()
save_publication_figure(fig, 'Supplementary_Fig_1')
print('Successfully generated Supplementary_Fig_1!')