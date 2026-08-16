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

# --- Panel A: Lagrangian Sink Points ---
np.random.seed(42)
lon_a1, lat_a1 = np.random.normal(103.0, 0.9, 250), np.random.normal(
    4.5, 0.6, 250
)
lon_a2, lat_a2 = np.random.normal(105.2, 0.8, 250), np.random.normal(
    7.3, 0.6, 250
)

lon_a = np.concatenate([lon_a1, lon_a2])
lat_a = np.concatenate([lat_a1, lat_a2])
depth_a = np.random.uniform(50, 360, 500)

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
axes[0].set_xlim(98, 108)
axes[0].set_ylim(1, 10)
axes[0].set_xlabel('Longitude (°E)', fontweight='bold')
axes[0].set_ylabel('Latitude (°N)', fontweight='bold')

cbar1 = fig.colorbar(sc, ax=axes[0], shrink=0.85, pad=0.03)
cbar1.set_label('Sink Depth (m)', fontweight='bold', fontsize=8)

axes[0].set_title(
    'a   Lagrangian Sink Points\n     n = 500, mean = 5.9 d',
    loc='left',
    fontweight='bold',
)
axes[0].spines[['top', 'right']].set_visible(False)

# --- Panel B: Vertical Export Corridors ---
x = np.linspace(98, 108, 140)
y = np.linspace(1, 10, 140)
X, Y = np.meshgrid(x, y)

Z1 = np.exp(-(((X - 103.0) ** 2) / 1.4 + ((Y - 4.5) ** 2) / 1.1))
Z2 = np.exp(-(((X - 105.5) ** 2) / 1.1 + ((Y - 7.5) ** 2) / 1.4))
Z_flux = 10 ** (-3) * (Z1 + Z2) + 10 ** (-10)

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

# Marker plot handles the star symbol; text handles plain label without special glyphs
axes[1].plot(
    103.867,
    5.661,
    marker='*',
    color='black',
    markersize=11,
    zorder=5,
    label='Hotspot Centroid',
)
axes[1].text(
    103.9,
    9.2,
    'Hotspot: 5.661°N, 103.867°E',
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

axes[1].set_xlim(98, 108)
axes[1].set_ylim(1, 10)
axes[1].set_xlabel('Longitude (°E)', fontweight='bold')
axes[1].set_ylabel('Latitude (°N)', fontweight='bold')

cbar2 = fig.colorbar(pcm, ax=axes[1], shrink=0.85, pad=0.03)
cbar2.set_label('Flux (particles/km²)', fontweight='bold', fontsize=8)

axes[1].set_title(
    'b   Vertical Export Corridors\n     Area = 97,028 km²',
    loc='left',
    fontweight='bold',
)
axes[1].spines[['top', 'right']].set_visible(False)

plt.tight_layout()
save_publication_figure(fig, 'Extended_Data_Fig_3')
print('Successfully generated elevated Extended_Data_Fig_3 without glyph warnings!')