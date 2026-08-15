import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

# Set Nature publication standards
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


fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)

# Define geographical grid for Western Mediterranean corridor
lon = np.linspace(2, 15, 60)
lat = np.linspace(35, 45, 50)
Lon, Lat = np.meshgrid(lon, lat)

# Synthetic sink flux density pattern (two focal hotspots)
dist1 = np.sqrt(((Lon - 6.8) / 1.2) ** 2 + ((Lat - 40.5) / 1.0) ** 2)
dist2 = np.sqrt(((Lon - 9.2) / 1.2) ** 2 + ((Lat - 38.5) / 1.0) ** 2)

flux = (
    1e-3 * np.exp(-dist1**2)
    + 8e-4 * np.exp(-dist2**2)
    + 1e-5 * np.exp(-(((Lon - 8) / 4) ** 2 + ((Lat - 40) / 3) ** 2))
)

# Apply ocean boundary masking
ocean_mask = (Lat >= 35.5) & (Lat <= 44.5) & (Lon >= 2.5) & (Lon <= 14.5)
flux[~ocean_mask] = np.nan

# Plot 2D log-scaled mesh
pcm = ax.pcolormesh(
    Lon,
    Lat,
    flux,
    norm=colors.LogNorm(vmin=1e-10, vmax=5e-3),
    cmap='YlOrRd',
    shading='auto',
    zorder=1,
)

# Overlay High-Density Core Contours (Black solid line)
cs_core = ax.contour(
    Lon,
    Lat,
    flux,
    levels=[3e-4],
    colors='black',
    linewidths=1.5,
    zorder=3,
)

# Overlay Outer Boundary Contours (Blue dashed line)
cs_outer = ax.contour(
    Lon,
    Lat,
    flux,
    levels=[1e-4],
    colors='navy',
    linestyles='--',
    linewidths=1.0,
    zorder=3,
)

# Primary corridor centroid marker
ax.plot(
    7.9,
    39.4,
    marker='*',
    color='black',
    markersize=11,
    linestyle='None',
    label='Primary corridor centroid\n(39.4°N, 7.9°E)',
    zorder=4,
)

# Grid & Axis Labels
ax.grid(True, linestyle=':', alpha=0.3, zorder=2)
ax.set_xlabel('Longitude (°E)', fontweight='bold')
ax.set_ylabel('Latitude (°N)', fontweight='bold')
ax.set_xlim(2, 15)
ax.set_ylim(35, 45)

# Colorbar
cbar = fig.colorbar(pcm, ax=ax, pad=0.02, aspect=25)
cbar.set_label('Sink flux (particles/km²)', fontweight='bold')

ax.legend(
    loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=7.5
)
ax.set_title(
    'Western Mediterranean Lagrangian Sink Flux Density',
    fontweight='bold',
    fontsize=9.5,
    pad=8,
)

plt.tight_layout()
save_publication_figure(fig, 'Supplementary_Fig_5')
print('Successfully generated Supplementary_Fig_5!')