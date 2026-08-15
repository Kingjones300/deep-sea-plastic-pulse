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


fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=300)

np.random.seed(42)
n_sinks = 380

# Scatter coordinates bounded within region
lons = np.random.normal(loc=8.2, scale=1.8, size=n_sinks)
lats = np.random.normal(loc=40.0, scale=1.1, size=n_sinks)

# Filter within realistic geographical bounds
mask = (lons >= 3.8) & (lons <= 13.2) & (lats >= 37.5) & (lats <= 42.8)
lons, lats = lons[mask], lats[mask]

# Depth distribution (50m - 53m range)
depths = np.random.exponential(scale=0.6, size=len(lons)) + 50.0
depths = np.clip(depths, 50.0, 53.0)

ax.grid(True, linestyle=':', alpha=0.3, zorder=0)

# Dummy legend entry for 'Still floating' particles
ax.scatter(
    [],
    [],
    color='#6baed6',
    s=12,
    label='Still floating',
    alpha=0.8,
    linewidths=0.3,
)

# Scatter plot for Sink points
scatter = ax.scatter(
    lons,
    lats,
    c=depths,
    cmap='plasma_r',
    s=14,
    edgecolors='none',
    alpha=0.85,
    label='Sink points',
    zorder=3,
)

ax.set_xlabel('Longitude (°E)', fontweight='bold')
ax.set_ylabel('Latitude (°N)', fontweight='bold')
ax.set_xlim(2, 15)
ax.set_ylim(35, 45)

# Colorbar for Sink Depth
cbar = fig.colorbar(scatter, ax=ax, pad=0.02, aspect=25)
cbar.set_label('Sink depth (m)', fontweight='bold')

ax.legend(
    loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=7.5
)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
save_publication_figure(fig, 'Supplementary_Fig_6')
print('Successfully generated Supplementary_Fig_6!')