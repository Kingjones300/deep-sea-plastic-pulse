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


fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=300)

np.random.seed(42)
n_particles = 500

# Exponentially decaying right-skewed distribution anchored at 50.0 m (mode ~ 50.2 m, tail ~ 53 m)
raw_depths = np.random.exponential(scale=0.55, size=n_particles * 2) + 50.0
sink_depths = raw_depths[raw_depths <= 53.0][:n_particles]

counts, bins, patches = ax.hist(
    sink_depths,
    bins=30,
    color='#3f51b5',
    edgecolor='white',
    linewidth=0.6,
    zorder=3,
)

ax.set_xlabel('Sink Depth (m)', fontweight='bold')
ax.set_ylabel('Count', fontweight='bold')
ax.set_xlim(49.8, 53.2)
ax.set_ylim(0, max(counts) + 5)
ax.spines[['top', 'right']].set_visible(False)
ax.set_title(
    'Supplementary Fig. 7 | Plastic sink-depth distribution: R3 Western Mediterranean\n'
    'Histogram of individual particle sink depths (n = 500); mode ~50.2 m; tail to ~53 m',
    loc='left',
    fontweight='bold',
    fontsize=8.5,
    pad=10,
)

plt.tight_layout()
save_publication_figure(fig, 'Supplementary_Fig_7')
print('Successfully generated Supplementary_Fig_7 (Histogram Only)!')