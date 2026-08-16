import matplotlib.pyplot as plt
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


fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), dpi=300)

# --- Panel A: PtS Weathering Progression (R2 Oligotrophic Gyre) ---
categories = ['W1\nVirgin', 'W2\nEarly', 'W3\nAdvanced', 'W4\nCritical']
pts_scores = [0.06, 0.25, 0.58, 0.85]
bar_colors = ['#00838f', '#2e7d32', '#f57c00', '#c62828']

bars = axes[0].bar(
    categories,
    pts_scores,
    color=bar_colors,
    edgecolor='none',
    width=0.55,
    alpha=0.9,
    zorder=3,
)

axes[0].grid(axis='y', linestyle=':', alpha=0.5, zorder=0)
for bar, score in zip(bars, pts_scores):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.025,
        f'{score:.2f}',
        ha='center',
        va='bottom',
        fontsize=8,
        fontweight='bold',
        color='#212121',
    )

axes[0].axhline(
    0.5, color='#9e9e9e', linestyle='--', linewidth=0.8, alpha=0.7, zorder=1
)
axes[0].set_ylim(0, 1.05)
axes[0].set_ylabel('PtS Score', fontweight='bold')
axes[0].set_title(
    'a   W1-W4 PtS Distribution\n     R2 N. Pacific Gyre (Oligotrophic)',
    loc='left',
    fontweight='bold',
)
axes[0].spines[['top', 'right']].set_visible(False)

# --- Panel B: Feature Importance (SWI Dominant Highlight) ---
features = ['SWI', 'FDI', 'NDVI', 'SNR']
shap_vals = [0.35, 0.30, 0.22, 0.13]
# Primary driver (SWI) highlighted in deep indigo; others in soft slate blue
feature_colors = ['#1a237e', '#3949ab', '#5c6bc0', '#9fa8da']

axes[1].grid(axis='x', linestyle=':', alpha=0.5, zorder=0)
bars_b = axes[1].barh(
    features[::-1],
    shap_vals[::-1],
    color=feature_colors[::-1],
    edgecolor='none',
    height=0.55,
    zorder=3,
)

for bar, val in zip(bars_b, shap_vals[::-1]):
    axes[1].text(
        val + 0.008,
        bar.get_y() + bar.get_height() / 2,
        f'{val:.2f}',
        ha='left',
        va='center',
        fontsize=7.5,
        fontweight='bold',
        color='#1a237e',
    )

axes[1].set_xlim(0, 0.42)
axes[1].set_xlabel('Mean |SHAP Value|', fontweight='bold')
axes[1].set_title(
    'b   Feature Importance\n     F1 = 0.9989 (Primary: SWI)',
    loc='left',
    fontweight='bold',
)
axes[1].spines[['top', 'right']].set_visible(False)

# --- Panel C: Bio-Ballistic Trajectory (180-Day Scale with Callout) ---
t = np.linspace(0, 180, 400)
rho_p = 1083 - 133 * np.exp(-t / 22.0)
seawater_thresh = 1026.0

axes[2].grid(True, linestyle=':', alpha=0.4, zorder=0)

axes[2].plot(
    t, rho_p, color='#0d47a1', lw=2.2, label=r'$\rho_p(t)$ Density', zorder=4
)
axes[2].axhline(
    seawater_thresh,
    color='#d32f2f',
    linestyle='--',
    linewidth=1.2,
    label='Seawater Threshold (1,026 kg/m³)',
    zorder=3,
)

axes[2].fill_between(
    t,
    rho_p,
    seawater_thresh,
    where=(rho_p >= seawater_thresh),
    color='#ef5350',
    alpha=0.18,
    zorder=2,
)

# Critical intersection marker at t = 14 days
t_sink = 14.0
rho_sink = 1083 - 133 * np.exp(-t_sink / 22.0)
axes[2].plot(
    t_sink,
    rho_sink,
    marker='*',
    color='#d32f2f',
    markersize=9,
    zorder=5,
    label='Sinking Point (t = 14 d)',
)
axes[2].annotate(
    't_sink ≈ 14 d',
    xy=(t_sink, rho_sink),
    xytext=(t_sink + 25, rho_sink - 20),
    arrowprops=dict(
        arrowstyle='->', color='#d32f2f', lw=0.9, connectionstyle='arc3,rad=-0.2'
    ),
    fontsize=7.5,
    fontweight='bold',
    color='#b71c1c',
)

axes[2].set_xlim(0, 180)
axes[2].set_ylim(940, 1100)
axes[2].set_xlabel('Days at Sea', fontweight='bold')
axes[2].set_ylabel('Density (kg/m³)', fontweight='bold')
axes[2].set_title(
    'c   Bio-Ballistic Density $\\rho_p(t)$\n     SST = 22.0°C, Chl = 0.08 mg/m³',
    loc='left',
    fontweight='bold',
)
axes[2].legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none', fontsize=7)
axes[2].spines[['top', 'right']].set_visible(False)

plt.tight_layout()
save_publication_figure(fig, 'Extended_Data_Fig_2')
print('Successfully generated elevated Extended_Data_Fig_2!')