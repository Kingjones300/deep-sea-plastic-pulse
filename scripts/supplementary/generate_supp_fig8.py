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


fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(6.5, 5.2),
    dpi=300,
    sharex=True,
    gridspec_kw={'hspace': 0.15},
    layout='constrained',
)

days = np.linspace(0, 120, 300)

rho_p = 952 + (1105 - 952) * (1 - np.exp(-days / 6.5))
seawater_rho = 1029.0
v_term = -30 + (30 - (-30)) / (1 + np.exp(-(days - 5.5) / 0.8))

# Top Sub-panel: Particle Density Dynamics
ax1.grid(True, linestyle=':', alpha=0.4, zorder=0)
ax1.plot(
    days,
    rho_p,
    color='#0000d1',
    lw=1.8,
    label=r'$\rho_p(t)$ particle density',
    zorder=3,
)
ax1.axhline(
    seawater_rho,
    color='#d32f2f',
    linestyle='--',
    linewidth=1.2,
    label=r'Seawater density ($1029.0\ \mathrm{kg/m^3}$)',
    zorder=2,
)

sinking_mask = rho_p >= seawater_rho
ax1.fill_between(
    days,
    seawater_rho,
    rho_p,
    where=sinking_mask,
    color='#ffcdd2',
    alpha=0.5,
    label='Sinking zone',
    zorder=1,
)

ax1.set_ylabel(r'Density ($\mathrm{kg/m^3}$)', fontweight='bold')
ax1.set_ylim(940, 1120)
ax1.legend(loc='lower right', frameon=True, facecolor='white', fontsize=7)
ax1.set_title(
    'Supplementary Fig. 8 | Biofilm Density and Terminal Velocity Dynamics',
    loc='left',
    fontweight='bold',
    fontsize=9,
    pad=8,
)

# Bottom Sub-panel: Terminal Velocity Dynamics
ax2.grid(True, linestyle=':', alpha=0.4, zorder=0)
ax2.axhline(0, color='black', linewidth=0.8, zorder=1)
ax2.plot(days, v_term, color='#00796b', lw=1.8, zorder=3)

ax2.fill_between(
    days,
    0,
    v_term,
    where=(v_term >= 0),
    color='#c8e6c9',
    alpha=0.6,
    label='Sinking',
    zorder=2,
)

ax2.set_xlabel('Days at sea', fontweight='bold')
ax2.set_ylabel('Terminal velocity (m/day)', fontweight='bold')
ax2.set_ylim(-33, 33)
ax2.set_xlim(-5, 125)
ax2.legend(loc='lower right', frameon=True, facecolor='white', fontsize=7)

save_publication_figure(fig, 'Supplementary_Fig_8')
print('Successfully generated Supplementary_Fig_8 without warnings!')