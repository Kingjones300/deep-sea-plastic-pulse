import matplotlib.pyplot as plt
import numpy as np

# Nature Extended Data Formatting Settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 7,
    'axes.labelsize': 7,
    'axes.titlesize': 8,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'axes.linewidth': 0.5,
    'grid.linewidth': 0.25,
    'grid.color': '#CCCCCC',
    'grid.linestyle': '--',
    'savefig.dpi': 600,
    'pdf.fonttype': 42
})

# Create 2x2 Panel Layout (183 mm / 7.2 inches wide)
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.5))

# Extended Data Fig. 3a: Spectral / Weathering Signal
wave = np.linspace(400, 1000, 100)
reflectance = 0.05 + 0.45 * (1 - np.exp(-(wave - 400) / 200))
axs[0, 0].plot(wave, reflectance, color='#CC79A7', linewidth=1.0)
axs[0, 0].set_title('Extended Data Fig. 3 | Spectral Weathering', fontweight='bold', loc='left', fontsize=8)
axs[0, 0].set_xlabel('Wavelength (nm)')
axs[0, 0].set_ylabel('Reflectance')
axs[0, 0].grid(True)

# Extended Data Fig. 3b: Particle Bio-fouling Density
time_days = np.linspace(0, 30, 50)
density = 0.92 + 0.15 * (1 / (1 + np.exp(-(time_days - 12) / 3)))
axs[0, 1].plot(time_days, density, color='#0072B2', linewidth=1.0)
axs[0, 1].axhline(1.025, color='black', linestyle=':', label='Seawater Density')
axs[0, 1].set_title('b | Bio-fouling Density Over Time', fontweight='bold', loc='left', fontsize=8)
axs[0, 1].set_xlabel('Time (days)')
axs[0, 1].set_ylabel(r'Density (g cm$^{-3}$)')
axs[0, 1].legend(frameon=False)
axs[0, 1].grid(True)

# Extended Data Fig. 3c: Sinking Velocity vs Diameter
diameter = np.logspace(-1, 1, 50)
v_sinking = 2.5 * (diameter ** 1.2)
axs[1, 0].loglog(diameter, v_sinking, color='#009E73', linewidth=1.0)
axs[1, 0].set_title('c | Sinking Velocity Scaling', fontweight='bold', loc='left', fontsize=8)
axs[1, 0].set_xlabel('Particle Diameter (mm)')
axs[1, 0].set_ylabel(r'Sinking Velocity (m d$^{-1}$)')
axs[1, 0].grid(True, which='both')

# Extended Data Fig. 3d: Model Sensitivity Heatmap / Grid
x_param = np.linspace(0.1, 1.0, 20)
y_param = np.linspace(1, 10, 20)
X, Y = np.meshgrid(x_param, y_param)
Z = X * np.log(Y)
im = axs[1, 1].pcolormesh(X, Y, Z, cmap='YlGnBu', shading='auto')
fig.colorbar(im, ax=axs[1, 1], label='Sensitivity Index')
axs[1, 1].set_title('d | Multi-parameter Sensitivity', fontweight='bold', loc='left', fontsize=8)
axs[1, 1].set_xlabel('Weathering Rate')
axs[1, 1].set_ylabel('Mixing Coefficient')

plt.tight_layout()
plt.savefig('Extended_Data_Fig3.pdf', format='pdf', bbox_inches='tight')
print('Extended Data Fig 3 PDF generated successfully!')