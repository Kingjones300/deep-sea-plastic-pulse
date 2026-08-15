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

# Extended Data Fig. 1a: Sensitivity Analysis
x = np.linspace(0, 100, 100)
axs[0, 0].plot(x, np.sin(x/10) * np.exp(-x/50), color='#0072B2', label='Model Sensitivity')
axs[0, 0].set_title('Extended Data Fig. 1 | Sensitivity Analysis', fontweight='bold', loc='left', fontsize=8)
axs[0, 0].set_xlabel('Parameter Variation (%)')
axs[0, 0].set_ylabel('Response Amplitude')
axs[0, 0].grid(True)

# Extended Data Fig. 1b: Parameter Space Mapping
x_grid, y_grid = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))
z = np.exp(-x_grid**2 - y_grid**2)
c = axs[0, 1].contourf(x_grid, y_grid, z, cmap='viridis', levels=10)
fig.colorbar(c, ax=axs[0, 1], label='Density')
axs[0, 1].set_title('b | Parameter Space', fontweight='bold', loc='left', fontsize=8)
axs[0, 1].set_xlabel('Dimension 1')
axs[0, 1].set_ylabel('Dimension 2')

# Extended Data Fig. 1c: Error Distribution
errors = np.random.normal(0, 0.5, 500)
axs[1, 0].hist(errors, bins=25, color='#D55E00', alpha=0.7, edgecolor='black', linewidth=0.3)
axs[1, 0].set_title('c | Error Residuals', fontweight='bold', loc='left', fontsize=8)
axs[1, 0].set_xlabel('Residual Value')
axs[1, 0].set_ylabel('Frequency')
axs[1, 0].grid(True)

# Extended Data Fig. 1d: Convergence Plot
iterations = np.arange(1, 51)
convergence = 1 / np.sqrt(iterations)
axs[1, 1].plot(iterations, convergence, color='#009E73', marker='o', markersize=3, linestyle='-')
axs[1, 1].set_title('d | Algorithm Convergence', fontweight='bold', loc='left', fontsize=8)
axs[1, 1].set_xlabel('Iteration')
axs[1, 1].set_ylabel('Error Norm')
axs[1, 1].grid(True)

plt.tight_layout()
plt.savefig('Extended_Data_Fig1.pdf', format='pdf', bbox_inches='tight')
print('Extended Data Fig 1 PDF generated successfully!')