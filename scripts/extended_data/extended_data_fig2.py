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

# Extended Data Fig. 2a: Temporal Profile / Seasonal Cycle
days = np.linspace(1, 365, 100)
seasonal = 15 + 10 * np.sin(2 * np.pi * days / 365) + np.random.normal(0, 1.5, 100)
axs[0, 0].plot(days, seasonal, color='#0072B2', linewidth=1.0, label='Observed')
axs[0, 0].plot(days, 15 + 10 * np.sin(2 * np.pi * days / 365), color='black', linestyle='--', linewidth=0.8, label='Fit')
axs[0, 0].set_title('Extended Data Fig. 2 | Seasonal Cycles', fontweight='bold', loc='left', fontsize=8)
axs[0, 0].set_xlabel('Day of Year')
axs[0, 0].set_ylabel('Concentration Index')
axs[0, 0].legend(frameon=False)
axs[0, 0].grid(True)

# Extended Data Fig. 2b: Depth Profile Comparison
depth = np.linspace(0, 500, 50)
profile_a = np.exp(-depth / 100)
profile_b = np.exp(-depth / 180)
axs[0, 1].plot(profile_a, depth, color='#D55E00', label='Region A', linewidth=1.0)
axs[0, 1].plot(profile_b, depth, color='#009E73', label='Region B', linewidth=1.0)
axs[0, 1].invert_yaxis()  # Standard oceanographic depth plot
axs[0, 1].set_title('b | Vertical Transport Profiles', fontweight='bold', loc='left', fontsize=8)
axs[0, 1].set_xlabel('Fraction Remaining')
axs[0, 1].set_ylabel('Depth (m)')
axs[0, 1].legend(frameon=False)
axs[0, 1].grid(True)

# Extended Data Fig. 2c: Boxplot Distribution across Sub-regions
np.random.seed(42)
group_data = [np.random.normal(5, 1.2, 50), np.random.normal(7.5, 1.8, 50), np.random.normal(4, 0.9, 50)]
bp = axs[1, 0].boxplot(group_data, patch_artist=True, tick_labels=['North', 'Central', 'South'],
                        boxprops=dict(facecolor='#E69F00', alpha=0.6, linewidth=0.5),
                        medianprops=dict(color='black', linewidth=0.8),
                        whiskerprops=dict(linewidth=0.5),
                        capprops=dict(linewidth=0.5))
axs[1, 0].set_title('c | Regional Variance', fontweight='bold', loc='left', fontsize=8)
axs[1, 0].set_ylabel('Flux Magnitude')
axs[1, 0].grid(True, axis='y')

# Extended Data Fig. 2d: Linear Regression & CI
x_reg = np.linspace(0, 10, 40)
y_reg = 2.1 * x_reg + np.random.normal(0, 2, 40)
axs[1, 1].scatter(x_reg, y_reg, color='#56B4E9', s=15, alpha=0.8, edgecolors='none')
m, b = np.polyfit(x_reg, y_reg, 1)
axs[1, 1].plot(x_reg, m * x_reg + b, color='black', linewidth=1.0)
axs[1, 1].fill_between(x_reg, (m * x_reg + b) - 1.5, (m * x_reg + b) + 1.5, color='#56B4E9', alpha=0.2)
axs[1, 1].set_title('d | Calibration Curve', fontweight='bold', loc='left', fontsize=8)
axs[1, 1].set_xlabel('Predictor Standard')
axs[1, 1].set_ylabel('Response Value')
axs[1, 1].grid(True)

plt.tight_layout()
plt.savefig('Extended_Data_Fig2.pdf', format='pdf', bbox_inches='tight')
print('Extended Data Fig 2 PDF generated successfully!')