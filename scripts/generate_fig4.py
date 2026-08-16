import matplotlib.pyplot as plt
import numpy as np

# Nature Geoscience Formatting Settings
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

# Create Double-Column Figure (183 mm / 7.2 inches wide)
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.0))

# Panel a: Satellite Disappearance
t = np.linspace(0, 30, 100)
axs[0, 0].plot(t, np.exp(-0.1 * t), color='#1f77b4', label='Surface Signal')
axs[0, 0].set_title('(a) Satellite Disappearance Signal', fontweight='bold', loc='left')
axs[0, 0].set_xlabel('Time (days)')
axs[0, 0].set_ylabel('Normalized Intensity')
axs[0, 0].grid(True)

# Panel b: Sediment Trap Lag
lags = np.linspace(-10, 10, 50)
corr = np.exp(-0.1 * (lags - 3)**2)
axs[0, 1].plot(lags, corr, color='#d62728')
axs[0, 1].axvline(3, color='black', linestyle=':', linewidth=0.8)
axs[0, 1].set_title('(b) Cross-Correlation Lag', fontweight='bold', loc='left')
axs[0, 1].set_xlabel('Lag (days)')
axs[0, 1].set_ylabel('Correlation Coefficient')
axs[0, 1].grid(True)

# Panel c: Null Model Distribution
null_data = np.random.normal(0, 1, 1000)
axs[1, 0].hist(null_data, bins=30, density=True, color='#2ca02c', alpha=0.4, edgecolor='none')
axs[1, 0].axvline(2.5, color='black', linewidth=1.0)
axs[1, 0].set_title('(c) Null Model Distribution', fontweight='bold', loc='left')
axs[1, 0].set_xlabel('Test Statistic')
axs[1, 0].set_ylabel('Density')
axs[1, 0].grid(True)

# Panel d: Validation Metric Cards (Clean Vector Layout)
ax_d = axs[1, 1]
ax_d.axis('off')
ax_d.set_title('(d) Validation Workflow Summary', fontweight='bold', loc='left')

# Replace the values below with your exact calculated metrics
metrics = [
    ('Lag Correlation', 'r = 0.84 (p < 0.001)', r'Status: Passed $\checkmark$'),
    ('Granger Causality', 'F = 14.2 (p = 0.002)', r'Status: Passed $\checkmark$'),
    ('Null Significance', 'p_null < 0.005', r'Status: Passed $\checkmark$')
]

for i, (title, val, status) in enumerate(metrics):
    x_pos = 0.02 + i * 0.33
    ax_d.text(x_pos, 0.70, title, fontsize=7, fontweight='bold')
    ax_d.text(x_pos, 0.45, val, fontsize=6.5)
    ax_d.text(x_pos, 0.20, status, fontsize=7, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig('Figure4_Nature_Standard.pdf', format='pdf', bbox_inches='tight')
print('Figure 4 PDF generated successfully!')