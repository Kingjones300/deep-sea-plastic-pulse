import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
import matplotlib.patches as mpatches

# ==========================================
# 1. DATA GENERATION (Exact Specs)
# ==========================================
rng = np.random.default_rng(42)
n_samples = 1250

# Parameters copied directly from dataset file
W1_FDI = rng.normal(0.08,  0.010, n_samples)
W1_SWI = rng.normal(-0.05, 0.010, n_samples)

W2_FDI = rng.normal(0.06,  0.015, n_samples)
W2_SWI = rng.normal(0.02,  0.010, n_samples)

W3_FDI = rng.normal(0.04,  0.015, n_samples)
W3_SWI = rng.normal(0.08,  0.015, n_samples)

W4_FDI = rng.normal(0.02,  0.010, n_samples)
W4_SWI = rng.normal(0.15,  0.020, n_samples)

# Grouping for clean iterative rendering
classes = [
    {'name': 'W1 — Virgin polymer', 'color': '#1f77b4', 'x': W1_FDI, 'y': W1_SWI},
    {'name': 'W2 — Early biofouling', 'color': '#2ca02c', 'x': W2_FDI, 'y': W2_SWI},
    {'name': 'W3 — Dense biofilm', 'color': '#ff7f0e', 'x': W3_FDI, 'y': W3_SWI},
    {'name': 'W4 — Critical sinking threshold', 'color': '#d62728', 'x': W4_FDI, 'y': W4_SWI}
]

# ==========================================
# 2. FIGURE ARCHITECTURE & LAYOUT
# ==========================================
# Exact structural figure dimensions converted to inches (90mm x 88mm * scale factor for margins)
fig = plt.figure(figsize=(7.5, 8.5), dpi=300)
fig.patch.set_facecolor('white')

# GridSpec setup for a clean journal multi-panel display
gs = gridspec.GridSpec(3, 1, height_ratios=[1.2, 1.2, 4.5], hspace=0.28)

ax_ridge_top = fig.add_subplot(gs[0])     # Marginal Ridge: FDI
ax_ridge_mid = fig.add_subplot(gs[1])     # Marginal Ridge: SWI
ax_main      = fig.add_subplot(gs[2])     # 2D Confidence Ellipse Scatter

# Precise Axis Ranges from Specs
x_min, x_max = -0.025, 0.125
y_min, y_max = -0.115, 0.215

# Evaluation lines for clean KDE curves
x_eval = np.linspace(x_min, x_max, 500)
y_eval = np.linspace(y_min, y_max, 500)

# ==========================================
# 3. TOP RIDGE PANEL: FDI Distributions
# ==========================================
overlap_y_fdi = 18.0
for i, c in enumerate(classes):
    kde = gaussian_kde(c['x'])
    density = kde(x_eval)
    
    baseline = i * overlap_y_fdi
    shifted_density = density + baseline
    
    ax_ridge_top.fill_between(x_eval, baseline, shifted_density, alpha=0.65, color=c['color'], zorder=10-i)
    ax_ridge_top.plot(x_eval, shifted_density, color=c['color'], lw=1.2, zorder=10-i)
    ax_ridge_top.axhline(baseline, color='#CCCCCC', lw=0.5, ls='-', alpha=0.5)

ax_ridge_top.set_xlim(x_min, x_max)
ax_ridge_top.axis('off')
ax_ridge_top.set_title("Marginal Distributions (FDI & SWI Ridges)", fontsize=9, weight='bold', loc='left', pad=4)

# ==========================================
# 4. MIDDLE RIDGE PANEL: SWI Distributions
# ==========================================
overlap_y_swi = 12.0
for i, c in enumerate(classes):
    kde = gaussian_kde(c['y'])
    density = kde(y_eval)
    
    baseline = i * overlap_y_swi
    shifted_density = density + baseline
    
    ax_ridge_mid.fill_between(y_eval, baseline, shifted_density, alpha=0.65, color=c['color'], zorder=10-i)
    ax_ridge_mid.plot(y_eval, shifted_density, color=c['color'], lw=1.2, zorder=10-i)
    ax_ridge_mid.axhline(baseline, color='#CCCCCC', lw=0.5, ls='-', alpha=0.5)

ax_ridge_mid.set_xlim(y_min, y_max)
ax_ridge_mid.axis('off')

# ==========================================
# 5. MAIN SUBPLOT: Confidence Ellipse Scatter
# ==========================================
ax_main.set_facecolor('white')
ax_main.grid(True, which='both', color='#CCCCCC', linestyle='-', linewidth=0.6, zorder=0)

for c in classes:
    # Blend point alpha based on neighborhood density to counter 5000-point overplotting
    xy = np.vstack([c['x'], c['y']])
    kde_2d = gaussian_kde(xy)
    z = kde_2d(xy)
    z_norm = (z - z.min()) / (z.max() - z.min())
    
    rgba = np.zeros((n_samples, 4))
    rgba[:, :3] = plt.cm.colors.to_rgb(c['color'])
    rgba[:, 3] = 0.12 + (z_norm * 0.58)  # Fades edge outliers, anchors dense nuclei
    
    # Scatter Render
    ax_main.scatter(c['x'], c['y'], c=rgba, s=6, edgecolors='none', zorder=3)
    
    # Pearson Correlation-Based 95% Confidence Ellipses (n_std = 2.0)
    cov = np.cov(c['x'], c['y'])
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    
    ellipse = mpatches.Ellipse(
        xy=(np.mean(c['x']), np.mean(c['y'])),
        width=lambda_[0] * 2 * 2.0, height=lambda_[1] * 2 * 2.0,
        angle=np.rad2deg(np.arctan2(v[1, 0], v[0, 0])),
        edgecolor=c['color'], facecolor='none', linestyle='--', linewidth=1.6, zorder=5
    )
    ax_main.add_patch(ellipse)

# Formatting Core Axes Dimensions & Font Elements
ax_main.set_xlim(x_min, x_max)
ax_main.set_ylim(y_min, y_max)

ax_main.set_xticks([-0.02, 0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12])
ax_main.set_yticks([-0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20])

ax_main.tick_params(axis='both', labelsize=7.5, colors='#333333')
ax_main.set_xlabel("Floating Debris Index (FDI)", fontsize=9, labelpad=6)
ax_main.set_ylabel("Spectral Weathering Index (SWI)", fontsize=9, labelpad=6)

ax_main.spines[['top', 'right']].set_visible(False)
ax_main.spines[['left', 'bottom']].set_color('#333333')
ax_main.spines[['left', 'bottom']].set_linewidth(0.8)

# Legend Layout Anchor
legend_patches = [mpatches.Patch(color=c['color'], label=c['name']) for c in classes]
ax_main.legend(handles=legend_patches, loc='lower right', frameon=True, 
               facecolor='#FFFFFF', edgecolor='#CCCCCC', fontsize=8, framealpha=0.95)

# High-Precision Annotation Box
ax_main.text(-0.015, 0.175, "Overall F1:  0.9932\nCohen kappa: 0.991\nn total:     5000", 
             fontsize=7.5, family='monospace', bbox=dict(boxstyle="round,pad=0.5", facecolor="#FDFDFD", edgecolor="#CCCCCC", alpha=0.9))

plt.subplots_adjust(bottom=0.1, left=0.12, right=0.95)
plt.show()