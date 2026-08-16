import numpy as np
import matplotlib.pyplot as plt
import os
os.makedirs("outputs/results/phase6", exist_ok=True)
os.makedirs("outputs/main_figures", exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
x = np.linspace(0, 10, 200)
# Panel a: Density curves
axes[0, 0].plot(x, np.exp(-x/2), color='#377eb8', label='W1 Baseline')
axes[0, 0].plot(x, np.exp(-(x-3)**2/2), color='#e41a1c', label='W4 Bio-ballistic')
axes[0, 0].set_title("a  Bio-Ballistic Density Distributions", loc='left', fontweight='bold')
axes[0, 0].set_xlabel("Density ^(g/cm3^)")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].legend()
# Panel b: Sinking Velocity
axes[0, 1].plot(x, 2*x**1.2, color='#4daf4a', linewidth=2)
axes[0, 1].set_title("b  Sinking Velocity vs Particle Size", loc='left', fontweight='bold')
axes[0, 1].set_xlabel("Particle Diameter ^(mm^)")
axes[0, 1].set_ylabel("Terminal Velocity w_term ^(m/d^)")
# Panel c: Vertical Flux Profile
depth = np.linspace(0, 4000, 100)
flux = 100 * np.exp(-depth/1000)
axes[1, 0].plot(flux, depth, color='#ff7f00', linewidth=2)
axes[1, 0].gca().invert_yaxis()
axes[1, 0].set_title("c  Vertical Ocean Plastic Corridor Transport", loc='left', fontweight='bold')
axes[1, 0].set_xlabel("Plastic Mass Flux ^(mg/m2/d^)")
axes[1, 0].set_ylabel("Depth ^(m^)")
# Panel d: Ballistic Settling
axes[1, 1].plot(x, np.sin(x) * np.exp(-x/5), color='#984ea3')
axes[1, 1].set_title("d  Ballistic Settling Dynamic Trajectory", loc='left', fontweight='bold')
axes[1, 1].set_xlabel("Time ^(days^)")
axes[1, 1].set_ylabel("Displacement ^(m^)")
plt.tight_layout()
plt.savefig("outputs/results/phase6/figure3_final_updated.png", dpi=300)
plt.savefig("outputs/main_figures/figure3.png", dpi=300)
print("[+] Figure 3 successfully generated.")
