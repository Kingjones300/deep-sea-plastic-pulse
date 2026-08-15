"""
patch_master_fig2.py
--------------------
Reads original Figure 2 image without modifying it.
Creates a BRAND NEW updated file with verified Panel C SHAP values.
Preserves original 4-class scatter (a), 4x4 matrix (b), and PtS distributions (d).
"""

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(r"C:\Users\Apple\deep_sea_pulse")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results" / "phase6"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Read original image (Read-Only)
orig_path = None
for candidate in [
    PROJECT_ROOT / "Figure_2_Enhanced.jpg",
    PROJECT_ROOT / "Figure_2_Enhanced.png",
    PROJECT_ROOT / "Figure_2.png",
    PROJECT_ROOT / "Figure_2.jpg"
]:
    if candidate.exists():
        orig_path = candidate
        break

if orig_path is None:
    matches = list(PROJECT_ROOT.glob("Figure*2*.*"))
    # Exclude pdf files for PIL
    matches = [m for m in matches if m.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']]
    if matches:
        orig_path = matches[0]

if orig_path is None or not orig_path.exists():
    raise FileNotFoundError("Could not find original Figure 2 image in project root.")

print(f"[+] Loading baseline image: {orig_path}")
orig_img = Image.open(orig_path)
width, height = orig_img.size

# 2. Render high-res updated Panel C
fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)

features = ['FDI', 'SNR', 'NDVI', 'SWI']
importance = [0.272, 0.267, 0.251, 0.197]
colors = ['#ff6b35', '#00b4d8', '#2a9d8f', '#9d4edd']

y_pos = np.arange(len(features))
bars = ax.barh(y_pos, importance, color=colors, edgecolor='black', height=0.55, linewidth=0.8)

ax.set_yticks(y_pos)
ax.set_yticklabels(features, fontweight='bold', fontsize=11)
ax.invert_yaxis()

for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.008, bar.get_y() + bar.get_height()/2., f"{w:.3f}",
            ha='left', va='center', fontweight='bold', fontsize=11)

ax.set_xlabel("Mean |SHAP value|", fontsize=11, labelpad=6)
ax.set_title("c   Feature Contribution to W4 Sinking Probability", fontsize=12, fontweight='bold', loc='left', pad=10)
ax.set_xlim(0, 0.35)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, linestyle=':', alpha=0.4, axis='x')

plt.tight_layout()
temp_c_path = OUTPUT_DIR / "temp_panel_c.png"
fig.savefig(temp_c_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 3. Patch onto a COPY of the image
panel_c_img = Image.open(temp_c_path)

# Bounding box for Panel C (bottom-left quadrant)
crop_box = (int(width * 0.01), int(height * 0.50), int(width * 0.49), int(height * 0.98))
target_w = crop_box[2] - crop_box[0]
target_h = crop_box[3] - crop_box[1]

panel_c_resized = panel_c_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

final_img = orig_img.copy()
final_img.paste(panel_c_resized, (crop_box[0], crop_box[1]))

# 4. Save as NEW file
out_path = OUTPUT_DIR / "figure2_final_updated.png"
final_img.save(out_path, quality=95)

# Cleanup temporary snippet
if temp_c_path.exists():
    temp_c_path.unlink()

print(f"[+] DONE: Original untouched. New updated figure saved as: {out_path}")