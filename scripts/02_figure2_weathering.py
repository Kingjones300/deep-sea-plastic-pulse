from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(r"C:\Users\Apple\deep_sea_pulse")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

orig_path = PROJECT_ROOT / "outputs" / "main_figures" / "figure2.png"
if not orig_path.exists():
    raise FileNotFoundError(f"Could not find baseline image at {orig_path}")

print(f"[+] Loading baseline image: {orig_path}")
orig_img = Image.open(orig_path)
width, height = orig_img.size

fig, ax = plt.subplots(figsize=(6, 4.2), dpi=300)

features = ["FDI", "SNR", "NDVI", "SWI"]
importance = [0.272, 0.267, 0.251, 0.197]
colors = ["#ff6b35", "#00b4d8", "#2a9d8f", "#9d4edd"]

y_pos = np.arange(len(features))
bars = ax.barh(y_pos, importance, color=colors, edgecolor="black", height=0.55, linewidth=0.8)

ax.set_yticks(y_pos)
ax.set_yticklabels(features, fontweight="bold", fontsize=11)
ax.invert_yaxis()

for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.008, bar.get_y() + bar.get_height()/2., f"{w:.3f}", ha="left", va="center", fontweight="bold", fontsize=11)

ax.set_xlabel(chr(77)+chr(101)+chr(97)+chr(110)+chr(32)+chr(124)+chr(83)+chr(72)+chr(65)+chr(80)+chr(32)+chr(118)+chr(97)+chr(108)+chr(117)+chr(101)+chr(124), fontsize=11, fontweight="bold", labelpad=8)
ax.set_title("c   Feature Contribution to W4 Sinking Probability", fontsize=12, fontweight="bold", loc="left", pad=12)
ax.set_xlim(0, 0.35)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, linestyle=":", alpha=0.4, axis="x")

temp_c_path = OUTPUT_DIR / "temp_panel_c.png"
fig.savefig(temp_c_path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.1)
plt.close()

panel_c_img = Image.open(temp_c_path)
crop_box = (int(width * 0.01), int(height * 0.50), int(width * 0.49), int(height * 0.99))
target_w = crop_box[2] - crop_box[0]
target_h = crop_box[3] - crop_box[1]

panel_c_resized = panel_c_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

final_img = orig_img.copy()
final_img.paste(panel_c_resized, (crop_box[0], crop_box[1]))

out_path = OUTPUT_DIR / "Figure_2.png"
final_img.save(out_path, quality=95)

if temp_c_path.exists():
    temp_c_path.unlink()

print(f"[+] DONE: Updated Figure 2 PNG saved directly to: {out_path}")
