# ════════════════════════════════════════════════════════════════════
# ASSEMBLE — strict equal 2×2 grid with borders and labels
# Layout : a (top-left)   b (top-right)
#          c (bottom-left) d (bottom-right)
# ════════════════════════════════════════════════════════════════════
print("Assembling Figure 2...")

TITLE_H = 120   # height reserved for overall figure title
TOTAL_W = CANVAS*2 + GAP*3
TOTAL_H = CANVAS*2 + GAP*3 + TITLE_H
combined = Image.new("RGB", (TOTAL_W, TOTAL_H), (255, 255, 255))

positions = {
    "a": (GAP,            TITLE_H + GAP),
    "b": (GAP*2 + CANVAS, TITLE_H + GAP),
    "c": (GAP,            TITLE_H + GAP*2 + CANVAS),
    "d": (GAP*2 + CANVAS, TITLE_H + GAP*2 + CANVAS),
}
panels = {"a": canvas_a, "b": canvas_b,
          "c": canvas_c, "d": canvas_d}

draw = ImageDraw.Draw(combined)

# Load fonts
try:
    font_title = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/"
        "DejaVuSans-Bold.ttf", 56)
    font_lbl = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/"
        "DejaVuSans-Bold.ttf", 52)
except Exception:
    font_title = ImageFont.load_default()
    font_lbl   = ImageFont.load_default()

# Overall figure title — centred at top
title_text = ("Figure 2  |  Weathering state classification "
              "and XGBoost explainability")
bbox_t = draw.textbbox((0, 0), title_text, font=font_title)
tw = bbox_t[2] - bbox_t[0]
tx = (TOTAL_W - tw) // 2
ty = (TITLE_H - (bbox_t[3] - bbox_t[1])) // 2
draw.text((tx, ty), title_text, fill="#1A1A2E", font=font_title)

# Panels + borders + labels
for lbl, (px, py) in positions.items():
    img = panels[lbl]
    combined.paste(img, (px, py))
    # Border around each panel
    draw.rectangle(
        [px - BORDER, py - BORDER,
         px + CANVAS + BORDER, py + CANVAS + BORDER],
        outline="#BBBBBB", width=BORDER)
    # Label badge — white box then bold letter
    draw.rectangle(
        [px + 12, py + 12, px + 80, py + 80],
        fill=(255, 255, 255))
    draw.text((px + 18, py + 14), lbl,
               fill="#1A1A2E", font=font_lbl)

# Save
out_png  = os.path.join(OUTPUT_DIR, "Fig2_FINAL.png")
out_tiff = os.path.join(OUTPUT_DIR, "Fig2_FINAL.tiff")

combined.save(out_png,  dpi=(300, 300))
combined.save(out_tiff, dpi=(600, 600), compression="tiff_lzw")

print(f"\n{'='*60}")
print(f"Figure 2 FINAL saved:")
print(f"  PNG  → {out_png}")
print(f"  TIFF → {out_tiff}")
print(f"  Canvas per panel : {CANVAS} × {CANVAS} px")
print(f"  Combined size    : {TOTAL_W} × {TOTAL_H} px")
print(f"{'='*60}")