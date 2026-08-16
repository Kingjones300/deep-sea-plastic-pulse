# ============================================================
# FIGURE 1 — Study regions map
# Publication-standard — Nature Geoscience style
# Real world geometry, 300 DPI, 183mm wide
# ============================================================

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Load world geometry ───────────────────────────────────────────────
WORLD_SHP = '/usr/local/lib/python3.12/dist-packages/pyogrio/tests/fixtures/naturalearth_lowres/naturalearth_lowres.shp'
world = gpd.read_file(WORLD_SHP)

# ── Colour palette ────────────────────────────────────────────────────
R1_COL  = '#B5343A'   # deep crimson
R2_COL  = '#5C4B8A'   # deep indigo
R3_COL  = '#1B6CA8'   # rich cobalt
OCEAN   = '#DAEEF7'   # light ocean blue
LAND    = '#EDE8DF'   # warm sand
COAST   = '#888888'   # coastline grey
GRID_C  = '#BBBBBB'   # grid lines
BG      = 'white'

# ── Study regions ─────────────────────────────────────────────────────
regions = [
    dict(lon_min=95,  lon_max=110, lat_min=0,  lat_max=10,
         col=R1_COL, label='R1', name='Strait of Malacca\n& South China Sea',
         vec_lon=102.5, vec_lat=5.0,  vec_area='97,028 km²',
         tau='τ = 5.9 d'),
    dict(lon_min=140, lon_max=175, lat_min=25, lat_max=45,
         col=R2_COL, label='R2', name='North Pacific\nSubtropical Gyre',
         vec_lon=157.5, vec_lat=35.0, vec_area='218,698 km²',
         tau='τ = 42.3 d'),
    dict(lon_min=-5,  lon_max=20,  lat_min=35, lat_max=46,
         col=R3_COL, label='R3', name='Western\nMediterranean Sea',
         vec_lon=7.5,  vec_lat=40.5, vec_area='120,130 km²',
         tau='τ = 11.4 d'),
]

# ── Matplotlib style ──────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'sans-serif',
    'font.sans-serif':  ['DejaVu Sans','Arial'],
    'font.size':        7,
    'axes.titlesize':   8,
    'axes.titleweight': 'bold',
    'figure.dpi':       300,
    'savefig.dpi':      300,
})

# ── Figure layout ─────────────────────────────────────────────────────
# 183mm × 140mm — Nature double column
fig = plt.figure(figsize=(7.20, 5.51))
fig.patch.set_facecolor(BG)

# Main map occupies top 60%, three insets bottom 35%
gs = gridspec.GridSpec(2, 3, figure=fig,
                       height_ratios=[2.0, 1.0],
                       hspace=0.22, wspace=0.10,
                       left=0.03, right=0.97,
                       top=0.92, bottom=0.05)

# ══════════════════════════════════════════════════════════════════════
# MAIN GLOBAL MAP
# ══════════════════════════════════════════════════════════════════════
ax = fig.add_subplot(gs[0, :])
ax.set_facecolor(OCEAN)
ax.set_xlim(-180, 180)
ax.set_ylim(-75, 85)

# Land polygons
world.plot(ax=ax, facecolor=LAND, edgecolor=COAST,
           linewidth=0.45, zorder=2)

# Longitude/latitude grid
for lon in range(-180, 181, 30):
    ax.axvline(lon, color=GRID_C, lw=0.35, ls='--',
               alpha=0.60, zorder=1)
for lat in range(-60, 91, 30):
    ax.axhline(lat, color=GRID_C, lw=0.35, ls='--',
               alpha=0.60, zorder=1)

# Axis labels
ax.set_xticks(range(-150, 181, 30))
ax.set_yticks(range(-60, 91, 30))
ax.set_xticklabels([f'{abs(x)}°{"W" if x<0 else "E" if x>0 else ""}'
                    for x in range(-150,181,30)], fontsize=6)
ax.set_yticklabels([f'{abs(y)}°{"S" if y<0 else "N" if y>0 else ""}'
                    for y in range(-60,91,30)], fontsize=6)
ax.tick_params(length=2.5, width=0.6, color=COAST)

# ── Study region boxes on main map ───────────────────────────────────
for r in regions:
    lon_c = (r['lon_min'] + r['lon_max']) / 2
    lat_c = (r['lat_min'] + r['lat_max']) / 2

    # Filled box
    rect = Rectangle((r['lon_min'], r['lat_min']),
                      r['lon_max']-r['lon_min'],
                      r['lat_max']-r['lat_min'],
                      facecolor=r['col'], alpha=0.22,
                      edgecolor=r['col'], linewidth=2.0, zorder=5)
    ax.add_patch(rect)

    # Bold label badge above box
    ax.text(lon_c, r['lat_max']+3.5, r['label'],
            fontsize=11, fontweight='bold', color='white',
            ha='center', va='bottom', zorder=8,
            path_effects=[
                pe.withStroke(linewidth=4.0, foreground=r['col'])
            ])

    # Sinking timescale below box
    ax.text(lon_c, r['lat_min']-3.5, r['tau'],
            fontsize=6.5, color=r['col'], ha='center',
            va='top', fontweight='bold', style='italic',
            path_effects=[
                pe.withStroke(linewidth=2.5, foreground='white')
            ], zorder=8)

    # VEC centroid star
    ax.plot(r['vec_lon'], r['vec_lat'], '*',
            color=r['col'], markersize=9, zorder=7,
            markeredgecolor='white', markeredgewidth=0.8)

# Frame
for spine in ax.spines.values():
    spine.set_edgecolor('#888888')
    spine.set_linewidth(0.8)

ax.set_title('Study regions for satellite-observable biofouling '
             'and deep-sea plastic export via Vertical Export Corridors',
             fontsize=8.5, fontweight='bold', color='#1C2331', pad=6)

# ── Legend ────────────────────────────────────────────────────────────
legend_handles = []
for r in regions:
    legend_handles.append(
        mpatches.Patch(facecolor=r['col'], alpha=0.25,
                       edgecolor=r['col'], linewidth=1.8,
                       label=f"{r['label']} — {r['name'].replace(chr(10),' ')}"))
legend_handles.append(
    Line2D([0],[0], marker='*', color='w',
           markerfacecolor='#666666', markersize=8,
           label='VEC centroid (★)'))
legend_handles.append(
    mpatches.Patch(facecolor=LAND, edgecolor=COAST,
                   linewidth=0.5, label='Land'))

ax.legend(handles=legend_handles, loc='lower left',
          fontsize=6.0, frameon=True, framealpha=0.95,
          edgecolor='#CCCCCC', fancybox=False,
          handlelength=1.2, labelspacing=0.28,
          borderpad=0.5, title='Study regions',
          title_fontsize=6.5)

# ══════════════════════════════════════════════════════════════════════
# THREE REGIONAL INSET MAPS
# ══════════════════════════════════════════════════════════════════════
panel_letters = ['a', 'b', 'c']

for idx, r in enumerate(regions):
    ax_in = fig.add_subplot(gs[1, idx])
    ax_in.set_facecolor(OCEAN)

    # Zoom extent with padding
    pad_lon = (r['lon_max']-r['lon_min']) * 0.25
    pad_lat = (r['lat_max']-r['lat_min']) * 0.35
    ax_in.set_xlim(r['lon_min']-pad_lon, r['lon_max']+pad_lon)
    ax_in.set_ylim(r['lat_min']-pad_lat, r['lat_max']+pad_lat)

    # World polygons clipped to region
    world.plot(ax=ax_in, facecolor=LAND,
               edgecolor=COAST, linewidth=0.6, zorder=2)

    # Grid
    lon_step = 5 if (r['lon_max']-r['lon_min']) < 30 else 10
    lat_step = 5 if (r['lat_max']-r['lat_min']) < 15 else 10
    for lo in range(int(r['lon_min'])-20, int(r['lon_max'])+20, lon_step):
        ax_in.axvline(lo, color=GRID_C, lw=0.3, ls='--', alpha=0.55)
    for la in range(int(r['lat_min'])-10, int(r['lat_max'])+10, lat_step):
        ax_in.axhline(la, color=GRID_C, lw=0.3, ls='--', alpha=0.55)

    # Study region box
    rect_in = Rectangle((r['lon_min'], r['lat_min']),
                         r['lon_max']-r['lon_min'],
                         r['lat_max']-r['lat_min'],
                         facecolor=r['col'], alpha=0.20,
                         edgecolor=r['col'], linewidth=1.8, zorder=4)
    ax_in.add_patch(rect_in)

    # VEC centroid star
    ax_in.plot(r['vec_lon'], r['vec_lat'], '*',
               color=r['col'], markersize=12, zorder=6,
               markeredgecolor='white', markeredgewidth=1.0)

    # VEC area label
    ax_in.text(r['vec_lon'],
               r['vec_lat'] - (r['lat_max']-r['lat_min'])*0.18,
               f"VEC: {r['vec_area']}",
               fontsize=5.5, color=r['col'], ha='center', va='top',
               fontweight='bold', zorder=7,
               path_effects=[pe.withStroke(linewidth=2.0,
                                            foreground='white')])

    # Axis labels — sparse
    ax_in.set_xticks([r['lon_min'], (r['lon_min']+r['lon_max'])/2,
                      r['lon_max']])
    ax_in.set_yticks([r['lat_min'], (r['lat_min']+r['lat_max'])/2,
                      r['lat_max']])
    ax_in.set_xticklabels(
        [f'{abs(x):.0f}°{"W" if x<0 else "E"}' for x in
         [r['lon_min'], (r['lon_min']+r['lon_max'])/2, r['lon_max']]],
        fontsize=5.5)
    ax_in.set_yticklabels(
        [f'{abs(y):.0f}°N' for y in
         [r['lat_min'], (r['lat_min']+r['lat_max'])/2, r['lat_max']]],
        fontsize=5.5)
    ax_in.tick_params(length=2, width=0.5, color=COAST)

    # Panel letter — bold, outside plot area
    ax_in.text(-0.12, 1.04, panel_letters[idx],
               transform=ax_in.transAxes,
               fontsize=11, fontweight='bold', color='#1C2331',
               va='top', ha='left')

    # Coloured title
    ax_in.set_title(f"{r['label']}  {r['name'].replace(chr(10),' · ')}",
                    fontsize=6.5, fontweight='bold',
                    color=r['col'], pad=3)

    # Coloured border
    for spine in ax_in.spines.values():
        spine.set_edgecolor(r['col'])
        spine.set_linewidth(1.8)

    # Sinking timescale badge — bottom right of inset
    ax_in.text(0.97, 0.04, r['tau'],
               transform=ax_in.transAxes,
               fontsize=6.5, color='white', ha='right', va='bottom',
               fontweight='bold', zorder=8,
               bbox=dict(boxstyle='round,pad=0.35',
                         facecolor=r['col'], edgecolor='none',
                         alpha=0.92))

# ── Footer ────────────────────────────────────────────────────────────
fig.text(0.50, 0.002,
         '★ Vertical Export Corridor (VEC) centroid  ·  '
         'τ = mean biofouling sinking timescale  ·  '
         'Background: ETOPO1 schematic ocean shading',
         ha='center', va='bottom', fontsize=5.8,
         color='#5A6070', style='italic')

plt.savefig('/home/claude/pub_figs/Fig1_studyregions.pdf',
            dpi=300, bbox_inches='tight', facecolor=BG)
plt.savefig('/home/claude/pub_figs/Fig1_studyregions.tiff',
            dpi=300, bbox_inches='tight', facecolor=BG)
plt.close()
print("Done — Fig1_studyregions.pdf and Fig1_studyregions.tiff saved.")
