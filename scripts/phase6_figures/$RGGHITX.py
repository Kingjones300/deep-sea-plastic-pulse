"""
Figure 1
Ocean Endmember framework for satellite-observable biofouling
and deep-sea plastic export via Vertical Export Corridors

Nature Geoscience style reconstruction

"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ============================
# Publication settings
# ============================

mpl.rcParams.update({

    "font.family": "Arial",
    "font.size": 8,

    "axes.linewidth": 0.6,

    "pdf.fonttype": 42,
    "ps.fonttype": 42

})


# ============================
# Colour palette
# ============================

colors = {

    "R1": "#009E73",
    "R2": "#7B2CBF",
    "R3": "#D62828",

    "ocean": "#DCEAF7"

}


# ============================
# Study regions
# ============================

regions = {

"R1\nStrait of Malacca\nSouth China Sea":
{
"lon":102,
"lat":5,
"tau":"τ = 5.9 d",
"color":colors["R1"]
},

"R2\nNorth Pacific\nSubtropical Gyre":
{
"lon":158,
"lat":35,
"tau":"τ = 42.3 d",
"color":colors["R2"]
},

"R3\nWestern\nMediterranean Sea":
{
"lon":8,
"lat":40,
"tau":"τ = 11.4 d",
"color":colors["R3"]
}

}



# ============================
# Figure canvas
# ============================


fig = plt.figure(

    figsize=(7.2,6),

    dpi=600

)


# ============================
# Global map
# ============================


ax = fig.add_axes(

    [0.05,0.42,0.9,0.5],

    projection=ccrs.Robinson()

)


ax.set_global()


ax.add_feature(

    cfeature.OCEAN,

    facecolor=colors["ocean"]

)


ax.add_feature(

    cfeature.LAND,

    facecolor="#D9D9D9"

)


ax.add_feature(

    cfeature.COASTLINE,

    linewidth=0.35

)



# regional markers

for label,data in regions.items():

    ax.scatter(

        data["lon"],

        data["lat"],

        s=55,

        color=data["color"],

        edgecolor="black",

        linewidth=0.4,

        transform=ccrs.PlateCarree(),

        zorder=5

    )



# labels

ax.text(

    -150,

    55,

    "Ocean Endmember study regions",

    fontsize=10,

    weight="bold",

    transform=ccrs.PlateCarree()

)



for label,data in regions.items():

    ax.text(

        data["lon"]+8,

        data["lat"],

        label,

        fontsize=7,

        transform=ccrs.PlateCarree()

    )


# ============================
# Three regional summary boxes
# ============================


positions=[0.05,0.36,0.67]


for pos,(label,data) in zip(
    positions,
    regions.items()
):

    box = fig.add_axes(

        [pos,0.08,0.27,0.23],

        projection=ccrs.PlateCarree()

    )


    box.set_extent(

        [

        data["lon"]-8,

        data["lon"]+8,

        data["lat"]-8,

        data["lat"]+8

        ]

    )


    box.add_feature(

        cfeature.OCEAN,

        facecolor=colors["ocean"]

    )


    box.add_feature(

        cfeature.LAND,

        facecolor="#D9D9D9"

    )


    box.add_feature(

        cfeature.COASTLINE,

        linewidth=0.4

    )


    box.scatter(

        data["lon"],

        data["lat"],

        marker="*",

        s=80,

        color=data["color"],

        edgecolor="black",

        transform=ccrs.PlateCarree()

    )


    box.set_title(

        label,

        fontsize=8

    )


    box.text(

        0.5,

        -0.15,

        data["tau"],

        ha="center",

        transform=box.transAxes,

        fontsize=8

    )



# ============================
# Explanation
# ============================


fig.text(

    0.05,

    0.02,

    "★ VEC centroid   |   τ = mean biofouling sinking timescale",

    fontsize=8

)



fig.canvas.draw()


# ============================
# Export
# ============================


plt.savefig(

    "Fig1_NatureGeoscience_600dpi.tiff",

    dpi=600,

    bbox_inches="tight"

)


plt.savefig(

    "Fig1_vector.pdf",

    bbox_inches="tight"

)


plt.savefig(

    "Fig1_preview.png",

    dpi=300,

    bbox_inches="tight"

)


plt.close()


print("Figure 1 generated successfully")