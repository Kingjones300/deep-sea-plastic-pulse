"""
Figure 1 v2
Nature Geoscience style refinement

Ocean Endmember study regions for
satellite-observable biofouling and
deep-sea plastic export via Vertical Export Corridors

"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# -----------------------------
# Publication style
# -----------------------------

mpl.rcParams.update({

    "font.family": "Arial",
    "font.size": 7.5,
    "axes.linewidth":0.5,
    "pdf.fonttype":42,
    "ps.fonttype":42

})


# -----------------------------
# Colours
# -----------------------------

COLORS={

"R1":"#009E73",
"R2":"#7B2CBF",
"R3":"#D62828",
"ocean":"#DCEAF7",
"land":"#D9D9D9"

}


# -----------------------------
# Regions
# -----------------------------

regions={

"R1":{
"lon":102,
"lat":5,
"name":"Strait of Malacca\n& South China Sea",
"tau":"τ = 5.9 d",
"color":COLORS["R1"]
},

"R2":{
"lon":158,
"lat":35,
"name":"North Pacific\nSubtropical Gyre",
"tau":"τ = 42.3 d",
"color":COLORS["R2"]
},

"R3":{
"lon":8,
"lat":40,
"name":"Western\nMediterranean Sea",
"tau":"τ = 11.4 d",
"color":COLORS["R3"]
}

}



# -----------------------------
# Figure
# -----------------------------

fig=plt.figure(

figsize=(7.2,5.8),

dpi=600

)



# -----------------------------
# Global map
# -----------------------------


ax=fig.add_axes(

[0.05,0.45,0.90,0.45],

projection=ccrs.Robinson()

)


ax.set_global()


ax.add_feature(

cfeature.OCEAN,

facecolor=COLORS["ocean"]

)


ax.add_feature(

cfeature.LAND,

facecolor=COLORS["land"]

)


ax.add_feature(

cfeature.COASTLINE,

linewidth=0.35

)



# markers

for r,data in regions.items():

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



# Leader annotations

annotations={

"R1":(125,12),

"R2":(175,48),

"R3":(-20,50)

}


for r,data in regions.items():

    ax.annotate(

        f"{r}\n{data['name']}",

        xy=(data["lon"],data["lat"]),

        xytext=annotations[r],

        fontsize=7,

        ha="left",

        transform=ccrs.PlateCarree(),

        arrowprops=dict(

            arrowstyle="-",

            linewidth=0.5

        )

    )



# -----------------------------
# Insets
# -----------------------------


positions=[0.07,0.365,0.66]


for pos,(r,data) in zip(

positions,

regions.items()

):

    inset=fig.add_axes(

        [pos,0.10,0.25,0.25],

        projection=ccrs.PlateCarree()

    )


    inset.set_extent(

        [

        data["lon"]-8,

        data["lon"]+8,

        data["lat"]-8,

        data["lat"]+8

        ]

    )


    inset.add_feature(

        cfeature.OCEAN,

        facecolor=COLORS["ocean"]

    )


    inset.add_feature(

        cfeature.LAND,

        facecolor=COLORS["land"]

    )


    inset.add_feature(

        cfeature.COASTLINE,

        linewidth=0.35

    )


    inset.scatter(

        data["lon"],

        data["lat"],

        marker="*",

        s=70,

        color=data["color"],

        edgecolor="black",

        transform=ccrs.PlateCarree()

    )


    inset.set_title(

        f"{r} | {data['name']}",

        fontsize=7

    )


    inset.text(

        0.5,

        -0.17,

        data["tau"],

        ha="center",

        transform=inset.transAxes,

        fontsize=8

    )



# -----------------------------
# Legend
# -----------------------------


fig.text(

0.05,

0.03,

"★ VEC centroid     τ  Mean biofouling sinking timescale",

fontsize=7.5

)



fig.canvas.draw()


# -----------------------------
# Export
# -----------------------------


plt.savefig(

"Fig1_NG_final_v2_600dpi.tiff",

dpi=600,

bbox_inches="tight"

)


plt.savefig(

"Fig1_NG_final_v2.pdf",

bbox_inches="tight"

)


plt.savefig(

"Fig1_NG_final_v2_preview.png",

dpi=300,

bbox_inches="tight"

)


plt.close()


print("Figure 1 final v2 generated")