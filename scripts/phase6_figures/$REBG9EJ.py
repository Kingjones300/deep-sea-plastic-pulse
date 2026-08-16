"""
Figure 2
Spectral discrimination of plastic weathering states
and prediction of sinking probability

Nature Geoscience style reconstruction

"""

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.patches import Ellipse
from scipy.stats import gaussian_kde

import seaborn as sns


# =====================================================
# Publication settings
# =====================================================

mpl.rcParams.update({

    "font.family": "Arial",
    "font.size": 7,

    "axes.linewidth": 0.5,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "xtick.direction": "out",
    "ytick.direction": "out"

})


# =====================================================
# Paths
# =====================================================

DATA_FILE = "Fig2_complete_data.xlsx"

OUTPUT_DIR = "output"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)



# =====================================================
# Nature-style colour palette
# =====================================================

COLORS = {

    "W1":"#0072B2",
    "W2":"#009E73",
    "W3":"#E69F00",
    "W4":"#D55E00"

}


ORDER = [

    "W1",
    "W2",
    "W3",
    "W4"

]



# =====================================================
# Read Excel workbook
# =====================================================

xls = pd.ExcelFile(DATA_FILE)


print(
    "Available sheets:",
    xls.sheet_names
)


# =====================================================
# Load panels
# =====================================================

panel_A = pd.read_excel(

    DATA_FILE,

    sheet_name="Panel_A_Scatter"

)


panel_B = pd.read_excel(

    DATA_FILE,

    sheet_name="Panel_B_ConfusionMatrix"

)


panel_C = pd.read_excel(

    DATA_FILE,

    sheet_name="Panel_C_SHAP"

)


panel_D = pd.read_excel(

    DATA_FILE,

    sheet_name="Panel_D_PtS"

)



# =====================================================
# Figure canvas
# =====================================================

fig = plt.figure(

    figsize=(7.2,7.0),

    dpi=600

)



gs = fig.add_gridspec(

    2,

    2,

    width_ratios=[1,1],

    height_ratios=[1,1],

    wspace=0.28,

    hspace=0.32

)



axA = fig.add_subplot(gs[0,0])

axB = fig.add_subplot(gs[0,1])

axC = fig.add_subplot(gs[1,0])

axD = fig.add_subplot(gs[1,1])



# =====================================================
# PANEL A
# Spectral weathering state space
# =====================================================


for state in ORDER:

    subset = panel_A[
        panel_A["State"] == state
    ]


    x = subset["FDI"].values

    y = subset["SWI"].values



    # scatter points

    axA.scatter(

        x,

        y,

        s=8,

        alpha=0.35,

        color=COLORS[state],

        linewidth=0

    )



    # centroid

    axA.scatter(

        np.mean(x),

        np.mean(y),

        s=45,

        marker="o",

        color=COLORS[state],

        edgecolor="black",

        linewidth=0.5,

        zorder=5

    )



    # KDE contour

    if len(x) > 5:

        values=np.vstack([x,y])

        kde=gaussian_kde(values)



        xi,yi=np.mgrid[

            x.min():x.max():100j,

            y.min():y.max():100j

        ]


        zi=kde(

            np.vstack([

                xi.flatten(),

                yi.flatten()

            ])

        )


        axA.contour(

            xi,

            yi,

            zi.reshape(xi.shape),

            levels=3,

            colors=[COLORS[state]],

            linewidths=0.8,

            alpha=0.8

        )



axA.set_xlabel(

    "FDI (Floating Debris Index)"

)


axA.set_ylabel(

    "SWI (Spectral Weathering Index)"

)


axA.text(

    -0.15,

    1.05,

    "a",

    transform=axA.transAxes,

    fontweight="bold",

    fontsize=10

)



axA.legend(

    ORDER,

    frameon=False,

    fontsize=6,

    loc="best"

)



# =====================================================
# PANEL B
# Confusion matrix
# =====================================================


cm = panel_B.values


sns.heatmap(

    cm,

    annot=True,

    fmt=".0f",

    cmap="Reds",

    cbar=False,

    square=True,

    linewidths=0.4,

    ax=axB

)


axB.set_xlabel(

    "Predicted weathering state"

)


axB.set_ylabel(

    "True weathering state"

)


axB.set_xticklabels(

    ORDER,

    rotation=0

)


axB.set_yticklabels(

    ORDER,

    rotation=0

)


axB.text(

    -0.15,

    1.05,

    "b",

    transform=axB.transAxes,

    fontweight="bold",

    fontsize=10

)# =====================================================
# PANEL C
# SHAP feature importance
# =====================================================

# Sort features according to SHAP value

panel_C = panel_C.sort_values(

    by="SHAP",

    ascending=True

)


axC.barh(

    panel_C["Feature"],

    panel_C["SHAP"],

    height=0.55,

    color="#4C78A8"

)


axC.set_xlabel(

    "Mean |SHAP value|"

)


axC.set_ylabel("")


axC.text(

    -0.15,

    1.05,

    "c",

    transform=axC.transAxes,

    fontweight="bold",

    fontsize=10

)



# Remove unnecessary borders

axC.spines["top"].set_visible(False)

axC.spines["right"].set_visible(False)



# =====================================================
# PANEL D
# PtS probability distributions
# =====================================================


for state in ORDER:


    subset = panel_D[

        panel_D["State"] == state

    ]


    pts = subset["PtS"].dropna().values


    if len(pts) > 5:


        kde = gaussian_kde(pts)


        x_grid = np.linspace(

            pts.min(),

            pts.max(),

            300

        )


        y_grid = kde(x_grid)


        axD.plot(

            x_grid,

            y_grid,

            linewidth=1.8,

            color=COLORS[state],

            label=state

        )



        axD.fill_between(

            x_grid,

            y_grid,

            alpha=0.12,

            color=COLORS[state]

        )



# sinking threshold

axD.axvline(

    0.70,

    linestyle="--",

    linewidth=1,

    color="black"

)


axD.text(

    0.70,

    axD.get_ylim()[1]*0.85,

    "PtS = 0.70\nthreshold",

    fontsize=6,

    ha="left"

)



axD.set_xlabel(

    "Probability of sinking (PtS)"

)


axD.set_ylabel(

    "Density"

)



axD.legend(

    frameon=False,

    fontsize=6

)



axD.text(

    -0.15,

    1.05,

    "d",

    transform=axD.transAxes,

    fontweight="bold",

    fontsize=10

)



# =====================================================
# Global formatting
# =====================================================


for ax in [

    axA,

    axB,

    axC,

    axD

]:

    ax.tick_params(

        labelsize=6,

        width=0.5

    )



# =====================================================
# Export
# =====================================================


plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "Fig2_NatureGeoscience_600dpi.tiff"

    ),

    dpi=600,

    bbox_inches="tight"

)



plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "Fig2_NatureGeoscience.pdf"

    ),

    bbox_inches="tight"

)



plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "Fig2_preview.png"

    ),

    dpi=300,

    bbox_inches="tight"

)



plt.close()



print(

    "Figure 2 generated successfully"

)