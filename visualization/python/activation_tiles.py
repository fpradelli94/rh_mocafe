"""
Generates the tiles plot shown in the original manuscript by Pradelli F. et al.

Uses the `../../sim_index/sim_index.csv` table as input.

To regenerate:

    python3 activation_tiles.py

You might need to adjust the output image.

"""
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd
from pint import UnitRegistry


def load_arbitrary_units(
        local_ureg: UnitRegistry,
        parameters_df: pd.DataFrame,
        sau_name: str,
        tau_name: str,
        afau_name: str
):
    # define sau according to dataframe
    local_ureg.define(f"{sau_name} = "
                      f"{parameters_df.loc[sau_name, 'real_value']} * {parameters_df.loc[sau_name, 'real_um']} = "
                      f"sau")

    # define tau according to dataframe
    local_ureg.define(f"{tau_name} = "
                      f"{parameters_df.loc[tau_name, 'real_value']} * {parameters_df.loc[tau_name, 'real_um']} = "
                      f"tau")

    # define afau according to dataframe
    local_ureg.define(f"af concentration arbitrary unit = "
                      f"{parameters_df.loc[afau_name, 'real_value']} * {parameters_df.loc[afau_name, 'real_um']} = "
                      f"afau")

    return local_ureg


# load data
indata = pd.read_csv("./sim_index.csv")

# define ureg and load arbitrary units
ureg = UnitRegistry()
load_arbitrary_units(ureg, pd.read_csv("../../notebooks/out/g_parameters.csv", index_col="name"),
                     sau_name="Space Arbitrary Unit",
                     tau_name="Time Arbitrary Unit",
                     afau_name="T_s")

# convert colums
td_col_name, = [col_name for col_name in indata.columns if "tumor_diameter" in col_name]
td_unit = "um"
indata[td_col_name] = indata[td_col_name].map(lambda td: (td * ureg.sau).to(td_unit).magnitude)
D_af_col_name, = [col_name for col_name in indata.columns if "D_af" in col_name]
D_af_unit = "mm^2 / s"
indata[D_af_col_name] = indata[D_af_col_name].map(lambda D_af: (D_af * ureg("sau^2 / tau")).to(D_af_unit).magnitude)
V_pT_af_col_name, = [col_name for col_name in indata.columns if "V_pT" in col_name]
V_pt_af_unit = "pg /(mL * s)"
indata[V_pT_af_col_name] = indata[V_pT_af_col_name].map(
    lambda V_pt_af: (V_pt_af * ureg("afau / tau")).to(V_pt_af_unit).magnitude
)
V_uc_af_col_name, = [col_name for col_name in indata.columns if "V_uc" in col_name]
V_uc_af_unit = "1 / s"
indata[V_uc_af_col_name] = indata[V_uc_af_col_name].map(
    lambda V_uc_af: (V_uc_af * ureg("1 / tau")).to(V_uc_af_unit).magnitude
)

# get values for tumor diameter
td_values = indata[td_col_name].unique()
n_td_values = len(td_values)
# get col labels
col_labels = [fr"$d_0$ = {td:.6g} $\mu m$" for td in td_values]

# get values for af diffusivity
D_af_values = indata[D_af_col_name].unique()
n_D_af_values = len(D_af_values)
# get row labels
row_labels = [fr"$D_{{af}}$ = {D_af:.2e} [$mm^2$ / s]" for D_af in D_af_values]

# get values for V_pt
V_pT_af_values = indata[V_pT_af_col_name].unique()
n_V_pT_af_values = len(V_pT_af_values)
# get labels from V_pT_af_values
V_pT_af_labels = [f"{V_pT_af:.2e}" for V_pT_af in V_pT_af_values]

# get values for V_uc_af
V_uc_af_values = indata[V_uc_af_col_name].unique()
n_V_uc_af_values = len(V_uc_af_values)
# get labels from V_uc_af
V_uc_af_labels = [f"{V_uc_af:.2e}" for V_uc_af in V_uc_af_values]

# create subplot (rows = diffusivity, cols = diameter)
fig, axes = plt.subplots(nrows=n_D_af_values, ncols=n_td_values, figsize=(12, 8))

# create discrete colormap (snow = False; royalblue = True)
cmap = colors.ListedColormap(['snow', 'royalblue'])

# iterate on axes
for row in range(n_D_af_values):
    for col in range(n_td_values):
        # get ax
        ax = axes[row][col]
        # get corresponding td and D_af value
        D_af = D_af_values[row]
        td = td_values[col]
        # init boolean matrix
        tca_matrix = np.full((n_V_uc_af_values, n_V_pT_af_values), True)
        # check tip_cell_state
        for i, V_uc_af in enumerate(V_uc_af_values[::-1]):
            for j, V_pT_af in enumerate(V_pT_af_values):
                # get subset of indata
                subset = indata[np.isclose(indata[D_af_col_name], D_af) &
                                np.isclose(indata[td_col_name], td) &
                                np.isclose(indata[V_uc_af_col_name], V_uc_af) &
                                np.isclose(indata[V_pT_af_col_name], V_pT_af)]
                # check if tip cell is activated
                tca_matrix[i][j] = subset["tip_cell_activated"].values[0]
                


        # set labels
        ax.set_xticks([])
        ax.set_yticks([])
        pad = 5
        if col == 0:
            ax.set_ylabel(fr"$V_{{uc}} [{V_uc_af_unit}]$")
            ax.set_yticks(np.arange(n_V_uc_af_values))
            ax.set_yticklabels(V_uc_af_labels[::-1])
            ax.annotate(row_labels[row], xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center')
        if row == 0:
            ax.annotate(col_labels[col], xy=(0.5, 1), xytext=(0, pad),
                        xycoords='axes fraction', textcoords='offset points',
                        size='large', ha='center', va='baseline')
        if row == (n_D_af_values - 1):
            ax.set_xlabel(fr"$V_{{pT}}$ [{V_pt_af_unit}]")
            ax.set_xticks(np.arange(n_V_pT_af_values))
            ax.set_xticklabels(V_pT_af_labels)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")

        # set grid
        ax.set_xticks(np.arange(-0.5, n_V_pT_af_values, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_V_uc_af_values, 1), minor=True)
        ax.grid(which="minor", color="k", linestyle='-', linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)

        # set imshow
        im = ax.imshow(tca_matrix, cmap=cmap, vmin=0, vmax=1)

# set colorbax
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5)
cax = cbar.ax  # get axis
cax.get_yaxis().set_ticks([0.25, 0.75])  # place two middle ticks
cax.get_yaxis().set_ticklabels(["No Angiogenesis", "Angiogenesis"], rotation=90, va="center")
cax.set_position([0.9, 0.25, 0.1, 0.5])

fig.subplots_adjust(top=0.95, right=0.85)
plt.show()
