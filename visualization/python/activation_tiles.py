"""
Generates the tiles plot shown in the original manuscript by Pradelli F. et al.

Uses the `../../sim_index/sim_index.csv` table as input.

To regenerate:

    python3 activation_tiles.py

You might need to adjust the output image.

"""
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import colors
import matplotlib.pyplot as plt
from mocafe.fenut.parameters import Parameters
from src.simulation import get_ureg_with_arbitrary_units


logger = logging.getLogger(__name__)


def plot_activation_tiles(tip_cell_activation_csv: str, save_to: Path):
    # load data
    tip_cell_activation_csv_path = Path(tip_cell_activation_csv)
    indata = pd.read_csv(tip_cell_activation_csv)

    # load parameters
    sim_parameters = pd.read_csv(tip_cell_activation_csv_path.parent / Path("sim_info/sim_parameters.csv"),
                                 index_col="name")
    with open(tip_cell_activation_csv_path.parent / Path("sim_info/patient_parameters.json"), "r") as infile:
        patient_parameters = json.load(infile)

    # load arbitrary units
    ureg = get_ureg_with_arbitrary_units(Parameters(sim_parameters))

    # Convert each column to correct units
    # 1. tdf_i (converted to actual tumor volume)
    tdf_i_col_name, = [col_name for col_name in indata.columns if "tdf_i" in col_name]
    patient_tumor_lateral_semiax = ((patient_parameters["tumor_lateral_ax"]["value"] / 2) *
                                    ureg(patient_parameters["tumor_lateral_ax"]["mu"]))
    patient_tumor_axial_semiax = ((patient_parameters["tumor_axial_ax"]["value"] / 2) *
                                  ureg(patient_parameters["tumor_axial_ax"]["mu"]))
    patient_tumor_volume = (4 / 3) * np.pi * patient_tumor_axial_semiax * (patient_tumor_lateral_semiax ** 2)
    patient_tumor_volume_mu = "mm ** 3"
    patient_tumor_volume = patient_tumor_volume.to(patient_tumor_volume_mu)
    indata[tdf_i_col_name] = indata[tdf_i_col_name].map(lambda tdf: (tdf * patient_tumor_volume).magnitude)
    # 2. T_c
    T_c_col_name, = [col_name for col_name in indata.columns if "T_c" in col_name]
    T_c_unit = "pg / mL"
    indata[T_c_col_name] = indata[T_c_col_name].map(lambda T_c: (T_c * ureg("afau")).to(T_c_unit).magnitude)
    # 3. V_pT_af
    V_pT_af_col_name, = [col_name for col_name in indata.columns if "V_pT" in col_name]
    V_pt_af_unit = "pg /(mL * s)"
    indata[V_pT_af_col_name] = indata[V_pT_af_col_name].map(
        lambda V_pt_af: (V_pt_af * ureg("afau / tau")).to(V_pt_af_unit).magnitude
    )
    # 4. V_uc_af
    V_uc_af_col_name, = [col_name for col_name in indata.columns if "V_uc" in col_name]
    V_uc_af_unit = "1 / s"
    indata[V_uc_af_col_name] = indata[V_uc_af_col_name].map(
        lambda V_uc_af: (V_uc_af * ureg("1 / tau")).to(V_uc_af_unit).magnitude
    )

    # Get labels for plot
    # 1. tdf_i
    tv_values = indata[tdf_i_col_name].unique()
    n_tv_values = len(tv_values)
    # get col labels
    col_labels = [fr"$d_0$ = {tv:.2g} $mm^3$" for tv in tv_values]
    # 2. T_c
    T_c_values = indata[T_c_col_name].unique()
    n_T_c_values = len(T_c_values)
    # get row labels
    newline = "\n"
    row_labels = [fr"$T_{{c}}${newline}{T_c:.1e}{newline}[{T_c_unit}]" for T_c in T_c_values]
    # 3. V_pT_af
    V_pT_af_values = indata[V_pT_af_col_name].unique()
    n_V_pT_af_values = len(V_pT_af_values)
    # get labels from V_pT_af_values
    V_pT_af_labels = [f"{V_pT_af:.2e}" for V_pT_af in V_pT_af_values]
    # 4. V_uc_af
    V_uc_af_values = indata[V_uc_af_col_name].unique()
    n_V_uc_af_values = len(V_uc_af_values)
    # get labels from V_uc_af
    V_uc_af_labels = [f"{V_uc_af:.2e}" for V_uc_af in V_uc_af_values]

    # create subplot (rows = T_c values, cols = diameter)
    fig, axes = plt.subplots(nrows=n_T_c_values, ncols=n_tv_values, figsize=(12, 8))

    # create discrete colormap (snow = False; royalblue = True)
    cmap = colors.ListedColormap(['snow', 'royalblue'])

    # iterate on axes
    for row in range(n_T_c_values):
        for col in range(n_tv_values):
            # get ax
            ax = axes[row][col]
            # get corresponding td and D_af value
            T_c = T_c_values[row]
            tv = tv_values[col]
            # init boolean matrix
            tca_matrix = np.full((n_V_uc_af_values, n_V_pT_af_values), True)
            # check tip_cell_state
            for i, V_uc_af in enumerate(V_uc_af_values[::-1]):
                for j, V_pT_af in enumerate(V_pT_af_values):
                    # get subset of indata
                    subset = indata[np.isclose(indata[T_c_col_name], T_c) &
                                    np.isclose(indata[tdf_i_col_name], tv) &
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
            if row == (n_T_c_values - 1):
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

    fig.subplots_adjust(top=0.95, right=0.85, left=0.17)
    plt.savefig(save_to)


def plot_activation_tiles2(tip_cell_activation_csv: str, save_to: str):
    # load data
    tip_cell_activation_csv_path = Path(tip_cell_activation_csv)
    indata = pd.read_csv(tip_cell_activation_csv)

    # load parameters
    sim_parameters = pd.read_csv(tip_cell_activation_csv_path.parent / Path("sim_info/sim_parameters.csv"),
                                 index_col="name")
    with open(tip_cell_activation_csv_path.parent / Path("sim_info/patient_parameters.json"), "r") as infile:
        patient_parameters = json.load(infile)

    # load arbitrary units
    ureg = get_ureg_with_arbitrary_units(Parameters(sim_parameters))

    # Convert each column to correct units
    # 1. V_pT_af
    V_pT_af_col_name, = [col_name for col_name in indata.columns if "V_pT" in col_name]
    V_pt_af_unit = "pg /(mL * s)"
    indata[V_pT_af_col_name] = indata[V_pT_af_col_name].map(
        lambda V_pt_af: (V_pt_af * ureg("afau / tau")).to(V_pt_af_unit).magnitude
    )
    # 2. V_uc_af
    V_uc_af_col_name, = [col_name for col_name in indata.columns if "V_uc" in col_name]
    V_uc_af_unit = "1 / s"
    indata[V_uc_af_col_name] = indata[V_uc_af_col_name].map(
        lambda V_uc_af: (V_uc_af * ureg("1 / tau")).to(V_uc_af_unit).magnitude
    )

    # Get labels for plot
    # 1. V_pT_af
    V_pT_af_values = indata[V_pT_af_col_name].unique()
    n_V_pT_af_values = len(V_pT_af_values)
    # get labels from V_pT_af_values
    V_pT_af_labels = [f"{V_pT_af:.2e}" for V_pT_af in V_pT_af_values]
    # 2. V_uc_af
    V_uc_af_values = indata[V_uc_af_col_name].unique()
    n_V_uc_af_values = len(V_uc_af_values)
    # get labels from V_uc_af
    V_uc_af_labels = [f"{V_uc_af:.2e}" for V_uc_af in V_uc_af_values]

    # create subplot (rows = T_c values, cols = diameter)
    n_grid_rows = 1
    n_grid_cols = 1
    fig, ax = plt.subplots(nrows=n_grid_rows, ncols=n_grid_cols, figsize=(12, 8))

    # create discrete colormap (snow = False; royalblue = True)
    cmap = colors.ListedColormap(['snow', 'royalblue'])

    # init boolean matrix
    tca_matrix = np.full((n_V_uc_af_values, n_V_pT_af_values), True)
    # check tip_cell_state
    for i, V_uc_af in enumerate(V_uc_af_values[::-1]):
        for j, V_pT_af in enumerate(V_pT_af_values):
            # get subset of indata
            subset = indata[np.isclose(indata[V_uc_af_col_name], V_uc_af) &
                            np.isclose(indata[V_pT_af_col_name], V_pT_af)]
            # check if tip cell is activated
            tca_matrix[i][j] = subset["tip_cell_activated"].values[0]

    # set labels
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_ylabel(fr"$V_{{uc}} [{V_uc_af_unit}]$")
    ax.set_yticks(np.arange(n_V_uc_af_values))
    ax.set_yticklabels(V_uc_af_labels[::-1])

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
    cbar = fig.colorbar(im, ax=ax, shrink=0.5)
    cax = cbar.ax  # get axis
    cax.get_yaxis().set_ticks([0.25, 0.75])  # place two middle ticks
    cax.get_yaxis().set_ticklabels(["No Angiogenesis", "Angiogenesis"], rotation=90, va="center")
    cax.set_position([0.9, 0.25, 0.1, 0.5])

    fig.subplots_adjust(top=0.95, right=0.85, left=0.17)
    plt.savefig(save_to)
