"""
Code used only in 2D simulations
"""

import fenics
import pandas as pd
import pint
import numpy as np
from typing import Dict
from mocafe.fenut.parameters import Parameters

# get process rank
comm_world = fenics.MPI.comm_world
rank = comm_world.Get_rank()


def compute_2d_mesh_for_patient(
        parameters_df: pd.DataFrame,
        local_ureg: pint.UnitRegistry,
        patient_parameters: Dict
):
    """
    Generate a ``RectangleMesh`` suitable for the simulation, given the patient parameters listed in the given
    dictionary.

    :param parameters_df: simulation parameters dataframe
    :param local_ureg: Unit registry
    :param patient_parameters: parameters for the patient
    :return: the computed mesh and a Dict object containing the mesh data
    """
    # load Lx and Ly form patient_parameters
    Lx_real: pint.Quantity = float(patient_parameters["simulation_box"]["Lx"]["value"]) \
                             * local_ureg(patient_parameters["simulation_box"]["Lx"]["mu"])
    Ly_real: pint.Quantity = float(patient_parameters["simulation_box"]["Ly"]["value"]) \
                             * local_ureg(patient_parameters["simulation_box"]["Ly"]["mu"])

    # convert Lx and Ly to sau
    Lx: pint.Quantity = Lx_real.to("sau")
    Ly: pint.Quantity = Ly_real.to("sau")

    # compute nx and ny based on R_c size
    R_c: float = parameters_df.loc['R_c', 'sim_value']
    nx: int = int(np.floor(Lx.magnitude / (R_c / 2)))  # each element should be about R_c / 2
    ny: int = int(np.floor(Ly.magnitude / (R_c / 2)))

    # define mesh
    mesh: fenics.RectangleMesh = fenics.RectangleMesh(fenics.Point(0., 0.),
                                                      fenics.Point(Lx.magnitude, Ly.magnitude),
                                                      nx,
                                                      ny)

    # define mesh parameters
    mesh_parameters: Dict = {
        "name": ["Lx",
                 "Ly",
                 "nx",
                 "ny"],
        "info": [f"Mesh x length based on input image (estimated from tumor diameter)",
                 f"Mesh x length based on input image (estimated from tumor diameter)",
                 "n elements along the x axes (each element is about R_c / 2 long)",
                 "n elements along the y axes (each element is about R_c / 2 long)"],
        "real_value": [Lx_real.magnitude,
                       Ly_real.magnitude,
                       None,
                       None],
        "real_um": [Lx_real.units,
                    Ly_real.units,
                    None,
                    None],
        "sim_value": [Lx.magnitude,
                      Ly.magnitude,
                      nx,
                      ny],
        "sim_um": ["sau",
                   "sau",
                   None,
                   None],
        "references": [None,
                       None,
                       None,
                       None]
    }

    # return both
    return mesh, Parameters(pd.DataFrame(mesh_parameters))
