"""
Code used only in 3D simulations
"""
import fenics
import pandas as pd
import pint
import numpy as np
from PIL import Image
from typing import Dict
import skimage.io as io
from mocafe.fenut.parameters import Parameters
from src.expressions import Vessel3DReconstruction

# get process rank
comm_world = fenics.MPI.comm_world
rank = comm_world.Get_rank()


def compute_3d_mesh_for_patient(
        parameters_df: pd.DataFrame,
        local_ureg: pint.UnitRegistry,
        patient_parameters: Dict
):
    """
    Generate a ``BoxMesh`` suitable for the simulation, given the patient parameters listed in the given
    dictionary

    :param parameters_df: parameters dataframe
    :param local_ureg: Unit registry
    :param patient_parameters: parameters for the patient
    :return: the computed mesh and a Dict containing the mesh data
    """
    # load Lx and Ly form patient_parameters
    Lx_real: pint.Quantity = float(patient_parameters["simulation_box"]["Lx"]["value"]) \
                             * local_ureg(patient_parameters["simulation_box"]["Lx"]["mu"])
    Ly_real: pint.Quantity = float(patient_parameters["simulation_box"]["Ly"]["value"]) \
                             * local_ureg(patient_parameters["simulation_box"]["Ly"]["mu"])
    Lz_real: pint.Quantity = float(patient_parameters["simulation_box"]["Lz"]["value"]) \
                             * local_ureg(patient_parameters["simulation_box"]["Lz"]["mu"])

    # convert Lx and Ly to sau
    Lx: pint.Quantity = Lx_real.to("sau")
    Ly: pint.Quantity = Ly_real.to("sau")
    Lz: pint.Quantity = Lz_real.to("sau")

    # compute nx and ny based on R_c size
    R_c: float = parameters_df.loc['R_c', 'sim_value']
    nx: int = int(np.floor(Lx.magnitude / (R_c*0.7)))
    ny: int = int(np.floor(Ly.magnitude / (R_c*0.7)))
    nz: int = int(np.floor(Lz.magnitude / (R_c*0.7)))

    # define mesh
    mesh: fenics.BoxMesh = fenics.BoxMesh(comm_world,
                                          fenics.Point(0., 0., 0.),
                                          fenics.Point(Lx.magnitude, Ly.magnitude, Lz.magnitude),
                                          nx, ny, nz)

    # define mesh parameters
    mesh_parameters: Dict = {
        "name": ["Lx",
                 "Ly",
                 "Lz",
                 "nx",
                 "ny",
                 "nz"],
        "info": [f"Mesh x length based on input image (estimated from tumor diameter)",
                 f"Mesh y length based on input image (estimated from tumor diameter)",
                 f"Mesh z length based on input image (estimated from tumor diameter)",
                 "n elements along the x axes (each element is about R_c / 2 long)",
                 "n elements along the y axes (each element is about R_c / 2 long)",
                 "n elements along the z axes (each element is about R_c / 2 long)"],
        "real_value": [Lx_real.magnitude,
                       Ly_real.magnitude,
                       Lz_real.magnitude,
                       None,
                       None,
                       None],
        "real_um": [Lx_real.units,
                    Ly_real.units,
                    Lz_real.units,
                    None,
                    None,
                    None],
        "sim_value": [Lx.magnitude,
                      Ly.magnitude,
                      Lz.magnitude,
                      nx,
                      ny,
                      nz],
        "sim_um": ["sau",
                   "sau",
                   "sau",
                   None,
                   None,
                   None],
        "references": [None,
                       None,
                       None,
                       None,
                       None,
                       None]
    }

    # return both
    return mesh, Parameters(pd.DataFrame(mesh_parameters))


def compute_3d_c_0(c_old: fenics.Function,
                   mesh_parameters: Parameters):
    """
    Computes c initial condition in 3D
    """
    # load images
    binary = io.imread("./notebooks/out/RH_vessels_binary_ND.png")
    skeleton = io.imread("./notebooks/out/RH_vessels_skeleton_ND.png")
    edges = io.imread("./notebooks/out/RH_vessels_edges_ND.png")
    half_depth = 0.1
    c_old_expression = Vessel3DReconstruction(z_0=mesh_parameters.get_value("Lz") - half_depth,
                                              skeleton_array=skeleton,
                                              edge_array=edges,
                                              binary_array=binary,
                                              mesh_Lx=mesh_parameters.get_value("Lx"),
                                              mesh_Ly=mesh_parameters.get_value("Ly"),
                                              half_depth=half_depth)
    del binary, skeleton, edges

    c_old.interpolate(c_old_expression)
