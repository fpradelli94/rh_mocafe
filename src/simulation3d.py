"""
Code used only in 3D simulations
"""
import fenics
import pandas as pd
import pint
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple
import skimage.io as io
from mocafe.fenut.parameters import Parameters
from src.expressions import Vessel3DReconstruction

# get process rank
comm_world = fenics.MPI.comm_world
rank = comm_world.Get_rank()

# get logger
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
formatter = logging.Formatter(f"p{rank}:%(name)s:%(levelname)s: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


def get_3d_mesh_for_patient(
        parameters_df: pd.DataFrame,
        local_ureg: pint.UnitRegistry,
        patient_parameters: Dict,
        mesh_file: Path,
        mesh_parameters_file: Path,
        recompute_mesh) -> Tuple[fenics.BoxMesh, Parameters]:
    """
    Get a ``BoxMesh`` suitable for the simulation, given the patient parameters listed in the given
    dictionary.

    Under the hood, it generates the mesh if it does not exist in the `mesh_file` otherwise, it loads it
    directly from the mesh file.

    :param parameters_df: parameters dataframe
    :param local_ureg: Unit registry
    :param patient_parameters: parameters for the patient
    :param mesh_file: the mesh file. It is empty if the mesh has never been computed.
    :param mesh_parameters_file: the file containing the mesh_parameters. It is empty if the mesh has never been
    computed.
    :param recompute_mesh: set to True if it is necessary to recompute the mesh, even if the mesh_file contains a mesh.
    :return: the computed mesh and a Dict containing the mesh data
    """

    # if mesh_file and mesh_parameters_file exist, and it is not necesssary to recompute the mesh,
    # just load the mesh parameters. Else, compute the mesh and store it.
    if (mesh_file.exists() and mesh_parameters_file.exists()) and (recompute_mesh is False):
        # load mesh parameters
        mesh_parameters: Parameters = Parameters(pd.read_csv(mesh_parameters_file))

    else:
        # generate mesh
        logger.info(f"Computing mesh ...")
        mesh, mesh_parameters = _compute_3d_mesh_for_patient(parameters_df,
                                                             local_ureg,
                                                             patient_parameters)
        # save mesh
        with fenics.XDMFFile(str(mesh_file.resolve())) as outfile:
            outfile.write(mesh)

        # save mesh parameters
        mesh_parameters.as_dataframe().to_csv(mesh_parameters_file)

        # delete mesh
        del mesh

    # in any case, reload the mesh (to ensure consistent distribution of the dofs)
    mesh = fenics.Mesh()
    with fenics.XDMFFile(str(mesh_file.resolve())) as infile:
        infile.read(mesh)

    return mesh, mesh_parameters


def _compute_3d_mesh_for_patient(
        parameters_df: pd.DataFrame,
        local_ureg: pint.UnitRegistry,
        patient_parameters: Dict) -> Tuple[fenics.BoxMesh, Parameters]:
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


def get_3d_c0(c_old: fenics.Function,
              mesh_parameters: Parameters,
              c0_xdmf: Path,
              recompute_c0: bool) -> None:
    """
    Get the 3d initial condition for c0. If c0_xdmf is empty, it computes the initial condition and saves it in the
    file for future reuse.

    :param c_old: FEniCS function to store initial condition
    :param mesh_parameters: mesh parameters (used in the computation)
    :param c0_xdmf: file containing c0 (if already computed once)
    :param recompute_c0: force the function to recompute c0 even if c0_xdmf is not empty.
    """

    # define c0_label in xdmf file
    c0_label = "c0"

    # if c0 file exists and it is not necessary to recompute it, load it. Else compute it.
    if c0_xdmf.exists() and (recompute_c0 is False):
        with fenics.XDMFFile(str(c0_xdmf.resolve())) as infile:
            infile.read_checkpoint(c_old, c0_label, 0)
    else:
        # compute c0
        logger.info("Computing c0...")
        _compute_3d_c_0(c_old, mesh_parameters)
        # store c0
        with fenics.XDMFFile(str(c0_xdmf.resolve())) as outfile:
            outfile.write_checkpoint(c_old, c0_label, 0, fenics.XDMFFile.Encoding.HDF5, False)


def _compute_3d_c_0(c_old: fenics.Function,
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
