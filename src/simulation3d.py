"""
Code used only in 3D simulations
"""
import dolfinx
from mpi4py import MPI
import pandas as pd
import pint
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple
from mocafe.fenut.parameters import Parameters
from src.ioutils import read_parameters

# get process rank
comm_world = MPI.COMM_WORLD
rank = comm_world.rank

# get logger
logger = logging.getLogger(__name__)


def get_3d_mesh_for_patient(
        parameters_df: pd.DataFrame,
        local_ureg: pint.UnitRegistry,
        patient_parameters: Dict,
        mesh_file: Path,
        mesh_parameters_file: Path,
        recompute_mesh: bool) -> Tuple[dolfinx.mesh.Mesh, Parameters]:
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

    # if mesh_file and mesh_parameters_file exist, and it is not necessary to recompute the mesh,
    # just load the mesh parameters. Else, compute the mesh and store it.
    if mesh_file.exists() and mesh_parameters_file.exists() and (recompute_mesh is False):
        # load mesh parameters
        mesh_parameters: Parameters = read_parameters(mesh_parameters_file)
        logger.info(f"Found existing mesh.")

    else:
        # generate mesh
        logger.info(f"Computing mesh ...")
        mesh, mesh_parameters = _compute_3d_mesh_for_patient(parameters_df,
                                                             local_ureg,
                                                             patient_parameters)
        # # save mesh
        # with fenics.XDMFFile(str(mesh_file.resolve())) as outfile:
        #     outfile.write(mesh)
        #
        # # save mesh parameters
        # write_parameters(mesh_parameters, mesh_parameters_file)
        #
        # # delete mesh
        # del mesh

    # # in any case, reload the mesh (to ensure consistent distribution of the dofs)
    # logger.info(f"Loading mesh...")
    # mesh = fenics.Mesh()
    # with fenics.XDMFFile(str(mesh_file.resolve())) as infile:
    #     infile.read(mesh)

    return mesh, mesh_parameters


def _compute_3d_mesh_for_patient(
        parameters_df: pd.DataFrame,
        local_ureg: pint.UnitRegistry,
        patient_parameters: Dict) -> Tuple[dolfinx.mesh.Mesh, Parameters]:
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
    Lx: float = Lx_real.to("sau").magnitude
    Ly: float = Ly_real.to("sau").magnitude
    Lz: float = Lz_real.to("sau").magnitude

    # compute nx and ny based on R_c size
    R_c: float = parameters_df.loc['R_c', 'sim_value']
    nx: int = int(np.floor(Lx / (R_c*0.7)))
    ny: int = int(np.floor(Ly / (R_c*0.7)))
    nz: int = int(np.floor(Lz / (R_c*0.7)))

    # define mesh
    mesh = dolfinx.mesh.create_box(comm_world,
                                   points=[[0., 0., 0.], [Lx, Ly, Lz]],
                                   n=[nx, ny, nz])

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
        "sim_value": [Lx,
                      Ly,
                      Lz,
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
