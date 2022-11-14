"""
Code used only in 3D simulations
"""
import fenics
import pandas as pd
import pint
import numpy as np
from PIL import Image
import skimage.io as io
from mocafe.fenut.parameters import Parameters
from src.expressions import Vessel3DReconstruction

# get process rank
comm_world = fenics.MPI.comm_world
rank = comm_world.Get_rank()


def compute_3d_mesh_from_image(
        parameters_df: pd.DataFrame,
        local_ureg: pint.UnitRegistry,
        image_path: str,
        tumor_diameter: pint.Quantity
):
    """
    Generate a ``BoxMesh`` suitable for the simulation, estimating the dimension from the input image (stored at the
    image path), given that the pixel size is contained in the unit registry.

    :param parameters_df: parameters dataframe
    :param local_ureg: Unit registry
    :param image_path: image to use as basis for the mesh
    :param tumor_diameter: diameter of the tumor
    :return: the computed mesh and a ``Parameters`` object containing the mesh data
    """
    # get tumor image size
    tumor_image_size = Image.open(image_path).convert('L').size

    # convert size to space units
    image_size = tumor_image_size * local_ureg.pxls
    image_size = image_size.to("um")

    # compute Lx and Ly based on image size
    sau = local_ureg(f"{parameters_df.loc['Space Arbitrary Unit', 'real_value']} "
                     f"{parameters_df.loc['Space Arbitrary Unit', 'real_um']}")
    Lx = float(f"{image_size[1].magnitude / sau.magnitude:.2g}")  # length of x side
    Ly = float(f"{image_size[0].magnitude / sau.magnitude:.2g}")  # length of y side

    # compute Lz based on tumor diameter
    Lz = 2 * tumor_diameter.to("sau").magnitude

    # compute nx and ny based on R_c size
    R_c = parameters_df.loc['R_c', 'sim_value']
    nx = int(np.floor(Lx / (R_c*0.7)))
    ny = int(np.floor(Ly / (R_c*0.7)))
    nz = int(np.floor(Lz / (R_c*0.7)))

    # define mesh
    mesh = fenics.BoxMesh(comm_world,
                          fenics.Point(0., 0., 0.),
                          fenics.Point(Lx, Ly, Lz),
                          nx, ny, nz)

    # define mesh parameters
    mesh_parameters = {
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
        "real_value": [image_size[1].magnitude,
                       image_size[0].magnitude,
                       (Lz * local_ureg("sau")).to("um"),
                       None,
                       None,
                       None],
        "real_um": [image_size[1].units,
                    image_size[0].units,
                    image_size[1].units,
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


def compute_3d_c_0(c_old: fenics.Function,
                   mesh_parameters: Parameters,
                   no_discontinuities: bool = False):
    """
    Computes c initial condition in 3D
    """
    # load images
    if no_discontinuities:
        binary = io.imread("./notebooks/out/RH_vessels_binary_ND.png")
        skeleton = io.imread("./notebooks/out/RH_vessels_skeleton_ND.png")
        edges = io.imread("./notebooks/out/RH_vessels_edges_ND.png")
    else:
        binary = io.imread("./notebooks/out/RH_vessels_binary.png")
        skeleton = io.imread("./notebooks/out/RH_vessels_skeleton.png")
        edges = io.imread("./notebooks/out/RH_vessels_edges.png")
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
