"""
Code used only in 2D simulations
"""

import fenics
import pandas as pd
import pint
import numpy as np
from PIL import Image
from mocafe.fenut.parameters import Parameters

# get process rank
comm_world = fenics.MPI.comm_world
rank = comm_world.Get_rank()


def compute_2d_mesh_from_image(
        parameters_df: pd.DataFrame,
        local_ureg: pint.UnitRegistry,
        image_path: str
):
    """
    Generate a ``RectangleMesh`` suitable for the simulation, estimating the dimension from the input image (stored at
    the image path), given that the pixel size is contained in the unit registry.

    :param parameters_df: parameters dataframe
    :param local_ureg: Unit registry
    :param image_path: image to use as basis for the mesh
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

    # compute nx and ny based on R_c size
    R_c = parameters_df.loc['R_c', 'sim_value']
    nx = int(np.floor(Lx / (R_c / 2)))  # each element should be about R_c / 2
    ny = int(np.floor(Ly / (R_c / 2)))

    # define mesh
    mesh = fenics.RectangleMesh(fenics.Point(0., 0.),
                                fenics.Point(Lx, Ly),
                                nx,
                                ny)

    # define mesh parameters
    mesh_parameters = {
        "name": ["Lx",
                 "Ly",
                 "nx",
                 "ny"],
        "info": [f"Mesh x length based on input image (estimated from tumor diameter)",
                 f"Mesh x length based on input image (estimated from tumor diameter)",
                 "n elements along the x axes (each element is about R_c / 2 long)",
                 "n elements along the y axes (each element is about R_c / 2 long)"],
        "real_value": [image_size[1].magnitude,
                       image_size[0].magnitude,
                       None,
                       None],
        "real_um": [image_size[1].units,
                    image_size[0].units,
                    None,
                    None],
        "sim_value": [Lx,
                      Ly,
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
