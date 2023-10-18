"""
File containing the FEniCS Expressions used throughout the simulation
"""
import fenics
import numpy as np
import pint
from typing import Dict
from PIL import Image
from mocafe.math import sigmoid
from mocafe.fenut.parameters import Parameters


def get_growing_RH_expression(sim_parameters: Parameters,
                              patient_parameters: Dict,
                              mesh_parameters: Parameters,
                              local_ureg: pint.UnitRegistry,
                              initial_t: float,
                              spatial_dimension: int) -> fenics.Expression:
    """
    Returns the FEniCS expression of the growing RH, modeled as a growing ellipsoid. The expression is written in c++
    to improve efficiency.

    :param sim_parameters: simulation parameters.
    :param mesh_parameters: mesh parameters.
    :param initial_t: time at which the tumor is at the ``min_tumor_diameter`` (initial condition)
    :param spatial_dimension: 2 if 2D, 3 if 3D.
    """
    # get patient axes for the full grown tumor
    lateral_ax: pint.Quantity = float(patient_parameters["tumor_lateral_ax"]["value"]) \
                                * local_ureg(patient_parameters["tumor_lateral_ax"]["mu"])
    axial_ax: pint.Quantity = float(patient_parameters["tumor_axial_ax"]["value"]) \
                              * local_ureg(patient_parameters["tumor_axial_ax"]["mu"])

    # convert to sau
    lateral_ax = lateral_ax.to("sau")
    axial_ax = axial_ax.to("sau")

    # compute initial semiaxes
    tdf_i = sim_parameters.get_value("tdf_i")
    semiax_x0 = (tdf_i ** (1/3)) * (lateral_ax.magnitude / 2)
    semiax_y0 = (tdf_i ** (1/3)) * (lateral_ax.magnitude / 2)
    semiax_z0 = (tdf_i ** (1/3)) * (axial_ax.magnitude / 2)

    # write cpp code for time variant semiaxes
    cpp_semiax_x = "(sa_xi * pow(tgr, t / 3))"
    cpp_semiax_y = "(sa_yi * pow(tgr, t / 3))"
    cpp_semiax_z = "(sa_zi * pow(tgr, t / 3))"

    # derive cpp code for center
    cpp_cx = "(Lx / 2)"
    cpp_cy = "(Ly / 2)"
    # build code for ellipsoid
    if spatial_dimension == 2:
        cpp_ellipse = f"((pow(((x[0] - {cpp_cx}) / {cpp_semiax_x}), 2) + " \
                      f"  pow(((x[1] - {cpp_cy}) / {cpp_semiax_y}), 2)) <= 1.) ? 1. : 0."
        rh_exp = fenics.Expression(cpp_ellipse,
                                   degree=1,
                                   sa_xi=semiax_x0,
                                   sa_yi=semiax_y0,
                                   tgr=sim_parameters.get_value("tgr"),
                                   t=initial_t,
                                   Lx=mesh_parameters.get_value("Lx"),
                                   Ly=mesh_parameters.get_value("Ly"))
    elif spatial_dimension == 3:
        cpp_cz = f"(Lz - {cpp_semiax_z})"
        cpp_ellipsoid = f"((pow(((x[0] - {cpp_cx}) / {cpp_semiax_x}), 2) + " \
                        f"  pow(((x[1] - {cpp_cy}) / {cpp_semiax_y}), 2) + " \
                        f"  pow(((x[2] - {cpp_cz}) / {cpp_semiax_z}), 2)) <= 1.) ? 1. : 0."
        rh_exp = fenics.Expression(cpp_ellipsoid,
                                   degree=1,
                                   sa_xi=semiax_x0,
                                   sa_yi=semiax_y0,
                                   sa_zi=semiax_z0,
                                   tgr=sim_parameters.get_value("tgr"),
                                   t=initial_t,
                                   Lx=mesh_parameters.get_value("Lx"),
                                   Ly=mesh_parameters.get_value("Ly"),
                                   Lz=mesh_parameters.get_value("Lz"))
    else:
        raise RuntimeError(f"This expression can be computed for spatial dimension 2 (2D) or 3 (3D). "
                           f"Spatial dimension {spatial_dimension} was given.")
    return rh_exp


class BWImageExpression(fenics.UserExpression):
    """
    Generates the 2D putative initial capillay network for 2D simulations.
    """
    def __init__(self,
                 image_path,
                 value_range,
                 local_ureg: pint.UnitRegistry):
        super(BWImageExpression, self).__init__()
        # get image as numpy array
        image_array = np.array(Image.open(image_path).convert('L'))
        # rescale numpy array
        max_value = max(value_range)
        min_value = min(value_range)
        image_array = (image_array / np.amax(image_array)) * (max_value - min_value) + min_value
        # save as attribute
        self.image_array = image_array
        # get image array shape
        self.image_array_shape = image_array.shape
        # get local registry
        self.ureg = local_ureg

    def eval(self, values, x):
        # convert to sau
        x_sau = x * self.ureg.sau
        # convert to pixel side
        x_pxls = x_sau.to("pxls")
        # get i coordinate for the corresponding pixel
        i = self.image_array_shape[0] - 1 - int(np.round(x_pxls[1].magnitude))
        i = int(i) if i >= 0 else 0
        # get j coordinate for the corresponding pixel
        j = int(np.round(x_pxls[0].magnitude))

        # compute value
        values[0] = self.image_array[i][j]

    def value_shape(self):
        return ()


class Vessel3DReconstruction(fenics.UserExpression):
    """
    Generate the 3D reconstruction of the putative initial capillary network.
    """
    def __init__(self,
                 z_0: float,
                 skeleton_array: np.ndarray,
                 edge_array: np.ndarray,
                 binary_array: np.ndarray,
                 mesh_Lx,
                 mesh_Ly,
                 half_depth):
        super(Vessel3DReconstruction, self).__init__()
        # save z0 value
        self.z_0 = z_0

        # save Lx and Ly
        self.mesh_Lx = mesh_Lx
        self.mesh_Ly = mesh_Ly
        self.half_depth = half_depth

        # check if shape of arrays are equal
        if np.any(skeleton_array.shape != edge_array.shape) and np.any(binary_array.shape != edge_array.shape):
            raise RuntimeError("skeleton, binary, and edge array mush have the same shape.")

        # check if image ratio is preserved
        if not np.isclose(mesh_Lx / mesh_Ly, binary_array.shape[1] / binary_array.shape[0]):
            raise RuntimeError("Mesh and image must have the same ratio")

        # get number of pixel
        self.n_p_x = skeleton_array.shape[1]
        self.n_p_y = skeleton_array.shape[0]

        # get center points
        normalized_skeleton_array = skeleton_array / np.amax(skeleton_array)
        self.center_points = self._get_ones_spatial_coordinates(normalized_skeleton_array)

        # get edge points
        normalized_edge_array = edge_array / np.amax(edge_array)
        self.edge_points = self._get_ones_spatial_coordinates(normalized_edge_array)

        # save binary array
        self.binary_array = binary_array / np.amax(binary_array)

    def _get_ones_spatial_coordinates(self, input_image: np.ndarray):
        """
        Given an input image, returns the spatial coordinates of the pixels equal to one.
        By default, the spatial dimension of the input image is x: [0, 1] ; y: [0, 1]

        :return: list of spatial coordinates
        """
        # get non zero points indices
        nonzero_indices = input_image.nonzero()
        # `convert indices to spatial coord
        sc_list = [
            [self.mesh_Lx * (ind_x / self.n_p_x), self.mesh_Ly * (1 - (ind_y / self.n_p_y)), self.z_0]
            for ind_x, ind_y in zip(nonzero_indices[1], nonzero_indices[0])
        ]
        # convert spatial coordinates to ndarray
        sc_array = np.array(sc_list)
        return sc_array

    def eval(self, values, x):
        # check if point is out of depth
        if (self.z_0 - self.half_depth) < x[2] < (self.z_0 + self.half_depth):
            # check if point correspond to 1 in binary array
            j = int(np.floor((x[0] / self.mesh_Lx) * (self.n_p_x - 1)))
            i = (self.n_p_y - 1) - int(np.floor((x[1] / self.mesh_Ly) * (self.n_p_y - 1)))

            if np.isclose(self.binary_array[i, j], 1.):
                # get x distances from center points
                distance_to_c = np.sqrt(np.sum((x - self.center_points) ** 2, axis=1))
                # get min distance and indices of the closest c
                distance_to_closest_c = np.amin(distance_to_c)
                closest_c_index = np.argmin(distance_to_c)
                # get closest c
                closest_c = self.center_points[closest_c_index]

                # get x distance from edge points
                distance_to_e = np.sqrt(np.sum((x - self.edge_points) ** 2, axis=1))
                # get index of the closest edge point
                closest_e_index = np.argmin(distance_to_e)
                # get closest edge point
                closest_e = self.edge_points[closest_e_index]

                # distance between closest c and closest e
                distance_ce = np.sqrt(np.sum((closest_e - closest_c) ** 2))

                # set value accordingly
                values[0] = sigmoid(distance_to_closest_c, distance_ce, 1., -1., 100)
            else:
                values[0] = -1.
        else:
            values[0] = -1.

    def value_shape(self):
        return ()
