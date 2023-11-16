"""
File containing the FEniCS Expressions used throughout the simulation
"""
import pint
import logging
import numpy as np
from typing import Dict
from mocafe.fenut.parameters import Parameters
from skimage import morphology


logger = logging.getLogger(__name__)


class RHEllipsoid:
    def __init__(self,
                 sim_parameters: Parameters,
                 patient_parameters: Dict,
                 mesh_parameters: Parameters,
                 local_ureg: pint.UnitRegistry,
                 t: float,
                 spatial_dimension: int):
        """
        Class encapsulating the dolfinx expression of the growing RH, modeled as a growing ellipsoid.

        :param sim_parameters: simulation parameters.
        :param patient_parameters: patient parameters
        :param mesh_parameters: mesh parameters.
        :param local_ureg:UnitRegistry
        :param t: time at which the tumor is at the ``min_tumor_diameter`` (initial condition)
        :param spatial_dimension: 2 if 2D, 3 if 3D.
        """
        # get patient axes for the full-grown tumor
        lateral_ax: pint.Quantity = float(patient_parameters["tumor_lateral_ax"]["value"]) \
                                    * local_ureg(patient_parameters["tumor_lateral_ax"]["mu"])
        axial_ax: pint.Quantity = float(patient_parameters["tumor_axial_ax"]["value"]) \
                                  * local_ureg(patient_parameters["tumor_axial_ax"]["mu"])
        # convert to sau
        lateral_ax = lateral_ax.to("sau")
        axial_ax = axial_ax.to("sau")

        # compute initial semiaxes
        tdf_i = sim_parameters.get_value("tdf_i")
        self.semiax_x0 = (tdf_i ** (1 / 3)) * (lateral_ax.magnitude / 2)
        self.semiax_y0 = (tdf_i ** (1 / 3)) * (lateral_ax.magnitude / 2)
        self.semiax_z0 = (tdf_i ** (1 / 3)) * (axial_ax.magnitude / 2)

        # compute length of semiaxes considering time
        self.tgr = sim_parameters.get_value("tgr")
        self.semiax_x = self.semiax_x0 * (self.tgr ** (t / 3))
        self.semiax_y = self.semiax_y0 * (self.tgr ** (t / 3))
        self.semiax_z = self.semiax_z0 * (self.tgr ** (t / 3))

        # compute center of ellipsoid
        Lx = mesh_parameters.get_value("Lx")
        self.cx = Lx / 2
        Ly = mesh_parameters.get_value("Ly")
        self.cy = Ly / 2
        if spatial_dimension == 3:
            self.Lz = mesh_parameters.get_value("Lz")
            self.cz = self.Lz - self.semiax_z
        else:
            self.Lz = 0
            self.cz = 0

    def eval(self, x):
        ellipsoid = ((((x[0] - self.cx) / self.semiax_x) ** 2)
                     + (((x[1] - self.cy) / self.semiax_y) ** 2)
                     + (((x[2] - self.cz) / self.semiax_z) ** 2)
                     - 1)
        return np.where(ellipsoid <= 0, 1., 0.)

    def update_time(self, t: float):
        # update value of semiaxes
        self.semiax_x = self.semiax_x0 * (self.tgr ** (t / 3))
        self.semiax_y = self.semiax_y0 * (self.tgr ** (t / 3))
        self.semiax_z = self.semiax_z0 * (self.tgr ** (t / 3))
        # update value of cz
        self.cz = self.Lz - self.semiax_z


# class BWImageExpression(fenics.UserExpression):
#     """
#     Generates the 2D putative initial capillay network for 2D simulations.
#     """
#     def __init__(self,
#                  image_path,
#                  value_range,
#                  local_ureg: pint.UnitRegistry):
#         super(BWImageExpression, self).__init__()
#         # get image as numpy array
#         image_array = np.array(Image.open(image_path).convert('L'))
#         # rescale numpy array
#         max_value = max(value_range)
#         min_value = min(value_range)
#         image_array = (image_array / np.amax(image_array)) * (max_value - min_value) + min_value
#         # save as attribute
#         self.image_array = image_array
#         # get image array shape
#         self.image_array_shape = image_array.shape
#         # get local registry
#         self.ureg = local_ureg
#
#     def eval(self, values, x):
#         # convert to sau
#         x_sau = x * self.ureg.sau
#         # convert to pixel side
#         x_pxls = x_sau.to("pxls")
#         # get i coordinate for the corresponding pixel
#         i = self.image_array_shape[0] - 1 - int(np.round(x_pxls[1].magnitude))
#         i = int(i) if i >= 0 else 0
#         # get j coordinate for the corresponding pixel
#         j = int(np.round(x_pxls[0].magnitude))
#
#         # compute value
#         values[0] = self.image_array[i][j]
#
#     def value_shape(self):
#         return ()


# class VesselReconstruction:
#     """
#     Generate the 3D/2D reconstruction of the putative initial capillary network.
#     """
#     def __init__(self,
#                  z_0: float,
#                  skeleton_array: np.ndarray,
#                  edge_array: np.ndarray,
#                  binary_array: np.ndarray,
#                  mesh_Lx,
#                  mesh_Ly,
#                  half_depth):
#         """
#         :param z_0: height of the 3D vessel reconstruction. Set to 0 in 2D.
#         """
#         # save z0 value
#         self.z_0 = z_0
#
#         # save Lx and Ly
#         self.mesh_Lx = mesh_Lx
#         self.mesh_Ly = mesh_Ly
#         self.half_depth = half_depth
#
#         # check if shape of arrays are equal
#         if np.any(skeleton_array.shape != edge_array.shape) and np.any(binary_array.shape != edge_array.shape):
#             raise RuntimeError("skeleton, binary, and edge array mush have the same shape.")
#
#         # check if image ratio is preserved
#         if not np.isclose(mesh_Lx / mesh_Ly, binary_array.shape[1] / binary_array.shape[0]):
#             raise RuntimeError("Mesh and image must have the same ratio")
#
#         # get number of pixel
#         self.n_p_x = skeleton_array.shape[1]
#         self.n_p_y = skeleton_array.shape[0]
#
#         # get center points
#         boolean_skeleton_array = skeleton_array.astype(bool)
#         self.center_points = self._get_ones_spatial_coordinates(boolean_skeleton_array)
#         self.n_centers = self.center_points.shape[0]
#
#         # get edge points
#         boolean_edge_array = edge_array.astype(bool)
#         self.edge_points = self._get_ones_spatial_coordinates(boolean_edge_array)
#         self.n_edge_points = self.center_points.shape[0]
#
#         # get binary array as boolean
#         boolean_binary_array = binary_array.astype(bool)
#         # merge binary array with edge array.
#         # Otherwise, the estimation of the vessel radius will be higher in the axial plane.
#         bin_edges_composition = boolean_binary_array | boolean_edge_array
#         self.bin_edges_composition = bin_edges_composition
#
#     def _get_ones_spatial_coordinates(self, input_image: np.ndarray):
#         """
#         Given an input image, returns the spatial coordinates of the pixels equal to one.
#         By default, the spatial dimension of the input image is x: [0, 1] ; y: [0, 1]
#
#         :return: list of spatial coordinates
#         """
#         # get indices for nonzero points
#         nonzero_indices = input_image.nonzero()
#         # Convert indices to spatial coord
#         # Each index is converted to the spatial coordinates of the pixel midpoint
#         sc_list = [
#             [self.mesh_Lx * ((ind_x + 0.5) / self.n_p_x), self.mesh_Ly * (1 - ((ind_y + 0.5) / self.n_p_y)), self.z_0]
#             for ind_x, ind_y in zip(nonzero_indices[1], nonzero_indices[0])
#         ]
#         # convert spatial coordinates to ndarray
#         sc_array = np.array(sc_list)
#         return sc_array
#
#     def eval(self, values, x):
#         # check if point is out of depth
#         if (self.z_0 - self.half_depth) < x[2] < (self.z_0 + self.half_depth):
#             # check if point correspond to 1 in binary array
#             j = int(np.floor((x[0] / self.mesh_Lx) * (self.n_p_x - 1)))
#             i = (self.n_p_y - 1) - int(np.floor((x[1] / self.mesh_Ly) * (self.n_p_y - 1)))
#
#             if self.bin_edges_composition[i, j] > 0:
#                 # get x distances from center points
#                 distance_to_c = np.sqrt(np.sum((x - self.center_points) ** 2, axis=1))
#                 # get min distance and indices of the closest c
#                 distance_to_closest_c = np.amin(distance_to_c)
#                 closest_c_index = np.argmin(distance_to_c)
#                 # get closest c
#                 closest_c = self.center_points[closest_c_index]
#
#                 # get x distance from edge points
#                 distance_to_e = np.sqrt(np.sum((x - self.edge_points) ** 2, axis=1))
#                 # get index of the closest edge point
#                 closest_e_index = np.argmin(distance_to_e)
#                 # get closest edge point
#                 closest_e = self.edge_points[closest_e_index]
#
#                 # distance between closest c and closest e
#                 distance_ce = np.sqrt(np.sum((closest_e - closest_c) ** 2))
#
#                 # set value accordingly
#                 values[0] = sigmoid(distance_to_closest_c, distance_ce, 1., -1., 100)
#             else:
#                 values[0] = -1.
#         else:
#             values[0] = -1.
#
#     def value_shape(self):
#         return ()


class VesselReconstruction:
    """
    Generate the 3D reconstruction of the putative initial capillary network.
    """
    def __init__(self,
                 z_0: float,
                 binary_array: np.ndarray,
                 distance_transform_array: np.ndarray,
                 mesh_Lx,
                 mesh_Ly,
                 half_depth):
        # save z0 value
        self.z_0 = z_0

        # save Lx and Ly
        self.mesh_Lx = mesh_Lx
        self.mesh_Ly = mesh_Ly
        self.half_depth = half_depth

        # check if image ratio is preserved
        if not np.isclose(mesh_Lx / mesh_Ly, binary_array.shape[1] / binary_array.shape[0]):
            raise RuntimeError("Mesh and image must have the same ratio")

        # get number of pixel
        self.n_p_x = binary_array.shape[1]
        self.n_p_y = binary_array.shape[0]

        # save binary array as property
        boolean_binary_array = binary_array.astype(bool)
        # dilate binary (improves roundness)
        self.boolean_binary_array = morphology.binary_dilation(boolean_binary_array)

        # get center points and their distance to the edge
        self._build_center_points_and_radiuses(distance_transform_array)

    def _build_center_points_and_radiuses(self, distance_transform_array):
        # convert distance_transform to spatial dimensions
        converted_distance_transform_array = (distance_transform_array / self.n_p_x) * self.mesh_Lx

        # get nonzero indices
        nonzero_indices = distance_transform_array.nonzero()

        # convert indices to spatial coordinates (midpoint of the pixel)
        sc_list = [
            [self.mesh_Lx * ((ind_x + 0.5) / self.n_p_x), self.mesh_Ly * (1 - ((ind_y + 0.5) / self.n_p_y)), self.z_0]
            for ind_x, ind_y in zip(nonzero_indices[1], nonzero_indices[0])
        ]

        # set sc_list as property
        self.center_points = np.array(sc_list)
        self.n_centers = self.center_points.shape[0]
        logger.debug(f"Computed center points (shape: {self.center_points.shape})")

        # get radiuses
        self.distance_to_edge_for_center = converted_distance_transform_array[nonzero_indices]
        logger.debug(f"Computed distances (shape: {self.distance_to_edge_for_center.shape})")

    def eval(self, x):
        # get n_points
        n_points = x.shape[1]

        """
        1. For each x coordinate, evaluate the closest center and the distance from the closest center.
        """
        # init distances of closest c
        distance_of_closest_c = np.zeros((n_points,))
        # init array of distances between c and e
        distance_ce = np.zeros((n_points,))
        for point_index in range(n_points):
            # get current point
            current_point = x[:, point_index]
            # find its distance from all center points
            distance_to_c = np.zeros((self.n_centers,))
            for i in range(3):
                distance_to_c += (current_point[i] - self.center_points[:, i]) ** 2
            distance_to_c = np.sqrt(distance_to_c)
            # get the minimum distance
            index_of_closest_c = np.argmin(distance_to_c)
            # append the distance of closest c to array
            distance_of_closest_c[point_index] = distance_to_c[index_of_closest_c]
            # append the corresponding center-edge distance
            distance_ce[point_index] = self.distance_to_edge_for_center[index_of_closest_c]

        """
        2. Compute the value to assign at each coordinate using the sigmoid defined as follows:

                            +1  ---------------
                                               \
                                                \
                                                 \
                                                | -------------------- -1
                                                |
                                            distance_ce
        """
        value = ((np.exp(distance_ce * 100) + (-1 * np.exp(100 * distance_of_closest_c)))
                 / (np.exp(distance_ce * 100) + np.exp(100 * distance_of_closest_c)))

        """
        3. Filter out the values not included in the depth of z and for which the projection is not part of the 
           vessel images
        """
        # filter points not included in the vessel depth
        logger.debug(f"x shape: {x.shape}")
        logger.debug(f"value shape: {value.shape}")
        value[x[2] < (self.z_0 - self.half_depth)] = -1.
        value[x[2] > (self.z_0 + self.half_depth)] = -1.
        # filter points out of projection
        i = (self.n_p_y - 1) - np.floor((x[1] / self.mesh_Ly) * (self.n_p_y - 1))
        i = i.astype(int)
        j = np.floor((x[0] / self.mesh_Lx) * (self.n_p_x - 1))
        j = j.astype(int)
        value[~self.boolean_binary_array[i, j]] = -1

        return value
