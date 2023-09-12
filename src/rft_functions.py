"""This module is a collection of all functions used to calculate the forces
on an object according to the RFT frame work"""

import math

import numpy as np
from stl import mesh


## STL file import and geometric evaluation
def import_mesh(model):
    """Caluclating geometric properties of the 3D Object and initial alignment"""
    trg = mesh.Mesh.from_file(f"./models/{model}.stl")
    mins = np.amin(trg.centroids, axis=0)
    maxs = np.amax(trg.centroids, axis=0)
    object_dims = maxs - mins
    object_width_x, object_width_y, object_height = object_dims

    point_list = trg.centroids
    normal_list = trg.normals
    area_list = trg.areas

    point_list[:, 2] -= min(point_list[:, 2])
    depth_list = point_list[:, 2]

    return (
        point_list,
        normal_list,
        area_list,
        depth_list,
        object_width_x,
        object_width_y,
        object_height,
    )


## Movement vector determination
def calc_movement(
    points,
    depth,
    object_height,
    dir_angle_xz_deg,
    dir_angle_y_deg,
    linear_velocity,
    rotation,
    angular_velocity,
):
    """Calculates Movement based on various input parameters"""
    linear_direction_vector = np.array(
        [
            [
                round(math.cos(math.radians(dir_angle_xz_deg)), 12),
                round(math.cos(math.radians(dir_angle_y_deg)), 12),
                round(math.sin(math.radians(dir_angle_xz_deg)), 12),
            ]
        ]
    )

    n_elements = points.shape[0]
    elements = np.ones((n_elements, 3))
    movement = elements * (linear_velocity * linear_direction_vector * 1000).round(12)

    if rotation:
        radii_list = points.copy()
        radii_list[:, 2] += (depth * 1000) - (object_height / 2)
        angular_movement = np.cross(elements * angular_velocity, radii_list)
        movement += angular_movement.round(12)

    movement = (movement / np.linalg.norm(movement, axis=1, keepdims=True)).round(12)

    return movement
