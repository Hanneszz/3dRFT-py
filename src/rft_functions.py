"""This module is a collection of all functions used to calculate the forces
on an object according to the RFT frame work"""

import math

import numpy as np
from stl import mesh


def import_mesh(model: str):
    """Calculates the geometric properties of a 3D object from an STL file.

    Args:
        model (str): The path to the STL file.

    Returns:
        tuple: A tuple of the geometric properties of the object,
        including the point cloud, normals, areas, depths, width, height, and length.
    """
    try:
        trg = mesh.Mesh.from_file(f"./models/{model}.stl")
        mins = np.amin(trg.centroids, axis=0)
        maxs = np.amax(trg.centroids, axis=0)
        object_dims = maxs - mins
        object_width_x, object_width_y, object_height = object_dims

        point_list = trg.centroids
        normal_list = trg.normals
        normal_list = (
            normal_list / np.linalg.norm(normal_list, axis=1, keepdims=True)
        ).round(12)
        area_list = trg.areas

        point_list[:, 2] -= min(point_list[:, 2])
        depth_list = point_list[:, 2]

        vertices = trg.vectors.reshape(-1, 3)
        vertices, indices = np.unique(vertices, axis=0, return_inverse=True)
        faces = indices.reshape(-1, 3)

        return (
            point_list,
            normal_list,
            area_list,
            depth_list,
            object_width_x,
            object_width_y,
            object_height,
            vertices,
            faces,
            trg,
        )
    except FileNotFoundError as exc:
        print(exc)
        print("Please make sure the model is in the models folder and try again.")
        exit()


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
    """Calculates the movement of a set of points based on various input parameters.

    Args:
        points (np.ndarray): The set of points to move.
        depth (float): The depth of the object.
        object_height (float): The height of the object.
        dir_angle_xz_deg (float): The direction angle in the XZ plane in degrees.
        dir_angle_y_deg (float): The direction angle in the Y direction in degrees.
        linear_velocity (float): The linear velocity.
        rotation (bool): Whether or not the object is rotating.
        angular_velocity (float): The angular velocity.

    Returns:
        np.ndarray: The set of points after movement.
    """

    if not isinstance(rotation, bool):
        raise TypeError("rotation must be a boolean value.")

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


def check_conditions(point_list, normal_list, area_list, depth_list, movement):
    """This checks certain conditions for subsurfaces to be considered in the RFT method"""
    is_leading_edge = np.sum(normal_list * movement, axis=1) >= 0
    is_intruding = point_list[:, 2] < 0
    include = is_intruding & is_leading_edge

    point_list = point_list[include, :]
    normal_list = normal_list[include, :]
    area_list = area_list[include, :]
    depth_list = depth_list[include]
    movement = movement[include, :]

    return point_list, normal_list, area_list, depth_list, movement


def find_local_frame(normal_list, movement):
    """This determines the local coordinate frame for each subsurface of the object"""
    z_local = np.repeat([[0, 0, 1]], normal_list.shape[0], axis=0)

    r_local = np.zeros_like(z_local)

    dot_product_movement = np.einsum("ij,ij->i", movement, z_local)[:, np.newaxis]
    difference_movement = movement - dot_product_movement * z_local
    norms_movement = np.linalg.norm(difference_movement, axis=1, keepdims=True).round(
        12
    )

    dot_product_normal = np.einsum("ij,ij->i", normal_list, z_local)[:, np.newaxis]
    difference_normal = normal_list - dot_product_normal * z_local
    norms_normal = np.linalg.norm(difference_normal, axis=1, keepdims=True).round(12)

    temp_mask_1 = norms_movement == 0
    temp_mask_2 = norms_normal == 0

    mask_1 = np.squeeze(temp_mask_1 & temp_mask_2)
    mask_2 = np.squeeze(temp_mask_1 & ~temp_mask_2)
    mask_3 = np.squeeze(~mask_1 & ~mask_2)

    r_local[mask_1] = [1, 0, 0]
    r_local[mask_2] = difference_normal[mask_2] / norms_normal[mask_2]
    # check r_local
    print(r_local)
    print("###### initial state #####")
    # apply operation for mask_3 --> mask_3[:] = True
    r_local[mask_3] = difference_movement[mask_3] / norms_movement[mask_3]
    print(r_local)
    print("##### after trying with mask ######")
    # since r_local = 0 0 0, 0 0 0..., try operation without mask
    r_local = difference_movement / norms_movement
    print(r_local)
    print("##### without mask ######")
    # why it work now and not before man :(

    theta_local = np.cross(z_local, r_local, axis=1)

    return z_local, r_local, theta_local


def find_angles(
    normal_list: np.array,
    movement: np.array,
    z_local: np.array,
    r_local: np.array,
    theta_local: np.array,
) -> (np.array, np.array, np.array):
    """This determines the characteristic angles of the RFT method"""
    beta = np.zeros(normal_list.shape[0])
    gamma = np.zeros(normal_list.shape[0])
    psi = np.zeros(normal_list.shape[0])

    dot_normals_r = np.sum(normal_list * r_local, axis=1)
    dot_normals_z = np.sum(normal_list * z_local, axis=1)

    mask_1 = np.squeeze((dot_normals_r >= 0) & (dot_normals_z >= 0))
    mask_2 = np.squeeze((dot_normals_r >= 0) & (dot_normals_z < 0))
    mask_3 = np.squeeze((dot_normals_r < 0) & (dot_normals_z >= 0))

    beta[mask_1] = -np.round(np.arccos(dot_normals_z[mask_1]), 12)
    beta[mask_2] = np.round(np.pi - np.arccos(dot_normals_z[mask_2]), 12)
    beta[mask_3] = np.round(np.arccos(dot_normals_z[mask_3]), 12)
    beta[~(mask_1 | mask_2 | mask_3)] = np.round(
        -np.pi + np.arccos(dot_normals_z[~(mask_1 | mask_2 | mask_3)]), 12
    )

    dot_movement_r = np.sum(movement * r_local, axis=1)
    dot_movement_z = np.sum(movement * z_local, axis=1)

    gamma[:] = np.round(np.arccos(np.clip(dot_movement_r, -1, 1)), 12)
    gamma[dot_movement_z > 0] *= -1

    diff_normals = normal_list - dot_normals_z[:, np.newaxis] * z_local
    norm_diff_normals = np.linalg.norm(diff_normals, axis=1)

    nr0_inc = np.where(
        norm_diff_normals[:, np.newaxis] != 0,
        diff_normals / norm_diff_normals[:, np.newaxis],
        np.zeros_like(diff_normals),
    )
    dot_nr0_r = np.sum(nr0_inc * r_local, axis=1)
    dot_nr0_theta = np.sum(nr0_inc * theta_local, axis=1)

    mask_psi = np.squeeze((norm_diff_normals == 0) | (dot_nr0_r == 0))
    psi[~mask_psi] = np.round(
        np.arctan(dot_nr0_theta[~mask_psi] / dot_nr0_r[~mask_psi]), 12
    )

    return beta, gamma, psi


def find_fit(beta, gamma, psi):
    """This determines the fit of the RFT method"""
    x_1 = np.round(np.sin(gamma), 12)
    x_2 = np.round(np.cos(beta), 12)
    x_3 = np.round(
        np.cos(psi) * np.cos(gamma) * np.sin(beta) + np.sin(gamma) * np.cos(beta), 12
    )

    t_k = np.column_stack(
        (
            np.ones(beta.shape[0]),
            x_1,
            x_2,
            x_3,
            x_1**2,
            x_2**2,
            x_3**2,
            x_1 * x_2,
            x_2 * x_3,
            x_3 * x_1,
            x_1**3,
            x_2**3,
            x_3**3,
            x_1 * x_2**2,
            x_2 * x_1**2,
            x_2 * x_3**2,
            x_3 * x_2**2,
            x_3 * x_1**2,
            x_1 * x_3**2,
            x_1 * x_2 * x_3,
        )
    )

    c1k = np.array(
        [
            0.00212,
            -0.02320,
            -0.20890,
            -0.43083,
            -0.00259,
            0.48872,
            -0.00415,
            0.07204,
            -0.02750,
            -0.08772,
            0.01992,
            -0.45961,
            0.40799,
            -0.10107,
            -0.06576,
            0.05664,
            -0.09269,
            0.01892,
            0.01033,
            0.15120,
        ]
    )

    c2k = np.array(
        [
            -0.06796,
            -0.10941,
            0.04725,
            -0.06914,
            -0.05835,
            -0.65880,
            -0.11985,
            -0.25739,
            -0.26834,
            0.02692,
            -0.00736,
            0.63758,
            0.08997,
            0.21069,
            0.04748,
            0.20406,
            0.18589,
            0.04934,
            0.13527,
            -0.33207,
        ]
    )

    c3k = np.array(
        [
            -0.02634,
            -0.03436,
            0.45256,
            0.00835,
            0.02553,
            -1.31290,
            -0.05532,
            0.06790,
            -0.16404,
            0.02287,
            0.02927,
            0.95406,
            -0.00131,
            -0.11028,
            0.01487,
            -0.02730,
            0.10911,
            -0.04097,
            0.07881,
            -0.27519,
        ]
    )

    f_1 = t_k @ c1k
    f_2 = t_k @ c2k
    f_3 = t_k @ c3k

    return f_1, f_2, f_3


def find_alpha(
    normal_list,
    movement,
    beta,
    gamma,
    psi,
    z_local,
    r_local,
    theta_local,
    f_1,
    f_2,
    f_3,
    material_constant,
    friction_surface,
):
    """This determines the dimensionless response vectors alpha"""
    alpha_r_gen = np.round(f_1 * np.sin(beta) * np.cos(psi) + f_2 * np.cos(gamma), 12)
    alpha_theta_gen = np.round(f_1 * np.sin(beta) * np.sin(psi), 12)
    alpha_z_gen = np.round(-f_1 * np.cos(beta) - f_2 * np.sin(gamma) - f_3, 12)

    alpha_generic = (
        alpha_r_gen[:, np.newaxis] * r_local
        + alpha_theta_gen[:, np.newaxis] * theta_local
        + alpha_z_gen[:, np.newaxis] * z_local
    )

    dot_product_alpha_normals = np.einsum("ij,ij->i", alpha_generic, -normal_list)
    mask_1 = dot_product_alpha_normals < 0
    mask_2 = ~mask_1

    alpha_generic_n = np.zeros(normal_list.shape)
    alpha_generic_t = np.zeros(normal_list.shape)

    alpha_generic_n[mask_1] = (-dot_product_alpha_normals[mask_1][:, np.newaxis]) * (
        -normal_list[mask_1]
    )
    alpha_generic_t[mask_1] = alpha_generic[mask_1] + alpha_generic_n[mask_1]

    alpha_generic_n[mask_2] = (dot_product_alpha_normals[mask_2][:, np.newaxis]) * (
        -normal_list[mask_2]
    )
    alpha_generic_t[mask_2] = alpha_generic[mask_2] - alpha_generic_n[mask_2]

    dot_product_alpha_t_move = np.einsum("ij,ij->i", alpha_generic_t, -movement)
    mask_3 = dot_product_alpha_t_move < 0
    alpha_generic_t[mask_3] *= -1

    alpha = material_constant * (
        alpha_generic_n
        + np.min(
            friction_surface
            * np.linalg.norm(alpha_generic_n, axis=1, keepdims=True)
            / np.linalg.norm(alpha_generic_t, axis=1, keepdims=True),
            1,
        )[:, np.newaxis]
        * alpha_generic_t
    )

    return alpha_generic, alpha_generic_n, alpha_generic_t, alpha


def find_forces(alpha, depth_list, area_list):
    """This determines the forces on the object"""
    forces = alpha * abs(depth_list)[:, np.newaxis] * area_list[:, np.newaxis]
    pressure = forces / area_list / 1000000

    force_x = np.sum(forces[:, 0])
    force_y = np.sum(forces[:, 1])
    force_z = np.sum(forces[:, 2])
    resultant = np.linalg.norm([force_x, force_y, force_z])

    return forces, pressure, force_x, force_y, force_z, resultant


def find_torques(point_list, forces):
    """This determines the torques on the object"""
    torques = np.cross(point_list, forces)
    torque_x = np.sum(torques[:, 0])
    torque_y = np.sum(torques[:, 1])
    torque_z = np.sum(torques[:, 2])
    resultant_torque = np.linalg.norm([torque_x, torque_y, torque_z])

    return torques, torque_x, torque_y, torque_z, resultant_torque
