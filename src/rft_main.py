"""This module runs the 3D-RFT code based on the framework proposed by Agarwal et al.
--> https://doi.org/10.1073/pnas.2214017120"""


import math

import numpy as np
from stl import mesh


def import_mesh(model: str):
    """This imports the mesh from the models folder"""
    try:
        trg = mesh.Mesh.from_file(f"./models/{model}.stl")
        mins = np.amin(trg.centroids, axis=0)
        maxs = np.amax(trg.centroids, axis=0)
        object_dims = maxs - mins
        object_width_x, object_width_y, object_height = object_dims

        point_list = trg.centroids
        normal_list = trg.normals
        normal_list = normal_list / np.linalg.norm(normal_list, axis=1, keepdims=True)
        area_list = trg.areas

        point_list[:, 2] -= min(point_list[:, 2])

        vertices = trg.vectors.reshape(-1, 3)
        vertices, indices = np.unique(vertices, axis=0, return_inverse=True)
        vertices[:, 2] -= min(vertices[:, 2])
        faces = indices.reshape(-1, 3)

        return (
            point_list,
            normal_list,
            area_list,
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
    """This calculates the movement of the object"""
    if not isinstance(rotation, bool):
        raise TypeError("rotation must be a boolean value.")

    linear_direction_vector = np.array(
        [
            [
                math.cos(math.radians(dir_angle_xz_deg)),
                math.cos(math.radians(dir_angle_y_deg)),
                math.sin(math.radians(dir_angle_xz_deg)),
            ]
        ]
    )

    n_elements = points.shape[0]
    elements = np.ones((n_elements, 3))
    movement = elements * (linear_velocity * linear_direction_vector * 1000).round(12)

    if rotation:
        radii_list = points.copy()
        radii_list[:, 2] += (depth) - (object_height / 2)
        angular_movement = np.cross(elements * angular_velocity, radii_list)
        movement += angular_movement.round(12)

    movement = movement / np.linalg.norm(movement, axis=1, keepdims=True).round(12)

    return movement


def check_conditions(point_list, normal_list, area_list, depth_list, movement):
    """This checks certain conditions for subsurfaces to be considered in the RFT method"""
    is_leading_edge = np.einsum("ij,ij->i", normal_list, movement) >= -0.00000000000001
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

    dot_product_movement = np.einsum("ij,ij->i", movement, z_local)[:, np.newaxis]
    difference_movement = movement - dot_product_movement * z_local
    norms_movement = np.linalg.norm(difference_movement, axis=1, keepdims=True)

    dot_product_normal = np.einsum("ij,ij->i", normal_list, z_local)[:, np.newaxis]
    difference_normal = normal_list - dot_product_normal * z_local
    norms_normal = np.linalg.norm(difference_normal, axis=1, keepdims=True)

    r_local = np.zeros_like(z_local)

    # Initialize r_local
    r_local = np.zeros_like(difference_movement)

    # Case 1: norms_movement and norms_normal are both zero
    mask_1 = (norms_movement == 0) & (norms_normal == 0)
    r_local[mask_1.ravel()] = [1, 0, 0]

    # Case 2: norms_movement is zero and norms_normal is not
    mask_2 = (norms_movement == 0) & ~(norms_normal == 0)
    r_local[mask_2.ravel()] = (difference_normal / norms_normal)[mask_2.ravel()]

    # Case 3: norms_movement is not zero
    # This case implicitly includes scenarios where norms_normal can be either zero or not
    mask_3 = norms_movement != 0
    r_local[mask_3.ravel()] = (difference_movement / norms_movement)[mask_3.ravel()]

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
    beta = np.zeros(normal_list.shape[0])[:, np.newaxis]
    gamma = np.zeros(normal_list.shape[0])[:, np.newaxis]
    psi = np.zeros(normal_list.shape[0])[:, np.newaxis]

    dot_normals_r = np.einsum("ij,ij->i", normal_list, r_local)[:, np.newaxis]
    dot_normals_z = np.einsum("ij,ij->i", normal_list, z_local)[:, np.newaxis]

    beta_1 = np.where(
        (dot_normals_r >= 0) & (dot_normals_z >= 0), -np.arccos(dot_normals_z), 0
    )
    beta_2 = np.where(
        (dot_normals_r >= 0) & (dot_normals_z < 0), np.pi - np.arccos(dot_normals_z), 0
    )
    beta_3 = np.where(
        (dot_normals_r < 0) & (dot_normals_z >= 0), np.arccos(dot_normals_z), 0
    )
    beta_4 = np.where(
        (dot_normals_r < 0) & (dot_normals_z < 0), -np.pi + np.arccos(dot_normals_z), 0
    )

    beta = (beta_1 + beta_2 + beta_3 + beta_4).round(12)

    dot_movement_r = np.sum(movement * r_local, axis=1, keepdims=True)
    dot_movement_z = np.sum(movement * z_local, axis=1, keepdims=True)

    gamma = np.round(np.arccos(np.clip(dot_movement_r, -1, 1)), 12)
    gamma[dot_movement_z > 0] *= -1

    diff_normals = normal_list - dot_normals_z * z_local
    norm_diff_normals = np.linalg.norm(diff_normals, axis=1, keepdims=True).round(12)

    nr0_inc = (diff_normals / norm_diff_normals).round(12)
    dot_nr0_r = np.sum(nr0_inc * r_local, axis=1, keepdims=True)
    dot_nr0_theta = np.sum(nr0_inc * theta_local, axis=1, keepdims=True)

    psi = np.round(np.arctan(dot_nr0_theta / dot_nr0_r), 12)
    psi[(norm_diff_normals == 0) | (dot_nr0_r == 0)] = 0

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

    f_2 = (t_k @ c2k)[:, np.newaxis]
    f_3 = (t_k @ c3k)[:, np.newaxis]
    f_1 = (t_k @ c1k)[:, np.newaxis]

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
        alpha_r_gen * r_local + alpha_theta_gen * theta_local + alpha_z_gen * z_local
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

    norm_alpha_n = np.linalg.norm(alpha_generic_n, axis=1, keepdims=True)
    norm_alpha_t = np.linalg.norm(alpha_generic_t, axis=1, keepdims=True)
    ratio = friction_surface * (norm_alpha_n / norm_alpha_t)
    ratio[ratio > 1 | np.isnan(ratio)] = 1
    alpha = material_constant * (alpha_generic_n + ratio * alpha_generic_t)

    return alpha_generic, alpha_generic_n, alpha_generic_t, alpha


def find_forces(alpha, depth_list, area_list):
    """This determines the forces on the object"""
    forces = alpha * abs(depth_list) * area_list
    pressures = forces / area_list

    force_x = np.sum(forces[:, 0])
    force_y = np.sum(forces[:, 1])
    force_z = np.sum(forces[:, 2])
    resultant = np.linalg.norm([force_x, force_y, force_z])

    return forces, pressures, force_x, force_y, force_z, resultant


def find_torques(point_list, forces):
    """This determines the torques on the object in Nm"""
    torques = np.cross(point_list, forces) / 1000
    torque_x = np.sum(torques[:, 0])
    torque_y = np.sum(torques[:, 1])
    torque_z = np.sum(torques[:, 2])
    resultant_torque = np.linalg.norm([torque_x, torque_y, torque_z])

    return torques, torque_x, torque_y, torque_z, resultant_torque


def run_rft(
    model,
    bulk_density,
    friction_material,
    firction_surface,
    friction_type,
    gravity,
    linear_velocity,
    direction_angle_xz_deg,
    direction_angle_y_deg,
    rotation,
    angular_velocity,
    start_depth,
    end_depth,
    step_size,
):
    """This is the Python implementation of the 3D-RFT Code based on the framework"""
    if friction_type == "angle":
        friction_material = np.tan(np.deg2rad(friction_material))
        firction_surface = np.tan(np.deg2rad(firction_surface))
    elif friction_type == "coefficient":
        pass
    else:
        raise ValueError("FRICTION_UNIT must be either 'angle' or 'coefficient'")

    material_constant = (
        gravity
        * bulk_density
        * (
            894 * friction_material**3
            - 386 * friction_material**2
            + 89 * friction_material
        )
    ) * 10**-9

    angular_velocity = np.array([0, 0, -2 * np.pi])

    num_steps = int((end_depth + step_size - start_depth) / step_size)

    ## STL Processing
    (
        point_list,
        normal_list,
        area_list,
        object_width_x,
        object_width_y,
        object_height,
        vertices,
        faces,
        trg,
    ) = import_mesh(model)

    result_matrix = np.zeros((num_steps, 7))
    step = 0

    for depth in range(start_depth, end_depth + step_size, step_size):
        current_point_list = np.copy(point_list)
        current_point_list[:, 2] -= depth
        current_depth_list = current_point_list[:, 2][:, np.newaxis]
        current_normal_list = np.copy(normal_list)
        current_area_list = np.copy(area_list)

        # min_z = np.min(point_list[:, 2])
        # offset = min_z - 0
        # point_list[:, 2] -= offset
        # point_list[:, 2] -= depth
        # depth_list = point_list[:, 2][:, np.newaxis]
        # vertices[:, 2] -= offset
        # vertices[:, 2] -= depth
        # point_list[:, 2] -= step_size
        # depth_list = point_list[:, 2][:, np.newaxis]
        # vertices[:, 2] -= step_size

        ## Calculate movement
        movement = calc_movement(
            current_point_list,
            depth,
            object_height,
            direction_angle_xz_deg,
            direction_angle_y_deg,
            linear_velocity,
            rotation,
            angular_velocity,
        )

        ## Check conditions
        (
            current_point_list,
            current_normal_list,
            current_area_list,
            current_depth_list,
            movement,
        ) = check_conditions(
            current_point_list,
            current_normal_list,
            current_area_list,
            current_depth_list,
            movement,
        )

        ## Find local coordinate frame for each subsurface
        z_local, r_local, theta_local = find_local_frame(current_normal_list, movement)

        ## Find the characteristic angles of the RFT method
        beta, gamma, psi = find_angles(
            current_normal_list, movement, z_local, r_local, theta_local
        )

        ## Find empirical values for the RFT method
        f_1, f_2, f_3 = find_fit(beta, gamma, psi)

        ## Find dimensionless response vectors alpha
        alpha_generic, alpha_generic_n, alpha_generic_t, alpha = find_alpha(
            current_normal_list,
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
            firction_surface,
        )

        ## Find the resultant forces on object
        forces, pressures, force_x, force_y, force_z, resultant = find_forces(
            alpha, current_depth_list, current_area_list
        )

        ## Find the resultant torques on object
        (
            torques,
            torque_x,
            torque_y,
            torque_z,
            resultant_torque,
        ) = find_torques(current_point_list, forces)

        result_matrix[step, :] = [
            depth,
            force_x,
            force_y,
            force_z,
            torque_x,
            torque_y,
            torque_z,
        ]

        step += 1

    print("Processed movement from", start_depth, "to", end_depth, "mm")

    results = {
        "point_list": current_point_list,
        "normal_list": current_normal_list,
        "area_list": current_area_list,
        "depth_list": current_depth_list,
        "object_width_x": object_width_x,
        "object_width_y": object_width_y,
        "object_height": object_height,
        "vertices": vertices,
        "faces": faces,
        "trg": trg,
        "movement": movement,
        "z_local": z_local,
        "r_local": r_local,
        "theta_local": theta_local,
        "alpha_generic": alpha_generic,
        "alpha_generic_n": alpha_generic_n,
        "alpha_generic_t": alpha_generic_t,
        "alpha": alpha,
        "depth": depth,
        "forces": forces,
        "pressures": pressures,
        "force_x": force_x,
        "force_y": force_y,
        "force_z": force_z,
        "resultant": resultant,
        "torques": torques,
        "torque_x": torque_x,
        "torque_y": torque_y,
        "torque_z": torque_z,
        "resultant_torque": resultant_torque,
        "result_matrix": result_matrix,
    }

    return results
