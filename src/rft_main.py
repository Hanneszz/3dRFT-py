"""This module runs the 3D-RFT code based on the framework proposed by Agarwal et al. --> https://doi.org/10.1073/pnas.2214017120"""

######## IMPORTS ########
import numpy as np

import rft_functions


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
        depth_list,
        object_width_x,
        object_width_y,
        object_height,
        vertices,
        faces,
        trg,
    ) = rft_functions.import_mesh(model)

    result_matrix = np.zeros((num_steps, 7))
    step = 0

    for depth in range(start_depth, end_depth + step_size, step_size):
        point_list[:, 2] -= depth
        vertices[:, 2] -= depth

        ## Calculate movement
    movement = rft_functions.calc_movement(
        point_list,
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
        point_list,
        normal_list,
        area_list,
        depth_list,
        movement,
    ) = rft_functions.check_conditions(
        point_list, normal_list, area_list, depth_list, movement
    )

    ## Find local coordinate frame for each subsurface
    z_local, r_local, theta_local = rft_functions.find_local_frame(
        normal_list, movement
    )

    ## Find the characteristic angles of the RFT method
    beta, gamma, psi = rft_functions.find_angles(
        normal_list, movement, z_local, r_local, theta_local
    )

    ## Find empirical values for the RFT method
    f_1, f_2, f_3 = rft_functions.find_fit(beta, gamma, psi)

    ## Find dimensionless response vectors alpha
    alpha_generic, alpha_generic_n, alpha_generic_t, alpha = rft_functions.find_alpha(
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
        firction_surface,
    )

    ## Find the resultant forces on object
    forces, pressures, force_x, force_y, force_z, resultant = rft_functions.find_forces(
        alpha, depth_list, area_list
    )

    ## Find the resultant torques on object
    (
        torques,
        torque_x,
        torque_y,
        torque_z,
        resultant_torque,
    ) = rft_functions.find_torques(point_list, forces)

    result_matrix[step, :] = [
        depth,
        force_x,
        force_y,
        force_z,
        torque_x,
        torque_y,
        torque_z,
    ]

    print("Processed movement at depth:", depth, "mm")
    step += 1

    results = {
        "point_list": point_list,
        "normal_list": normal_list,
        "area_list": area_list,
        "depth_list": depth_list,
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
