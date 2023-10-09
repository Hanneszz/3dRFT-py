"""This is the Python implementation of the 3D-RFT Code based on the framework
proposed by Agarwal et al. --> https://doi.org/10.1073/pnas.2214017120"""

######## IMPORTS ########
import math

import numpy as np
import scipy as sp

from src import rft_functions

######## INPUTS ########

## Model selection
MODEL = "CylinderNormal"

## Physical properties
BULK_DENSITY = 1310
FRICTION_MATERIAL = 0.21
FRICTION_SURFACE = 0.4
FRICTION_TYPE = "coefficient"
if FRICTION_TYPE == "angle":
    FRICTION_MATERIAL = math.tan(np.deg2rad(FRICTION_MATERIAL))
    FRICTION_SURFACE = math.tan(np.deg2rad(FRICTION_SURFACE))
elif FRICTION_TYPE == "coefficient":
    pass
else:
    raise ValueError("FRICTION_UNIT must be either 'angle' or 'coefficient'")
GRAVITY = 9.81
MATERIAL_CONSTANT = (
    GRAVITY
    * BULK_DENSITY
    * (
        894 * FRICTION_MATERIAL**3
        - 386 * FRICTION_MATERIAL**2
        + 89 * FRICTION_MATERIAL
    )
)

## Movement defintions
LINEAR_VELOCITY = 0.1
DIR_ANGLE_XZ_DEG = -90
DIR_ANGLE_Y_DEG = -90

ROTATION = True
ANGULAR_VELOCITY = np.array([0, 0, -2 * sp.pi])

## Depth parameters in mm
START_DEPTH = 100
END_DEPTH = 125
STEP_SIZE = 5
NUM_STEPS = int((END_DEPTH + STEP_SIZE - START_DEPTH) / STEP_SIZE)


## Miscellaneous & Plots
THRESHOLD = 1.0e-12

######## METHOD ########

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
) = rft_functions.import_mesh(MODEL)

result_matrix = np.zeros((NUM_STEPS, 7))
STEP = 0

## Start loop
for depth in range(START_DEPTH, END_DEPTH + STEP_SIZE, STEP_SIZE):
    point_list[:, 2] -= depth
    ## Calculate movement
    movement = rft_functions.calc_movement(
        point_list,
        depth,
        object_height,
        DIR_ANGLE_XZ_DEG,
        DIR_ANGLE_Y_DEG,
        LINEAR_VELOCITY,
        ROTATION,
        ANGULAR_VELOCITY,
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
        MATERIAL_CONSTANT,
        FRICTION_SURFACE,
    )

    ## Find the resultant forces on object
    forces, pressure, force_x, force_y, force_z, resultant = rft_functions.find_forces(
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

    result_matrix[STEP, :] = [
        depth,
        force_x,
        force_y,
        force_z,
        torque_x,
        torque_y,
        torque_z,
    ]

    print("Processed movement at depth:", depth, "mm")
    STEP += 1
