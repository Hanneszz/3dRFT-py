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
BULK_DENSITY = 1520
FRICTION_MATERIAL_DEG = 20
FRICTION_MATERIAL = math.tan(np.deg2rad(FRICTION_MATERIAL_DEG))
FRICTION_SURFACE = 0.4
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

ROTATION = 1
ANGULAR_VELOCITY = np.array([0, 0, -2 * sp.pi])

## Depth parameters in mm
START_DEPTH = 0
END_DEPTH = 125
STEP_SIZE = 5
NUM_STEPS = (END_DEPTH - START_DEPTH) / STEP_SIZE


## Miscellaneous & Plots
THRESHOLD = 1.0e-12
SHOW_GEOMETRY = 0
SHOW_MOVEMENT = 0
SHOW_ALPHA = 0
SHOW_QUIVER = 0
SHOW_SCATTER = 1
SHOW_SCATTER_XYZ = 0
SHOW_RESULTS = 0
SAVE_FIGURES = 0

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
) = rft_functions.import_mesh(MODEL)

## Start loop
for depth in range(START_DEPTH, END_DEPTH + STEP_SIZE, STEP_SIZE):
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

    point_list[:, 2] -= STEP_SIZE

print("provisional end")
