"""This is the Python implementation of the 3D-RFT Code based on the framework
proposed by Agarwal et al. --> https://doi.org/10.1073/pnas.2214017120"""

######## IMPORTS ########
import math
import numpy as np
import scipy as sp

######## INPUTS ########

## Model selection
MODEL = "CylinderNormal"

## Physical Properties
BULK_DENSITY = 1520
FRICTION_MATERIAL_DEG = 20
FRICTION_MATERIAL = math.tan(math.radians(FRICTION_MATERIAL_DEG))
FRICTION_SURFACE = 0.4

print("The Value for the interface friction coefficient is: ", FRICTION_SURFACE)
