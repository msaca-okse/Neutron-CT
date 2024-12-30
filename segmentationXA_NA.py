import SimpleITK as sitk
import numpy as np
import os
os.chdir('/zhome/71/c/146676/Desktop/msaca/main/')  # Navigate into a subdirectory
import matplotlib.pyplot as plt
from cil.io import TIFFStackReader
import plot_library as pl
import importlib
import tifffile

# Assume 'fixed_image' and 'moving_image' are SimpleITK images
# Assume 'transform' is the transformation obtained from the registration

# Step 1: Define the original size, spacing, and origin of the moving image
moving_size = moving_image.GetSize()
moving_spacing = moving_image.GetSpacing()
moving_origin = moving_image.GetOrigin()
moving_direction = moving_image.GetDirection()

# Step 2: Create a new image with the same size, spacing, and direction
# But this will be in the coordinate system of the fixed image
new_image = sitk.Resample(
    moving_image, 
    fixed_image, 
    transform, 
    sitk.sitkLinear,  # Interpolation type, choose as appropriate
    0.0,              # Default value for empty regions
    moving_image.GetPixelID()  # Keep the pixel type of the moving image
)

# Step 3: The 'new_image' will have the same resolution as the moving image
# But it will be aligned according to the transformation with the fixed image's coordinates.