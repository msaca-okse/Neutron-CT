# README

This file contains the code for the script and some additional information:
- This script generates a transformation object by registration and saves it. To
 apply the registrating transformation, load in the data as is done in this script,
 generate a registrator object, and apply the registration.
- The script is self-contained and generates this README file.

==================================================

## Script Code:

### Registrator scripted:
## Registering full scale XA to NA

import numpy as np
import os
os.chdir('/zhome/71/c/146676/main/')  # Navigate into a subdirectory
import matplotlib.pyplot as plt
import plot_library as pl
import SimpleITK as sitk
import registrator
import plotly.io as pio
from cil.framework import ImageData, ImageGeometry
import tifffile
import stitcher_XA
from multiprocessing import Pool
import module_auxiliary as ma


# Load the images
paths_XA = stitcher_XA.generate_stitched_paths(dataset = 'fbp', folders = [1,2,3,4,5],
    start_indices = [300, 94, 94, 94, 94], end_indices = [693, 694, 694, 694, 788])[::2]


def loader(path):
    image = tifffile.imread(path)[::2,::2]
    ny,nx = np.shape(image)
    pad = 100
    image2 = np.zeros((ny, nx + 2*pad), dtype=np.float32)
    image2[:, pad:-pad] = image
    return image2

with Pool() as pool:
    recon_XA = pool.map(loader, paths_XA)

recon_XA = np.stack(recon_XA)
recon_XA = np.clip(recon_XA, a_min = -0.1, a_max = 0.1)

path_NA = '/dtu-compute/msaca/output/tv_recon/slice_tv_####.tiff'
A = np.arange(0, 1788)
paths_NA = ma.generate_paths(path_NA, A)

def loader(path):
    return tifffile.imread(path)[:1600,:1600]

with Pool() as pool:
    recon_NA = pool.map(loader, paths_NA)

recon_NA = np.stack(recon_NA)
recon_NA = np.clip(recon_NA, a_min = -0.1, a_max = 0.1)

# Initialice the registrator class and set thresholds
reg = registrator.Registrator()
reg.load_images(fixed_array=recon_NA,moving_array=recon_XA)
threshold = [0.02,0.02]

# Compute segmentation and meteorite moments
reg.compute_stone_boundaries(thresholds = threshold, factor = 5) # saved in reg.fixed_seg and reg.moving_seg
fixed_moments = reg.compute_moments(reg.fixed_seg)
moving_moments = reg.compute_moments(reg.moving_seg)
P_fixed = fixed_moments['axes']
P_moving = moving_moments['axes']
c_fixed = fixed_moments['centroid']
c_moving = moving_moments['centroid']



angle_radians = np.deg2rad(180)
rot_matrix = [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, -1]
            ]
matrix = [elem for row in rot_matrix for elem in row]
translation = (0,0,0)
transform = reg.transformation(type='Affine', matrix = matrix,translation=translation)
reg.resample(transform)
reg.compute_stone_boundaries(thresholds = threshold, factor = 5) # saved in reg.fixed_seg and reg.moving_seg



plt.imshow(1*(reg.as_array(reg.moving)[530]>threshold[1]))
plt.savefig('/dtu-compute/msaca/output/fig_02_01.png')
plt.close()

plt.imshow(1*(reg.as_array(reg.moving)[:,140]>threshold[1]))
plt.savefig('/dtu-compute/msaca/output/fig_02_02.png')
plt.close()

plt.imshow(1*(reg.as_array(reg.moving)[:,:,530]>threshold[1]))
plt.savefig('/dtu-compute/msaca/output/fig_02_03.png')
plt.close()


plt.imshow(1*(reg.as_array(reg.fixed)[530]>threshold[0]))
plt.savefig('/dtu-compute/msaca/output/fig_03_01.png')
plt.close()

plt.imshow(1*(reg.as_array(reg.fixed)[:,140]>threshold[0]))
plt.savefig('/dtu-compute/msaca/output/fig_03_02.png')
plt.close()

plt.imshow(1*(reg.as_array(reg.fixed)[:,:,530]>threshold[0]))
plt.savefig('/dtu-compute/msaca/output/fig_03_03.png')
plt.close()

# Location has now been initialized, now do loops of registration:

# Loop 1
transform_registration= reg.register(learning_rate = 1, sampling_percentage = 0.1, max_iter = 50,
                                        metric_type = 'ms', optimizer_type = 'gd',
                                        smoothing = 0, shrinking = 7, thresholds=threshold,
                                        smooth_fixed = False, smooth_moving = False)
reg.resample(transform_registration)

transform_registration= reg.register(learning_rate = 0.2, sampling_percentage = 0.1, max_iter = 10,
                                        metric_type = 'ms', optimizer_type = 'gd',
                                        smoothing = 0, shrinking = 3, thresholds=threshold,
                                        smooth_fixed = False, smooth_moving = False)
reg.resample(transform_registration)



plt.imshow(1*(reg.as_array(reg.moving)[530]>threshold[1]))
plt.savefig('/dtu-compute/msaca/output/fig_04_01.png')
plt.close()

plt.imshow(1*(reg.as_array(reg.moving)[:,140]>threshold[1]))
plt.savefig('/dtu-compute/msaca/output/fig_04_02.png')
plt.close()

plt.imshow(1*(reg.as_array(reg.moving)[:,:,530]>threshold[1]))
plt.savefig('/dtu-compute/msaca/output/fig_04_03.png')
plt.close()


plt.imshow(1*(reg.as_array(reg.fixed)[530]>threshold[0]))
plt.savefig('/dtu-compute/msaca/output/fig_05_01.png')
plt.close()

plt.imshow(1*(reg.as_array(reg.fixed)[:,140]>threshold[0]))
plt.savefig('/dtu-compute/msaca/output/fig_05_02.png')
plt.close()

plt.imshow(1*(reg.as_array(reg.fixed)[:,:,530]>threshold[0]))
plt.savefig('/dtu-compute/msaca/output/fig_05_03.png')
plt.close()


# Plot the result
plt.imshow(1*(reg.as_array(reg.moving)[430]>threshold[1]) - 1*(reg.as_array(reg.fixed)[430]>threshold[0]))
plt.savefig('/dtu-compute/msaca/output/fig_06_01.png')
plt.close()

plt.imshow(1*(reg.as_array(reg.moving)[:,140]>threshold[1]) - 1*(reg.as_array(reg.fixed)[:,140]>threshold[0]))
plt.savefig('/dtu-compute/msaca/output/fig_06_02.png')
plt.close()

plt.imshow(1*(reg.as_array(reg.moving)[:,:,430]>threshold[1]) - 1*(reg.as_array(reg.fixed)[:,:,430]>threshold[0]))
plt.savefig('/dtu-compute/msaca/output/fig_06_03.png')
plt.close()


# Plot the result
plt.imshow(1*(reg.as_array(reg.moving)[430]>threshold[1]) - 1*(reg.as_array(reg.fixed)[430]>threshold[0]))
plt.savefig('/dtu-compute/msaca/output/fig_04_01.png')
plt.close()

plt.imshow(1*(reg.as_array(reg.moving)[:,140]>threshold[1]) - 1*(reg.as_array(reg.fixed)[:,140]>threshold[0]))
plt.savefig('/dtu-compute/msaca/output/fig_04_02.png')
plt.close()

plt.imshow(1*(reg.as_array(reg.moving)[:,:,430]>threshold[1]) - 1*(reg.as_array(reg.fixed)[:,:,430]>threshold[0]))
plt.savefig('/dtu-compute/msaca/output/fig_04_03.png')
plt.close()


header = """\
# README

This file contains the code for the script and some additional information:
- This script generates a transformation object by registration and saves it. To
 apply the registrating transformation, load in the data as is done in this script,
 generate a registrator object, and apply the registration.
- The script is self-contained and generates this README file.
"""

# Name of the script (this file itself)
script_name = __file__

# Name of the output README file
readme_name = "/zhome/71/c/146676/main/Transformations/README_registrate_XA_to_NA.txt"

# Open the README file for writing
with open(readme_name, "w") as readme:
    # Write the header
    readme.write(header)
    readme.write("\n")
    readme.write("=" * 50 + "\n\n")  # Separator

    # Write the script code
    readme.write("## Script Code:\n\n")
    with open(script_name, "r") as script:
        for line in script:
            readme.write(line)

print(f"README file '{readme_name}' has been created.")

path = '/zhome/71/c/146676/main/Transformations/transformation_XA_to_NA.tfm'
sitk.WriteTransform(reg.transformation_history, path)