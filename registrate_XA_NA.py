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


# Load the images
paths_XA = generate_stitched_paths(dataset = 'fbp', folders = [1,2,3,4,5],
    start_indices = [350, 94, 94, 94, 94], end_indices = [693, 694, 694, 694, 788])

ny, nx = np.shape(tifffile.imread(paths_XA[0]))
recon_xray = np.zeros((len(paths_XA), ny, nx))

def loader(path):
    return tifffile.imread(path)

with Pool() as pool:
    recon_XA = pool.map(loader, paths_XA)

recon_XA = np.stack(recon_XA)
recon_XA = np.clip(recon_XA, a_min = -0.1, a_max = 0.1)

path_NA = '/dtu-compute/msaca/output/tv_recon/slice_tv_####.tiff'
A = np.arange(0, 1788)
paths_NA = ma.generate_paths(path_NA, A)

with Pool() as pool:
    recon_NA = pool.map(loader, paths_NA)

recon_NA = np.stack(recon_NA)
recon_NA = np.clip(recon_XA, a_min = -0.1, a_max = 0.1)

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
reg.plot3d(reg.fixed_seg, moments=fixed_moments, path = '/dtu-compute/msaca/output/fig_00_01.html')
reg.plot3d(reg.moving_seg, moments=moving_moments, path = '/dtu-compute/msaca/output/fig_00_02.html')
"""
# Location has now been initialized, now do loops of registration:

# Loop 1
transform_registration= reg.register(learning_rate = 1, sampling_percentage = 0.1, max_iter = 100,
                                        metric_type = 'ms', optimizer_type = 'gd',
                                        smoothing = 0, shrinking = 7, thresholds=threshold,
                                        smooth_fixed = False, smooth_moving = False)
reg.resample(transform_registration)


# Loop 2:
transform_registration= reg.register(learning_rate = 0.5, sampling_percentage = 0.1, max_iter = 100,
                                        metric_type = 'ms', optimizer_type = 'gd',
                                        smoothing = 0, shrinking = 7, thresholds=threshold,
                                        smooth_fixed = False, smooth_moving = False)
reg.resample(transform_registration)

# Loop 3:
transform_registration= reg.register(learning_rate = 0.2, sampling_percentage = 0.1, max_iter = 100,
                                        metric_type = 'ms', optimizer_type = 'gd',
                                        smoothing = 0, shrinking = 5, thresholds=threshold,
                                        smooth_fixed = False, smooth_moving = False)
reg.resample(transform_registration)

# Loop 4:
transform_registration= reg.register(learning_rate = 0.1, sampling_percentage = 0.2, max_iter = 100,
                                        metric_type = 'ms', optimizer_type = 'gd',
                                        smoothing = 0, shrinking = 4, thresholds=threshold,
                                        smooth_fixed = False, smooth_moving = False)
reg.resample(transform_registration)

# Loop 5:
transform_registration= reg.register(learning_rate = 0.1, sampling_percentage = 0.2, max_iter = 100,
                                        metric_type = 'ms', optimizer_type = 'gd',
                                        smoothing = 0, shrinking = 3, thresholds=threshold,
                                        smooth_fixed = False, smooth_moving = False)
reg.resample(transform_registration)


# Plot the result
plt.imshow(1*(reg.as_array(reg.moving)[430]>threshold[1]) - 1*(reg.as_array(reg.fixed)[430]>threshold[0]))
plt.savefig('/dtu-compute/msaca/output/fig_04_01.png')
plt.close()

plt.imshow(1*(reg.as_array(reg.moving)[:,340]>threshold[1]) - 1*(reg.as_array(reg.fixed)[:,340]>threshold[0]))
plt.savefig('/dtu-compute/msaca/output/fig_04_02.png')
plt.close()

plt.imshow(1*(reg.as_array(reg.moving)[:,:,430]>threshold[1]) - 1*(reg.as_array(reg.fixed)[:,:,430]>threshold[0]))
plt.savefig('/dtu-compute/msaca/output/fig_04_03.png')
plt.close()

# Save the transformation
path = '/dtu-compute/msaca/sliceA_xray_pc/estrids_recon/full_XA_NA.json'
registrator.save_registration_data(path, reg.fixed, reg.moving,reg.transformation_history)

# Save a readme file for loading in the registerred images
string = "Estrids full recon was used. Slices 0 to 2800 were used. See the script 'loader_fullXA_NA.py' to load the registered volumes"
file_path = '/dtu-compute/msaca/sliceA_xray_pc/estrids_recon/README_full_XA_NA.txt'
with open(file_path, 'w') as file:
    file.write(custom_string)

"""