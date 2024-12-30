### Registrator scripted:
## Registering full scale XA to NA

from cil.io import ZEISSDataReader
import numpy as np
import os
os.chdir('/zhome/71/c/146676/Desktop/msaca/main/')  # Navigate into a subdirectory
import matplotlib.pyplot as plt
from cil.io import TIFFStackReader, TIFFWriter
import plot_library as pl
import SimpleITK as sitk
import importlib
import registrator
import plotly.io as pio
from cil.framework import ImageData, ImageGeometry
import tifffile

# Define paths
reconpath0 = "/dtu-compute/msaca/sliceA_xray_pc/estrids_recon/xray_3um_stitch_total.tif"
reconpath_neutron = "/dtu-compute/msaca/sliceA_neutron_psi/My_reconstructions/"

# Load x-ray image
with tifffile.TiffFile(reconpath0) as tiff:
    recon_xray_0 = tiff.asarray()

recon_xray_ = recon_xray_0[:2800]
recon_xray = recon_xray_.astype(np.float32)
#recon_xray = pl.array_normalizer(recon_xray_.astype(np.float32))

# Load Neutron image
reader = TIFFStackReader(reconpath_neutron)
recon_neutron = pl.array_normalizer(reader.read())

# Initialice the registrator class and set thresholds
reg = registrator.Registrator()
reg.load_images(fixed_array=recon_neutron,moving_array=recon_xray)
threshold = [0.505,0.31]

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

# Align the moments. Since the xray image is around 3 times larger in each dimension, i decrease its size by 3
rot_matrix = [
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 1]
            ]
matrix = [elem for row in rot_matrix for elem in row]
translation = (0,0,0)
transform = reg.transformation(type='Affine', matrix = matrix,translation=translation)

# Do the resampling and plot
reg.resample(transform)
reg.compute_stone_boundaries(thresholds = threshold, factor = 5) # saved in reg.fixed_seg and reg.moving_seg
reg.plot3d(reg.fixed_seg, moments=fixed_moments, path = '/dtu-compute/msaca/output/fig_01_01.html')
reg.plot3d(reg.moving_seg, moments=moving_moments, path = '/dtu-compute/msaca/output/fig_01_02.html')

# Flip along axis 0
angle_radians = np.deg2rad(180)
rot_matrix = [
                [-1, 0, 0],
                [0, 1, 0],
                [0, 0, -1]
            ]
matrix = [elem for row in rot_matrix for elem in row]
translation = (0,0,0)
transform = reg.transformation(type='Affine', matrix = matrix,translation=translation)
reg.resample(transform)
reg.compute_stone_boundaries(thresholds = threshold, factor = 5) # saved in reg.fixed_seg and reg.moving_seg

reg.plot3d(reg.fixed_seg, moments=fixed_moments, path = '/dtu-compute/msaca/output/fig_02_01.html')
reg.plot3d(reg.moving_seg, moments=moving_moments, path = '/dtu-compute/msaca/output/fig_02_02.html')

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