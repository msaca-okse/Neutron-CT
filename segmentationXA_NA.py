import registrator
import plot_library as pl
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from cil.io import TIFFStackReader
import random
import SimpleITK as sitk


def create_heatmap(fixed_values, moving_values, path_image_log, path_image, path_matrix, bins = 300, nticks = 10,
    range_fixed = [0.35,0.65], range_moving = [-2,5], title = None):

    heatmap, xedges, yedges = np.histogram2d(fixed_values, moving_values, bins=(bins, bins),
        range=[range_fixed, range_moving])
    np.save(path_matrix, heatmap)
    log_heatmap = np.log1p(heatmap)

    # Plot the heatmap
    plt.figure(figsize=(15, 15))
    plt.imshow(log_heatmap.T[0:bins,0:bins], origin='lower', aspect='auto', cmap='YlGnBu')
    plt.colorbar(label='log-Frequency')
    if title is not None:
        plt.title(title)
    plt.xlabel('Fixed: Neutron')
    plt.ylabel('Moving: Xray')

    x_tick_positions = np.linspace(0, len(xedges) - 1, num=nticks) 
    x_tick_labels = np.linspace(range_fixed[0], range_fixed[1], num=nticks)

    # For the y-axis (moving values)
    y_tick_positions = np.linspace(0, len(yedges) - 1, num=nticks) 
    y_tick_labels = np.linspace(range_moving[0], range_moving[1], num=nticks)
    x_tick_labels = [f"{x:.3f}" for x in x_tick_labels]
    y_tick_labels = [f"{y:.3f}" for y in y_tick_labels]

    # Set the tick labels on both axes
    plt.xticks(x_tick_positions, x_tick_labels)
    plt.yticks(y_tick_positions, y_tick_labels)

    plt.savefig(path_image_log)
    plt.close()

    # Plot the heatmap
    plt.figure(figsize=(15, 15))
    plt.imshow(heatmap.T[0:bins,0:bins], origin='lower', aspect='auto', cmap='jet')
    plt.colorbar(label='Frequency')
    if title is not None:
        plt.title(title)
    plt.xlabel('Fixed: Neutron')
    plt.ylabel('Moving: Xray')

    x_tick_positions = np.linspace(0, len(xedges) - 1, num=nticks) 
    x_tick_labels = np.linspace(range_fixed[0], range_fixed[1], num=nticks)

    # For the y-axis (moving values)
    y_tick_positions = np.linspace(0, len(yedges) - 1, num=nticks) 
    y_tick_labels = np.linspace(range_moving[0], range_moving[1], num=nticks)
    x_tick_labels = [f"{x:.3f}" for x in x_tick_labels]
    y_tick_labels = [f"{y:.3f}" for y in y_tick_labels]

    # Set the tick labels on both axes
    plt.xticks(x_tick_positions, x_tick_labels)
    plt.yticks(y_tick_positions, y_tick_labels)

    plt.savefig(path_image)
    plt.close()


# Paths to images
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

threshold = [0.505,0.31]
reg = registrator.Registrator()
reg.load_images(fixed_array=recon_neutron,moving_array=recon_xray)

filename = '/dtu-compute/msaca/sliceA_xray_pc/estrids_recon/full_XA_NA.json'
meta_fixed, meta_moving, transform = registrator.load_data(filename)

reg.resample(transform,ignore_flag = True)

# Compute meteorite mask and resample it to the correct size
factor = 4
reg.compute_stone_boundaries(thresholds = threshold, factor = factor) # saved in reg.fixed_seg and reg.moving_seg
initial_mask = reg.fixed_seg
mask = sitk.Resample(
    initial_mask,
    reg.fixed,
    sitk.Transform(),
    sitk.sitkNearestNeighbor,  # Use nearest neighbor to preserve binary mask values
    initial_mask.GetPixelID()
)

mask = reg.as_array(mask)
fixed = reg.as_array(reg.fixed)
moving = reg.as_array(reg.moving)


sampling_rate = 1
shape = np.shape(mask)
random_mask = np.random.rand(*shape) < sampling_rate
mask = mask*random_mask
mask = mask.astype(bool)


fixed_values = fixed[mask]
moving_values = moving[mask]

prefix = '/dtu-compute/msaca/output/'

# Create 2d histogram/heatmap of non postfiltered images
path_image_log = prefix + 'heatmap_log_01.png'
path_image = prefix + 'heatmap_01.png'
path_matrix = prefix + 'heatmap_matrix_01.npy'
title = 'FBP recons, slice A, background masked, no postfilter'
create_heatmap(fixed_values, moving_values, path_image_log, path_image, path_matrix, bins = 400, nticks = 15,
    range_fixed = [0.35,0.7], range_moving = [-2,5], title = title)








# Apply gaussian filtering before making the heatmap
fixed_filtered_ = sitk.SmoothingRecursiveGaussian(reg.fixed, sigma=0.002)
moving_filtered_ = sitk.SmoothingRecursiveGaussian(reg.moving, sigma=0.002)
fixed_filtered = reg.as_array(fixed_filtered_)
moving_filtered = reg.as_array(moving_filtered_)
fixed_values = fixed_filtered[mask]
moving_values = moving_filtered[mask]

# Create 2d histogram/heatmap of postfiltered images
path_image_log = prefix + 'heatmap_log_02.png'
path_image = prefix + 'heatmap_02.png'
path_matrix = prefix + 'heatmap_matrix_02.npy'
title = 'FBP recons, slice A, background masked, gaussian filter, kernel=0.002'
create_heatmap(fixed_values, moving_values, path_image_log, path_image, path_matrix, bins = 400, nticks = 15,
    range_fixed = [0.35,0.7], range_moving = [-2,5], title = title)


k = 280
path = '/dtu-compute/msaca/output/vert_slice_N.png'
plt.imshow(fixed_filtered[:,k,:], cmap = 'gray')
plt.clim([0.425,0.7])
plt.title('Neutron reconstruction')
plt.savefig(path, dpi=600)
plt.close()

path ='/dtu-compute/msaca/output/vert_slice_X.png'
plt.imshow(moving_filtered[:,k,:], cmap = 'gray')
plt.clim([-1.5,3.5])
plt.title('X-ray reconstruction')
plt.savefig(path, dpi=600)
plt.close()


def value_segmenter(neutron, xray, N_limits, X_limits):
    seg_N = (neutron>N_limits[0]) * (neutron<N_limits[1])
    seg_X = (xray>X_limits[0]) * (xray<X_limits[1])
    out = seg_N * seg_X
    return out

segmentation = np.zeros(np.shape(fixed_filtered))
g1 = value_segmenter(fixed_filtered, moving_filtered, [0.450, 0.49], [-0.75, 1])
g2 = value_segmenter(fixed_filtered, moving_filtered, [0, 1], [-0.2, 0.2])
g3 = value_segmenter(fixed_filtered, moving_filtered, [0.48, 0.53], [0.3, 0.7])
g4 = value_segmenter(fixed_filtered, moving_filtered, [0.48, 0.55], [0.7, 1.5])
segmentation[g1*mask] = 1
segmentation[g2*mask] = 2
segmentation[g3*mask] = 3
segmentation[g4*mask] = 4
segmentation[g5*mask] = 5
segmentation[g6*mask] = 6
segmentation[~mask] = -1
path = '/dtu-compute/msaca/output/vert_slice_segm2.png'
plt.imshow(segmentation[:,k,:])
plt.colorbar()
plt.title('Segmentation, see script for color explanation')
plt.savefig(path, dpi=600)
plt.close()





"""

# Apply gaussian filtering before making the heatmap
fixed_filtered_ = sitk.SmoothingRecursiveGaussian(reg.fixed, sigma=0.005)
moving_filtered_ = sitk.SmoothingRecursiveGaussian(reg.moving, sigma=0.005)
fixed_filtered = reg.as_array(fixed_filtered_)
moving_filtered = reg.as_array(moving_filtered_)
fixed_values = fixed_filtered[mask]
moving_values = moving_filtered[mask]

# Create 2d histogram/heatmap of postfiltered images
path_image_log = prefix + 'heatmap_log_03.png'
path_image = prefix + 'heatmap_03.png'
path_matrix = prefix + 'heatmap_matrix_03.npy'
title = 'FBP recons, slice A, background masked, gaussian filter, kernel=0.005'
create_heatmap(fixed_values, moving_values, path_image_log, path_image, path_matrix, bins = 400, nticks = 15,
    range_fixed = [0.4,0.7], range_moving = [-2,5], title = title)









# Apply gaussian filtering before making the heatmap
fixed_filtered_ = sitk.SmoothingRecursiveGaussian(reg.fixed, sigma=0.01)
moving_filtered_ = sitk.SmoothingRecursiveGaussian(reg.moving, sigma=0.01)
fixed_filtered = reg.as_array(fixed_filtered_)
moving_filtered = reg.as_array(moving_filtered_)
fixed_values = fixed_filtered[mask]
moving_values = moving_filtered[mask]

# Create 2d histogram/heatmap of postfiltered images
path_image_log = prefix + 'heatmap_log_04.png'
path_image = prefix + 'heatmap_04.png'
path_matrix = prefix + 'heatmap_matrix_04.npy'
title = 'FBP recons, slice A, background masked, gaussian filter, kernel=0.01'
create_heatmap(fixed_values, moving_values, path_image_log, path_image, path_matrix, bins = 400, nticks = 15,
    range_fixed = [0.4,0.7], range_moving = [-1,4], title = title)










# Apply gaussian filtering before making the heatmap
fixed_filtered_ = sitk.SmoothingRecursiveGaussian(reg.fixed, sigma=0.02)
moving_filtered_ = sitk.SmoothingRecursiveGaussian(reg.moving, sigma=0.02)
fixed_filtered = reg.as_array(fixed_filtered_)
moving_filtered = reg.as_array(moving_filtered_)
fixed_values = fixed_filtered[mask]
moving_values = moving_filtered[mask]

# Create 2d histogram/heatmap of postfiltered images
path_image_log = prefix + 'heatmap_log_05.png'
path_image = prefix + 'heatmap_05.png'
path_matrix = prefix + 'heatmap_matrix_05.npy'
title = 'FBP recons, slice A, background masked, gaussian filter, kernel=0.02'
create_heatmap(fixed_values, moving_values, path_image_log, path_image, path_matrix, bins = 400, nticks = 15,
    range_fixed = [0.4,0.7], range_moving = [-1,4], title = title)






# Apply median filtering before making the heatmap
fixed_filtered_ = sitk.Median(reg.fixed, radius = [2,2,2])
moving_filtered_ = sitk.Median(reg.moving, radius = [2,2,2])
fixed_filtered = reg.as_array(fixed_filtered_)
moving_filtered = reg.as_array(moving_filtered_)
fixed_values = fixed_filtered[mask]
moving_values = moving_filtered[mask]

# Create 2d histogram/heatmap of postfiltered images
path_image_log = prefix + 'heatmap_log_06.png'
path_image = prefix + 'heatmap_06.png'
path_matrix = prefix + 'heatmap_matrix_06.npy'
title = 'FBP recons, slice A, background masked, median filter, kernel size=[2,2,2]'
create_heatmap(fixed_values, moving_values, path_image_log, path_image, path_matrix, bins = 400, nticks = 15,
    range_fixed = [0.4,0.7], range_moving = [-1,4], title = title)







# Apply median filtering before making the heatmap
fixed_filtered_ = sitk.Median(reg.fixed, radius = [3,3,3])
moving_filtered_ = sitk.Median(reg.moving, radius = [3,3,3])
fixed_filtered = reg.as_array(fixed_filtered_)
moving_filtered = reg.as_array(moving_filtered_)
fixed_values = fixed_filtered[mask]
moving_values = moving_filtered[mask]

# Create 2d histogram/heatmap of postfiltered images
path_image_log = prefix + 'heatmap_log_07.png'
path_image = prefix + 'heatmap_07.png'
path_matrix = prefix + 'heatmap_matrix_07.npy'
title = 'FBP recons, slice A, background masked, median filter, kernel size=[3,3,3]'
create_heatmap(fixed_values, moving_values, path_image_log, path_image, path_matrix, bins = 400, nticks = 15,
    range_fixed = [0.4,0.7], range_moving = [-1,4], title = title)

"""
