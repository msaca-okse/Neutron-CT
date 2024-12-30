import registrator
import plot_library as pl
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from cil.io import TIFFStackReader
import random
import SimpleITK as sitk

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


sampling_rate = 0.05  # Probability of True (30%)
shape = np.shape(mask)
random_mask = np.random.rand(*shape) < sampling_rate
mask = mask*random_mask
mask = mask.astype(bool)


fixed_values = fixed[mask]
moving_values = moving[mask]

bins = 300
range_fixed = [0.3, 0.7]
range_moving = [-3, 6]
heatmap, xedges, yedges = np.histogram2d(fixed_values, moving_values, bins=(bins, bins),
    range=[range_fixed, range_moving])
log_heatmap = np.log1p(heatmap)

# Plot the heatmap
plt.figure(figsize=(10, 10))
plt.imshow(log_heatmap.T[0:bins,0:bins], origin='lower', aspect='auto', cmap='YlGnBu')
plt.colorbar(label='Frequency')
plt.title('2D Heatmap/histogram of neutron/xray attenuation (arbitrary values)')
plt.xlabel('Fixed: Neutron')
plt.ylabel('Moving: Xray')

nticks = 10
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

plt.savefig('/dtu-compute/msaca/output/fig_07_01.png')
plt.close()


# Apply gaussian filtering before making the heatmap
fixed_filtered_ = sitk.SmoothingRecursiveGaussian(reg.fixed, sigma=0.005)
moving_filtered_ = sitk.SmoothingRecursiveGaussian(reg.moving, sigma=0.005)


fixed_filtered = reg.as_array(fixed_filtered_)
moving_filtered = reg.as_array(moving_filtered_)

fixed_values = fixed_filtered[mask]
moving_values = moving_filtered[mask]

bins = 300
range_fixed = [0.3, 0.7]
range_moving = [-3, 6]
heatmap, xedges, yedges = np.histogram2d(fixed_values, moving_values, bins=(bins, bins),
    range=[range_fixed, range_moving])
log_heatmap = np.log1p(heatmap)

# Plot the heatmap
plt.figure(figsize=(10, 10))
plt.imshow(log_heatmap.T[0:bins,0:bins], origin='lower', aspect='auto', cmap='YlGnBu')
plt.colorbar(label='Frequency')
plt.title('2D Heatmap/histogram of neutron/xray attenuation (arbitrary values)')
plt.xlabel('Fixed: Neutron')
plt.ylabel('Moving: Xray')

nticks = 10
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

plt.savefig('/dtu-compute/msaca/output/fig_08_01.png')
plt.close()


# Apply median filtering before making the heatmap
fixed_filtered_ = sitk.Median(reg.fixed, radius = [2,2,2])
moving_filtered_ = sitk.Median(reg.moving, radius = [2,2,2])


fixed_filtered = reg.as_array(fixed_filtered_)
moving_filtered = reg.as_array(moving_filtered_)

fixed_values = fixed_filtered[mask]
moving_values = moving_filtered[mask]

bins = 300
range_fixed = [0.3, 0.7]
range_moving = [-3, 6]
heatmap, xedges, yedges = np.histogram2d(fixed_values, moving_values, bins=(bins, bins),
    range=[range_fixed, range_moving])
log_heatmap = np.log1p(heatmap)

# Plot the heatmap
plt.figure(figsize=(10, 10))
plt.imshow(log_heatmap.T[0:bins,0:bins], origin='lower', aspect='auto', cmap='YlGnBu')
plt.colorbar(label='Frequency')
plt.title('2D Heatmap/histogram of neutron/xray attenuation (arbitrary values)')
plt.xlabel('Fixed: Neutron')
plt.ylabel('Moving: Xray')

nticks = 10
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

plt.savefig('/dtu-compute/msaca/output/fig_09_01.png')
plt.close()