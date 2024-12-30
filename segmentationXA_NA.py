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

reg_median = reg.copy()
reg.resample(transform,ignore_flag = True)

# Compute meteorite mask and resample it to the correct size
factor = 4
reg.compute_stone_boundaries(thresholds = threshold, factor = factor) # saved in reg.fixed_seg and reg.moving_seg
initial_mask = reg.moving_seg
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


sampling_rate = 0.01  # Probability of True (30%)
shape = np.shape(mask)
random_mask = np.random.rand(*shape) < sampling_rate
mask = mask*random_mask
mask = mask.astype(bool)


fixed_values = fixed[mask]
moving_values = moving[mask]

bins = 300
heatmap, xedges, yedges = np.histogram2d(fixed_values, moving_values, bins=(bins, bins),
    range=[[fixed_values.min(), fixed_values.max()], [moving_values.min(), moving_values.max()]])
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
x_tick_labels = np.linspace(fixed_values.min(), fixed_values.max(), num=nticks)

# For the y-axis (moving values)
y_tick_positions = np.linspace(0, len(yedges) - 1, num=nticks) 
y_tick_labels = np.linspace(moving_values.min(), moving_values.max(), num=nticks)
x_tick_labels = [f"{x:.3f}" for x in x_tick_labels]
y_tick_labels = [f"{y:.3f}" for y in y_tick_labels]

# Set the tick labels on both axes
plt.xticks(x_tick_positions, x_tick_labels)
plt.yticks(y_tick_positions, y_tick_labels)

plt.savefig('/dtu-compute/msaca/output/fig_07_01.png')
plt.close()


# Apply median filtering before making the heatmap
reg_median.moving = sitk.SmoothingRecursiveGaussian(reg_median.moving, sigma=0.005)
reg_median.fixed = sitk.SmoothingRecursiveGaussian(reg_median.fixed, sigma=0.01)

reg_median.resample(transform,ignore_flag = True)

fixed_median = reg_median.as_array(reg_median.fixed)
moving_median = reg_median.as_array(reg_median.moving)

fixed_values = reg_median.fixed[mask]
moving_values = reg_median.moving[mask]

bins = 300
heatmap, xedges, yedges = np.histogram2d(fixed_values, moving_values, bins=(bins, bins),
    range=[[fixed_values.min(), fixed_values.max()], [moving_values.min(), moving_values.max()]])
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
x_tick_labels = np.linspace(fixed_values.min(), fixed_values.max(), num=nticks)

# For the y-axis (moving values)
y_tick_positions = np.linspace(0, len(yedges) - 1, num=nticks) 
y_tick_labels = np.linspace(moving_values.min(), moving_values.max(), num=nticks)
x_tick_labels = [f"{x:.3f}" for x in x_tick_labels]
y_tick_labels = [f"{y:.3f}" for y in y_tick_labels]

# Set the tick labels on both axes
plt.xticks(x_tick_positions, x_tick_labels)
plt.yticks(y_tick_positions, y_tick_labels)

plt.savefig('/dtu-compute/msaca/output/fig_08_01.png')
plt.close()