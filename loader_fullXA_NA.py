import registrator
import plot_library as pl
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from cil.io import TIFFStackReader

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


# Plot the result
plt.imshow(1*(reg.as_array(reg.moving)[430]>threshold[1]) - 1*(reg.as_array(reg.fixed)[430]>threshold[0]))
plt.savefig('/dtu-compute/msaca/output/fig_05_01.png')
plt.close()

plt.imshow(1*(reg.as_array(reg.moving)[:,340]>threshold[1]) - 1*(reg.as_array(reg.fixed)[:,340]>threshold[0]))
plt.savefig('/dtu-compute/msaca/output/fig_05_02.png')
plt.close()

plt.imshow(1*(reg.as_array(reg.moving)[:,:,430]>threshold[1]) - 1*(reg.as_array(reg.fixed)[:,:,430]>threshold[0]))
plt.savefig('/dtu-compute/msaca/output/fig_05_03.png')
plt.close()

# Plot the segmented result:
factor = 2
reg.compute_stone_boundaries(thresholds = threshold, factor = factor) # saved in reg.fixed_seg and reg.moving_seg
plt.imshow(1*(reg.as_array(reg.moving_seg)[430//factor]>threshold[1]) - 1*(reg.as_array(reg.fixed_seg)[430//factor]>threshold[0]))
plt.savefig('/dtu-compute/msaca/output/fig_06_01.png')
plt.close()

plt.imshow(1*(reg.as_array(reg.moving_seg)[:,340//factor]>threshold[1]) - 1*(reg.as_array(reg.fixed_seg)[:,340//factor]>threshold[0]))
plt.savefig('/dtu-compute/msaca/output/fig_06_02.png')
plt.close()

plt.imshow(1*(reg.as_array(reg.moving_seg)[:,:,430//factor]>threshold[1]) - 1*(reg.as_array(reg.fixed_seg)[:,:,430//factor]>threshold[0]))
plt.savefig('/dtu-compute/msaca/output/fig_06_03.png')
plt.close()