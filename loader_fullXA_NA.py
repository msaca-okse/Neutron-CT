import registrator
import plot_library as pl
import tifffile
import numpy as np
import matplotlib.pyplot as plt

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

reg = registrator.Registrator()
reg.load_images(fixed_array=recon_neutron,moving_array=recon_xray)

filename = '/dtu-compute/msaca/sliceA_xray_pc/estrids_recon/full_XA_NA.json'
meta_fixed, meta_moving, transform = registrator.load_data(filename)

reg.resample(transform,ignore_flag = True)


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

