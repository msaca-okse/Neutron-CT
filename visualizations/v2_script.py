import os
import sys
os.chdir('/zhome/71/c/146676/main/')
sys.path.append('/zhome/71/c/146676/main/')
from helpers import module_auxiliary as ma
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from loaders import loader_XA_to_NA
import SimpleITK as sitk
xray_slices = range(300,2700)
neutron_slices = range(0,1600)
output_volume = [[0,1600],[None], [None]]
reg = loader_XA_to_NA.load_subset_of_registered_data(dataset_XA='tv', dataset_NA='tv' , h=1, xray_slices = xray_slices,
    neutron_slices = neutron_slices, output_volume = output_volume)

fixed = sitk.GetArrayFromImage(reg.fixed)[::2,::2,::2]
moving = sitk.GetArrayFromImage(reg.moving)[::2,::2,::2]
np.save( '/dtu-compute/msaca/cache/low_res_recon/neutron.npy',fixed)
np.save('/dtu-compute/msaca/cache/low_res_recon/xray.npy',moving)