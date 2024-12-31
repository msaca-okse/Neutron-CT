# This script loads in a reconstruction, does preprocessing (morph_spot_clean + rotation)
# Then a reconstruction is made using fbp. Both preprocessing and reconstruction is done
# in parallel, however the preprocessing must be finished once reconstruction can start

import sys
import os
import time

# Add the desired directory to the sys.path
path_to_add = '/dtu-compute/msaca/muhrec_folder2/build-imagingsuite/Release/lib/'
sys.path.append(path_to_add)
path_to_add = '/dtu-compute/msaca/muhrec_folder2/amglib/CBCTCalibration/'
sys.path.append(path_to_add)
path_to_add = '/dtu-compute/msaca/muhrec_folder2/imagingsuite/package/'
sys.path.append(path_to_add)
path_to_add = '/zhome/71/c/146676/Desktop/msaca/main/'
sys.path.append(path_to_add)

import SimpleITK as sitk
import imgalg
import numpy as np
from astropy.io import fits
import module_auxiliary as ma
import tifffile
from multiprocessing import Pool
import matplotlib.pyplot as plt
import image_utils as iu

start_time = time.time()
##########################################################
# Part 1: Preprocessing


path = '/dtu-compute/msaca/sliceA_neutron_psi/OB/OB_start_#####.fits'
A = range(1,31)
ob_paths = ma.generate_paths(path, A)


def load_file(path):
    with fits.open(path) as hdul:
        # Assume the data is in the primary HDU
        return hdul[0].data
with Pool() as pool:
    ob_data = pool.map(load_file, ob_paths)

# Now compute the mean across all loaded files
# Convert list of arrays to a single array, assuming the arrays are of the same shape
ob = np.stack(ob_data)
ob = np.mean(ob, axis=0)
ob = ob[180:1760,240:1820]

path = '//dtu-compute/msaca/sliceA_neutron_psi/DC/DC_#####.fits'
A = range(1,31)
dc_paths = ma.generate_paths(path, A)
with Pool() as pool:
    dc_data = pool.map(load_file, dc_paths)


# Now compute the mean across all loaded files
# Convert list of arrays to a single array, assuming the arrays are of the same shape
dc = np.stack(dc_data)
dc = np.mean(dc, axis=0)
dc = dc[180:1760,240:1820]

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Loading time: {elapsed_time:.2f} seconds")

path = '/dtu-compute/msaca/sliceA_neutron_psi/ct_3x1126_60s/ct_3x1126_60s_#####.fits'
path_cache = '/dtu-compute/msaca/output/cache/spot_cleaned_#####.tiff'

batch = np.linspace(1,1126,200).astype(np.uint16)

def preprocess_projection(batch_idx):
    a = range(batch[batch_idx],batch[batch_idx+1])
    Data = []
    for i in a:
        A = [3*i-2, 3*i-1, 3*i]
        data_paths = ma.generate_paths(path, A)
        data = ma.fits_loader(data_paths)
        data = data[:,180:1760,240:1820]
        data = np.median(data, axis=0)
        Data.append(data)

    Data = np.stack(Data)    
    Data = (-np.log((np.abs(Data-dc[np.newaxis])+1)/(1+np.abs(ob[np.newaxis]-dc[np.newaxis])))).astype(np.float32)
    
    plt.imshow(Data[0])
    plt.colorbar()
    plt.clim([-0.3,0.6])
    plt.savefig('/dtu-compute/msaca/output/cache/output_image.png', format='png')
    plt.close()

    Data = iu.morph_spot_clean(Data,th_peaks=0.999,th_holes=0.999,method=0)

    plt.imshow(Data[0])
    plt.colorbar()
    plt.clim([-0.3,0.6])
    plt.savefig('/dtu-compute/msaca/output/cache/output_image2.png', format='png')
    plt.close()

    count = 0
    for i in a:
        A = [i]
        temp_path = ma.generate_paths(path_cache, A)[0]
        tifffile.imwrite(temp_path, Data[count].astype(np.float32))
        count = count+1



# IMPORTaNT, set the environment variable "export NUM_PROCS=$LSB_DJOB_NUMPROC" in the job script. It should be the number of cores.
with Pool() as pool:
    pool.map(preprocess_projection, range(len(batch)-1))
#for i in range(1,1126):
#    preprocess_projection(i)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Preprocess time: {elapsed_time:.2f} seconds")