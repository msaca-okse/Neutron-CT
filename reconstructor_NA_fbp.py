# This script loads in a reconstruction, does preprocessing (morph_spot_clean + rotation)
# Then a reconstruction is made using fbp. Both preprocessing and reconstruction is done
# in parallel, however the preprocessing must be finished once reconstruction can start

import sys
import os

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


##########################################################
# Part 1: Preprocessing


path = '/dtu-compute/msaca/sliceA_neutron_psi/OB2_DC2/OB2_DC2_#####.fits'
A = range(1,241)
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


path = '/dtu-compute/msaca/sliceA_neutron_psi/OB2_DC2/OB2_DC2_#####.fits'
A = range(243,483)
dc_paths = ma.generate_paths(path, A)
with Pool() as pool:
    dc_data = pool.map(load_file, dc_paths)

# Now compute the mean across all loaded files
# Convert list of arrays to a single array, assuming the arrays are of the same shape
dc = np.stack(dc_data)
dc = np.mean(dc, axis=0)

path = '/dtu-compute/msaca/sliceA_neutron_psi/ct_3x1126_60s/ct_3x1126_60s_#####.fits'
path_cache = '/dtu-compute/msaca/output/cache/spot_cleaned_#####.tiff'

def preprocess_projection(i):
    A = [3*i-2, 3*i-1, 3*i]
    data_paths = ma.generate_paths(path, A)
    data = ma.fits_loader(data_paths)
    data = np.median(data, axis=0)
    norm = imgalg.NormalizeImage(True) # True for use logarithm
    norm.setReferences(ob,dc)
    norm.process(data)
    msc = imgalg.MorphSpotClean()
    msc.setCleanMethod(detectionMethod=imgalg.MorphDetectAllSpots, cleanMethod=imgalg.MorphCleanReplace);
    msc.setLimits(
        applyClamp=True,  # Apply clamping
        vmin=-0.1,         # Minimum pixel value (e.g., for normalization)
        vmax=12,       # Maximum pixel value (e.g., for normalization)
        maxarea=30       # Maximum blob area (e.g., to filter out large blobs)
    )
    msc.setEdgeConditioning(5)
    msc.process(data,th=[0.0, 0.0],sigma=[0.01, 0.01])
    A = [i]
    temp_path = ma.generate_paths(path_cache, A)[0]
    tifffile.imwrite(temp_path, data)



# IMPORTaNT, set the environment variable "export NUM_PROCS=$LSB_DJOB_NUMPROC" in the job script. It should be the number of cores.
with Pool() as pool:
    pool.map(preprocess_projection, range(1, 1126))