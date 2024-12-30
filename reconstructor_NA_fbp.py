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


##########################################################
# Part 1: Preprocessing


path = '/dtu-compute/msaca/sliceA_neutron_psi/OB2_DC2/OB2_DC2_#####.fits'
A = range(1,241)
ob_paths = ma.generate_paths(path, A)
ob = ma.fits_loader(ob_paths)
ob = median_arr = np.median(ob, axis=0)

path = '/dtu-compute/msaca/sliceA_neutron_psi/OB2_DC2/OB2_DC2_#####.fits'
A = range(243,483)
dc_paths = ma.generate_paths(path, A)
dc = ma.fits_loader(dc_paths)
dc = median_arr = np.median(dc, axis=0)

path = '/dtu-compute/msaca/sliceA_neutron_psi/ct_3x1126_60s/ct_3x1126_60s_#####.fits'
path_cache = '/dtu-compute/msaca/output/cache/spot_cleaned_#####.tiff'

def preprocess_projection(i):
    A = [3*i-2, 3*i-1, 3*i]
    data_paths = ma.generate_paths(path, A)
    data = ma.fits_loader(dc_paths)
    data = median_arr = np.median(data, axis=0)
    norm = imgalg.NormalizeImage(True) # True for use logarithm
    norm.setReferences(ob,dc)
    norm.process(data)
    msc = imgalg.MorphSpotClean()
    msc.setCleanMethod(detectionMethod=imgalg.MorphDetectAllSpots, cleanMethod=imgalg.MorphCleanReplace);
    msc.setLimits([-0.1,12])
    msc.setMaxArea(30)
    msc.setEdgeConditioning(5)
    msc.process(data,th=[0.0, 0.0],sigma=[0.01, 0.01])
    A = [i]
    temp_path = ma.generate_paths(path_cache, A)[0]
    tifffile.imwrite(temp_path, data)



num_procs = int(os.getenv("NUM_PROCS", 1))
# IMPORTaNT, set the environment variable "export NUM_PROCS=$LSB_DJOB_NUMPROC" in the job script. It should be the number of cores.
with Pool(processes=num_procs) as pool:
    pool.map(preprocess_projection, range(1, 1126))