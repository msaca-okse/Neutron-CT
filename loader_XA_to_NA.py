import numpy as np
import os
os.chdir('/zhome/71/c/146676/main/')  # Navigate into a subdirectory
import matplotlib.pyplot as plt
import plot_library as pl
import SimpleITK as sitk
import registrator
import plotly.io as pio
from cil.framework import ImageData, ImageGeometry
import tifffile
import stitcher_XA
from multiprocessing import Pool
import module_auxiliary as ma

def loader_XA(path):
    image = tifffile.imread(path)[::compression_v,::compression_v]
    ny,nx = np.shape(image)
    pad = 200
    image2 = np.zeros((ny, nx + 2*pad), dtype=np.float32)
    image2[:, pad:-pad] = image
    return image2

def loader_NA(path):
    return tifffile.imread(path)[:1600,:1600]

def load_registered_data(compression = 1):
    global compression_v 
    compression_v = compression
    # Load the images
    paths_XA = stitcher_XA.generate_stitched_paths(dataset = 'fbp', folders = [1,2,3,4,5],
        start_indices = [300, 94, 94, 94, 94], end_indices = [693, 694, 694, 694, 788])[::compression_v]




    with Pool() as pool:
        recon_XA = pool.map(loader_XA, paths_XA)

    recon_XA = np.stack(recon_XA)
    recon_XA = np.clip(recon_XA, a_min = -0.1, a_max = 0.1)

    path_NA = '/dtu-compute/msaca/output/tv_recon/slice_tv_####.tiff'
    A = np.arange(0, 1788)
    paths_NA = ma.generate_paths(path_NA, A)


    with Pool() as pool:
        recon_NA = pool.map(loader_NA, paths_NA)

    recon_NA = np.stack(recon_NA)
    recon_NA = np.clip(recon_NA, a_min = -0.1, a_max = 0.1)

    # Initialice the registrator class and set thresholds
    reg = registrator.Registrator()
    reg.load_images(fixed_array=recon_NA,moving_array=recon_XA)
    threshold = [0.02,0.02]
    path = 'transformation_XA_to_NA_compression' + str(compression) + '.tfm'
    transform = sitk.ReadTransform(path)
    reg.resample(transform,ignore_flag = True)

    return reg