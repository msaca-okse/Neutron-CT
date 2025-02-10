# This script loads in a reconstruction, does preprocessing (morph_spot_clean + rotation)
# Then a reconstruction is made using fbp. Both preprocessing and reconstruction is done
# in parallel, however the preprocessing must be finished once reconstruction can start

import sys
import os
import time
import numpy as np
import tracemalloc
# Add the desired directory to the sys.path
path_to_add = '/zhome/71/c/146676/Desktop/msaca/main/'
sys.path.append(path_to_add)

import SimpleITK as sitk
from astropy.io import fits
import module_auxiliary as ma
import tifffile
from multiprocessing import Pool
import matplotlib.pyplot as plt
import image_utils as iu
import extended_data as ed
import CTcorrector as ctc
from cil.plugins.astra import FBP
from cil.framework import AcquisitionGeometry, AcquisitionData, ImageGeometry, ImageData, BlockDataContainer
from cil.processors import PaganinProcessor

try:
    alpha = float(os.getenv("ALPHA"))
except:
    alpha = 0.2

try:
    stride = int(os.getenv("STRIDE"))
except:
    stride = 10

output = False
Total_angles = 6000
N_batches = 10
folders = np.array([1,2,3,4,5,6,7])

generic_pp = '/dtu-compute/msaca/sliceA_xray_pc/compressed_XA_###_.'
generic_pps = ma.generate_paths(generic_pp,folders)


generic_pc = '/dtu-compute/msaca/sliceA_xray_pc/output/cache/sino_###.'
generic_pcs = ma.generate_paths(generic_pc,folders)

generic_fn = '/compressed_XA_###.'
generic_fns = ma.generate_paths(generic_fn,folders)

for folder_idx in range(len(folders)):

    path_prefix = generic_pps[folder_idx][:-1]
    path_cache = generic_pcs[folder_idx][:-1] + '/sino_#####.tiff'
    path_dark = '/dark/dark.tiff'
    path_obeam = '/obeam/refHST6000.tiff'
    path_postfix = generic_fns[folder_idx][:-1] + '_####.tiff'


    start_time = time.time()
    ##########################################################
    # Part 1: Preprocessing


    dark = tifffile.imread(path_prefix + path_dark)
    obeam = tifffile.imread(path_prefix + path_obeam)


    N_angles = Total_angles//stride
    A = np.arange(1,Total_angles + 1, stride)
    paths = ma.generate_paths(path_prefix + path_postfix, A)
    def loader(i):
        return tifffile.imread(paths[i])

    with Pool() as pool:
        data = pool.map(loader, np.arange(0,N_angles))



    data = np.stack(data)
    N_slices, N_pixels = np.shape(data[0])
    data = np.array(-np.log((data - dark[np.newaxis,:,:])/(obeam[np.newaxis,:,:] - dark[np.newaxis,:,:])),dtype=np.float32)
    data = data - np.mean(data[:,:,0:200], axis = (1,2))[:,np.newaxis, np.newaxis]
    split_row = int(N_angles/2)
    split_col = N_pixels-248


    data = ma.rearrange_3d_block_matrix(data, split_row, split_col).astype(np.float32)  
    N_angles, N_slices, N_pixels = np.shape(data)


    print(f"Data reading time: {(time.time() - start_time):.2f} seconds")
    start_time = time.time()

    angles = np.linspace(0,180,N_angles)
    angles_split = np.array_split(angles, N_batches)

    angle_indices = np.array_split(np.arange(N_angles),N_batches)



    def preprocess_projection(i):
        angle = angles_split[i]
        ag = AcquisitionGeometry.create_Parallel3D(detector_position=[0,N_slices,0])\
                                .set_angles(angle)\
                                .set_panel((N_pixels,N_slices), pixel_size=(1,1))\
                                .set_labels(labels=('angle','vertical','horizontal'))
        data_batch = AcquisitionData(data[angle_indices[i]], geometry=ag)
        data_batch.reorder('cil')
        data_batch.geometry.config.units = 'mm'
        processor = PaganinProcessor(full_retrieval=False,pad=100)
        processor.set_input(data_batch)

        data_batch = processor.get_output(override_filter={'alpha':alpha})
        return data_batch.as_array()


    with Pool() as pool:
        data =  pool.map(preprocess_projection, range(N_batches))


    data = 100*np.concatenate(data, axis=0)

    print(f"Preprocess time: {(time.time() - start_time):.2f} seconds")
    start_time = time.time()

    if not output:
        A = np.arange(N_slices)
        paths = ma.generate_paths(path_cache, A)
        def writer(i):
            tifffile.imwrite(paths[i], data[:,i].astype(np.float32))

        with Pool() as pool:
            pool.map(writer, A)
        print(f"Data writing time: {(time.time() - start_time):.2f} seconds")


    if output:
        data2D = data[:,0]

        ag2D = AcquisitionGeometry.create_Parallel2D(detector_position=[0,N_pixels//2])\
                            .set_angles(angles)\
                            .set_panel((N_pixels), pixel_size=(1))\
                            .set_labels(labels=('angle','horizontal'))
        data2D = AcquisitionData(data2D, geometry=ag2D)
        data2D.reorder('astra')
        ag2D.set_angles(ag2D.angles,initial_angle=90)
        ig2D = ImageGeometry(voxel_num_x=3000, voxel_num_y=3000, voxel_size_x=1, voxel_size_y=1)
        device = 'cpu'
        fbp = FBP(ig2D,ag2D,device)
        recon_slice_FBP = fbp(data2D).as_array().astype(np.float32)
        tifffile.imwrite(ma.generate_unique_filename('/dtu-compute/msaca/sliceA_xray_pc/output/slice.tiff'), recon_slice_FBP)
