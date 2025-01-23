import sys
import os
import time
import numpy as np

try:
    alpha = float(os.getenv("ALPHA"))
except:
    print('Using default alpha!')
    alpha = 150

try:
    stride = int(os.getenv("STRIDE"))
except:
    print('Using default stride')
    stride = 200

try:
    N_iter = int(os.getenv("N_ITER"))
except:
    print('Using default N_iter')
    N_iter = 200

try:
    print('Using default N_batches')
    n_batches = int(os.getenv("N_batches"))
except:
    n_batches = 10


folder  = int(os.getenv("FOLDER"))
batch_id = int(os.getenv("BATCH_ID"))

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
from cil.plugins.ccpi_regularisation.functions import FGP_TV
from cil.optimisation.functions import L2NormSquared, L1Norm, BlockFunction, MixedL21Norm, IndicatorBox, TotalVariation, LeastSquares
from cil.optimisation.operators import BlockOperator, GradientOperator, IdentityOperator, FiniteDifferenceOperator
from cil.optimisation.algorithms import CGLS, SIRT, GD, FISTA, ISTA, PDHG, SPDHG
from cil.plugins.astra.operators import ProjectionOperator
from cil.optimisation.functions import IndicatorBox, MixedL21Norm, L2NormSquared, \
                                       BlockFunction, L1Norm, LeastSquares, \
                                       OperatorCompositionFunction, TotalVariation, \
                                       ZeroFunction
from cil.optimisation.operators import BlockOperator, GradientOperator,\
                                       GradientOperator
from cil.processors import PaganinProcessor, Slicer

############






### Delete folders





########

generic_pc = '/dtu-compute/msaca/sliceA_xray_pc/output/cache/sino_###.'
generic_pcs = ma.generate_paths(generic_pc,[folder])

generic_f_fbp ='/dtu-compute/msaca/sliceA_xray_pc/output/fbp_recon/r_###.'
generic_f_fbps = ma.generate_paths(generic_f_fbp,[folder])

generic_f_tv = '/dtu-compute/msaca/sliceA_xray_pc/output/tv_recon/r_###.'
generic_f_tvs = ma.generate_paths(generic_f_tv,[folder])




def recon_FBP_single(batch_id):
    for folder_idx in range(len(folders)):

        path_cache_sino = generic_pcs[folder_idx][:-1] + '/sino_#####.tiff'
        save_folder_fbp = generic_f_fbps[folder_idx][:-1]+'/'
        N_slices = ma.find_largest_number(path_cache_sino)


        subslices = np.arange(0,N_slices, stride)
        base_size = len(subslices) // n_batches
        remainder = len(subslices) % n_batches
        # Split the array into n_batches parts
        split_arr = []
        start_idx = 0
        for i in range(n_batches):
            # For the first N-1 parts, add an extra element if there's a remainder
            end_idx = start_idx + base_size + (1 if i < remainder else 0)
            split_arr.append(subslices[start_idx:end_idx])
            start_idx = end_idx

        read_path = ma.generate_paths(path_cache_sino, [0])
        image = tifffile.imread(read_path[0])
        N_angles,N_pixels = np.shape(image)


        angles = np.linspace(0, 180, N_angles, endpoint=True, dtype=np.float32)

        

        batch = split_arr[batch_id-1]
        read_path = ma.generate_paths(path_cache_sino, batch)
        write_path_FBP = ma.generate_paths(save_folder_fbp + 'slice_fbp_####.tiff',batch)

        for i in range(len(batch)):
            data2D = tifffile.imread(read_path[i])
            ag2D = AcquisitionGeometry.create_Parallel2D(detector_position=[0,N_pixels//2])\
                                .set_angles(angles)\
                                .set_panel((N_pixels), pixel_size=(1))\
                                .set_labels(labels=('angle','horizontal'))
            data2D = AcquisitionData(data2D, geometry=ag2D)
            data2D.reorder('astra')
            ag2D.set_angles(ag2D.angles, initial_angle=+270)
            ig2D = ImageGeometry(voxel_num_x=3000, voxel_num_y=750, voxel_size_x=1, voxel_size_y=1)
            device = 'gpu'
            fbp = FBP(ig2D,ag2D,device)
            recon_slice_FBP = fbp(data2D).as_array().astype(np.float32)
            tifffile.imwrite(write_path_FBP[i], recon_slice_FBP)

def recon_FBP_multi(batch_id):
    for folder_idx in range(len(folders)):

        path_cache_sino = generic_pcs[folder_idx][:-1] + '/sino_#####.tiff'
        save_folder_fbp = generic_f_fbps[folder_idx][:-1]+'/'
        N_slices = ma.find_largest_number(path_cache_sino)

        subslices = np.arange(0,N_slices, stride)
        base_size = len(subslices) // n_batches
        remainder = len(subslices) % n_batches
        # Split the array into n_batches parts
        split_arr = []
        start_idx = 0
        for i in range(n_batches):
            # For the first N-1 parts, add an extra element if there's a remainder
            end_idx = start_idx + base_size + (1 if i < remainder else 0)
            split_arr.append(subslices[start_idx:end_idx])
            start_idx = end_idx

        read_path = ma.generate_paths(path_cache_sino, [0])
        image = tifffile.imread(read_path[0])
        N_angles,N_pixels = np.shape(image)
        angles = np.linspace(0, 180, N_angles, endpoint=True, dtype=np.float32)

        print('Initiating batch FBP reconstruction from saved sinograms')
        batch = split_arr[batch_id-1]
        read_path = ma.generate_paths(path_cache_sino, batch)
        write_path_FBP = ma.generate_paths(save_folder_fbp + 'slice_fbp_####.tiff',batch)
        data_batch = np.empty((len(batch), N_angles, N_pixels), dtype = np.float32)

        start_time = time.time()
        for i in range(len(batch)):
            data_batch[i] = tifffile.imread(read_path[i])
        N_batch_slices = np.shape(data_batch)[0]


        print(f"Data loading time: {(time.time()-start_time):.2f} seconds")
        start_time = time.time()

        ag_batch = AcquisitionGeometry.create_Parallel3D(detector_position=[0,N_pixels//2,0])\
                                .set_angles(angles)\
                                .set_panel((N_pixels,N_batch_slices), pixel_size=(1,1))\
                                .set_labels(labels=('vertical','angle','horizontal'))
        data_batch = AcquisitionData(data_batch, geometry=ag_batch)
        data_batch.reorder('astra')

        ag_batch.set_angles(ag_batch.angles, initial_angle=+270)
        ig_batch = ImageGeometry(voxel_num_x=3000, voxel_num_y=750, voxel_num_z=N_batch_slices, voxel_size_x=1, voxel_size_y=1, voxel_size_z=1)
        device = 'gpu'
        fbp = FBP(ig_batch,ag_batch,device)

        recon_slice_FBP = fbp(data_batch).as_array().astype(np.float32)

        print(f"Reconstruction time: {(time.time()-start_time):.2f} seconds")
        start_time = time.time()

        for i in range(len(batch)):
            tifffile.imwrite(write_path_FBP[i], recon_slice_FBP[i])

        print(f"Writing time: {(time.time()-start_time):.2f} seconds")


def recon_TV_single(batch_id):
    for folder_idx in range(len(folders)):

        path_cache_sino = generic_pcs[folder_idx][:-1] + '/sino_#####.tiff'
        save_folder_tv = generic_f_tvs[folder_idx][:-1]+'/'
        N_slices = ma.find_largest_number(path_cache_sino)


        subslices = np.arange(0,N_slices, stride)
        base_size = len(subslices) // n_batches
        remainder = len(subslices) % n_batches
        # Split the array into n_batches parts
        split_arr = []
        start_idx = 0
        for i in range(n_batches):
            # For the first N-1 parts, add an extra element if there's a remainder
            end_idx = start_idx + base_size + (1 if i < remainder else 0)
            split_arr.append(subslices[start_idx:end_idx])
            start_idx = end_idx

        read_path = ma.generate_paths(path_cache_sino, [0])
        image = tifffile.imread(read_path[0])
        N_angles,N_pixels = np.shape(image)


        angles = np.linspace(0, 180, N_angles, endpoint=True, dtype=np.float32)

        

        batch = split_arr[batch_id-1]
        read_path = ma.generate_paths(path_cache_sino, batch)
        write_path_TV = ma.generate_paths(save_folder_tv + 'slice_tv_####.tiff',batch)

        for i in range(len(batch)):
            data2D = tifffile.imread(read_path[i])
            ag2D = AcquisitionGeometry.create_Parallel2D(detector_position=[0,N_pixels//2])\
                                .set_angles(angles)\
                                .set_panel((N_pixels), pixel_size=(1))\
                                .set_labels(labels=('angle','horizontal'))
            data2D = AcquisitionData(data2D, geometry=ag2D)
            data2D.reorder('astra')
            ag2D.set_angles(ag2D.angles, initial_angle=+270)
            ig2D = ImageGeometry(voxel_num_x=3000, voxel_num_y=750, voxel_size_x=1, voxel_size_y=1)
            device = 'gpu'

            initial = ig2D.allocate(0)
            A = ProjectionOperator(ig2D,ag2D,device)
            b = data2D
            F = LeastSquares(A,b)
            G = alpha*FGP_TV(device='gpu', nonnegativity=True)
            reconstructor = FISTA(f=F, g=G, initial=initial)
            reconstructor.run(N_iter)
            recon_slice_TV = reconstructor.solution.copy().as_array().astype(np.float32)
            tifffile.imwrite(write_path_TV[i], recon_slice_TV)



def recon_TV_multi():
    device='gpu'
    path_cache_sino = generic_pcs[0][:-1] + '/sino_#####.tiff'
    save_folder_tv = generic_f_tvs[0][:-1]+'/'
    N_slices = ma.find_largest_number(path_cache_sino)

    subslices = np.arange(0,N_slices, stride)
    base_size = len(subslices) // n_batches
    remainder = len(subslices) % n_batches
    # Split the array into n_batches parts
    split_arr = []
    start_idx = 0
    for i in range(n_batches):
        # For the first N-1 parts, add an extra element if there's a remainder
        end_idx = start_idx + base_size + (1 if i < remainder else 0)
        cor_start_idx = max(0,start_idx-10)
        cor_end_idx = min(end_idx + 10, N_slices)
        split_arr.append(subslices[cor_start_idx:cor_end_idx])
        start_idx = end_idx


    read_path = ma.generate_paths(path_cache_sino, [0])
    image = tifffile.imread(read_path[0])
    N_angles,N_pixels = np.shape(image)
    angles = np.linspace(0, 180, N_angles, endpoint=True, dtype=np.float32)

    print('Initiating batch TV reconstruction from saved sinograms')
    batch = split_arr[batch_id-1]
    read_path = ma.generate_paths(path_cache_sino, batch)
    in_path = save_folder_tv + 'slice_tv' + str(batch_id) + '_####.tiff'
    write_path_TV = ma.generate_paths(in_path,batch)
    data_batch = np.empty((len(batch), N_angles, N_pixels), dtype = np.float32)

    print('batch_id', batch_id)
    print('n_batches',n_batches)
    print('batch',batch)
    print('split_arr', split_arr)
    print('len(batch)', len(batch))

    start_time = time.time()
    for i in range(len(batch)):
        data_batch[i] = tifffile.imread(read_path[i])
    N_batch_slices = np.shape(data_batch)[0]

    print('N_batch_slices', N_batch_slices)

    print(f"Data loading time: {(time.time()-start_time):.2f} seconds")
    start_time = time.time()

    ag_batch = AcquisitionGeometry.create_Parallel3D(detector_position=[0,N_pixels//2,0])\
                            .set_angles(angles)\
                            .set_panel((N_pixels,N_batch_slices), pixel_size=(1,1))\
                            .set_labels(labels=('vertical','angle','horizontal'))
    data_batch = AcquisitionData(data_batch, geometry=ag_batch)
    data_batch.reorder('astra')

    ag_batch.set_angles(ag_batch.angles, initial_angle=+270)
    ig_batch = ImageGeometry(voxel_num_x=3000, voxel_num_y=750, voxel_num_z=N_batch_slices, voxel_size_x=1, voxel_size_y=1, voxel_size_z=1)

    initial = ig_batch.allocate(0)
    A = ProjectionOperator(ig_batch,ag_batch,device)
    b = data_batch
    F = LeastSquares(A,b)
    G = alpha*FGP_TV(device='gpu',nonnegativity=True)
    reconstructor = FISTA(f=F, g=G, initial=initial)
    reconstructor.run(N_iter)
    recon_slice_TV = reconstructor.solution.copy().as_array().astype(np.float32)


    print(f"Reconstruction time: {(time.time()-start_time):.2f} seconds")
    start_time = time.time()

    for i in range(len(batch)):
        print('Write_path_TV', write_path_TV[i], recon_slice_TV[i])
        tifffile.imwrite(write_path_TV[i], recon_slice_TV[i])

    print(f"Writing time: {(time.time()-start_time):.2f} seconds")




