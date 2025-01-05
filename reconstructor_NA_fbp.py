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
import extended_data as ed
import CTcorrector as ctc
from cil.plugins.astra import FBP
from cil.framework import AcquisitionGeometry, AcquisitionData, ImageGeometry, ImageData, BlockDataContainer



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
ob = ob[50:1910,90:1970]

path = '//dtu-compute/msaca/sliceA_neutron_psi/DC/DC_#####.fits'
A = range(1,31)
dc_paths = ma.generate_paths(path, A)
with Pool() as pool:
    dc_data = pool.map(load_file, dc_paths)


# Now compute the mean across all loaded files
# Convert list of arrays to a single array, assuming the arrays are of the same shape
dc = np.stack(dc_data)
dc = np.mean(dc, axis=0)
dc = dc[50:1910,90:1970]

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Loading time: {elapsed_time:.2f} seconds")

path = '/dtu-compute/msaca/sliceA_neutron_psi/ct_3x1126_60s/ct_3x1126_60s_#####.fits'
path_cache = '/dtu-compute/msaca/output/cache/spot_cleaned_#####.tiff'

batch = np.linspace(1,1127,200).astype(np.uint16)

def preprocess_projection(batch_idx):
    mode = 'save'
    size = 10

    a = range(batch[batch_idx],batch[batch_idx+1])

    if mode == 'load':
        count = 0
        Data = []
        for i in a:
            A = [i]
            temp_path = ma.generate_paths(path_cache,A)[0]
            Data.append(tifffile.imread(temp_path))
        Data = np.stack(Data)
        return Data.astype(np.float32)

    Data = []
    for i in a:
        A = [3*i-2, 3*i-1, 3*i]
        data_paths = ma.generate_paths(path, A)
        data = ma.fits_loader(data_paths)
        data = data[:,50:1910,90:1970]
        data = np.median(data, axis=0)
        Data.append(data)

    Data = np.stack(Data)    
    Data = (-np.log((np.abs(Data-dc[np.newaxis])+1)/(1+np.abs(ob[np.newaxis]-dc[np.newaxis])))).astype(np.float32)
   
    th = 0.2
    Data = iu.morph_spot_clean(Data,th_peaks=th,th_holes=th,method=0,size = size)

    if mode == 'save':
        count = 0
        for i in a:
            A = [i]
            temp_path = ma.generate_paths(path_cache, A)[0]
            tifffile.imwrite(temp_path, Data[count].astype(np.float32))
            count = count+1
        return Data.astype(np.float32)
    if mode is None:
        return Data.astype(np.float32)


# IMPORTaNT, set the environment variable "export NUM_PROCS=$LSB_DJOB_NUMPROC" in the job script. It should be the number of cores.
with Pool() as pool:
   Data_ =  pool.map(preprocess_projection, range(len(batch)-1))

Data = np.vstack(Data_)
Data = np.transpose(Data, [1,0,2])
N_slices, N_angles, N_pixels = np.shape(Data)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Preprocess time: {elapsed_time:.2f} seconds")

################################################################################
#
# Part II, Beam padding,  Cor estimation, tilt-correction and sinogram based preprocessing
#
#################################################################################
# Add beam padding with gaussian blur
angles = np.linspace(0, 360, N_angles, endpoint=True, dtype=np.float32)
slices = np.arange(0, N_slices)
ag = AcquisitionGeometry.create_Parallel3D(detector_position=[0,N_pixels//2,0])\
                            .set_angles(angles)\
                            .set_panel((N_pixels,N_slices), pixel_size=(1,1))\
                            .set_labels(labels=('vertical','angle','horizontal'))

sinograms = ed.ExtendedData()
sinograms.set_data(data = Data)
sinograms.set_slices(slices)
sinograms.pad_edges()
sinograms.acquisition_data(geometry=ag)
input_data = sinograms.data*100



# Cor estimation
corrector = ctc.CTcorrector()
angles = np.linspace(0,360,num=N_angles)
corrector.set_angles(angles = angles)
corrector.load_data(input_data)
corrector.set_labels()
k_angle = 20
corrector.get_projection_and_opposite(k_angle)
corrector.register(learning_rate = 1, sampling_percentage = 0.1,
                   max_iter = 200, metric_type = 'cor',
                   optimizer_type = 'gd', smoothing = 0,
                   shrinking = 1)

print('Estimated Cor: ', corrector.t, 'Estimated tilt: ', corrector.alpha)

############## Now do the cor and tilt correction printing the result

def show_slices(angle, translation, return_data = False, skip = 100, fig_path=None):
    angle_radians = np.deg2rad(angle)
    rotation_matrix = [
                    [np.cos(angle_radians),0,  -np.sin(angle_radians)],
                    [0,1,0],
                    [np.sin(angle_radians), 0,  np.cos(angle_radians)]
                ]
    matrix = [elem for row in rotation_matrix for elem in row]
    translation = [-translation, 0, 0]
    transform = corrector.transformation(matrix = matrix, translation = translation)
    data2 = corrector.resample(data=corrector.data, transform=transform)
    if return_data:
        return data2
    subslices = np.arange(0, N_slices, skip)
    sinograms.set_data(data2)
    sinograms.acquisition_data(geometry=ag)
    sinograms.set_subdata(subslices=subslices)
    sinograms.remove_ring(subdata = True)


    reconstruction = np.empty((len(sinograms.subslices), N_pixels,N_pixels))
    A = np.arange(len(sinograms.subslices))
    fig_paths = ma.generate_paths(fig_path + '_##.png',A) 
    for i in range(len(sinograms.subslices)):
        data2D = sinograms.subdata.get_slice(vertical=i)
        data2D.reorder('astra')
        ag2D = data2D.geometry
        ag2D.set_angles(ag2D.angles, initial_angle=0.0)
        ig2D = ag2D.get_ImageGeometry()
        device = 'gpu'
        fbp = FBP(ig2D,ag2D,device)
        reconstruction[i] = fbp(data2D).as_array()
        plt.figure(figsize=(10,10))
        plt.imshow(reconstruction[i])
        plt.clim([-0.1,0.1])
        plt.savefig(fig_paths[i])
        plt.close()


angle = 0.325
scale = corrector.fixed.GetSpacing()[0]
translation = -26
return_data = False
skip = 100
fig_path = '/dtu-compute/msaca/output/tilt_cor_corrector_slices/rec_slice'
show_slices(angle=angle, translation=translation, return_data = return_data, skip = skip, fig_path=fig_path)


end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total time: Including cor correction and plotting: {elapsed_time:.2f} seconds")


#return_data = True
#skip = None
#data = show_slices(angle, translation = translation, return_data = return_data, skip = skip, fig_path = fig_path)


###########################
#
# Part 3: Do the reconstruction (FBB, CGLS, TV...)
#
##################

#### FBP
#skip = 1
#subslices = np.arange(0, N_slices, skip)
#sinograms.set_data(data)
#sinograms.acquisition_data(geometry=ag)
#sinograms.set_subdata(subslices=subslices)
#sinograms.remove_ring(subdata = False)
#recon = sinograms.fbp(subdata = False)
#recon.as_array()
