import sys
import os
os.chdir('/zhome/71/c/146676/main/')
sys.path.append('/zhome/71/c/146676/main/')
import time
import SimpleITK as sitk
import numpy as np
from astropy.io import fits
from helpers import module_auxiliary as ma
import tifffile
from multiprocessing import Pool
import matplotlib.pyplot as plt
from helpers import extended_data as ed
from cil.recon import FBP
#from cil.plugins.astra import FBP
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
import h5py
from cil.utilities.display import show_geometry
from scipy.ndimage import uniform_filter1d
from cil.processors import Slicer
import nibabel as nib

import os

#method = os.getenv("METHOD", "sirt1")  # Default to "sirt1" if METHOD is not set
method = os.getenv("METHOD", "cgls1")  # Default to "sirt1" if METHOD is not set


c0 = 0
c1 = 2000

y0 = 130
y1 = 230

pad = 30



# Open the h5 file in read mode
with h5py.File('/dtu-compute/msaca/sliceA_diffraction/xrd_ct_integrated/scan-0339_pilatus_integrated.h5', 'r') as file:
    # Check the keys in the file (this shows the main datasets or groups)
    
    # Access a dataset or group (replace 'your_dataset' with the correct key)
    dataset = file['entry']['data1d']['I'][:][:,c0:c1]  # Replace 'your_dataset' with the actual name of the dataset
    dataset  = dataset.reshape(181, 362, c1-c0).transpose([2,0,1])
    two_theta = file['entry']['data1d']['2th'][:]

# Open the h5 file in read mode
with h5py.File('/dtu-compute/msaca/sliceA_diffraction/xrd_reconstructions/scan-339_recon.h5', 'r') as file:
    # Check the keys in the file (this shows the main datasets or groups)
    
    # Access a dataset or group (replace 'your_dataset' with the correct key)
    recon_gridrec = (file['reconstructed']['gridrec'][:]/500).transpose([2,0,1])[c0:c1,y0:y1]
     # Replace 'your_dataset' with the actual name of the dataset
     
    # Open the h5 file in read mode
with h5py.File('/dtu-compute/msaca/sliceA_diffraction/xrd_attenuation/scan-0339_xspress3-dtc-2d.h5', 'r') as file:
    # Check the keys in the file (this shows the main datasets or groups)
    
    # Access a dataset or group (replace 'your_dataset' with the correct key)
    dtc= file['entry/instrument/xspress3']
    data_all_events = dtc['all_events'][:].reshape(181,362)
    data_output_count_rate = dtc['output_count_rate'][:].reshape(181,362)
    data_att = dtc['window_counts'][:][:,0].reshape(181,362)




def zero_pad_3d(array: np.ndarray, N: int,value=0) -> np.ndarray:
    """
    Pads a 3D NumPy array with zeros along the last two dimensions by N pixels.
    The first dimension (axis=0) remains unchanged.

    Parameters:
    array (np.ndarray): Input 3D array.
    N (int): Number of pixels to pad along the last two dimensions.

    Returns:
    np.ndarray: Zero-padded 3D array.
    """
    if N < 0:
        raise ValueError("Padding size N must be non-negative")
    
    pad_width = ((0, 0), (0, 0), (N, N))  # No padding on axis=0, N on other axes
    return np.pad(array, pad_width, mode='constant', constant_values=0)


rel = 1e3*dataset/data_att.astype(np.float32)
data_nolog = rel# - np.mean(rel[:,-10:])
data_nolog_= zero_pad_3d(data_nolog, pad,value=np.mean(rel[:,-10:]))


data = data_nolog
data_ = data_nolog_
nchannels,ntheta, nx = np.shape(data)
nchannels,ntheta, nx_ = np.shape(data_)
angles = range(0,ntheta)

ag = AcquisitionGeometry.create_Parallel2D(detector_position=[0,nx_//2])\
                            .set_angles(angles)\
                            .set_channels(nchannels)\
                            .set_panel((nx_), pixel_size=(1))\
                            .set_labels(['channel','angle', 'horizontal'])
data_ = AcquisitionData(data_, geometry=ag)
data_.reorder('astra')
ig_ = ag.get_ImageGeometry()
roi = {'horizontal_x':(pad,-pad,1), 'horizontal_y':(pad+y0,pad+y1,1)}
processor = Slicer(roi)
processor.set_input(ig_)
ig = processor.get_output()
device = 'gpu'



fdx = FiniteDifferenceOperator(ig, direction='horizontal_x', bnd_cond='Neumann')
fdy = FiniteDifferenceOperator(ig, direction='horizontal_y', bnd_cond='Neumann')
fdc = FiniteDifferenceOperator(ig, direction='channel', bnd_cond='Neumann')

def cgls1(alpha_dc_):
    print('Saving to:\n')
    print('/dtu-compute/msaca/sliceA_diffraction/own_reconstructions/cgls_alphadc_'+str(round(alpha_dc_,3)) + '.nii')
    factor = 0.05
    alpha_id = 1*factor
    alpha_dx = 100*factor
    alpha_dy = 500*factor
    alpha_dc = alpha_dc_*factor
    L = alpha_id*IdentityOperator(ig)
    A = ProjectionOperator(ig, ag, 'gpu')
    FD = BlockOperator(alpha_dx * fdx,alpha_dy *  fdy,alpha_dc *  fdc)
    A_block = BlockOperator(L,FD, A)
    #A_block = BlockOperator(L,A_block_)


    g_block = BlockDataContainer(L.range.allocate(0), FD.range.allocate(0), data_)

    # set up and run CGLS
    optimizer = CGLS(operator=A_block, data=g_block, update_objective_interval=2)
    optimizer.run(15, verbose=1)
    recon_cgls = optimizer.solution.as_array().astype(np.float32)

    affine = np.eye(4)
    nii_img = nib.Nifti1Image(recon_cgls, affine)
    nib.save(nii_img, '/dtu-compute/msaca/sliceA_diffraction/own_reconstructions/cgls_alphadc_'+str(round(alpha_dc_,3)) + '.nii')


def sirt1(alpha_dc_):
    print('Saving to:\n')
    print('/dtu-compute/msaca/sliceA_diffraction/own_reconstructions/sirt_alphadc_'+str(round(alpha_dc_,3)) + '.nii')
    factor = 0.05
    alpha_id = 1*factor
    alpha_dx = 100*factor
    alpha_dy = 500*factor
    alpha_dc = alpha_dc_*factor
    L = alpha_id*IdentityOperator(ig)
    A = ProjectionOperator(ig, ag, 'gpu')
    FD = BlockOperator(alpha_dx * fdx,alpha_dy *  fdy,alpha_dc *  fdc)
    A_block = BlockOperator(L,FD, A)
    #A_block = BlockOperator(L,A_block_)


    g_block = BlockDataContainer(L.range.allocate(0), FD.range.allocate(0), data_)

    # set up and run CGLS
    constraint = IndicatorBox(lower=-0.00)
    optimizer = SIRT(operator=A_block, data=g_block, constraint = constraint)
    optimizer.run(150)
    recon_cgls = optimizer.solution.as_array().astype(np.float32)

    affine = np.eye(4)
    nii_img = nib.Nifti1Image(recon_cgls, affine)
    nib.save(nii_img, '/dtu-compute/msaca/sliceA_diffraction/own_reconstructions/sirt_alphadc_'+str(round(alpha_dc_,3)) + '.nii')




alpha_dc_array = np.array([3000,5000,10000])

if method == 'cgls1':
    with Pool() as pool:
        pool.map(cgls1, alpha_dc_array)



if method == 'sirt1':

    with Pool() as pool:
        pool.map(sirt1, alpha_dc_array)

