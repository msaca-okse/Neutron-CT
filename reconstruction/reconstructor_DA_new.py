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
from scipy.interpolate import interp1d
import cv2
import hdf5plugin
from scipy.ndimage import uniform_filter
from scipy.ndimage import label, sum_labels
from cil.plugins.astra import ProjectionOperator
from scipy.ndimage import label

print('Loaded packages')


pad = 200

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

def cgls1(data, data_att, alpha_dc_):
    data = (1e3*data/data_att).astype(np.float32)
    data_= zero_pad_3d(data, pad,value=np.mean(data[:,-10:]))

    print(9)
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
    print(10)
    fdx = FiniteDifferenceOperator(ig, direction='horizontal_x', bnd_cond='Neumann')
    fdy = FiniteDifferenceOperator(ig, direction='horizontal_y', bnd_cond='Neumann')
    fdc = FiniteDifferenceOperator(ig, direction='channel', bnd_cond='Neumann')
    factor = 0.03
    alpha_id = 1*factor
    alpha_dx = 100*factor
    alpha_dy = 500*factor
    alpha_dc = alpha_dc_*factor
    L = alpha_id*IdentityOperator(ig)
    A = ProjectionOperator(ig, ag, 'gpu')
    FD = BlockOperator(alpha_dx * fdx,alpha_dy *  fdy, alpha_dc *  fdc)
    A_block = BlockOperator(L,FD, A)

    g_block = BlockDataContainer(L.range.allocate(0), FD.range.allocate(0), data_)
    print(11)
    # set up and run CGLS
    optimizer = CGLS(operator=A_block, data=g_block)
    print(12)
    optimizer.run(30, verbose=1)
    print(13)
    recon_cgls = optimizer.solution.as_array().astype(np.float32)
    return recon_cgls


A1 = np.arange(339, 449+1, 2)
A2 = np.arange(455, 471+1, 2)
A3 = np.arange(475, 487+1, 2)
A = np.concatenate([A1,A2,A3])
# A = A1
data_path = '/dtu-compute/msaca/sliceA_diffraction/powder-crystal/integrated-####.h5'
data_files = ma.generate_paths(data_path, A)
data_att_path = '/dtu-compute/msaca/sliceA_diffraction/powder-crystal/scan-####_xspress3-dtc-2d.h5'
data_att_files = ma.generate_paths(data_att_path, A)
output_path = '/dtu-compute/msaca/sliceA_diffraction/powder-crystal/recons2-DA-####.h5'
output_files = ma.generate_paths(output_path, A)



c0, c1 = 0, 667
y0,y1 = 130, 230
alpha_dc_ = 10


N_labels = 20

recon = []

q_key = 'crystal_q-80'
p_key = 'powder_q-80'

for i in range(len(data_files)):


    print('Loading absorption data for slice', i)
    with h5py.File(data_att_files[i], 'r') as file:
        # Access a dataset or group (replace 'your_dataset' with the correct key)
        dtc= file['entry/instrument/xspress3']
        data_att = dtc['window_counts'][:][:,0].reshape(181,362).astype(np.float32)



    print('Loading integrated diffraction data')
    with h5py.File(data_files[i], 'r') as file:
        datasetq = file[q_key][:]
        datasetq = np.sum(datasetq,axis=2)

        datasetp = file[p_key][:]

    print('Loaded data for slice ', i)

    dataq = np.transpose(datasetq/200,[2,0,1])[:][c0:c1].astype(np.float32)
    datap = np.transpose(datasetp/200,[2,0,1])[:][c0:c1].astype(np.float32)
    nc, na, nx = np.shape(dataq)

    print('Ruuning cgls on crystal data')
    reconq = cgls1(dataq, data_att, alpha_dc_*0.2)
    print('Running cgls on powder data')
    reconp = cgls1(datap, data_att, alpha_dc_)

    angles = np.arange(0,181,1)
    device = 'gpu'


    d_channels = []
    intersector_recon = np.zeros((nc,N_labels, y1-y0,nx))
    print('Running crystal line intersection analysis')
    for i_c in range(0,nc):
        if not i_c%50:
            print('iteration',i_c)



        
        data_channel = dataq[i_c].astype(np.float32)
        blurred = cv2.GaussianBlur(data_channel, (105, 105), 0)  # Adjust kernel size based on spot size
        high_pass = cv2.subtract(data_channel, blurred)
        q = np.percentile(high_pass, 99.7)

        thresholded= (high_pass > q).astype(np.float32)  # Convert to 0 and 255
        thresholded_blurred = cv2.GaussianBlur(thresholded, (3, 3), 0) 

        labeled_array, num_features = label(thresholded_blurred)
        labeled_array = labeled_array*(thresholded.astype(bool))

        d_channels.append(data_channel)

        component_sizes = np.bincount(labeled_array.ravel())[1:]  # Ignore background (label=0)
        component_weights = sum_labels(data_channel, labeled_array, index=np.arange(1, num_features + 1))
        sorting_metric = component_sizes * component_weights
        sorted_labels = np.argsort(sorting_metric)[::-1]
        for i_l in range(N_labels):
            if not num_features >= i_l+1:
                break

            label_i_lth_largest = sorted_labels[i_l]
            mask_i_lth = (labeled_array == (label_i_lth_largest+1)).astype(np.float32)
            data_intensity = data_channel*mask_i_lth

            # Create a normalized array where each pixel is divided by its component size
            # for component_label in range(1, num_features + 1):  # Skip background (0)
            ### Define geometry
            ag2d = AcquisitionGeometry.create_Parallel2D(detector_position=[0,nx//2])\
                                .set_angles(angles)\
                                .set_panel((nx), pixel_size=(1))\
                                .set_labels(['angle', 'horizontal'])
            ig2d_ = ag2d.get_ImageGeometry()
            roi = {'horizontal_y':(y0,y1,1)}
            processor = Slicer(roi)
            processor.set_input(ig2d_)
            ig2d = processor.get_output()
            P0 = ProjectionOperator(ig2d, ag2d)

            data_ = AcquisitionData(mask_i_lth, geometry=ag2d)
            data_intensity_ = AcquisitionData(data_intensity, geometry=ag2d)
            data_.reorder('astra')
            data_intensity_.reorder('astra')
            im_back_proj = P0.adjoint(data_)
            im_back_proj_intensity = P0.adjoint(data_intensity_)


            # Construct the BlockOperator combining the two operators
            intersector_recon[i_c,i_l] = im_back_proj.as_array().astype(np.float32)
    intersector_recon = (intersector_recon > 1e-5).astype(np.uint8)
    intersector_recon = np.sum(intersector_recon,axis=1)


    output_file = output_files[i]
    print('Saving cgls recons and crystal intersectors to the dictionary', output_file)

    output_dictionary = {}
    output_dictionary['powder-recon'] = reconp
    output_dictionary['crystal-recon'] = reconq
    output_dictionary['crystal-mask'] = intersector_recon.astype(np.uint8)

    with h5py.File(output_file, "w") as f_out:
        for key, array in output_dictionary.items():
            f_out.create_dataset(key, data=array)
    print('Succesfully saved file', output_file)
        
