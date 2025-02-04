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

def load_registered_data(compression = 1,dataset_XA='tv', dataset_NA = 'tv'):
    global compression_v 
    compression_v = compression
    # Load the images
    if dataset_XA == 'tv':
        paths_XA = stitcher_XA.generate_batch_stitched_paths(dataset = 'tv', folders = [1,2,3,4,5],
            start_indices = [300, 94, 94, 94, 94], end_indices = [693, 694, 694, 694, 788])
    elif dataset_XA == 'fbp':
        paths_XA = stitcher_XA.generate_stitched_paths(dataset = 'fbp', folders = [1,2,3,4,5],
            start_indices = [300, 94, 94, 94, 94], end_indices = [693, 694, 694, 694, 788])[::compression_v]




    with Pool() as pool:
        recon_XA = pool.map(loader_XA, paths_XA)

    recon_XA = np.stack(recon_XA)
    recon_XA = np.clip(recon_XA, a_min = -0.1, a_max = 0.1)

    if dataset_NA == 'fbp':
        path_NA = '/dtu-compute/msaca/output/fbp_recon/slice_fbp_####.tiff'
    elif dataset_NA == 'tv':
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




def load_subset_of_registered_data(compression = 1, dataset='tv', h = 1, xray_slices = None, neutron_slices = None, output_volume = None):
    global compression_v 
    compression_v = compression
    # Load the images
    if dataset == 'tv':
        paths_XA = stitcher_XA.generate_batch_stitched_paths(dataset = 'tv', folders = [1,2,3,4,5],
            start_indices = [300, 94, 94, 94, 94], end_indices = [693, 694, 694, 694, 788])
    elif dataset == 'fbp':
        paths_XA = stitcher_XA.generate_stitched_paths(dataset = 'fbp', folders = [1,2,3,4,5],
            start_indices = [300, 94, 94, 94, 94], end_indices = [693, 694, 694, 694, 788])

    N_xray = len(paths_XA)
    paths_XA = [paths_XA[i] for i in xray_slices]

    with Pool() as pool:
        recon_XA = pool.map(loader_XA, paths_XA)

    recon_XA_B = np.stack(recon_XA)
    recon_XA_B = np.clip(recon_XA, a_min = -0.1, a_max = 0.1)
    ny_xray,nx_xray = np.shape(recon_XA_B[0])
    recon_XA_A = np.zeros((N_xray, ny_xray, nx_xray))
    recon_XA_A[xray_slices] = recon_XA_B

    path_NA = '/dtu-compute/msaca/output/tv_recon/slice_tv_####.tiff'
    #path_NA = '/dtu-compute/msaca/output/fbp_recon/slice_fbp_####.tiff'
    A = np.arange(0, 1788)
    paths_NA = ma.generate_paths(path_NA, A)
    N_neutron = len(paths_NA)
    paths_NA = [paths_NA[i] for i in neutron_slices]


    with Pool() as pool:
        recon_NA = pool.map(loader_NA, paths_NA)

    recon_NA_B = np.stack(recon_NA)
    recon_NA_B = np.clip(recon_NA, a_min = -0.1, a_max = 0.1)
    ny_neutron,nx_neutron = np.shape(recon_NA_B[0])
    recon_NA_A = np.zeros((N_neutron, ny_neutron, nx_neutron))
    recon_NA_A[neutron_slices] = recon_NA_B

    # Initialice the registrator class and set thresholds
    reg = registrator.Registrator()
    reg.load_images(fixed_array=recon_NA_A,moving_array=recon_XA_A)
    reg.fixed = scale_volume(reg.fixed, h)
    path = 'transformation_XA_to_NA_compression' + '1' + '.tfm'
    transform = sitk.ReadTransform(path)
    reg.resample(transform,ignore_flag = True)

    temp = output_volume
    if temp[0][0] is None:
        temp[0] = [0, N_neutron]
    if temp[1][0] is None:
        temp[1] = [0, ny_neutron]
    if temp[2][0] is None:
        temp[2] = [0, nx_neutron]

    start_index = [int(h*temp[2][0]), int(h*temp[1][0]), int(h*temp[0][0])]
    end_index = [int(h*temp[2][1]), int(h*temp[1][1]), int(h*temp[0][1])]
    # Calculate the size for the RegionOfInterest filter
    size = [end - start for start, end in zip(start_index, end_index)]
    # Use the RegionOfInterest filter to extract the subimage
    roi_filter = sitk.RegionOfInterestImageFilter()
    roi_filter.SetIndex(start_index)
    roi_filter.SetSize(size)
    reg.fixed = roi_filter.Execute(reg.fixed)
    reg.moving = roi_filter.Execute(reg.moving)
    return reg












def scale_volume(image, h):
    """
    Scales a volume by a factor h using SimpleITK.
    
    Parameters:
        image (sitk.Image): The input image to scale.
        h (float): Scaling factor (h > 0).
    
    Returns:
        sitk.Image: The scaled image.
    """
    assert h > 0, "Scaling factor h must be a positive number."

    # Get original image metadata
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    original_origin = image.GetOrigin()
    original_direction = image.GetDirection()

    # Compute new size and spacing
    new_size = [int(dim * h) for dim in original_size]
    new_spacing = [sp / h for sp in original_spacing]

    # Set up the resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(original_origin)
    resampler.SetOutputDirection(original_direction)
    resampler.SetInterpolator(sitk.sitkLinear)  # Use linear interpolation (adjust as needed)
    resampler.SetTransform(sitk.Transform())  # Identity transform

    # Resample the image
    scaled_image = resampler.Execute(image)
    
    return scaled_image