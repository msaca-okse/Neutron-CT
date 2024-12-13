###########################################
#
# TABLE OF CONTENTS
#
# 1. Threshold estimator:   Input: np array, cil image, sitk image. Options: sampling size. Output: a number, giving the otsu threshold value
#
# 2. Downsampler_direct: Input: np array, cil image, sitk image. Output: np array or sitk image, downsized. Downsamples using simpleITK
#
# 3. Downsampler_mean: Takes mans and downsamples: (has boxes of size (N,N,N))
#
# 4. 
#
#
#
###########################################

import numpy as np
import SimpleITK as sitk
import os

def compute_threshold(image, n_classes=1):
    if isinstance(image,np.ndarray):
        im_type = 'np'
        image = sitk.GetImageFromArray(image)
    elif isinstance(image,sitk.Image):
        im_type = 'sitk'

    image = downsample_direct(image, factor = 10)

    Otsu_method = sitk.OtsuMultipleThresholdsImageFilter()
    Otsu_method.SetNumberOfThresholds(n_classes) 

    # Perform the thresholding
    segmented_image = Otsu_method.Execute(image)
    return Otsu_method.GetThresholds()


def downsample_direct(image, factor):
    if isinstance(image,np.ndarray):
        im_type = 'np'
        image = sitk.GetImageFromArray(image)
    elif isinstance(image,sitk.Image):
        im_type = 'sitk'

    # Get the original size and spacing of the image
    size = image.GetSize()
    spacing = image.GetSpacing()

    # Calculate the new size (downsampling by factor)
    new_size = [int(size[0] / factor), int(size[1] / factor), int(size[2] / factor)]
    
    # Calculate the new spacing (enlarging the spacing to match the downsampled size)
    new_spacing = [s * factor for s in spacing]
    
    # Set the resampling transform (identity transform)
    transform = sitk.Transform(3, sitk.sitkIdentity)

    # Perform the resampling (using average interpolation for downsampling)
    downsampled_image = sitk.Resample(image,
                                    new_size,
                                    transform,
                                    sitk.sitkLinear,  # BSpline interpolation is good for downsampling
                                    image.GetOrigin(),
                                    new_spacing,
                                    image.GetDirection(),
                                    0)  # 0 is the background value for the resampling
    
    if im_type == 'np':
        downsampled_image = sitk.GetArrayFromImage(downsampled_image)

    return downsampled_image

def generate_paths(path, A):
    # Extract the folder and filename pattern
    folder, filename_pattern = os.path.split(path)
    
    # Determine the number of digits in the zero-padding
    num_digits = filename_pattern.count('#')
    filename_base = filename_pattern.replace('#' * num_digits, '{}')

    # Generate the list of paths
    paths = [
        os.path.join(folder, filename_base.format(str(num).zfill(num_digits)))
        for num in A
    ]
    
    return paths