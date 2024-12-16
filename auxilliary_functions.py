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


def cor_estimator(volume,angles, index0 = 0,t=0, learning_rate=1.0, max_iter = 50):
    # The volume should e a 3d volume with the labels ['angle', 'vertical', horizontal']

    index180 = (index0 +len(angles)//2)%len(angles)
    fixed = sitk.GetImageFromArray(volume[index0,:,:])
    moving = sitk.GetImageFromArray(volume[index180,:,:])


    f_size = fixed.GetSize()
    m_size = moving.GetSize()
    max_f = max(f_size)
    max_m = max(m_size)
    max_size = max(max_f,max_m)

    fixed.SetSpacing((1.5/max_size,1.5/max_size,1.5/max_size))
    moving.SetSpacing((1.5/max_size,1.5/max_size,1.5/max_size))


    size_fixed = fixed.GetSize()
    size_moving = moving.GetSize()

    spacing_fixed = fixed.GetSpacing()
    spacing_moving = moving.GetSpacing()

    # Compute the new origin (shift it to -N/2)
    new_origin_fixed = [-0.5 * (size_fixed[i] - 1) * spacing_fixed[i] for i in range(len(size_fixed))]
    new_origin_moving = [-0.5 * (size_moving[i] - 1) * spacing_moving[i] for i in range(len(size_moving))]

    # Set the origin to be the new center (-N/2)
    fixed.SetOrigin(new_origin_fixed)
    moving.SetOrigin(new_origin_moving)


    translation1 = create_translation_transform(-t, 0)
    flip = create_flip_transform([0,0])
    translation2 = create_translation_transform(t, 0)


    composite_transform = sitk.CompositeTransform(2)
    composite_transform.AddTransform(translation1)
    composite_transform.AddTransform(flip)
    composite_transform.AddTransform(translation2)

    registration = sitk.ImageRegistrationMethod()

    # Set similarity metric
    registration.SetMetricAsCorrelation()
    #registration.SetMetricAsMeanSquares()

    # Set optimizer
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.5)
    registration.SetOptimizerAsGradientDescent(
            learningRate=learning_rate,
            numberOfIterations=max_iter,
            convergenceMinimumValue=-1e-16,
            convergenceWindowSize=1000
        )

    # Set the initial transform
    registration.SetInitialTransform(composite_transform, inPlace=True)

    #
    registration.AddCommand(
        sitk.sitkIterationEvent,
        lambda: print_parameters(composite_transform, registration)
    )


    # Execute registration
    final_transform = registration.Execute(fixed, moving)
    print("Final x-translation:", final_transform.GetParameters()[0])


# Define translation
def create_translation_transform(tx, ty):
    translation_transform = sitk.TranslationTransform(2)
    translation_transform.SetOffset((tx, ty))
    return translation_transform

# Define flip along the y-axis as an affine transform
def create_flip_transform(center):
    flip_transform = sitk.AffineTransform(2)
    matrix = [-1, 0, 0, 1]  # Flipping along y-axis
    flip_transform.SetMatrix(matrix)
    flip_transform.SetCenter(center)
    return flip_transform


def print_parameters(transform, registration_method):
    # Print the optimizer iteration and metric value
    if not (registration_method.GetOptimizerIteration() % 10):
        print(f"Iteration: {registration_method.GetOptimizerIteration()}")
        print(f"Metric Value: {registration_method.GetMetricValue()}")
        # Print the updated parameters of the transform
        print(f"Transform Parameters: {transform.GetParameters()}")
        print()