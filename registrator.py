import numpy as np
import SimpleITK as sitk
import copy
import matplotlib.pyplot as plt
previous_metric_value = None
import plotly.graph_objects as go
import plot_library as pl
import random

class Registrator():
    # Class to load in two images: A fixed and a moving image. The class uses simple ITK. These are the things that should be implemented in the class: A loader, to load in two volumes. Read them as numpy files, save them as itk images.
    # Initializer: Initialize the grid and two empty images saved in self.fixed and self.moving.

    # Method self.copy(): Returns a copy of the class 
    # Saveclass: Method that saves a class: Input: A path to where the class is saved. 
    # Load class: A function that initializes and loads a class. Input: path file. Returns a Registrator class object.
    # Filter thresholding method: has two optional arguments fixed=float, moving=float. Everything abov the float is set to 1, everything below is set to zero. The thresholding is done inplace. Simply use the .copy() method before running the filter to save the old
    # methods.
    # Downsizing method: Has 2 arguments: fixed_factor = int, moving_factor = int. The method downsizes an image by a factor. In this class, the origican grid is kept, while the number of pixels is reduced. The class is done inplace.

    # Transformations: 
    # This method returns a transformation object. The input of the class are the transformation parameters.
    # There will be multiple methods with different transformation types. Output is a transformation object.
    
    # Register: Method outputs a transformation given by a registration algorithm, wich registers moving to fixed. Settings for the registration algorithm are set in this method, some of the options are arguments to the class. 

    # Resample: Takes as input a transformation object and does resampling of the moving image. There should be an argument to this class specifying if linear of nearest neighbor interpoilation is used. Per default, the class overwrites the .moving variable
    # but if inplace=False, the class instead outputs the resampled image.
    
    # Skeleton code for this class was generated using chatGPT, by prompting the description above.

    """
    A class to load two images (fixed and moving), apply transformations, and perform image registration.
    
    Attributes:
        fixed (SimpleITK.Image): The fixed image for registration.
        moving (SimpleITK.Image): The moving image to be registered.
        transform (SimpleITK.Transform): The transformation applied during registration.
    """
    
    def __init__(self):
        """
        Initializes the Registrator class with empty images for fixed and moving.
        """
        self.fixed = None
        self.moving = None
        self.called_inputs = set()
        self.fixed_smooth = {}
        self.moving_smooth = {}
        self.callback = 1

    def load_images(self, fixed_array, moving_array):
        """
        Loads two images from numpy arrays and converts them to SimpleITK images.
        
        Args:
            fixed_path (str): Pointer to the fixed image numpy array.
            moving_path (str): Pointer to the moving image numpy array.
        
        """
        # Convert numpy arrays to SimpleITK images
        self.fixed = sitk.GetImageFromArray(fixed_array)
        self.moving = sitk.GetImageFromArray(moving_array)


        f_size = self.fixed.GetSize()
        m_size = self.moving.GetSize()
        max_f = max(f_size)
        max_m = max(m_size)
        max_size = max(max_f,max_m)

        self.fixed.SetSpacing((1.5/max_size,1.5/max_size,1.5/max_size))
        self.moving.SetSpacing((1.5/max_size,1.5/max_size,1.5/max_size))


        size_fixed = self.fixed.GetSize()
        size_moving = self.moving.GetSize()

        spacing_fixed = self.fixed.GetSpacing()
        spacing_moving = self.moving.GetSpacing()

        # Compute the new origin (shift it to -N/2)
        new_origin_fixed = [-0.5 * (size_fixed[i] - 1) * spacing_fixed[i] for i in range(len(size_fixed))]
        new_origin_moving = [-0.5 * (size_moving[i] - 1) * spacing_moving[i] for i in range(len(size_moving))]

        # Set the origin to be the new center (-N/2)
        self.fixed.SetOrigin(new_origin_fixed)
        self.moving.SetOrigin(new_origin_moving)



    def copy(self):
        """
        Creates and returns a copy of the current Registrator instance.
        
        Returns:
            Registrator: A new instance of the Registrator class with the same images and attributes.
        """
        # Create a new Registrator instance
        new_instance = Registrator()

        # Copy all attributes of the current instance to the new instance
        new_instance.__dict__ = copy.deepcopy(self.__dict__)  # Deep copy all attributes

        return new_instance

    def save_class(self, path: str):
        """
        Saves the current Registrator instance to a file.
        
        Args:
            path (str): The file path where the class should be saved.
        """
        # You can save the class instance using pickle or any other method
        # For now, just saving images as an example
        sitk.WriteImage(self.fixed, f"{path}_fixed.nii")
        sitk.WriteImage(self.moving, f"{path}_moving.nii")

    @staticmethod
    def load_class(path: str):
        """
        Loads a Registrator class instance from a file.
        
        Args:
            path (str): Path to the file where the class was saved.
        
        Returns:
            Registrator: The loaded Registrator instance.
        """
        registrator = Registrator()
        registrator.fixed = sitk.ReadImage(f"{path}_fixed.nii")
        registrator.moving = sitk.ReadImage(f"{path}_moving.nii")
        return registrator

    def filter_threshold(self, fixed_threshold: float = 0.0, moving_threshold: float = 0.0, cast = True):
        """
        Applies thresholding to both fixed and moving images. Everything above the threshold is set to 1, and below is set to 0.
        
        Args:
            fixed_threshold (float, optional): Threshold for the fixed image. Default is 0.0.
            moving_threshold (float, optional): Threshold for the moving image. Default is 0.0.
        """
        # Apply thresholding to both fixed and moving images (inplace)
        self.fixed = sitk.BinaryThreshold(self.fixed, lowerThreshold=fixed_threshold, upperThreshold=float("inf"), insideValue=1, outsideValue=0)
        self.moving = sitk.BinaryThreshold(self.moving, lowerThreshold=moving_threshold, upperThreshold=float("inf"), insideValue=1, outsideValue=0)

        if cast:
            self.fixed =sitk.Cast(self.fixed, sitk.sitkFloat32)
            self.moving =sitk.Cast(self.moving, sitk.sitkFloat32)

    def downsizing(self, fixed_factor = 2, moving_factor = 2, smooth = False):
        """
        Downsizes both the fixed and moving images by a specified factor.
        
        Args:
            fixed_factor (int): The factor by which to downsize the fixed image.
            moving_factor (int): The factor by which to downsize the moving image.
        """
        # Resample the images (downsize) by the given factor, preserving the original grid structure
        # This will be done inplace, reducing the number of pixels
        self.fixed = self._downsample_image(input_image=self.fixed, factor = fixed_factor,smooth = smooth)
        self.moving = self._downsample_image(input_image=self.moving, factor = moving_factor, smooth = smooth)


    def apply_mean(self, input_image, smooth):
            mean_filter = sitk.MeanImageFilter()
            mean_filter.SetRadius(smooth)  # Adjust the radius as needed
            image = mean_filter.Execute(input_image)
            return image

    def _downsample_image(self, input_image, factor=10,smooth=False):
        """
        Downsamples a 3D volume by reducing its resolution by a specified factor in each dimension,
        effectively averaging values over blocks (e.g., 2x2x2 cubes when factor=2).

        Args:
            input_image (sitk.Image): The input SimpleITK image to be downsampled.
            factor (int, optional): The downsampling factor. Defaults to 2.
                - A factor of 2 will halve the size of the image in each dimension.
                - The new spacing will be adjusted accordingly.

        Returns:
            sitk.Image: The downsampled SimpleITK image with reduced resolution.

        Function Steps:
            1. Retrieve the original size and spacing of the input image.
            2. Compute the new size by dividing each dimension of the image size by the factor.
            3. Compute the new spacing by multiplying the original spacing by the factor.
            4. Use `sitk.Resample` to resample the image:
                - Use an identity transform to retain the original alignment.
                - Apply `sitk.sitkBSpline` interpolation to achieve smooth averaging.
            5. Return the downsampled image.

        Example:
            # Load a 3D image
            input_image = sitk.ReadImage('large_image.nii')
            
            # Downsample the image to half the size in each dimension
            downsampled_image = downsample_image(input_image, factor=2)
            
            # Save the downsampled image
            sitk.WriteImage(downsampled_image, 'downsampled_image.nii')
        """
        if smooth:
            mean_filter = sitk.MeanImageFilter()
            mean_filter.SetRadius(smooth)  # Adjust the radius as needed
            image = mean_filter.Execute(input_image)
        else:
            image = input_image


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
        
        return downsampled_image


    def center_transform(self):
        thresholds = [0.5, 0.5]
        binary_fixed = sitk.BinaryThreshold(self.fixed, lowerThreshold=thresholds[0], upperThreshold=float("inf"), insideValue=1, outsideValue=0)
        binary_moving = sitk.BinaryThreshold(self.moving, lowerThreshold=thresholds[1], upperThreshold=float("inf"), insideValue=1, outsideValue=0)
        label_shape_filter_fixed = sitk.LabelShapeStatisticsImageFilter()
        label_shape_filter_moving = sitk.LabelShapeStatisticsImageFilter()
        label_shape_filter_fixed.Execute(binary_fixed)
        label_shape_filter_moving.Execute(binary_moving)
        centroid_fixed = np.array(label_shape_filter_fixed.GetCentroid(1))
        centroid_moving = np.array(label_shape_filter_moving.GetCentroid(1))
        self.fixed.SetOrigin(-centroid_fixed)
        self.moving.SetOrigin(-centroid_moving)
        return -centroid_fixed, -centroid_moving




    def transformation(self, type='Similarity3Dtransform', matrix=None,translation=(0,0,0),
     thresholds = [0.5, 0.5], principal_inverse=False, principal_fixed=True):
        """
        Returns a transformation object based on the provided transformation parameters.
        
        Args:
            transform_parameters: The parameters that define the transformation (e.g., rotation, translation).
            input: type='Similarity3Dtransform, Centroid, Euler3DTransform'
        
        Returns:
            SimpleITK.Transform: The transformation object.
        """
        if type == 'Centroid':
            transform = sitk.CenteredTransformInitializer(
            self.fixed,                  # Reference (fixed) image
            self.moving,                     # Moving (adjust) image
            sitk.Similarity3DTransform(),    # Use similarity transformation (translation, rotation, scaling)
            sitk.CenteredTransformInitializerFilter.GEOMETRY  # Align the centroids (geometry)
            )
        elif type == 'Similarity3DTransform':
            if (translation is None) or (matrix is None):
                raise ValueError('Specify Matrix and Translation')
            transform = sitk.Similarity3DTransform()
            transform.SetMatrix(matrix)
            transform.SetTranslation(translation)
        elif type == 'Euler3DTransform':
            transform = sitk.Euler3DTransform()
            transform.SetMatrix(matrix)
            transform.SetTranslation(translation)
        elif type == 'Affine':
            transform = sitk.AffineTransform(3)
            transform.SetMatrix(matrix)
            transform.SetTranslation(translation)
        elif type == 'Principal':
            transform = sitk.Euler3DTransform()
            factor = 5
            if not principal_inverse:
                if principal_fixed:
                    fixed_d = self._downsample_image(self.fixed,factor = factor)
                    binary_fixed = sitk.BinaryThreshold(fixed_d, lowerThreshold=thresholds[0], upperThreshold=float("inf"), insideValue=1, outsideValue=0)
                    label_shape_filter_fixed = sitk.LabelShapeStatisticsImageFilter()
                    label_shape_filter_fixed.Execute(binary_fixed)
                    principal_axes_fixed = label_shape_filter_fixed.GetPrincipalAxes(1)  # Eigenvectors (flattened)
                    matrix = np.array(principal_axes_fixed).reshape((3, 3)).T.flatten()
                    translation = -np.array(label_shape_filter_fixed.GetCentroid(1))
                    transform.SetMatrix(matrix)
                    transform.SetTranslation(translation)
            else:
                if principal_fixed:
                    fixed_d = self._downsample_image(self.fixed,factor = factor)
                    binary_fixed = sitk.BinaryThreshold(fixed_d, lowerThreshold=thresholds[0], upperThreshold=float("inf"), insideValue=1, outsideValue=0)
                    label_shape_filter_fixed = sitk.LabelShapeStatisticsImageFilter()
                    label_shape_filter_fixed.Execute(binary_fixed)
                    principal_axes_fixed = label_shape_filter_fixed.GetPrincipalAxes(1)  # Eigenvectors (flattened)
                    matrix = np.array(principal_axes_fixed).reshape((3, 3)).flatten()
                    translation = np.array(label_shape_filter_fixed.GetCentroid(1))
                    transform.SetMatrix(matrix)
                    transform.SetTranslation(translation)

        random_integer = random.randint(1, 1000000)
        transform.ID = random_integer
        transform.FLAG = False

        return transform

    def register(self, learning_rate = 0.1, sampling_percentage = 0.1,
                     max_iter = 50,
                    metric_type = 'ms', optimizer_type = 'gd',
                     smoothing = 0,
                      shrinking = 1, thresholds=None, histogram_bins = 50,
                      smooth_fixed = False, smooth_moving = False):
        """
        Perform the image registration using the provided transformation parameters.
        
        Args:
            transform_params: The transformation parameters (e.g., translation, rotation).
            metric_type (str): The metric used for the registration (e.g., 'MeanSquares', 'Mattes').
            optimizer_type (str): The optimizer used for the registration (e.g., 'GradientDescent').
        
        Returns:
            SimpleITK.Transform: The resulting transformation after registration.
        """


        initial_transform = sitk.Similarity3DTransform()

        # Explicitly set the matrix to the identity matrix
        initial_transform.SetMatrix([1.0, 0.0, 0.0,
                                    0.0, 1.0, 0.0,
                                    0.0, 0.0, 1.0])

        # Set the translation to zero
        initial_transform.SetTranslation([0.0, 0.0, 0.0])

        # Set the scale to 1.0
        initial_transform.SetScale(1.0)

        registration = sitk.ImageRegistrationMethod()

        self.first_metric_value = None
        self.second_metric_value = None

        registration.SetInitialTransform(initial_transform)

        fixed_d, moving_d = self.choose_registration_images(smoothing, shrinking, smooth_fixed, smooth_moving)

        if self.callback==1:
            (n1,n2,n3) = fixed_d.GetSize()
            N_pixels = n1*n2*n3
            k_flops = N_pixels*sampling_percentage*max_iter
            print('Estimated time for registration optimization',k_flops/750000000, 'min. Calculated on hpc, linuxsh node with Gradient Descent')


        if metric_type == 'mmi':
            registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=histogram_bins)
        elif metric_type == 'ms' or metric_type == 'ls':
            registration.SetMetricAsMeanSquares()
            fixed_d = sitk.BinaryThreshold(fixed_d, lowerThreshold=thresholds[0], upperThreshold=float("inf"), insideValue=1, outsideValue=0)
            moving_d = sitk.BinaryThreshold(moving_d, lowerThreshold=thresholds[1], upperThreshold=float("inf"), insideValue=1, outsideValue=0)
            fixed_d =sitk.Cast(fixed_d, sitk.sitkFloat32)
            moving_d =sitk.Cast(moving_d, sitk.sitkFloat32)

        else:
            raise ValueError('Specify metric type: Either "mmi" or "ms"/"ls"')


        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(sampling_percentage)

        if optimizer_type == 'gd':
            registration.SetOptimizerAsGradientDescent(
            learningRate=learning_rate,
            numberOfIterations=max_iter,
            convergenceMinimumValue=-1e-16,
            convergenceWindowSize=1000
        )


        else:
            raise ValueError('Specify optimization algorithm: Only "gd" currently supported')
                            
        #registration.SetOptimizerScalesFromPhysicalShift(smallParameterVariation = 0.01)
        registration.SetInterpolator(sitk.sitkLinear)


        self.metric_values = []
        self.iterations = []
        if self.callback==1:
            self.metric_values = []
            self.iterations = []
            registration.AddCommand(sitk.sitkIterationEvent, lambda: self.registration_callback(
            registration.GetOptimizerIteration(),
            registration.GetMetricValue(), every_N = max_iter//10,
            learning_rate =registration.GetOptimizerLearningRate()))
            
        registration.Execute(fixed_d, moving_d)
        transform = registration.GetInitialTransform()
        initial_transform = sitk.Similarity3DTransform(transform)

        if self.callback==1:
            print('Transformation has matrix', initial_transform.GetMatrix(),' and translation', initial_transform.GetTranslation())
            print('Learning rate after execution:', f"{learning_rate:.15f}")



        random_integer = random.randint(1, 1000000)
        transform.ID = random_integer
        transform.FLAG = self.metric_value > self.first_metric_value
        return transform

    def resample(self, transform=None, interpolation_type: str = 'linear', inplace: bool = True,fixed=False, padding = None):
        """
        Resample the moving image using the given transformation object.
        
        Args:
            transform (SimpleITK.Transform): The transformation object.
            interpolation_type (str, optional): The interpolation type ('linear' or 'nearest'). Default is 'linear'.
            inplace (bool, optional): Whether to overwrite the moving image or return a new resampled image. Default is True.
        
        Returns:
            SimpleITK.Image: The resampled image (if inplace=False).
        """
        print(transform)
        if transform is None:
            transform = sitk.Similarity3DTransform()
            rotation_matrix = [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ]
            matrix = [elem for row in rotation_matrix for elem in row]
            transform.SetMatrix(matrix)
        else:
            if transform.FLAG:
                print("Refusing to apply transformation with worse loss. Set transform.FLAG=False and resample again to overwrite")
                return

            if transform.ID in self.called_inputs:
                print(f"Method already called with ID: {transform.ID}")
                return
            self.called_inputs.add(transform.ID)
        
        # Perform the operation
        print(f"Resampling the transformation")

        if interpolation_type == 'linear':
            interpolation_method = sitk.sitkLinear
        elif interpolation_type == 'nearest':
            interpolation_method = sitk.sitk.NearestNeighbor

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(self.fixed)  # Reference image (fixed)
        resampler.SetInterpolator(interpolation_method)   # Interpolation method
        resampler.SetTransform(transform)    # Apply the initial transform (aligned centroids)
        resampler.SetOutputPixelType(self.fixed.GetPixelID())
        resampler.SetOutputSpacing(self.fixed.GetSpacing())  # Ensure the spacing is preserved
        resampler.SetOutputOrigin(self.fixed.GetOrigin())  # Preserve origin
        resampler.SetOutputDirection(self.fixed.GetDirection())

        if padding is not None:
            origin = self.fixed.GetOrigin()
            spacing = self.fixed.GetSpacing()
            direction = self.fixed.GetDirection()
            size = self.fixed.GetSize()
            new_size = [size[i] + 2 * padding[i] for i in range(3)]
            new_origin = [origin[i] - padding[i] * spacing[i] for i in range(3)]
            resampler.SetSize(new_size)
            resampler.SetOutputOrigin(new_origin)
        
        if fixed:
            if inplace:
                print('Resampling fixed image')
                self.fixed = resampler.Execute(self.fixed)  # Resample the moving image
                for key, value in self.fixed_smooth.items():
                    print('Resampling smoothed fixed images')
                    self.fixed_smooth[key] = resampler.Execute(self.fixed_smooth[key])
            else:
                print('Returning resampled fixed image')
                return resampler.Execute(self.fixed)
        else:
            
            if inplace:
                print('Resampling moving image')
                self.moving = resampler.Execute(self.moving)  # Resample the moving image
                for key, value in self.moving_smooth.items():
                    self.moving_smooth[key] = resampler.Execute(self.moving_smooth[key])
                    print('Resampling smoothed moving images')
            else:
                print('Returning resampled moving image')
                return resampler.Execute(self.moving)
            
        
        

    def compute_principal_moments_and_axes(self, thresholds,factor = 5):
        """
        Compute the principal moments and axes of inertia for a 3D image.

        Args:
            image (sitk.Image): The input 3D image (non-zero values are considered part of the region).

        Returns:
            tuple: A tuple containing:
                - principal_moments (list of float): The eigenvalues of the inertia matrix.
                - principal_axes (list of list of float): The eigenvectors of the inertia matrix.
                
        """

        fixed_d = self._downsample_image(self.fixed,factor = factor)
        moving_d = self._downsample_image(self.moving,factor = factor)



        # Create a binary mask of the region
        binary_fixed = sitk.BinaryThreshold(fixed_d, lowerThreshold=thresholds[0], upperThreshold=float("inf"), insideValue=1, outsideValue=0)
        binary_moving = sitk.BinaryThreshold(moving_d, lowerThreshold=thresholds[1], upperThreshold=float("inf"), insideValue=1, outsideValue=0)

        # Use LabelShapeStatisticsImageFilter to calculate shape properties
        label_shape_filter_fixed = sitk.LabelShapeStatisticsImageFilter()
        label_shape_filter_moving = sitk.LabelShapeStatisticsImageFilter()
        label_shape_filter_fixed.Execute(binary_fixed)
        label_shape_filter_moving.Execute(binary_moving)

        if label_shape_filter_fixed.GetNumberOfLabels() == 0:
            raise ValueError("The image has no non-zero pixels to calculate moments of inertia.")

        # Get principal moments and axes for label 1 (the only label in this case)
        principal_moments_fixed = label_shape_filter_fixed.GetPrincipalMoments(1)  # Eigenvalues
        principal_moments_moving = label_shape_filter_moving.GetPrincipalMoments(1)  # Eigenvalues
        

        principal_axes_fixed = label_shape_filter_fixed.GetPrincipalAxes(1)  # Eigenvectors (flattened)
        principal_axes_moving = label_shape_filter_moving.GetPrincipalAxes(1)  # Eigenvectors (flattened)

        # Reshape principal axes into a 3x3 matrix
        principal_axes_matrix_fixed = np.array(principal_axes_fixed).reshape((3, 3))
        principal_axes_matrix_moving = np.array(principal_axes_moving).reshape((3, 3))

        centroid_fixed = np.array(label_shape_filter_fixed.GetCentroid(1))
        centroid_moving = np.array(label_shape_filter_moving.GetCentroid(1))

        out = {'fixed': {"moment": principal_moments_fixed, "axes": principal_axes_matrix_fixed, 'centroid': centroid_fixed} , 'moving': {'moment': principal_moments_moving, 'axes': principal_axes_matrix_moving, 'centroid': centroid_moving}}

        return out

    def compute_moments(self, image):
        #binary_fixed = sitk.BinaryThreshold(image, lowerThreshold=thresholds0.5, upperThreshold=float("inf"), insideValue=1, outsideValue=0)
        label_shape_filter_fixed = sitk.LabelShapeStatisticsImageFilter()
        label_shape_filter_fixed.Execute(image)
        principal_moments_fixed = label_shape_filter_fixed.GetPrincipalMoments(1)  # Eigenvalues
        principal_axes_fixed = label_shape_filter_fixed.GetPrincipalAxes(1)  # Eigenvectors (flattened)
        principal_axes_matrix_fixed = np.array(principal_axes_fixed).reshape((3, 3))
        centroid_fixed = np.array(label_shape_filter_fixed.GetCentroid(1))

        out = {"moment": principal_moments_fixed, "axes": principal_axes_matrix_fixed, 'centroid': centroid_fixed}

        return out
    


    def transform_mirror(self,point, vector1, vector2):
        """
        Create a mirroring transformation around a plane specified by a point and two vectors.

        Args:
            point (tuple): A point on the plane (x0, y0, z0).
            vector1 (tuple): First vector spanning the plane (v1x, v1y, v1z).
            vector2 (tuple): Second vector spanning the plane (v2x, v2y, v2z).

        Returns:
            sitk.AffineTransform: A SimpleITK affine transformation for mirroring around the plane.
        """
        # Convert inputs to NumPy arrays
        point = np.array(point)
        vector1 = np.array(vector1)
        vector2 = np.array(vector2)

        # Calculate the normal vector of the plane
        normal = np.cross(vector1, vector2)
        normal = normal / np.linalg.norm(normal)  # Normalize the normal vector

        # Construct the reflection matrix
        n = normal.reshape(3, 1)  # Make the normal a column vector
        reflection_matrix = np.eye(3) - 2 * (n @ n.T)  # Reflection formula: I - 2 * (n * n^T)

        b = point - reflection_matrix@point

        # Convert to a SimpleITK affine transform
        transform = sitk.AffineTransform(3)
        transform.SetMatrix(reflection_matrix.flatten())  # Set the 3x3 matrix
        transform.SetTranslation(b)  # Set the translation vector
        return transform
    
    def registration_callback(self,iteration, metric_value,every_N, learning_rate):
        global previous_metric_value

        self.metric_value = metric_value

        if iteration == 2:
            self.first_metric_value = metric_value

        if not iteration % every_N:
            self.metric_values.append(metric_value)
            self.iterations.append(iteration)
            if previous_metric_value is not None:
                metric_difference = abs(previous_metric_value - metric_value)
                print(f"Iteration {iteration}: Metric Value = {metric_value}, Metric Difference = {metric_difference}")
            else:
                print(f"Iteration {iteration}: Metric Value = {metric_value}")

            print('The current learning rate is', learning_rate)

            # Update previous_metric_value for the next iteration
            previous_metric_value = metric_value
            self


    def plot2d(self, thresholds, difference=False, show_factor = 4,moving=False):
        factor = 5
        fixed_d = self._downsample_image(self.fixed,factor = factor)
        moving_d = self._downsample_image(self.moving,factor = factor)
        if moving:
            temp = fixed_d
            fixed_d = moving_d
            moving_d = temp

        binary_fixed = sitk.BinaryThreshold(fixed_d, lowerThreshold=thresholds[0], upperThreshold=float("inf"), insideValue=1, outsideValue=0)
        label_shape_filter_fixed = sitk.LabelShapeStatisticsImageFilter()
        label_shape_filter_fixed.Execute(binary_fixed)
        principal_axes_fixed = label_shape_filter_fixed.GetPrincipalAxes(1)  # Eigenvectors (flattened)
        matrix = np.array(principal_axes_fixed).reshape((3, 3)).T.flatten()
        transform_coordinate_axes = self.transformation(type='Euler3DTransform',matrix=matrix)


        # Resample

        # Get the original image bounds (physical corners)
        original_size = np.array(fixed_d.GetSize())
        original_spacing = np.array(fixed_d.GetSpacing())
        original_origin = np.array(fixed_d.GetOrigin())
        bounds = [
            original_origin + np.multiply(original_spacing, [0, 0, 0]),
            original_origin + np.multiply(original_spacing, [original_size[0], 0, 0]),
            original_origin + np.multiply(original_spacing, [0, original_size[1], 0]),
            original_origin + np.multiply(original_spacing, [original_size[0], original_size[1], 0]),
            original_origin + np.multiply(original_spacing, [0, 0, original_size[2]]),
            original_origin + np.multiply(original_spacing, [original_size[0], 0, original_size[2]]),
            original_origin + np.multiply(original_spacing, [0, original_size[1], original_size[2]]),
            original_origin + np.multiply(original_spacing, original_size)
        ]

        # Transform the bounds using the rotation transform
        transformed_bounds = [transform_coordinate_axes.TransformPoint(pt) for pt in bounds]

        # Compute the new size and origin
        transformed_bounds = np.array(transformed_bounds)
        min_bounds = np.min(transformed_bounds, axis=0)
        max_bounds = np.max(transformed_bounds, axis=0)

        new_origin = min_bounds
        new_spacing = original_spacing
        new_size = np.ceil((max_bounds - min_bounds) / new_spacing).astype(int)

        # Resample the fixed_d
        resample = sitk.ResampleImageFilter()
        resample.SetSize(new_size.tolist())
        resample.SetOutputSpacing(new_spacing.tolist())
        resample.SetOutputOrigin(new_origin.tolist())
        resample.SetOutputDirection(fixed_d.GetDirection())
        resample.SetTransform(transform_coordinate_axes)
        resample.SetDefaultPixelValue(0)
        fixed_d = resample.Execute(fixed_d)  # Resample the moving image
        moving_d = resample.Execute(moving_d)  # Resample the moving image
        fixed_d = sitk.GetArrayFromImage(fixed_d)
        moving_d = sitk.GetArrayFromImage(moving_d)
        index = np.argmax(np.sum(fixed_d,axis=(0,1)))

        fixed_d = self._downsample_image(self.fixed,factor = show_factor)
        moving_d = self._downsample_image(self.moving,factor = show_factor)

        # Get the original image bounds (physical corners)
        original_size = np.array(fixed_d.GetSize())
        original_spacing = np.array(fixed_d.GetSpacing())
        original_origin = np.array(fixed_d.GetOrigin())
        bounds = [
            original_origin + np.multiply(original_spacing, [0, 0, 0]),
            original_origin + np.multiply(original_spacing, [original_size[0], 0, 0]),
            original_origin + np.multiply(original_spacing, [0, original_size[1], 0]),
            original_origin + np.multiply(original_spacing, [original_size[0], original_size[1], 0]),
            original_origin + np.multiply(original_spacing, [0, 0, original_size[2]]),
            original_origin + np.multiply(original_spacing, [original_size[0], 0, original_size[2]]),
            original_origin + np.multiply(original_spacing, [0, original_size[1], original_size[2]]),
            original_origin + np.multiply(original_spacing, original_size)
        ]

        # Transform the bounds using the rotation transform
        transformed_bounds = [transform_coordinate_axes.TransformPoint(pt) for pt in bounds]

        # Compute the new size and origin
        transformed_bounds = np.array(transformed_bounds)
        min_bounds = np.min(transformed_bounds, axis=0)
        max_bounds = np.max(transformed_bounds, axis=0)

        new_origin = min_bounds
        new_spacing = original_spacing
        new_size = np.ceil((max_bounds - min_bounds) / new_spacing).astype(int)

        # Resample the fixed_d
        resample = sitk.ResampleImageFilter()
        resample.SetSize(new_size.tolist())
        resample.SetOutputSpacing(new_spacing.tolist())
        resample.SetOutputOrigin(new_origin.tolist())
        resample.SetOutputDirection(fixed_d.GetDirection())
        resample.SetTransform(transform_coordinate_axes)
        resample.SetDefaultPixelValue(0)

        #resampler = sitk.ResampleImageFilter()
        #resampler.SetReferenceImage(fixed_d)  # Reference image (fixed)
        #resampler.SetInterpolator(sitk.sitkLinear)   # Interpolation method
        #resampler.SetTransform(transform_coordinate_axes)    # Apply the initial transform (aligned centroids)
        #resampler.SetOutputPixelType(fixed_d.GetPixelID())
        #resampler.SetOutputSpacing(fixed_d.GetSpacing())  # Ensure the spacing is preserved
        #resampler.SetOutputOrigin(fixed_d.GetOrigin())  # Preserve origin
        #resampler.SetOutputDirection(fixed_d.GetDirection())
        fixed_d = resample.Execute(fixed_d)  # Resample the fixed image
        moving_d = resample.Execute(moving_d)  # Resample the moving image



        fixed_d = sitk.GetArrayFromImage(fixed_d)
        moving_d = sitk.GetArrayFromImage(moving_d)

        binary_fixed= (fixed_d>thresholds[0])*1
        binary_moving = (moving_d > thresholds[1])*1
        ny, nx,nz = np.shape(binary_moving)
        if difference:
            plt.figure()
            print(ny, nx)
            np.set_printoptions(precision=2)
            plt.imshow(binary_fixed[:,:,factor*index//show_factor] - binary_moving[:,:,factor*index//show_factor], cmap='viridis', alpha=0.5)
            x, y = 0.95*nx, 0.94*ny  # Coordinates (in data units)
            plt.text(x, y, 'v_1 ->' + str(matrix[0::3]), color='black', fontsize=8, ha='right', va='bottom')
            x, y = 0.05*nx, 0.04*ny  # Coordinates (in data units)
            plt.text(x, y, 'v_2 ->' + str(matrix[1::3]), color='black', fontsize=8, ha='left', va='bottom')
            x, y = 0.05*nx, 0.97*ny  # Coordinates (in data units)
            plt.text(x, y, 'Yellow: Fixed, Purple: Moving', color='black', fontsize=8, ha='left', va='bottom')
            plt.show()

        else:
            plt.figure()
            plt.imshow(fixed_d[:,:,factor*index//show_factor], cmap='viridis', alpha=0.5)
            plt.imshow(moving_d[:,:,factor*index//show_factor], cmap='plasma', alpha=0.5)
            plt.show()



    def choose_registration_images(self, smoothing_i, shrinking_i, smooth_fixed, smooth_moving):
        if (shrinking_i is not None) and smooth_fixed:
            if str(smoothing_i) in self.fixed_smooth:
                print('Using precalculated smoothed fixed image with shrinking')
                fixed_d = self._downsample_image(self.fixed_smooth[str(smoothing_i)], factor=shrinking_i,smooth = 0)
            else:
                print('Calculating smoothed fixed image with shrinking')
                fixed_d = self._downsample_image(self.fixed, factor=shrinking_i,smooth=smoothing_i)

        if (shrinking_i is not None) and smooth_moving:
            if str(smoothing_i) in self.moving_smooth:
                print('Using precalculated smoothed moving image with shrinking')
                moving_d = self._downsample_image(self.moving_smooth[str(smoothing_i)], factor=shrinking_i,smooth = 0)
            else:
                print('Calculating smoothed moving image with shrinking')
                moving_d = self._downsample_image(self.moving, factor=shrinking_i,smooth=smoothing_i)


        if (shrinking_i is not None) and (not smooth_fixed):
            print('Using non-smoothed fixed image with shrinking')
            fixed_d = self._downsample_image(self.fixed, factor=shrinking_i,smooth=0)

        if (shrinking_i is not None) and (not smooth_moving):
                print('Using non-smoothed moving image with shrinking')
                moving_d = self._downsample_image(self.moving, factor=shrinking_i,smooth=0)

        if (shrinking_i is None):
            print('Using non-smoothed moving image without shrinking')
            fixed_d = self.fixed
            moving_d = self.moving

        return fixed_d, moving_d


    def resample_to_unit_spacing(self, image):
        current_spacing = image.GetSpacing()
        current_size = image.GetSize()

        # Compute the average physical spacing
        average_spacing = sum(current_spacing) / len(current_spacing)

        # Normalize spacing to 1.0
        desired_spacing = [1.0] * len(current_spacing)

        # Compute the new size to maintain the resolution
        new_size = [
            int(round(current_size[i] * (current_spacing[i] / desired_spacing[i])))
            for i in range(len(current_spacing))
        ]

        # Resample the image with normalized spacing
        resampled_image = sitk.Resample(
            image,
            new_size,
            sitk.Transform(),
            sitk.sitkLinear,  # Interpolation method
            image.GetOrigin(),
            desired_spacing,
            image.GetDirection(),
            0,  # Default pixel value for areas outside original image
            image.GetPixelID(),
        )
        return resampled_image

    def compute_stone_boundaries(self,thresholds = [0.5,0.5], factor = 2,connected_size = 0.01):
        fixed_d = self._downsample_image(self.fixed, factor=factor)
        moving_d = self._downsample_image(self.moving,factor=factor)
        fixed_d = sitk.SmoothingRecursiveGaussian(fixed_d, sigma=0.01)
        moving_d = sitk.SmoothingRecursiveGaussian(moving_d, sigma=0.01)


        # Step 2: Threshold to generate markers for background and foreground
        foreground_marker_fixed = fixed_d > thresholds[0]+0.0  # High intensity for the blob
        foreground_marker_moving = moving_d > thresholds[1]+0.0  # High intensity for the blob



        labeled_image_fixed = sitk.ConnectedComponent(foreground_marker_fixed)
        labeled_image_moving = sitk.ConnectedComponent(foreground_marker_moving)



        relabeled_image = sitk.RelabelComponent(labeled_image_fixed, sortByObjectSize=True)
        # Step 3: Create a binary mask for components larger than the minimum size
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(relabeled_image)

        # Create a new binary image
        cleaned_binary_image = sitk.Image(foreground_marker_fixed.GetSize(), sitk.sitkUInt8)
        cleaned_binary_image.CopyInformation(foreground_marker_fixed)

        for label in stats.GetLabels():
            if stats.GetPhysicalSize(label) >= connected_size:
                cleaned_binary_image = cleaned_binary_image | (relabeled_image == label)

        cleaned_binary_image_fixed = cleaned_binary_image

        relabeled_image = sitk.RelabelComponent(labeled_image_moving, sortByObjectSize=True)
        # Step 3: Create a binary mask for components larger than the minimum size
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(relabeled_image)

        # Create a new binary image
        cleaned_binary_image = sitk.Image(foreground_marker_moving.GetSize(), sitk.sitkUInt8)
        cleaned_binary_image.CopyInformation(foreground_marker_moving)

        for label in stats.GetLabels():
            if stats.GetPhysicalSize(label) >= connected_size:
                cleaned_binary_image = cleaned_binary_image | (relabeled_image == label)

        cleaned_binary_image_moving = cleaned_binary_image

        return cleaned_binary_image_fixed, cleaned_binary_image_moving


    def plot3d(self,image, moments=None):

        size = image.GetSize()            # Image size (number of pixels in each dimension)
        origin = image.GetOrigin()        # Physical coordinate of the first voxel
        spacing = image.GetSpacing()      # Physical size of each voxel
        direction = image.GetDirection()  # Image direction cosines
        min_physical = origin

        # Compute the physical coordinates of the image corners
        # Min corner (always the origin)

        # Max corner (computed as origin + size * spacing in each direction)
        # Incorporating direction cosines for non-orthogonal axes
        max_physical = [
            origin[i] + spacing[i] * (size[i] - 1) * direction[i * 3 + i]
            for i in range(3)
        ]

        volume_np = sitk.GetArrayFromImage(image)
        nx,ny,nz = volume_np.shape
        N = max(nx,ny,nz)
        stride = int(N//60)+1

        matrix_d = volume_np[::stride,::stride,::stride]

        x = np.linspace(origin[0], max_physical[0], matrix_d.shape[0])
        y = np.linspace(origin[1], max_physical[1], matrix_d.shape[1])
        z = np.linspace(origin[2], max_physical[2], matrix_d.shape[2])

        data = go.Volume(
        x=np.repeat(x, matrix_d.shape[1] * matrix_d.shape[2]),
        y=np.tile(np.repeat(y, matrix_d.shape[2]), matrix_d.shape[0]),
        z=np.tile(z, matrix_d.shape[0] * matrix_d.shape[1]),
        value=matrix_d.flatten(),  # Flatten the matrix to get values
        opacity=0.2,  # Lower opacity for a better 3D effect
        isomin=0.5,   # Minimum threshold for volume rendering
        isomax=1,   # Maximum threshold for volume rendering
        surface_count=15,  # Number of surfaces in the volume rendering
        colorscale="Viridis")

        fig = go.Figure(data=data)

        fig.update_layout(
            scene=dict(
                xaxis=dict(nticks=4, range=[origin[0], max_physical[0]], title='X Axis'),
                yaxis=dict(nticks=4, range=[origin[1], max_physical[1]], title='Y Axis'),
                zaxis=dict(nticks=4, range=[origin[2], max_physical[2]], title='Z Axis'),
                aspectmode='manual',  # Set manual aspect ratio
                aspectratio=dict(
                    x=matrix_d.shape[0] / matrix_d.shape[2],
                    y=matrix_d.shape[1] / matrix_d.shape[2],
                    z=1  # Use 1 as the reference dimension for scaling
                )
            ),
            title="3D rendering"
        )
        if moments is not None:
            # Define the start point for all arrows (x, y, z)
            x = moments['centroid']

            # Define the direction vectors for the 3 arrows (you can customize these)
            directions = moments['axes']

            # Define scaling factors for the arrow lengths
            scales = np.array(moments['moment'])  # Length of each arrow, can be adjusted

            # Create the arrows by scaling the direction vectors
            arrow_endpoints = [x + scale * direction for scale, direction in zip(scales, directions)]

            # Create a 3D scatter plot for the arrows


            # Add each arrow as a line segment
            for endpoint in arrow_endpoints:
                fig.add_trace(go.Scatter3d(
                    x=[x[0], endpoint[0]],
                    y=[x[1], endpoint[1]],
                    z=[x[2], endpoint[2]],
                    mode='lines+text',
                    line=dict(color='red', width=5),
                    text=["", "Arrow"],
                    textposition="top center"
                ))

        fig.show()