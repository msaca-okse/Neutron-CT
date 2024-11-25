import numpy as np
import SimpleITK as sitk
import copy

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
            print('hello')
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
            mean_filter.SetRadius(factor)  # Adjust the radius as needed
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




    def transformation(self, type='Similarity3Dtransform', matrix=None,translation=(0,0,0)):
        """
        Returns a transformation object based on the provided transformation parameters.
        
        Args:
            transform_parameters: The parameters that define the transformation (e.g., rotation, translation).
        
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

        return transform

    def register(self, transform_params=None, metric_type: str=None, optimizer_type: str=None,inPlace=False):
        """
        Perform the image registration using the provided transformation parameters.
        
        Args:
            transform_params: The transformation parameters (e.g., translation, rotation).
            metric_type (str): The metric used for the registration (e.g., 'MeanSquares', 'Mattes').
            optimizer_type (str): The optimizer used for the registration (e.g., 'GradientDescent').
        
        Returns:
            SimpleITK.Transform: The resulting transformation after registration.
        """
        # Registration algorithm logic will go here (in the future)
        registration = sitk.ImageRegistrationMethod()
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(0.1)

        # Use a multi-resolution pyramid
        registration.SetShrinkFactorsPerLevel([2, 2, 1])
        registration.SetSmoothingSigmasPerLevel([10, 5, 0])

        registration.SetInitialTransform(sitk.AffineTransform(3), inPlace=inPlace)
        registration.SetOptimizerAsGradientDescent(learningRate=0.1,
                                        numberOfIterations=2000,
                                        convergenceMinimumValue=1e-6,
                                            convergenceWindowSize=10)
        
        registration.SetOptimizerScalesFromPhysicalShift()
        registration.SetInterpolator(sitk.sitkLinear)
        transform = registration.Execute(self.fixed, self.moving)

        return transform.GetNthTransform(0)

    def resample(self, transform, interpolation_type: str = 'linear', inplace: bool = True,fixed=False):
        """
        Resample the moving image using the given transformation object.
        
        Args:
            transform (SimpleITK.Transform): The transformation object.
            interpolation_type (str, optional): The interpolation type ('linear' or 'nearest'). Default is 'linear'.
            inplace (bool, optional): Whether to overwrite the moving image or return a new resampled image. Default is True.
        
        Returns:
            SimpleITK.Image: The resampled image (if inplace=False).
        """
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
        
        if fixed:
            if inplace:
                self.fixed = resampler.Execute(self.fixed)  # Resample the moving image
            else:
                return resampler.Execute(self.fixed)
        else:
            if inplace:
                self.moving = resampler.Execute(self.moving)  # Resample the moving image
            else:
                return resampler.Execute(self.moving)
        
        

    def compute_principal_moments_and_axes(self, thresholds):
        """
        Compute the principal moments and axes of inertia for a 3D image.

        Args:
            image (sitk.Image): The input 3D image (non-zero values are considered part of the region).

        Returns:
            tuple: A tuple containing:
                - principal_moments (list of float): The eigenvalues of the inertia matrix.
                - principal_axes (list of list of float): The eigenvectors of the inertia matrix.
        """
        # Create a binary mask of the region
        binary_fixed = sitk.BinaryThreshold(self.fixed, lowerThreshold=thresholds[0], upperThreshold=float("inf"), insideValue=1, outsideValue=0)
        binary_moving = sitk.BinaryThreshold(self.moving, lowerThreshold=thresholds[1], upperThreshold=float("inf"), insideValue=1, outsideValue=0)

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