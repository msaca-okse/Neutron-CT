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

    def load_images(self, fixed_array: str, moving_array: str):
        """
        Loads two images from numpy arrays and converts them to SimpleITK images.
        
        Args:
            fixed_path (str): Pointer to the fixed image numpy array.
            moving_path (str): Pointer to the moving image numpy array.
        
        """
        # Convert numpy arrays to SimpleITK images
        self.fixed = sitk.GetImageFromArray(fixed_array)
        self.moving = sitk.GetImageFromArray(moving_array)

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

    def filter_threshold(self, fixed_threshold: float = 0.0, moving_threshold: float = 0.0):
        """
        Applies thresholding to both fixed and moving images. Everything above the threshold is set to 1, and below is set to 0.
        
        Args:
            fixed_threshold (float, optional): Threshold for the fixed image. Default is 0.0.
            moving_threshold (float, optional): Threshold for the moving image. Default is 0.0.
        """
        # Apply thresholding to both fixed and moving images (inplace)
        self.fixed = self.fixed > fixed_threshold
        self.moving = self.moving > moving_threshold

    def downsizing(self, fixed_factor: int, moving_factor: int):
        """
        Downsizes both the fixed and moving images by a specified factor.
        
        Args:
            fixed_factor (int): The factor by which to downsize the fixed image.
            moving_factor (int): The factor by which to downsize the moving image.
        """
        # Resample the images (downsize) by the given factor, preserving the original grid structure
        # This will be done inplace, reducing the number of pixels
        self.fixed = self._downsample_image(self.fixed, fixed_factor)
        self.moving = self._downsample_image(self.moving, moving_factor)


    def _downsample_image(self, input_image, factor=10):
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
        # Get the original size and spacing of the image
        size = input_image.GetSize()
        spacing = input_image.GetSpacing()

        # Calculate the new size (downsampling by factor)
        new_size = [int(size[0] / factor), int(size[1] / factor), int(size[2] / factor)]
        
        # Calculate the new spacing (enlarging the spacing to match the downsampled size)
        new_spacing = [s * factor for s in spacing]
        
        # Set the resampling transform (identity transform)
        transform = sitk.Transform(3, sitk.sitkIdentity)

        # Perform the resampling (using average interpolation for downsampling)
        downsampled_image = sitk.Resample(input_image,
                                        new_size,
                                        transform,
                                        sitk.sitkBSpline,  # BSpline interpolation is good for downsampling
                                        input_image.GetOrigin(),
                                        new_spacing,
                                        input_image.GetDirection(),
                                        0)  # 0 is the background value for the resampling
        
        return downsampled_image


    def transformation(self, type='Similarity3Dtransform', matrix=None):
        """
        Returns a transformation object based on the provided transformation parameters.
        
        Args:
            transform_parameters: The parameters that define the transformation (e.g., rotation, translation).
        
        Returns:
            SimpleITK.Transform: The transformation object.
        """
        # Transformation creation logic will go here (in the future)
        pass

    def register(self, transform_params, metric_type: str, optimizer_type: str):
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
        pass

    def resample(self, transform, interpolation_type: str = 'linear', inplace: bool = True):
        """
        Resample the moving image using the given transformation object.
        
        Args:
            transform (SimpleITK.Transform): The transformation object.
            interpolation_type (str, optional): The interpolation type ('linear' or 'nearest'). Default is 'linear'.
            inplace (bool, optional): Whether to overwrite the moving image or return a new resampled image. Default is True.
        
        Returns:
            SimpleITK.Image: The resampled image (if inplace=False).
        """
        # Resampling logic will go here (in the future)
        pass
