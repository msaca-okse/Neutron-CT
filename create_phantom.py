###
import numpy as np
from cil.framework import ImageGeometry, ImageData, AcquisitionGeometry, AcquisitionData
from cil.utilities.display import show2D
import matplotlib.pyplot as plt
from cil.plugins.astra import ProjectionOperator
from astropy.io import fits


def circular_mask(shape, r):
    """
    Create a mask for a quadratic 2D array with ones inside a circle and zeros outside.
    
    Parameters:
    shape (tuple): Shape of the quadratic array (n, n).
    r (float): Radius for the mask. r=1 means all values are set to one, r=0 means all values are set to zero.
    
    Returns:
    numpy.ndarray: Mask array with ones inside the circle and zeros outside.
    """
    n = shape[0]
    
    # Create a grid of coordinates
    y, x = np.ogrid[:n, :n]
    
    # Calculate the center of the array
    center = (n - 1) / 2  # Works for both even and odd dimensions
    
    # Calculate the distance of each point from the center
    distance_from_center = np.sqrt((x - center)**2 + (y - center)**2)
    
    # Create the mask: values inside the circle (<=r*n/2) are 1, outside are 0
    mask = distance_from_center <= r * (n / 2)
    
    return mask.astype(np.uint8)

# Create a 3D Shepp-Logan phantom
def create_3d_shepp_logan(n_slice=128, size_x=256, size_y=256):

    # Parameters for the ellipsoids in the Shepp-Logan phantom
    ellipsoids = [
        # (value, a, b, c, x0, y0, z0, phi, theta, psi)
        [4, 0.69*0.65, 0.92*0.65, 0.9*0.65, 0, 0, 0, 0, 0, 0],        # large ellipse
        [3, 0.6624*0.65, 0.874*0.65, 0.88*0.65, 0, 0, 0, 0, 0, 0],     # inner ellipse
        [2, 0.31, 0.11, 0.22, 0.22, 0, 0, -18, 0, 10],  # top right ellipse
        [2, 0.41*0.8, 0.16, 0.15, -0.22, 0, 0, 18, 0, 10],  # top left ellipse
        [1.5, 0.25, 0.21, 0.25, 0, 0.35, -0.15, 0, 0, 0], # bottom middle
        [0.5, 0.046, 0.046, 0.046, 0, 0.1, 0.25, 0, 0, 0], # small middle
        [1, 0.046, 0.023, 0.02, -0.08, -0.605, 0, 0, 0, 0], # bottom left
        [1, 0.023, 0.023, 0.02, 0, -0.605, 0, 0, 0, 0],   # bottom middle
        [1, 0.023, 0.046, 0.02, 0.06, -0.605, 0, 0, 0, 0], # bottom right
    ]

    # Create a 3D grid of coordinates
    x = np.linspace(-1, 1, size_x)
    y = np.linspace(-1, 1, size_y)
    z = np.linspace(-1, 1, n_slice)
    X, Y, Z = np.meshgrid(z, x, y, indexing='ij')

    # Initialize the phantom volume
    phantom = np.zeros((n_slice, size_x, size_y))

    # Function to add ellipsoids to the phantom
    for ellipsoid in ellipsoids:
        value, a, b, c, x0, y0, z0, phi, theta, psi = ellipsoid

        # Convert angles from degrees to radians
        phi, theta, psi = np.radians([phi, theta, psi])

        # Calculate distances based on ellipsoid equation
        distances = (((X - x0) / a) ** 2 + ((Y - y0) / b) ** 2 + ((Z - z0) / c) ** 2)
        
        # Add ellipsoid contribution to the phantom
        phantom[distances <= 1] += value

    ig = ImageGeometry(voxel_num_z=n_slice, 
                    voxel_num_y=size_x, 
                    voxel_num_x=size_y,
                    voxel_size_z=1/n_slice, 
                    voxel_size_y=1/size_y,
                    voxel_size_x= 1/size_x)
    
    out = ImageData(array=phantom, deep_copy=False, geometry=ig)

    return out


def simulate_projections(image_data, angles, beam_radius = None,dose = None,intensity = 1e4,dose_variation = 0.1):
    
    ig = image_data.geometry
    size_x = ig.voxel_num_x
    size_y = ig.voxel_num_y
    if dose == None:
        dose = np.random.uniform(low = -dose_variation, high = dose_variation, size=len(angles)) + np.linspace(20, 80, num=len(angles))
    elif dose == 1:
        dose = np.ones(len(angles))

    if beam_radius == None:
        mask = np.ones((size_x,size_y))
    else:
        mask = circular_mask((size_x, size_y), beam_radius)
    
    ag = AcquisitionGeometry.create_Parallel3D(ray_direction=[0, 1, 0], detector_position=[0, 0, 0], 
                                        detector_direction_x=[1, 0, 0], detector_direction_y=[0, 0, 1],
                                        rotation_axis_position=[0, 0, 0], rotation_axis_direction=[0, 0, 1],
                                        units='units distance')\
                                        .set_angles(angles)\
                                        .set_panel((size_x,size_y),pixel_size=(1/size_x,1/size_y))\
                                        .set_labels(['vertical','angle', 'horizontal'])
    
    radon = ProjectionOperator(image_geometry=ig, acquisition_geometry=ag, device='gpu')
    noiseless_data = radon.direct(image_data)


    noiseless_data = noiseless_data.sapyb(-1,ag.allocate(1),0.1)

    noiseless_data = noiseless_data.exp() # Get the exponential data...

    noisy_data = noiseless_data.clone()
    # Add dose over each projection (angle)
    

    # Reshape the 1D array to broadcast along the second dimension
    dose = dose[np.newaxis, :, np.newaxis]  # Shape: (1, 256, 1)
    # Multiply the 1D array across the 3D array along the second axis
    noiseless_data.fill(noiseless_data.as_array() * dose)
    #poisson_noise.fill(np.random.poisson(poisson_noise_level, size=(size_x, len(angles),size_y)))
    noisy_data.fill(np.random.poisson(intensity*noiseless_data.as_array()))
    noiseless_data.fill(noiseless_data.as_array()*intensity)
    # Rescale to uint (divided by 2)
    scale = np.max(noisy_data.as_array())
    noisy_data = noisy_data.sapyb(65535/2/scale,ag.allocate(1),0)
    noiseless_data = noiseless_data.sapyb(65535/2/scale,ag.allocate(1),0)


    noisy_data.fill(noisy_data.as_array().astype(np.uint16)*mask[:,np.newaxis,:]+1)
    noiseless_data.fill(noiseless_data.as_array().astype(np.uint16)*mask[:,np.newaxis,:]+1)

    
    return noisy_data, noiseless_data




if __name__=='__main__':
    size_x, size_y, n_slice = 256, 256, 128  # Custom size: 256x256x128
    phantom_3d = create_3d_shepp_logan(n_slice = n_slice,size_x=size_x, size_y=size_y)
    phantom_3d.reorder('astra')
    
    angles = np.linspace(0, 360, 100, endpoint=False, dtype=np.float32)
    angles = np.repeat(angles,3) # Take 3 measurements per angle.
    print(angles)
    print(len(angles))

    ndata, data = simulate_projections(phantom_3d, angles, dose = None,dose_variation=5,intensity=1e2,beam_radius=0.85)
    ndata.reorder(('angle','horizontal','vertical'))
    data.reorder(('angle','horizontal','vertical'))
    print(ndata)
    print(data)
    current_image = show2D(data.log(),slice_list = ('horizontal',220))
    current_image.save('/work3/msaca/current_figures/current.png')
    current_image = show2D(ndata.log(),slice_list = ('horizontal',220))
    current_image.save('/work3/msaca/current_figures/current2.png')
    

    save = True
    if save == True:
        ndata = ndata.as_array()
        for i_angle in range(len(angles)):
            # Create a FITS file for each slice
            hdu = fits.PrimaryHDU(ndata[i_angle])  # Create a PrimaryHDU object with the 2D slice
            filename = 'proj_simx3_'+ f'{i_angle+1:05}.fits'  # Filename for each FITS file
            hdu.writeto('/work3/msaca/simulation_data/experiment_1/' + filename, overwrite=True) 