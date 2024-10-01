###
import numpy as np
from cil.framework import ImageGeometry, ImageData, AcquisitionGeometry, AcquisitionData
from cil.utilities.display import show2D
import matplotlib.pyplot as plt
from cil.plugins.astra import ProjectionOperator

# Create a 3D Shepp-Logan phantom
def create_3d_shepp_logan(n_slice=128, size_x=256, size_y=256):

    # Parameters for the ellipsoids in the Shepp-Logan phantom
    ellipsoids = [
        # (value, a, b, c, x0, y0, z0, phi, theta, psi)
        [10, 0.69*0.8, 0.92*0.8, 0.9*0.8, 0, 0, 0, 0, 0, 0],        # large ellipse
        [8, 0.6624*0.8, 0.874*0.8, 0.88*0.8, 0, 0, 0, 0, 0, 0],     # inner ellipse
        [5, 0.31, 0.11, 0.22, 0.22, 0, 0, -18, 0, 10],  # top right ellipse
        [5, 0.41, 0.16, 0.15, -0.22, 0, 0, 18, 0, 10],  # top left ellipse
        [6, 0.25, 0.21, 0.25, 0, 0.35, -0.15, 0, 0, 0], # bottom middle
        [3, 0.046, 0.046, 0.046, 0, 0.1, 0.25, 0, 0, 0], # small middle
        [4, 0.046, 0.023, 0.02, -0.08, -0.605, 0, 0, 0, 0], # bottom left
        [4, 0.023, 0.023, 0.02, 0, -0.605, 0, 0, 0, 0],   # bottom middle
        [4, 0.023, 0.046, 0.02, 0.06, -0.605, 0, 0, 0, 0], # bottom right
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


def simulate_projections(image_data, angles, noise_level = 0, beam_radius = 1):

    


    ig = image_data.geometry
    size_x = ig.voxel_num_x
    size_y = ig.voxel_num_y
    ag = AcquisitionGeometry.create_Parallel3D(ray_direction=[0, 1, 0], detector_position=[0, 0, 0], 
                                            detector_direction_x=[1, 0, 0], detector_direction_y=[0, 0, 1],
                                            rotation_axis_position=[0, 0, 0], rotation_axis_direction=[0, 0, 1],
                                            units='units distance')\
                                            .set_angles(angles)\
                                            .set_panel((size_x,size_y),pixel_size=(1/size_x,1/size_y))\
                                            .set_labels(['vertical','angle', 'horizontal'])
    radon = ProjectionOperator(image_geometry=ig, acquisition_geometry=ag, device='gpu')
    out = radon.direct(image_data)
    return out

# 



if __name__=='__main__':
    size_x, size_y, n_slice = 256, 256, 128  # Custom size: 256x256x128
    phantom_3d = create_3d_shepp_logan(n_slice = n_slice,size_x=size_x, size_y=size_y)
    phantom_3d.reorder('astra')
    
    print(phantom_3d)

    angles = np.linspace(0, 360, 10, endpoint=False, dtype=np.float32)

    
    data = simulate_projections(phantom_3d, angles)
    print(data)
    show2D(data)
    plt.savefig('/work3/msaca/current_figures/current.png')

