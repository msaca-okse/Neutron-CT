## Select packages
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from cil.processors import Normaliser
from cil.framework import AcquisitionGeometry, AcquisitionData, ImageGeometry, ImageData
from cil.utilities.display import show2D, show_geometry
from cil.io import TIFFWriter, TIFFStackReader
from cil.recon import FBP
import create_phantom as cph

# Path settings
upper_path = '/work3/msaca/Mars_data_1/02_filtereddata/'
subfolder_normalizations = 'sinogram_data/'
device='gpu'

N_pixels = 2048

reader = TIFFStackReader(file_name=upper_path + subfolder_normalizations + 'sinogram_file__idx_0100.tiff')
sin1 = reader.read()


angles = np.linspace(0, 360, 1125, endpoint=False, dtype=np.float32)
ig = ImageGeometry(voxel_num_x=N_pixels, 
                   voxel_num_y=N_pixels, 
                   voxel_size_x=1/N_pixels, 
                   voxel_size_y=1/N_pixels)
ag = AcquisitionGeometry.create_Parallel2D(detector_position = [0, 10])\
                            .set_angles(angles)\
                            .set_panel(N_pixels, pixel_size=1/N_pixels)

sin1 = AcquisitionData(sin1, geometry=ag)
sin1.reorder('tigre')
fbp1 = FBP(sin1, ig, backend='tigre')
fbp1.set_filter(filter = 'ram-lak', cutoff=0.5)
recon_fbp1 = fbp1.run()

sin2 = sin1.copy()
temp = sin2.as_array()
temp[:,0:25] = 0
temp[:,-25:-1] = 0
sin2.fill(temp)
sin2.reorder('tigre')
fbp2 = FBP(sin2, ig, backend='tigre')
fbp2.set_filter(filter = 'ram-lak', cutoff=0.5)
recon_fbp2 = fbp2.run()


# Plot images to current figures folder
cmap = 'viridis'
size_x, size_y, n_slice = 256, 256, 128  # Custom size: 256x256x128
phantom_3d = cph.create_3d_shepp_logan(n_slice = n_slice,size_x=size_x, size_y=size_y,flat=False)
true_slice = int(100/256*128)
phantom_slice = phantom_3d.get_slice(vertical=true_slice-1)


results = [recon_fbp1,recon_fbp2, phantom_slice]
current_image = show2D(results, ["FBP-recon","Modified FBP recon","True slice"], fix_range=True, cmap=cmap, size=(10,10))
current_image.save('/work3/msaca/current_figures/current_slice.png')
sinograms = [sin1,sin2]
current_image = show2D(sinograms, ['Original Sinogram', "Modified Sinogram"], fix_range=True, cmap=cmap, size=(10,10))
current_image.save('/work3/msaca/current_figures/current_sinogram.png')