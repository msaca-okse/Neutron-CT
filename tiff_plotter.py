import tifffile
import numpy as np
import matplotlib.pyplot as plt
import module_auxiliary as ma
from multiprocessing import Pool


path_recon_tv = '/dtu-compute/msaca/output/tv_recon/slice_tv_####.tiff'
path_recon_fbp = '/dtu-compute/msaca/output/fbp_recon/slice_fbp_####.tiff'
fig_path = '/dtu-compute/msaca/output/reconstruction_.png'
N_slices = 1858

A = np.arange(N_slices)
paths = ma.generate_paths(path_recon_fbp, A)
ny,nx = np.shape(tifffile.imread(paths[0]))
image_fbp = np.empty((N_slices, ny, nx))
"""
for i in range(N_slices):
    image_fbp[i] = tifffile.imread(paths[i])


image_tv = np.empty((N_slices, ny, nx))
paths = ma.generate_paths(path_recon_tv, A)
for i in range(N_slices):
    image_tv[i] = tifffile.imread(paths[i])
"""

def loader(i):
    return tifffile.imread(paths[i])

with Pool() as pool:
    image_fbp = pool.map(loader, range(N_slices))

image_fbp = np.stack(image_fbp)

paths = ma.generate_paths(path_recon_tv, A)
def loader(i):
    return tifffile.imread(paths[i])


with Pool() as pool:
    image_tv = pool.map(loader, range(N_slices))

image_tv = np.stack(image_tv)


plt.figure(figsize=(10,10))
plt.plot(image_fbp[:,1000,600])
plt.plot(image_tv[:,1000,600])
plt.title('TV and fbp recon profile (horizontal profile)')
full_path = ma.generate_unique_filename(fig_path)
plt.savefig(full_path, dpi = 500)
plt.close()


plt.figure(figsize=(10,10))
plt.plot(image_fbp[600,1000,:])
plt.plot(image_tv[600,1000,:])
plt.title('TV and fbp recon profile (vertical profile)')
full_path = ma.generate_unique_filename(fig_path)
plt.savefig(full_path, dpi = 500)
plt.close()


plt.figure(figsize=(10,10))
plt.imshow(image_fbp[1000],cmap='gray')
plt.clim([-0.1,0.1])
plt.colorbar()
plt.title('FBP-recon_slice, script: preprocessor_NA + reconstructor_NA, date 14/01/2025, horizontal slice 1000')
full_path = ma.generate_unique_filename(fig_path)
plt.savefig(full_path, dpi = 500)
plt.close()


plt.figure(figsize=(10,10))
plt.imshow(image_tv[1000],cmap='gray')
plt.clim([-0.1,0.1])
plt.colorbar()
plt.title('TV-recon_slice, alpha=100, script: preprocessor_NA + reconstructor_NA, date 14/01/2025, horizontal slice 1000')
full_path = ma.generate_unique_filename(fig_path)
plt.savefig(full_path, dpi = 500)
plt.close()


plt.figure(figsize=(10,10))
plt.imshow(image_fbp[:,1000],cmap='gray')
plt.clim([-0.1,0.1])
plt.colorbar()
plt.title('FBP-recon_slice, script: preprocessor_NA + reconstructor_NA, date 14/01/2025, horizontal slice 1000')
full_path = ma.generate_unique_filename(fig_path)
plt.savefig(full_path, dpi = 500)
plt.close()


plt.figure(figsize=(10,10))
plt.imshow(image_tv[:,1000],cmap='gray')
plt.clim([-0.1,0.1])
plt.colorbar()
plt.title('TV-recon_slice, alpha=100, script: preprocessor_NA + reconstructor_NA, date 14/01/2025, horizontal slice 1000')
full_path = ma.generate_unique_filename(fig_path)
plt.savefig(full_path, dpi = 500)
plt.close()



plt.figure(figsize=(10,10))
plt.imshow(image_fbp[:,:,1000],cmap='gray')
plt.clim([-0.1,0.1])
plt.colorbar()
plt.title('FBP-recon_slice, script: preprocessor_NA + reconstructor_NA, date 14/01/2025, horizontal slice 1000')
full_path = ma.generate_unique_filename(fig_path)
plt.savefig(full_path, dpi = 500)
plt.close()


plt.figure(figsize=(10,10))
plt.imshow(image_tv[:,:,1000],cmap='gray')
plt.clim([-0.1,0.1])
plt.colorbar()
plt.title('TV-recon_slice, alpha=100, script: preprocessor_NA + reconstructor_NA, date 14/01/2025, horizontal slice 1000')
full_path = ma.generate_unique_filename(fig_path)
plt.savefig(full_path, dpi = 500)
plt.close()
