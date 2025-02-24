import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir('/zhome/71/c/146676/main/')
import SimpleITK as sitk
from loaders import loader_XA_to_NA
import importlib
importlib.reload(loader_XA_to_NA)
import tifffile
import SimpleITK
from loaders import stitcher_XA
from helpers import module_auxiliary as ma
from PIL import Image
import imageio.v3 as iio
Image.MAX_IMAGE_PIXELS = None 
from PIL import Image
import IPython.display as display
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label, find_objects
from matplotlib_scalebar.scalebar import ScaleBar
from cil.framework import ImageGeometry, ImageData # For extract edge function
from cil.optimisation.operators import GradientOperator # For extract edge function
from multiprocessing import Pool

conductivity = 10
smoothing_iter = 200

def execute(array):
    image_ = sitk.GetImageFromArray(array.astype(np.float32))
    diffusion = sitk.GradientAnisotropicDiffusionImageFilter()
    diffusion.SetTimeStep(0.0625)
    diffusion.SetConductanceParameter(conductivity*1.0)
    diffusion.SetNumberOfIterations(int(smoothing_iter))
    image_fil_ = diffusion.Execute(image_)
    image_fil = sitk.GetArrayFromImage(image_fil_)
    return image_fil

def run_diffusion():
    path = '/dtu-compute/msaca/sliceA_eds/EDS_BB_A_raw_files/'
    pre = 'Mosaic element_'
    pre = 'eds'
    images = []
    names = [pre + 'Al.tiff', pre + 'Ca.tiff',pre + 'Cl.tiff', pre + 'Cr.tiff', pre + 'Fe.tiff',
                        pre + 'K.tiff', pre + 'Mg.tiff', pre + 'Na.tiff', pre + 'Ni.tiff',
                        pre + 'O.tiff', pre + 'P.tiff', pre + 'S.tiff', pre + 'Si.tiff',
                        pre + 'Ti.tiff', pre + 'Zr.tiff']
    for i in range(len(names)):
        images.append(tifffile.imread(path + names[i]))
    images = np.stack(images)
    images = np.stack(images)
    images[images>150]=0
    images[0,images[0]>15]=0
    images_d = images[:,::2,::2]
    n_chan, ny, nx = np.shape(images_d)
    factor = np.linspace(10,30,nx)
    images_d[0] = images_d[0]*factor[np.newaxis,:]
    factor = np.linspace(1,1.5,nx)
    images_d[1:] = images_d[1:]*factor[np.newaxis, np.newaxis,:]

    # Create a multiprocessing pool
    input_list = [images_d[i] for i in range(len(names))]
    with Pool() as pool:
        filtered = pool.map(execute, input_list)

    np.save('/dtu-compute/msaca/cache/eds_diffusioned_200.npy', np.stack(filtered).astype(np.float32))


def place_mean_in_watersheds():
    data_diffused = np.load('/dtu-compute/msaca/cache/eds_diffusioned_200.npy')
    nc, ny, nx = np.shape(data_diffused)
    def xi_vector_field(image_s,eta,weigths):
        nc, ny, nx = np.shape(image_s)
        out_grad = np.zeros((ny,nx))
        for i in range(nc):
                print('Channel', i)
                ig = ImageGeometry(voxel_num_x=nx, voxel_num_y=ny)

                image_ = ImageData(image_s[i].astype(np.float32), geometry=ig)
                G = GradientOperator(ig)
                numerator_ = G.direct(image_)
                denominator_ = np.sqrt(eta**2 + numerator_.get_item(0)**2 + numerator_.get_item(1)**2)
                xi_ = numerator_/denominator_
                dy = xi_.get_item(0).as_array()
                dx = xi_.get_item(1).as_array()
                out_grad = out_grad + (dy**2 + dx**2)*weights[i]**2

        return np.sqrt(out_grad)/(np.sum(weights))

    eta = 10
    weights = np.ones(nc)
    edges = xi_vector_field(data_diffused,eta,weights)
    edges_ = sitk.GetImageFromArray(edges)
    watershed_filter = sitk.MorphologicalWatershedImageFilter()
    watershed_filter.SetMarkWatershedLine(True)#watershed_line)  # Prevent marking edges as lines
    watershed_filter.SetFullyConnected(False)  # Use fully connected components for better segmentation
    watershed_filter.SetLevel(0.005)

    # Perform watershed segmentation using seeds
    watershed_classes_ = watershed_filter.Execute(edges_)
    watershed_classes = sitk.GetArrayFromImage(watershed_classes_)

    watershedded = np.empty((nc, ny, nx),dtype=np.float32)

    for i in range(nc):
        image = data_diffused[i]
        mask = watershed_classes > 0
        flat_B = watershed_classes[mask]
        flat_A = image[mask]

        sums = np.bincount(flat_B, weights=flat_A)
        counts = np.bincount(flat_B)
        
        means = np.zeros(np.shape(sums))
        means[counts > 0] = sums[counts > 0] / counts[counts > 0]
        
        pwc = np.zeros(np.shape(image))
        
        pwc[mask] = means[flat_B]
        watershedded[i] = pwc

    np.save('/dtu-compute/msaca/cache/eds_watershedded.npy', np.stack(filtered).astype(np.float32))