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

def load_eds():
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
    images_d = images[:,4000::2,::2]
    for i in range(len(images[:,0,0])):
        images_d[i] = np.fliplr(images_d[i])
        
    n_chan, ny, nx = np.shape(images_d)
    factor = np.linspace(30,10,nx)
    images_d[0] = images_d[0]*factor[np.newaxis,:]
    factor = np.linspace(1.5,1,nx)
    images_d[1:] = images_d[1:]*factor[np.newaxis, np.newaxis,:]
    return images_d

def run_diffusion():
    images_d = load_eds()
    nc, ny, nx = np.shape(images_d)

    # Create a multiprocessing pool
    input_list = [images_d[i] for i in range(nc)]
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
    watershed_filter.SetMarkWatershedLine(False)#watershed_line)  # Prevent marking edges as lines
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

    np.save('/dtu-compute/msaca/cache/eds_watershedded.npy', watershedded)


def get_segmentation_from_watersheds(thresholds = np.array([15, 15, 15, 10 , 20, 10, 30, 12, 5 , 30, 10, 13 , 92, 10, 5]),
    output=True):
    data_watershedded = np.load('/dtu-compute/msaca/cache/eds_watershedded.npy')    
    elements = ['Al', 'Ca', 'Cl', 'Cr', 'Fe',  'K', 'Mg', 'Na', 'Ni',  'O',  'P',  'S', 'Si', 'Ti', 'Zr']
    nc, ny, nx = np.shape(data_watershedded)
    for i in range(nc):
        f_segm = data_watershedded>thresholds[:,np.newaxis, np.newaxis]

    elem_segm = {}
    for i in range(nc):
        elem_segm[elements[i]] = f_segm[i].astype(bool)

    elem_segm['Mg_high'] = data_watershedded[6]>60
    full_seg = np.zeros((ny,nx))
    mask_feldspar_an_al = (elem_segm["Al"]*elem_segm["Na"]*np.logical_not(elem_segm['Mg'])).astype(bool)
    mask_feldspar_al_or = (np.logical_not(elem_segm['Mg'])*elem_segm['K']).astype(bool)
    mask_pyroxene_cpx = elem_segm["Ca"].astype(bool)
    mask_pyroxene_opx = elem_segm['Mg_high'].astype(bool)
    mask_apatite = (elem_segm['P']*elem_segm['Cl']*elem_segm['Ca']).astype(bool)
    mask_chromite = elem_segm['Cr'].astype(bool)
    mask_ilminite = elem_segm['Ti'].astype(bool)
    mask_pyrite = elem_segm['S'].astype(bool)
    mask_baddelyite = elem_segm['Zr'].astype(bool)
    full_seg = np.zeros((ny,nx),dtype=np.uint8)
    full_seg[mask_feldspar_an_al] = 1
    full_seg[mask_feldspar_al_or] =2
    full_seg[mask_pyroxene_cpx] =3
    full_seg[mask_pyroxene_opx] =4
    full_seg[mask_apatite] = 5
    full_seg[mask_ilminite] = 6
    full_seg[mask_chromite] = 7
    full_seg[mask_pyrite] = 8
    full_seg[mask_baddelyite] = 9
    segmentation = {}
    segmentation['labels'] = ['feldspar_an_al', 'feldspar_al_or', 'pyroxene_cpx',
                               'pyroxene_opx' , 'apatite' , 'ilminite',
                               'chromite', 'pyrite', 'baddelyite']
    segmentation['feldspar_an_al'] = mask_feldspar_an_al
    segmentation['feldspar_al_or'] = mask_feldspar_al_or
    segmentation['pyroxene_cpx'] = mask_pyroxene_cpx
    segmentation['pyroxene_opx'] = mask_pyroxene_opx
    segmentation['apatite'] = mask_apatite
    segmentation['ilminite'] = mask_ilminite
    segmentation['chromite'] = mask_chromite
    segmentation['pyrite'] = mask_pyrite
    segmentation['baddelyite'] = mask_baddelyite
    if output:
        return full_seg, segmentation, elem_segm


def transform_eds(eds):
    def resampler(fixed, moving, transform):
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)  # Reference image (fixed)
        resampler.SetInterpolator(sitk.sitkLinear)   # Interpolation method
        resampler.SetTransform(transform)    # Apply the initial transform (aligned centroids)
        resampler.SetOutputPixelType(fixed.GetPixelID())
        resampler.SetOutputSpacing(fixed.GetSpacing())  # Ensure the spacing is preserved
        resampler.SetOutputOrigin(fixed.GetOrigin())  # Preserve origin
        resampler.SetOutputDirection(fixed.GetDirection())
        moving = resampler.Execute(moving)
        return moving
    nz, nx = 1788, 1600
    _, nx_eds = np.shape(eds)
    factor = 3
    XA_surf = np.zeros((nz,nx))
    fixed = sitk.GetImageFromArray(XA_surf)
    moving = sitk.GetImageFromArray(eds)

    fixed.SetSpacing((1/nx,1/nx))
    moving.SetSpacing((1/nx_eds,1/nx_eds))

    size_fixed = fixed.GetSize()
    size_moving = moving.GetSize()

    spacing_fixed = fixed.GetSpacing()
    spacing_moving = moving.GetSpacing()

    # Compute the new origin (shift it to -N/2)
    new_origin_fixed = [-0.5 * (size_fixed[i] - 1) * spacing_fixed[i] for i in range(len(size_fixed))]
    new_origin_moving = [-0.5 * (size_moving[i] - 1) * spacing_moving[i] for i in range(len(size_moving))]


    fixed.SetOrigin(new_origin_fixed)
    moving.SetOrigin(new_origin_moving)

    transform = sitk.Transform(2, sitk.sitkIdentity)
    # Get the original size and spacing of the image
    size = moving.GetSize()
    spacing = moving.GetSpacing()

    # Calculate the new size (downsampling by factor)
    new_size = [int(size[0] / factor), int(size[1] / factor)]
    # Calculate the new spacing (enlarging the spacing to match the downsampled size)
    new_spacing = [s * factor for s in spacing]

    # Perform the resampling (using average interpolation for downsampling)
    moving_d = sitk.Resample(moving,
                                    new_size,
                                    transform,
                                    sitk.sitkLinear,  # BSpline interpolation is good for downsampling
                                    moving.GetOrigin(),
                                    new_spacing,
                                    moving.GetDirection(),
                                    0)  # 0 is the background value for the resampling

    path = 'Transformations/transformation_EDS_to_NA.tfm'
    transform = sitk.ReadTransform(path)
    moving_new = resampler(fixed, moving, transform)
    return sitk.GetArrayFromImage(moving_new)