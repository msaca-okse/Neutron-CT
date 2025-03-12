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
import matplotlib
#matplotlib.use('Agg')
import matplotlib.colors as mcolors
import copy
from segmentation import s2_watershed_filter
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import mode

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
    images_d = load_eds(h=h)
    nc, ny, nx = np.shape(images_d)

    # Create a multiprocessing pool
    input_list = [images_d[i] for i in range(nc)]
    with Pool() as pool:
        filtered = pool.map(execute, input_list)

    np.save('/dtu-compute/msaca/cache/eds_diffusioned_200.npy', np.stack(filtered).astype(np.float32))


def save_to_array():
    images_d = load_eds()

    np.save('/dtu-compute/msaca/cache/eds_excerpt.npy', images_d.astype(np.float32))


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
    output=True, raw=False):
    if raw:
        data_watershedded = np.load('/dtu-compute/msaca/cache/eds_excerpt.npy') 
    else:
        data_watershedded = np.load('/dtu-compute/msaca/cache/eds_watershedded.npy')    
    elements = ['Al', 'Ca', 'Cl', 'Cr', 'Fe',  'K', 'Mg', 'Na', 'Ni',  'O',  'P',  'S', 'Si', 'Ti', 'Zr']
    nc, ny, nx = np.shape(data_watershedded)
    for i in range(nc):
        f_segm = data_watershedded>thresholds[:,np.newaxis, np.newaxis]

    elem_segm = {}
    elems = {}
    for i in range(nc):
        elem_segm[elements[i]] = f_segm[i].astype(bool)
        elems[elements[i]] = data_watershedded[i].astype(np.float32)

    elem_segm['Mg_high'] = data_watershedded[6]>60
    full_seg = np.zeros((ny,nx))
    mask_feldspar_an_al = (elem_segm["Al"]*elem_segm["Na"]*np.logical_not(elem_segm['Mg'])).astype(bool)
    mask_feldspar_al_or = (np.logical_not(elem_segm['Mg'])*elem_segm['K']).astype(bool)
    mask_pyroxene_cpx = elem_segm["Ca"].astype(bool) # Not Al (check this)
    mask_pyroxene_opx = elem_segm['Mg_high'].astype(bool) # Mg, Si, Fe...
    mask_apatite = (elem_segm['P']*elem_segm['Cl']*elem_segm['Ca']).astype(bool)
    mask_chromite = elem_segm['Cr'].astype(bool) # * Fe
    mask_ilminite = elem_segm['Ti'].astype(bool) # * Fe
    mask_pyrite = elem_segm['S'].astype(bool)    # * Fe
    mask_baddelyite = elem_segm['Zr'].astype(bool)
    # Iron oxide: Meget jern, ikke meget andet (udover oxygen) Specifikt: Ikke silicium
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
        return full_seg, segmentation, elem_segm, elems



def transform_eds(eds,h=1):
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

    def resampler2(fixed, moving, transform, h=1):
        # Get the spacing and size for the high-resolution image
        moving_spacing = moving.GetSpacing()
        high_res_size = moving.GetSize()
        low_res_spacing = fixed.GetSpacing()
        low_res_size = fixed.GetSize()
        resampler = sitk.ResampleImageFilter()
        # Set the output size (double the size)
        new_size = [int(dim * h) for dim in fixed.GetSize()]
        # Set the output spacing (half the spacing)
        new_spacing = [spacing / h for spacing in fixed.GetSpacing()]
        # Configure the resampler
        resampler.SetReferenceImage(fixed)  # Use the high-res image as reference
        resampler.SetSize(new_size)  # Set the new size
        resampler.SetOutputSpacing(new_spacing)  # Correct way to set spacing
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(transform)  # Apply the transformation
        resampler.SetOutputPixelType(fixed.GetPixelID())
        resampler.SetOutputOrigin(fixed.GetOrigin())  # Preserve origin
        resampler.SetOutputDirection(fixed.GetDirection())
        # Perform resampling
        return resampler.Execute(moving)


    nz, nx = 1788, 1600
    _, nx_eds = np.shape(eds)
    factor = 1
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
    moving_new = resampler2(fixed, moving, transform,h=h)
    return sitk.GetArrayFromImage(moving_new).astype(np.uint8)



def run_eds_segmentation(i_p):
    h=1
    raw = True # Use watershed (false, or raw data (True))
    full_seg_, segmentation_, elem_segm_, elems_ = get_segmentation_from_watersheds(
        thresholds = np.array([15, 15, 15, 10 , 20, 20, 30, 12, 5 , 30, 10, 13 , 92, 15, 5]),
        output=True, raw=False)
    segmentation = {}
    for key in segmentation_:
        if key != "labels":
            
            moving = sitk.GetImageFromArray(segmentation_[key].astype(np.float32))
            segmentation[key] = transform_eds(segmentation_[key].astype(np.float32),h=h)

    elem_segm = {}
    for key in elem_segm_:
        if key != "labels":
            
            moving = sitk.GetImageFromArray(elem_segm_[key].astype(np.float32))
            elem_segm[key] = transform_eds(elem_segm_[key].astype(np.float32),h=h)

    elems = {}
    for key in elems_:
        if key != "labels":
            
            moving = sitk.GetImageFromArray(elems_[key].astype(np.float32))
            elems[key] = transform_eds(elems_[key].astype(np.float32),h=h)

    moving = sitk.GetImageFromArray(full_seg_)
    full_seg = transform_eds(full_seg_, h=h)


    def plot_segmentations(background, ground_truth_seg, test_seg = None, class_labels = None,transparencies = [0.7, 0.5],
        path = '/dtu-compute/msaca/cache/plot.png'):
        #plt.close('all)')
        Ns = np.max(ground_truth_seg)
        if class_labels is not None:
            Ns = 11
        if class_labels is None:
            class_labels = "".join(map(str, range(0, Ns)))
        # Add black as the first color

        color_strings = ['blue',  'purple', 'pink', 'lime', 'cyan', 'yellow', 'red',
                'magenta','brown','black', 'white']

        # Convert RGB to normalized values and prepend a transparent color
        colors_d = [mcolors.to_rgb(color) for color in color_strings]
        colors_d = [(0, 0, 0, 0.7)] + [(r, g, b, 0.5) for r, g, b in colors_d[:Ns]]

        # Create the colormap
        cmap_d = mcolors.ListedColormap(colors_d)
        boundaries = np.arange(-0.5, Ns+1, 1)  # [-0.5, 0.5, 1.5, ..., 14.5]
        norm = mcolors.BoundaryNorm(boundaries, cmap_d.N)
        # Plot the matrix with the custom colormap
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(background, cmap="gray", vmin = 0.01, vmax = 50)  # Adjust alpha for blending

        im = ax.imshow(ground_truth_seg, cmap=cmap_d, norm=norm)

        # Create a colorbar next to the plot
        cbar = fig.colorbar(im, ax=ax, ticks=[], fraction=0.05, pad=0.04)
        bounds = np.arange(len(colors_d) + 1)  # Define color boundaries
        # Manually add text labels next to the colorbar
        cbar_ax = cbar.ax  # Get the colorbar axis
        for i, label in enumerate(class_labels):
            y_pos = (bounds[i] + bounds[i + 1]-1) / 2  # Center text at each color
            cbar_ax.text(1.3, y_pos, label, va='center', ha='left', fontsize=12)
        cbar_ax.set_frame_on(False)  # Remove border



        # Overlay the grayscale image using transparency
        
        #ax.axis('off')
        # Add scale bar (adjust values based on real-world scale)
        scalebar = ScaleBar(6, "µm", location="lower right", color="white", scale_loc="bottom", box_alpha=0.5)
        ax.add_artist(scalebar)
        fig.savefig(path, dpi=200)
        plt.close('all')
    
    elems_n = copy.deepcopy(elems)
    elements = ['Al', 'Ca', 'Cl', 'Cr', 'Fe',  'K', 'Mg', 'Na', 'Ni',  'O',  'P',  'S', 'Si', 'Ti', 'Zr']
    sum_elems = np.zeros(np.shape(elems['Al']))
    for element in elements:
        sum_elems = sum_elems + elems[element]

    mask = elems['O']>10
    for element in elements:
        elems_n[element] = elems_n[element].astype(np.float32)
        elems_n[element][mask] = elems[element][mask]/sum_elems[mask]
        elems_n[element][np.logical_not(mask)] = 0

    def normalize_dict(data_dict):
        """Normalizes each 2D array in the dictionary to zero mean and unit standard deviation.
        
        Returns:
            normalized_dict: Dictionary with normalized 2D arrays.
            means: Dictionary of means for each key.
            stds: Dictionary of standard deviations for each key.
        """
        means = {key: np.mean(value) for key, value in data_dict.items()}
        stds = {key: np.std(value) for key, value in data_dict.items()}
        
        normalized_dict = {key: (value - means[key]) / (stds[key] + 1e-8)  # Avoid division by zero
                        for key, value in data_dict.items()}
        
        return normalized_dict, means, stds
    elems_nn, means, stds = normalize_dict(elems_n)


    def augment_with_region(data_dict, box=None, m_mask = None, p_samples=0.01):
        """
        Augments the flattened data by oversampling from a selected region.

        Args:
            flattened_data (ndarray): Original flattened dataset (shape: num_pixels x num_features).
            data_dict (dict): Dictionary where keys are channel names and values are 2D arrays.
            box (tuple): (x_0, x_1, y_0, y_1) defining the region to sample from.
            n_samples (int): Number of samples to draw (with replacement) from the selected region.

        Returns:
            augmented_data (ndarray): Flattened data with oversampled points appended.
        """
        n_pixels = np.sum(mask)
        n_samples = int(np.ceil(p_samples*n_pixels))
        # Extract pixels from the selected region
        if box is not None:
            x_0, x_1, y_0, y_1 = box
            selected_points = np.array([data_dict[key][x_0:x_1, y_0:y_1].flatten() for key in sorted(data_dict.keys())])
            selected_points = selected_points.T  # Shape (num_selected_pixels, num_features)
        if m_mask is not None:
            selected_points = np.array([data_dict[key][m_mask] for key in sorted(data_dict.keys())])
            selected_points = selected_points.T  # Shape (num_selected_pixels, num_features)

        # Oversample by randomly selecting points with replacement
        sampled_indices = np.random.choice(selected_points.shape[0], size=n_samples, replace=True)
        oversampled_data = selected_points[sampled_indices]

        return oversampled_data

    def flatten_data(data_dict):
        data = np.array([data_dict[key][mask] for key in sorted(data_dict.keys())])
        flattened_data = data.reshape(len(data_dict), -1).T
        return flattened_data

    m_masks = {}
    no_metal_mask = (elems['Ti']<10)*(elems['Cr']<5)*(elems['S']<8)*(elems['Zr']<4)*(elems['P']<5)*(elems['Cl']<15)
    m_masks['Feld-An-Al'] = np.logical_and(elems['Al']>20,elems['Na']>20)
    m_masks['Feld-Al-Or'] = np.logical_and(elems['Al']>20,elems['K']>20)
    m_masks['Pyrox'] = np.logical_or(np.logical_or(elems['Ca']>20,elems['Mg']>80),elems['Fe']>25)*no_metal_mask
    m_masks['Apatite'] = elems['P']>15
    m_masks['Chromite'] = elems['Cr']>10
    m_masks['Ilminite'] = elems['Ti']>25
    m_masks['Pyrite'] = elems['S']>15
    m_masks['Zircon'] = elems['Zr']>10
    m_masks['Iron_oxide'] = ((elems['Si'] < 35)*(elems['Fe']>60)*(elems['Cr'] < 15)*(elems['Ti'] < 15)*(elems['S'] < 10)).astype(bool)
    m_masks['Artifact'] = elems['Al']>50

    pre_labelled_data = {}
    data_flat = flatten_data(elems_nn)
    box = 750,850,960,1040 # Feldspar An-Al
    p_samples = {}
    p_samples['Feld-An-Al'] = 0.3
    p_samples['Feld-Al-Or'] =0.3
    p_samples['Pyrox'] = 0.3
    p_samples['Apatite'] =0.0
    p_samples['Chromite'] = 0.0
    p_samples['Ilminite'] = 0.0
    p_samples['Pyrite'] = 0.0
    p_samples['Zircon'] = 0.0
    p_samples['Iron_oxide'] = 0.0
    p_samples['Artifact'] = 0.0
    for key in m_masks.keys():
        pre_labelled_data[key] = augment_with_region(elems_nn, m_mask = m_masks[key], p_samples=p_samples[key])

    data_flat_with_augm = np.vstack([data_flat] + list(pre_labelled_data.values()))

    p_samples = {}
    p_samples['Feld-An-Al'] = 0.01
    p_samples['Feld-Al-Or'] =0.01
    p_samples['Pyrox'] = 0.01
    p_samples['Apatite'] =0.001
    p_samples['Chromite'] = 0.001
    p_samples['Ilminite'] = 0.001
    p_samples['Pyrite'] = 0.001
    p_samples['Zircon'] = 0.001
    p_samples['Iron_oxide'] = 0.001
    p_samples['Artifact'] = 0.001
    for key in m_masks.keys():
        pre_labelled_data[key] = augment_with_region(elems_nn, m_mask = m_masks[key], p_samples=p_samples[key])
        

    n_components = 15

    from sklearn.decomposition import PCA
    from sklearn.decomposition import SparsePCA
    #pca = SparsePCA(n_components=n_components,alpha=10, max_iter=50)  # Keep all 15 components
    pca = PCA(n_components=n_components)  # Keep all 15 components
    pca.fit(data_flat_with_augm)  # Shape (1000000, 15)
    orig_transform = pca.transform(data_flat)

    aug_transform = {}
    for key in pre_labelled_data.keys():
        aug_transform[key] = pca.transform(pre_labelled_data[key])

    elements = ['Al', 'Ca', 'Cl', 'Cr', 'Fe',  'K', 'Mg', 'Na', 'Ni',  'O',  'P',  'S', 'Si', 'Ti', 'Zr']
    principal_components = pca.components_ 

    p_array = [0,0.25,0.5,0.75,0.9,1,1.5,4]
    p = p_array[i_p]
    def normalize_rows(arr, p):
        # Compute L2 norm for each row
        norms = np.linalg.norm(arr, axis=1, keepdims=True)  # Shape: (N, 1)
        
        # Raise norms to the power of p
        norms_power_p = norms ** p
        
        # Avoid division by zero (if any row is all zeros)
        norms_power_p[norms_power_p == 0] = 1
        
        # Normalize each row
        return arr / norms_power_p


    # Define "Unsure" class label as a unique number
    unsure_label =12  # Use -1 for unsure

    # Prepare training data
    class_labels = ['Feld-An-Al', 'Feld-Al-Or', 'Pyrox',
                    'Apatite', 'Chromite', 'Ilminite', 'Pyrite',
                    'Zircon', 'Iron_oxide', 'Artifact', 'Unsure']

    # Create a mapping from class names to numeric labels
    class_to_number = {label: i+1 for i, label in enumerate(class_labels)}
    

    X_test = orig_transform[:]
    X_train = np.vstack(list(aug_transform.values()))  # Stack labeled data
    y_train = np.concatenate([np.full(len(aug_transform[key]), class_to_number[key]) for key in aug_transform])
    X_train = normalize_rows(X_train, p)
    X_test = normalize_rows(X_test, p)


    # Define and fit the KNN classifier
    k = 40  # Number of neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Predict the k-nearest neighbors for each test point
    print('Doing KNN')
    _, indices = knn.kneighbors(X_test)
    neighbor_labels = y_train[indices]  # Shape (num_test_samples, k)

    # Compute the most frequent label (majority vote)
    majority_labels, counts = mode(neighbor_labels, axis=1, keepdims=True)
    majority_labels = majority_labels.flatten()  # Convert to 1D array

    # Count how many neighbors differ from the majority label
    diff_counts = np.sum(neighbor_labels != majority_labels[:, None], axis=1)

    # Assign labels: "Unsure" if 4 or more neighbors are different
    y_pred_numeric = np.where(diff_counts >= 15, unsure_label, majority_labels)

    from collections import Counter
    print("Predicted Class Distribution:", Counter(y_pred_numeric))




    import matplotlib.patches as mpatches
    N = 5
    # Reverse the mapping: {1: 'Feld-An-Al', 2: 'Feld-Al-Or', ..., 11: 'Artifact'}
    number_to_class = {v: k for k, v in class_to_number.items()}

    # Convert numeric predictions to string labels
    y_pred = np.array([number_to_class[num] if num in number_to_class else "Unsure" for num in y_pred_numeric])

    # Create a 5x5 grid of plots
    print('Generating plot')
    fig, axes = plt.subplots(N, N, figsize=(4*N, 4*N))

    # Set figure background to light gray
    fig.patch.set_facecolor("lightgray")

    M = 100000
    indices = np.random.choice(len(X_test), M, replace=True)  # Random sample with replacement
    X_test_sampled = X_test[indices]
    y_pred_sampled = y_pred[indices]
    label_to_name = {
        1: 'Feld-An-Al', 2: 'Feld-Al-Or', 3: 'Pyrox',
        4: 'Apatite', 5: 'Chromite', 6: 'Ilminite', 7: 'Pyrite',
        8: 'Zircon', 9: 'Iron_oxide', 10: 'Artifact', 11: 'Unsure'
    }

    # Define the custom colors
    color_dict = {
        'Feld-An-Al': 'blue', 'Feld-Al-Or': 'purple', 'Pyrox': 'pink',
        'Apatite': 'lime', 'Chromite': 'cyan', 'Ilminite': 'yellow', 'Pyrite': 'red',
        'Zircon': 'magenta', 'Iron_oxide': 'brown', 'Artifact': 'black', 'Unsure': 'white'
    }

    # Create a numeric-label-to-color mapping
    pred_colors_sampled = [color_dict[label] for label in y_pred_sampled]
    label_to_color = {num: color_dict[name] for num, name in label_to_name.items()}


    for k in range(1, N+1):  # Start from 1 to only show lower triangle of 6 PCs
        for j in range(k):  # j < k ensures only lower triangle
            ax = axes[k - 1, j]  # Shift index since grid is 5x5

            # Scatter original transformed data
            ax.scatter(X_test[:, k], X_test[:, j], color='grey', s=1, alpha=0.5)
            
            # Scatter each labeled dataset

            #for key, data in aug_transform.items():
            #    ax.scatter(data[:, k], data[:, j], c=color_dict[key], label=key, alpha=0.7, s=1, marker='x')




            for label in np.unique(y_train):  # Iterate through unique labels
                mask_ = y_train == label  # Get indices where y_train equals the label
                ax.scatter(X_train[mask_, k], X_train[mask_, j], 
                    c=label_to_color[label], label=label_to_name[label], 
                        alpha=0.7, s=1, marker='x')

            ax.scatter(X_test_sampled[:, k], X_test_sampled[:, j], c=pred_colors_sampled, s=2, alpha=0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'PC{k} vs PC{j}', fontsize=14)

    # Fill upper triangle and diagonal with gray
    for k in range(N):
        for j in range(k+1, N):
            axes[k, j].set_facecolor('gray')
            axes[k, j].set_xticks([])
            axes[k, j].set_yticks([])

    plt.tight_layout()
    # Create a separate legend for the color labels
    legend_patches = [mpatches.Patch(color=color, label=label) for label, color in color_dict.items()]
    fig.legend(handles=legend_patches, loc='upper right', title="Class Labels", fontsize=25, title_fontsize=40)

    plt.savefig('/dtu-compute/msaca/cache/plot_wts_h_' + str(h) +'_n_'+str(i_p)+ '_p='+str(p) + '_.png',dpi=250)
    plt.close('all')


    segmentation_full = np.zeros_like(mask).astype(np.int8)
    segmentation_full[mask] = y_pred_numeric


    box = 0,int(h*1474),0,int(h*1595)

    y0, y1, x0, x1 = box
    background = elems['Mg'][y0:y1,x0:x1]*1.0+0.0
    ground_truth_seg = segmentation_full[y0:y1,x0:x1]
    class_labels=['Background', 'Feld-An-Al', 'Feld-Al-Or', 'Pyrox',
                'Apatite', 'Chromite', 'Ilminite', 'Pyrite',
                'Zircon', 'Iron_oxide', 'Artifact','Unsure']
    plot_segmentations(background, ground_truth_seg, class_labels = class_labels,transparencies = [0, 0],
    path = '/dtu-compute/msaca/cache/plot_wts_im_h_' + str(h) +'_n_' +str(i_p)+ '_p='+str(p)+'_.png')




def parallel_run_eds_segmentation():
    p_array = [0,0.25,0.5,0.75,0.9,1,1.5,4]
    with Pool() as pool:
        filtered = pool.map(run_eds_segmentation, range(len(p_array)))