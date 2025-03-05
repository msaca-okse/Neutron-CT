import numpy as np
import matplotlib.pyplot as plt
import os
import sys
os.chdir('/zhome/71/c/146676/main/')
sys.path.append(os.getcwd())
import SimpleITK as sitk
import importlib
from helpers import module_auxiliary as ma
from matplotlib_scalebar.scalebar import ScaleBar
from cil.optimisation.operators import GradientOperator # For extract edge function
import matplotlib
matplotlib.use('Agg')
from segmentation import s3_watershed_eds
importlib.reload(s3_watershed_eds)
import matplotlib.colors as mcolors
import copy
from segmentation import s2_watershed_filter
from scipy.optimize import differential_evolution
from multiprocessing import Pool



lam = 1
center = 0.5
power = 100
narrowness= 1
element = 'Al'




NA_surf_volume_ = np.load('/dtu-compute/msaca/cache/NA_surf1.npy')
XA_surf_volume_ = np.load('/dtu-compute/msaca/cache/XA_surf1.npy')
nz, ny, nx = np.shape(XA_surf_volume_)
subvol = [0,nz,30, 40, 0, nx]

NA_surf_volume = NA_surf_volume_[subvol[0]:subvol[1],subvol[2]:subvol[3],subvol[4]:subvol[5]]
XA_surf_volume = XA_surf_volume_[subvol[0]:subvol[1],subvol[2]:subvol[3],subvol[4]:subvol[5]]
XA, NA = s2_watershed_filter.place_medians_in_watersheds(XA_surf_volume, NA_surf_volume,
    eta_edge = 0.005,level = 0.15, conductivity = 0.5, smoothing_iter = 15, watershed_line = False)


full_seg_, segmentation_, elem_segm_, elems_ = s3_watershed_eds.get_segmentation_from_watersheds(
    thresholds = np.array([13, 13, 13, 8 , 18, 8, 28, 10, 3 , 28, 8, 11 , 90, 8, 3]),
    output=True)
segmentation = {}
for key in segmentation_:
    if key != "labels":
        
        moving = sitk.GetImageFromArray(segmentation_[key].astype(np.float32))
        segmentation[key] = s3_watershed_eds.transform_eds(segmentation_[key].astype(np.float32))

elem_segm = {}
for key in elem_segm_:
    if key != "labels":
        
        moving = sitk.GetImageFromArray(elem_segm_[key].astype(np.float32))
        elem_segm[key] = s3_watershed_eds.transform_eds(elem_segm_[key].astype(np.float32))

elems = {}
for key in elems_:
    if key != "labels":
        
        moving = sitk.GetImageFromArray(elems_[key].astype(np.float32))
        elems[key] = s3_watershed_eds.transform_eds(elems_[key].astype(np.float32))


elems_n = copy.deepcopy(elems)
elements = ['Al', 'Ca', 'Cl', 'Cr', 'Fe',  'K', 'Mg', 'Na', 'Ni',  'O',  'P',  'S', 'Si', 'Ti', 'Zr']
sum_elems = np.zeros(np.shape(elems['Al']))
for element in elements:
    sum_elems = sum_elems + elems[element]

mask = sum_elems>10
for element in elements:
    elems_n[element] = elems_n[element].astype(np.float32)
    elems_n[element][mask] = elems[element][mask]/sum_elems[mask]
    elems_n[element][np.logical_not(mask)] = 0

moving = sitk.GetImageFromArray(full_seg_)
full_seg = s3_watershed_eds.transform_eds(full_seg_)


def analyze_element_segmentation(element):
    eds_threshold_arrays = {}
    eds_threshold_arrays['Al'] = np.arange(3,15)
    eds_threshold_arrays['Ca'] = np.arange(3,20)
    eds_threshold_arrays['Cl'] = np.arange(6,20)
    eds_threshold_arrays['Cr'] = np.arange(1,8)
    eds_threshold_arrays['Fe'] = np.arange(3,30)
    eds_threshold_arrays['K'] = np.arange(3,30)
    eds_threshold_arrays['Mg'] = np.arange(7,90,2)
    eds_threshold_arrays['Na'] = np.arange(3,25)
    eds_threshold_arrays['Ni'] = np.arange(3,4)
    eds_threshold_arrays['O'] = np.arange(40,65)
    eds_threshold_arrays['P'] = np.arange(11,2)
    eds_threshold_arrays['S'] = np.arange(6,10)
    eds_threshold_arrays['Si'] = np.arange(60,100)
    eds_threshold_arrays['Ti'] = np.arange(3,21)
    eds_threshold_arrays['Zr'] = np.arange(1,5)


    eds_threshold_array = eds_threshold_arrays[element]
    narrowness = 0.5
    for eds_threshold_ in eds_threshold_array:
        eds_threshold = eds_threshold_/1000
        def plot_segmentations(background, ground_truth_seg, test_seg = None, class_labels = None,transparencies = [0.7, 0.5]):
            plt.close('all)')
            Ns = 1
            if class_labels is None:
                class_labels = "".join(map(str, range(0, Ns)))
            # Add black as the first color

            colors_d = [
                (0, 0, 255),      # Blue Plagio
                (0, 255, 0),    # Purple Alk
                (255, 50, 150),   # Pink   #cpx
                (0, 255, 0),      # Green   # opx
                (255, 165, 0),    # Orange     #apatite
                (160, 255, 0),    # Brown Ilminite
                (0, 170, 255),    # light blue, chromite
                (255, 0, 0),    #  pyrite   
                (255, 0, 255),   # Magenta, Baddelyite
                (0, 255, 255) # Cyan
            ]

            # Convert RGB to normalized values and prepend a transparent color
            colors_d = [(0, 0, 0, 0.7)] + [(r/255, g/255, b/255, 0.5) for r, g, b in colors_d[:Ns+1]]

            # Create the colormap
            cmap_d = mcolors.ListedColormap(colors_d)
            boundaries = np.arange(-0.5, Ns+1, 1)  # [-0.5, 0.5, 1.5, ..., 14.5]
            norm = mcolors.BoundaryNorm(boundaries, cmap_d.N)
            # Plot the matrix with the custom colormap
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(background, cmap="gray", vmin = 0.01, vmax = 0.04)  # Adjust alpha for blending

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
            fig.tight_layout()
            if test_seg is not None:
                temp = ground_truth_seg.copy()
                temp[test_seg.astype(bool)] = Ns+2
                temp[test_seg.astype(bool)*ground_truth_seg.astype(bool)] = Ns+1
                colors_d = [(r, g, b, alpha) for r, g, b, alpha in colors_d] + [(1.0, 1.0, 0.0, 0.35)]
                cmap_d = mcolors.ListedColormap(colors_d)
                boundaries = np.arange(-0.5, Ns+3, 1)  # [-0.5, 0.5, 1.5, ..., 14.5]
                norm = mcolors.BoundaryNorm(boundaries, cmap_d.N)
                # Plot the matrix with the custom colormap
                fig2, ax2 = plt.subplots(figsize=(10, 10))
                ax2.imshow(background, cmap="gray", vmin = 0.01, vmax = 0.04)  # Adjust alpha for blending
                im2 = ax2.imshow(temp, cmap=cmap_d, norm=norm)

                # Create a colorbar next to the plot
                cbar = fig2.colorbar(im2, ax=ax2, ticks=[], fraction=0.05, pad=0.04)
                bounds = np.arange(len(colors_d) + 1)  # Define color boundaries
                # Manually add text labels next to the colorbar
                cbar_ax = cbar.ax  # Get the colorbar axis
                class_labels.append('Test_seg')
                for i, label in enumerate(class_labels):
                    y_pos = (bounds[i] + bounds[i + 1]-1) / 2  # Center text at each color
                    cbar_ax.text(1.3, y_pos, label, va='center', ha='left', fontsize=12)

                cbar_ax.set_frame_on(False)  # Remove border
                # Overlay the grayscale image using transparency
                #ax.axis('off')
                # Add scale bar (adjust values based on real-world scale)
                scalebar = ScaleBar(6, "µm", location="lower right", color="white", scale_loc="bottom", box_alpha=0.5)
                ax2.add_artist(scalebar)
                fig2.tight_layout()
                plt.close('all')
                return fig, fig2
            else:
                return fig



        def create_segmentation(i1, i2, t1, t2, b1, b2,mask):
            """Create segmentation mask with thresholds and inequality directions"""
            segm1 = i1 < t1 if b1 else i1 > t1
            segm2 = i2 < t2 if b2 else i2 > t2
            segm = np.logical_and(segm1,segm2)
            segm[np.logical_not(mask)] = False
            return segm


        def create_segmentation_x(i_x, t_x, b_x, mask):
            """Create segmentation mask for i_x"""
            segm_x = i_x < t_x if b_x else i_x > t_x
            segm_x[np.logical_not(mask)] = False
            return segm_x

        def class_balance_penalty(segm, mask, lam=1,center = 0.5, power = 6, narrowness = 1):
            """Penalizes unbalanced segmentation inside the mask."""
            segm_in_mask = segm[mask]  # Only consider pixels inside the stone
            p1 = np.mean(segm_in_mask)  # Fraction of pixels classified as True
            penalty =  lam * (narrowness*2*np.abs((p1 - center)))**power
            return penalty

        def dice_loss(segm, segm_x):
            intersection = np.sum(segm * segm_x)
            return 1 - (2 * intersection + 1e-6) / (np.sum(segm) + np.sum(segm_x) + 1e-6)


        def symmetric_dice_loss(segm1, segm2):
            """Computes a symmetric version of Dice loss."""
            intersection = np.sum(segm1 * segm2)
            intersection_complement = np.sum((1 - segm1) * (1 - segm2))
            total_pixels = segm1.size  # Total number of pixels in the image

            return 1 - (intersection + intersection_complement) / (2 * total_pixels)



        i1 = XA[:,7]
        i2 = NA[:,7]
        i_x = elems_n[element]

        i1 = i1[200:-500,200:-300]
        i2 = i2[200:-500,200:-300]
        i_x = i_x[200:-500,200:-300]
        i_x = i_x/1000

        mask = np.logical_or(i1>0.005, i2>0.0)

        def loss_function(params):
            t1, t2, b1, b2 = params
            segm = create_segmentation(i1, i2, t1, t2, round(b1), round(b2), mask)
            segm_x = create_segmentation_x(i_x, eds_threshold, 0, mask)
            #r1 = class_balance_penalty(segm, mask, lam=1,center = center, power = 100, narrowness = narrowness)
            #r2 = class_balance_penalty(segm_x, mask, lam=1,center = center, power = 100, narrowness = narrowness)
            return symmetric_dice_loss(segm, segm_x)# + r1 + r2

        bounds = [(0, 0.06),  # Bounds for t1
                (0, 0.06),  # Bounds for t2
                (0, 1), (0, 1)]  # Binary variables (will be rounded)

        result = differential_evolution(
            loss_function, 
            bounds, 
            strategy='rand1bin',  # Try 'rand1bin', 'best1bin', or 'best1exp'
            popsize=15,            # Increase population size (default is 15)
            mutation=(0.8, 1.2),    # Tune mutation range
            recombination=0.7,      # Adjust recombination
            maxiter=60,            # Increase max iterations
            disp=False,               # Show progress
            tol = 1e-4
        )
        t1_opt, t2_opt, b1_opt, b2_opt= result.x

        segm = create_segmentation(i1, i2, t1_opt, t2_opt, round(b1_opt), round(b2_opt), mask)
        segm_x = create_segmentation_x(i_x, eds_threshold, 0, mask)

        background = XA[:,7,:]
        ground_truth_seg = full_seg
        test_seg = segm.astype(np.uint8)
        ground_truth_seg = segm_x.astype(np.uint8)
        class_labels=['Background', element + ' (EDS)', 'Both']#, 'Feldspar al-or', 'Pyroxene (Cpx)', 'Pyroxene (Opx)', 'Apatite', 'Ilminite', 'Chromite', 'Pyrite', 'Baddelyite']
        fig, fig2= plot_segmentations(background[200:-500,200:-300], ground_truth_seg, test_seg = test_seg, class_labels = class_labels)


        val_opt = loss_function((t1_opt, t2_opt, b1_opt, b2_opt))
        segm_in_mask = segm[mask]  # Only consider pixels inside the stone
        p = np.mean(segm_in_mask)
        segm_in_mask = segm_x[mask]  # Only consider pixels inside the stone
        p_x = np.mean(segm_in_mask)


        p1 = np.linspace(0,1,1001)
        p1_mask = lam * (narrowness*2*np.abs((p1 - center)))**power<0.5
        interval = [round(np.min(p1[p1_mask]),3), round(np.max(p1[p1_mask]),3)]

        b1_opt_str = '<' if round(b1_opt) else '>'
        b2_opt_str = '<' if round(b2_opt) else '>'
        threshold_str = f"{int(eds_threshold_):02d}"
        text_str = "Comparison of EDS segmentation with XA_NA segmentation. Optimizing over thresholds values: \n XA th: "\
                + b1_opt_str +" " + str(round(t1_opt,4)) + " NA th: " + b2_opt_str + " " + str(round(t2_opt,4)) +\
                "EDS threshold fixed at > " + threshold_str  +\
                " \nOptimization using symmetric dice loss. Optimal value: " + str(round(val_opt,4))
                
        ax = fig2.gca()
        # Position the text inside the plot
        ax.text(0.5, -0.1, text_str, transform=ax.transAxes, ha='center', va='center',
                fontsize=10, bbox=dict(facecolor='gray', alpha=0.5, edgecolor='red'))
        index = element_array.index(element)
        index_str = f"{index:02d}"
        fig2.savefig('/dtu-compute/msaca/cache/plots2/output_' + index_str+element + '_' + threshold_str + '.png', dpi = 200)

element_array = ['Al', 'Ca', 'Cl', 'Cr', 'Fe',  'K', 'Mg', 'Na', 'Ni',  'O',  'P',  'S', 'Si', 'Ti', 'Zr']
with Pool() as pool:
    pool.map(analyze_element_segmentation, element_array)