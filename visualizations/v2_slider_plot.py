import os
import sys
os.chdir('/zhome/71/c/146676/main/')
# Add the main directory to the Python path
sys.path.append(os.getcwd())
import SimpleITK as sitk
import importlib
print("Current Working Directory:", os.getcwd())
#from helpers import module_auxiliary as ma
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib
from segmentation import s3_watershed_eds
importlib.reload(s3_watershed_eds)
import matplotlib.colors as mcolors
import copy
from segmentation import s2_watershed_filter
from matplotlib.widgets import Slider, Button
import numpy as np
import matplotlib.pyplot as plt

load = False

if not load:
    NA_surf_volume_ = np.load('/dtu-compute/msaca/cache/NA_surf1.npy')
    XA_surf_volume_ = np.load('/dtu-compute/msaca/cache/XA_surf1.npy')
    nz, ny, nx = np.shape(XA_surf_volume_)
    subvol = [0,nz,30, 40, 0, nx]
    #subvol = [200,1400,30, 40, 200, 1400]
    #subvol = [200,500,30, 40, 200, 500]

    NA_surf_volume = NA_surf_volume_[subvol[0]:subvol[1],subvol[2]:subvol[3],subvol[4]:subvol[5]]
    XA_surf_volume = XA_surf_volume_[subvol[0]:subvol[1],subvol[2]:subvol[3],subvol[4]:subvol[5]]
    XA, NA = s2_watershed_filter.place_medians_in_watersheds(XA_surf_volume, NA_surf_volume,
        eta_edge = 0.005,level = 0.15, conductivity = 0.5, smoothing_iter = 15, watershed_line = False)

    full_seg_, segmentation_, elem_segm_ = s3_watershed_eds.get_segmentation_from_watersheds(
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


    moving = sitk.GetImageFromArray(full_seg_)
    full_seg = s3_watershed_eds.transform_eds(full_seg_)[subvol[0]:subvol[1],subvol[4]:subvol[5]]


    background = XA[:,7,:]
    ground_truth_seg = full_seg
    ground_truth_seg = elem_segm['Al']*1
    #class_labels=['Background', 'Feldspar an-al', 'Feldspar al-or', 'Pyroxene (Cpx)',
    #            'Pyroxene (Opx)', 'Apatite', 'Ilminite', 'Chromite', 'Pyrite', 'Baddelyite']
    class_labels = None
    transparencies = [0.7, 0.5]


    Ns = np.max(ground_truth_seg)
    if class_labels is None:
        class_labels = "".join(map(str, range(0, Ns)))
    # Add black as the first color
    colors_d = [
        (0, 0, 255),      # Blue Plagio
        (128, 0, 128),    # Purple Alk
        (255, 50, 150),   # Pink   #cpx
        (0, 255, 0),      # Green   # opx
        (255, 165, 0),    # Orange     #apatite
        (160, 255, 0),    # Brown Ilminite
        (0, 170, 255),     # light blue, chromite
        (255, 0, 0),    #  pyrite   
        (255, 0, 255),   # Magenta, Baddelyite
        (0, 255, 255) # Cyan
    ]
    #colors = [(0, 0, 0)] + colors
    # Create a ListedColormap
    colors_d = [(0, 0, 0)] + [(r/255, g/255, b/255) for r, g, b in colors_d[:Ns]]
    cmap_d = mcolors.ListedColormap(colors_d)
    boundaries = np.arange(-0.5, Ns+1, 1)  # [-0.5, 0.5, 1.5, ..., 14.5]
    norm = mcolors.BoundaryNorm(boundaries, cmap_d.N)
    # Plot the matrix with the custom colormap
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(left=0.1, bottom=0.25)
    im = ax.imshow(ground_truth_seg, cmap=cmap_d, norm=norm,  alpha=transparencies[0])

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
    ax.imshow(background, cmap="gray", alpha=transparencies[1], vmin = -0.01, vmax = 0.07)  # Adjust alpha for blending

    #ax.axis('off')
    # Add scale bar (adjust values based on real-world scale)
    scalebar = ScaleBar(6, "µm", location="lower right", color="white", scale_loc="bottom", box_alpha=0.5)
    ax.add_artist(scalebar)
    fig.tight_layout()


# Initial threshold values
threshold_XA = 0.03
threshold_NA = 0.03

# Initial inequalities
inequality_XA = '>'
inequality_NA = '>'


def apply_segmentation(XA, NA, t_XA, t_NA, inequality_XA='>', inequality_NA='>'):
    m1 = XA < t_XA if inequality_XA == '<' else XA > t_XA
    m2 = NA < t_NA if inequality_NA == '<' else NA > t_NA
    return (m1 * m2).astype(np.float32)


# Initial segmentation plot
segmented_image = apply_segmentation(XA[:,7,:], NA[:,7,:], threshold_XA, threshold_NA, inequality_XA, inequality_NA)

from matplotlib.colors import LinearSegmentedColormap

# Create a custom colormap: [transparent, reddish]
colors = [(1, 1, 1, 0), (1, 0, 0, 0.5)]  # RGBA
cmap_segmentation = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
segmented_display = ax.imshow(segmented_image, cmap=cmap_segmentation, alpha=0.8, vmin=0, vmax=1)  # Show seg
# Add sliders for threshold adjustment
ax_XA = plt.axes([0.1, 0.1, 0.03, 0.65])  # (x, y, width, height)
ax_NA = plt.axes([0.15, 0.1, 0.03, 0.65])

slider_XA = Slider(ax_XA, "Threshold XA", 0.0, 0.05, valinit=threshold_XA, valstep=0.001, orientation='vertical')
slider_NA = Slider(ax_NA, "Threshold NA", 0.0, 0.05, valinit=threshold_NA, valstep=0.001, orientation='vertical')


def update(val):
    low = slider_XA.val
    high = slider_NA.val
    
    new_segmentation = apply_segmentation(XA[:,7,:], NA[:,7,:], low, high, inequality_XA, inequality_NA)
    segmented_display.set_data(new_segmentation)  # Update the displayed segmentation
    fig.canvas.draw_idle()  # Refresh the plot


# Connect sliders to update function
slider_XA.on_changed(update)
slider_NA.on_changed(update)
# Show the plot

# Button to toggle segmentation visibility
ax_button = plt.axes([0.05, 0.05, 0.02, 0.02])  # (x, y, width, height)
button = Button(ax_button, 'Segm On/Off')

# Transparency state
is_visible = True

def toggle_segmentation(event):
    global is_visible
    if is_visible:
        segmented_display.set_alpha(0)  # Hide segmentation
    else:
        segmented_display.set_alpha(0.5)  # Show segmentation with 50% transparency
    is_visible = not is_visible
    fig.canvas.draw_idle()  # Refresh the plot

# Connect button to toggle function
button.on_clicked(toggle_segmentation)


# Buttons to toggle inequalities
ax_inequality_XA = plt.axes([0.115, 0.8, 0.02, 0.02])  # (x, y, width, height)
button_inequality_XA = Button(ax_inequality_XA, 'XA: >')
ax_inequality_NA = plt.axes([0.1515, 0.8, 0.02, 0.02])  # (x, y, width, height)
button_inequality_NA = Button(ax_inequality_NA, 'NA: >')

def toggle_inequality_XA(event):
    global inequality_XA
    inequality_XA = '<' if inequality_XA == '>' else '>'
    button_inequality_XA.label.set_text(f'XA: {inequality_XA}')
    update(None)  # Refresh the segmentation with new inequality

def toggle_inequality_NA(event):
    global inequality_NA
    inequality_NA = '<' if inequality_NA == '>' else '>'
    button_inequality_NA.label.set_text(f'NA: {inequality_NA}')
    update(None)  # Refresh the segmentation with new inequality

# Connect inequality buttons to toggle functions
button_inequality_XA.on_clicked(toggle_inequality_XA)
button_inequality_NA.on_clicked(toggle_inequality_NA)


plt.show()