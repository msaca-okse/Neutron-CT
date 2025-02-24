import numpy as np
from cil.optimisation.operators import GradientOperator
from cil.framework import ImageGeometry, ImageData
import SimpleITK as sitk
from time import time
import nibabel as nib



#### Settings
subvol = [0,1600,30, 40, 0, 1600]

T1 = time()
NA_surf_volume_ = np.load('/dtu-compute/msaca/cache/NA_surf1.npy')
XA_surf_volume_ = np.load('/dtu-compute/msaca/cache/XA_surf1.npy')
T2 = time()
print('Loading time', T2-T1)

NA_surf_volume = NA_surf_volume_[subvol[0]:subvol[1],subvol[2]:subvol[3],subvol[4]:subvol[5]]
XA_surf_volume = XA_surf_volume_[subvol[0]:subvol[1],subvol[2]:subvol[3],subvol[4]:subvol[5]]

def xi_vector_field(image_XA_, image_NA_,eta):
        nz, ny, nx = np.shape(image_XA_)
        ig = ImageGeometry(voxel_num_x=nx, voxel_num_y=ny, voxel_num_z=nz)
        image_XA = ImageData(image_XA_.astype(np.float32), geometry=ig)
        image_NA = ImageData(image_NA_.astype(np.float32), geometry=ig)
        G = GradientOperator(ig)
        numerator_XA = G.direct(image_XA)
        numerator_NA = G.direct(image_NA)
        denominator_XA = np.sqrt(eta**2 + numerator_XA.get_item(0)**2 + numerator_XA.get_item(1)**2 + numerator_XA.get_item(2)**2)
        denominator_NA = np.sqrt(eta**2 + numerator_NA.get_item(0)**2 + numerator_NA.get_item(1)**2 + numerator_NA.get_item(2)**2)
        xi_XA = numerator_XA/denominator_XA
        xi_NA = numerator_NA/denominator_NA
        return np.sqrt(xi_XA.get_item(0).as_array()**2 + xi_XA.get_item(1).as_array()**2
             + xi_XA.get_item(2).as_array()**2 + xi_NA.get_item(0).as_array()**2 
             + xi_NA.get_item(1).as_array()**2 + xi_NA.get_item(2).as_array()**2)

def place_medians_in_watersheds(image_XA, image_NA,eta_edge = 0.005,level = 0.15, conductivity = 0.5, smoothing_iter = 15, watershed_line = False, verbose=False):
    image_XA_ = sitk.GetImageFromArray(image_XA)
    image_NA_ = sitk.GetImageFromArray(image_NA)

    T1 = time()
    diffusion = sitk.GradientAnisotropicDiffusionImageFilter()
    diffusion.SetTimeStep(0.0625)
    diffusion.SetConductanceParameter(conductivity*1.0)
    diffusion.SetNumberOfIterations(smoothing_iter)
    image_fil_XA_ = diffusion.Execute(image_XA_)
    image_fil_NA_ = diffusion.Execute(image_NA_)
    T2 = time()
    print('Diffusion time: ', T2-T1)


    edges = xi_vector_field(sitk.GetArrayFromImage(image_fil_XA_), sitk.GetArrayFromImage(image_fil_XA_) ,eta_edge)
    T3 = time()
    print('Gradient time: ', T3-T2)
    edges_ = sitk.GetImageFromArray(edges)
    watershed_filter = sitk.MorphologicalWatershedImageFilter()
    watershed_filter.SetMarkWatershedLine(True)#watershed_line)  # Prevent marking edges as lines
    watershed_filter.SetFullyConnected(False)  # Use fully connected components for better segmentation
    watershed_filter.SetLevel(level)

    # Perform watershed segmentation using seeds
    watershed_classes_ = watershed_filter.Execute(edges_)
    watershed_classes = sitk.GetArrayFromImage(watershed_classes_)
    T4 = time()
    print('Watershed time: ', T4-T3)
    #pwc = np.zeros(np.shape(watershed_classes))
    #for i in range(1,np.max(watershed_classes)):
    #    pwc[watershed_classes==i] = np.mean(image_np[watershed_classes==i])


    mask = watershed_classes > 0
    flat_B = watershed_classes[mask]
    flat_A_XA = image_XA[mask]
    flat_A_NA = image_NA[mask]

    sums_XA = np.bincount(flat_B, weights=flat_A_XA)
    sums_NA = np.bincount(flat_B, weights=flat_A_NA)
    counts = np.bincount(flat_B)
    
    means_XA = np.zeros(np.shape(sums_XA))
    means_NA = np.zeros(np.shape(sums_NA))
    means_XA[counts > 0] = sums_XA[counts > 0] / counts[counts > 0]
    means_NA[counts > 0] = sums_NA[counts > 0] / counts[counts > 0]
    
    pwc_XA = np.ones(np.shape(image_XA))
    pwc_NA = np.ones(np.shape(image_NA))
    
    pwc_XA[mask] = means_XA[flat_B]
    pwc_NA[mask] = means_NA[flat_B]

    T5 = time()
    print('Assignment time: ', T5-T4)
    print('Number of watersheds', np.max(watershed_classes))
    return pwc_XA, pwc_NA


XA_pwc, NA_pwc = place_medians_in_watersheds(XA_surf_volume, NA_surf_volume, level = 0.3)



neutron = [[-0.01, 0.05],
            [-0.01,0.02], 
            [0.03, 0.1], 
            [0.02, 0.03], 
            [-0.01, 0.02],
            [0.03, 0.1],
            [-0.01,0.02], 
            [0.03, 0.1],
            [0.02,0.03],
            [0.9,1.1]]
xray = [[-0.01,0.0], 
        [0, 0.025],
        [0,0.025],
        [0,0.025],
        [0.025, 0.035].
        [0.025, 0.035],
        [0.035, 0.1],
        [0.035, 0.1],
        [0.035, 0.1],
        [0.9,1.1]]

nz, ny, nx = np.shape(XA_pwc)
segm = np.empty((len(neutron), nz, ny, nx), dtype=bool)
for i in range(len(neutron)):
    segm[i] = np.logical_and(NA_pwc>neutron[i][0], NA_pwc<neutron[i][1])*np.logical_and(XA_pwc>xray[i][0],XA_pwc<xray[i][1])

segm_classes = np.empty((nz, ny, nx))
for i in range(len(neutron)):
    segm_classes[segm[i]] = i+1

T1 = time()
export_data = np.zeros(np.shape(NA_surf_volume_))
export_data[subvol[0]:subvol[1],subvol[2]:subvol[3],subvol[4]:subvol[5]] = segm_classes
affine = np.eye(4)
nii_img = nib.Nifti1Image(export_data, affine)

# Save the image to a file
nib.save(nii_img, "/dtu-compute/msaca/cache/s1_watershed_threshold.nii") 
nii_img = nib.Nifti1Image(XA_surf_volume_, affine)

# Save the image to a file
nib.save(nii_img, "/dtu-compute/msaca/cache/s1_reference.nii") 
T2 = time()
print('Data writing time', T2-T1)