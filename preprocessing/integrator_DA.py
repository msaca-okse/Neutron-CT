import sys
import os
import numpy as np
from multiprocessing import Pool
import h5py
import cv2
import hdf5plugin
from cil.utilities.display import show_geometry
from scipy.ndimage import uniform_filter1d
import os
import cupy as cp



def cartesian_to_polar_cupy(matrix, num_phi=360, num_rad=None,factor = 6):
    """ Convert a 2D CuPy matrix from Cartesian to Polar coordinates. """
    num_phi_old = num_phi
    num_rad_old = num_rad
    num_phi = factor*num_phi
    num_rad = factor*num_rad
    rows, cols = matrix.shape
    bias_col, bias_row = 28, 33
    center_x, center_y = cols // 2 + bias_col, rows // 2 + bias_row
    max_radius = (min(cols, rows) // 2) - 210
    
    if num_rad is None:
        num_rad = max_radius

    # Create polar coordinate grid
    theta = cp.linspace(0, 2 * cp.pi, num_phi)  # Angles
    r = cp.linspace(0, max_radius, num_rad)  # Radii
    R, Theta = cp.meshgrid(r, theta)  # Create grid

    # Convert polar to Cartesian coordinates
    X = center_x + R * cp.cos(Theta)
    Y = center_y + R * cp.sin(Theta)

    # Bilinear interpolation
    X = cp.clip(X, 0, cols - 1)
    Y = cp.clip(Y, 0, rows - 1)
    x0, y0 = X.astype(cp.int32), Y.astype(cp.int32)  # Floor values
    x1, y1 = cp.clip(x0 + 1, 0, cols - 1), cp.clip(y0 + 1, 0, rows - 1)  # Ceiling values

    # Get pixel values from the original image
    Ia = matrix[y0, x0]
    Ib = matrix[y0, x1]
    Ic = matrix[y1, x0]
    Id = matrix[y1, x1]

    # Compute bilinear interpolation weights
    wa = (x1 - X) * (y1 - Y)
    wb = (X - x0) * (y1 - Y)
    wc = (x1 - X) * (Y - y0)
    wd = (X - x0) * (Y - y0)

    # Compute interpolated values
    polar_matrix = wa * Ia + wb * Ib + wc * Ic + wd * Id
    polar_matrix = polar_matrix.reshape(num_phi_old, factor, num_rad_old, factor)

    # Take the mean over the (4,4) blocks
    polar_matrix = polar_matrix.mean(axis=(1, 3))

    return polar_matrix


def DA_loader_gpu(batch_id,file_path, Nx=362):

    print(batch_id)
    q_values = [60, 75, 80, 85, 90, 95, 98, 99, 100]
    q_powder_strings = [f"powder_q-{q}" for q in q_values]
    q_crystal_strings = [f"crystal_q-{q}" for q in q_values]
    q_dict = {f"powder_q-{q}": [] for q in q_values}
    q_dict.update({f"crystal_q-{q}": [] for q in q_values})



    r_nonunif = cp.arcsin(cp.linspace(0,0.35,667))
    r_unif = cp.linspace(0,1, 667)*0.35



    with h5py.File(file_path, 'r') as file:

        for i in range(Nx):
            dataset_ = file['entry']['instrument']['pilatus']['data'][Nx*batch_id + i]
            dataset = cp.pad(cp.asarray(dataset_, dtype=cp.float32), 600, mode='constant', constant_values=0)
            dataset[dataset < 0] = 0
            
            polar_matrix = cartesian_to_polar_cupy(dataset, num_phi=360, num_rad=667, factor = 4)
            polar_copy = polar_matrix.copy()
            col_means = cp.mean(polar_matrix[45:62], axis=0)  # Shape (num_rad,)
            zero_mask = (polar_matrix < 0.001)
            # Replace zero values with the computed column means
            col_means_broadcasted = cp.broadcast_to(col_means, polar_copy.shape)

            # Replace zero values with the computed column means
            polar_copy[zero_mask] = col_means_broadcasted[zero_mask]

            for i in range(len(q_values)):
                q = q_values[i]
                q_powder_key = q_powder_strings[i]
                q_crystal_key = q_crystal_strings[i]

                q_max = cp.percentile(polar_copy, q=q, axis=0)
                
                polar_powder = cp.clip(polar_matrix, 0, q_max)
                polar_crystal = polar_matrix - polar_powder

                integral_powder = cp.sum(polar_powder, axis=0)
                integral_crystal = cp.sum(polar_crystal, axis=0)

                integral_powder_corrected  = cp.interp(r_nonunif, r_unif, integral_powder).astype(cp.float32)
                integral_crystal_corrected  = cp.interp(r_nonunif, r_unif, integral_crystal).astype(cp.float32)

                q_dict[q_powder_key].append(integral_powder_corrected)
                q_dict[q_crystal_key].append(integral_crystal_corrected)
        

    for i in range(len(q_values)):
        q_powder_key = q_powder_strings[i]
        q_crystal_key = q_crystal_strings[i]
        array_powder = cp.asnumpy(cp.stack(q_dict[q_powder_key]))
        array_crystal = cp.asnumpy(cp.stack(q_dict[q_crystal_key]))

        q_dict[q_powder_key] = array_powder
        q_dict[q_crystal_key] = array_crystal

    return q_dict


if __name__ == '__main__':
    batch_ids = range(181)  # Generate batch_id values
    file_path = '/dtu-compute/msaca/sliceA_diffraction/xrd_non_integrated/scan-0339_pilatus.h5'
    output_file = 'integrated-0339.h5'
    output_file = os.path.join('/work3/msaca/sliceA_DA_cache', output_file)
    with Pool(processes=12) as pool:
        results = pool.starmap(DA_loader_gpu, [(batch_id, file_path) for batch_id in batch_ids])

    stacked_arrays = {key: [] for key in results[0].keys()}

    for result in results:
        for key in result:
            stacked_arrays[key].append(result[key])

    # Stack the arrays along axis=0
    for key in stacked_arrays:
        stacked_arrays[key] = np.stack(stacked_arrays[key], axis=0).astype(np.float32)



    with h5py.File(output_file, "w") as f_out:
        for key, array in stacked_arrays.items():
            f_out.create_dataset(key, data=array)