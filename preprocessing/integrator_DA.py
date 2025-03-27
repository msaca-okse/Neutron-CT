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
from scipy.interpolate import interp1d
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


def DA_loader_gpu(batch_id,file_path, q_value=0.75, Nx=362):
    with h5py.File(file_path, 'r') as file:
        all_polar3 = []
        print(batch_id)
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

            q_max = cp.percentile(polar_copy, q=q_value*100, axis=0)
            
            polar2 = cp.clip(polar_matrix, 0, q_max)
            polar3 = cp.sum(polar2, axis=0)
            
            all_polar3.append(polar3)
        
        return cp.asnumpy(cp.stack(all_polar3))