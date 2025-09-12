import os
import sys
os.chdir('/zhome/71/c/146676/main/')
sys.path.append('/zhome/71/c/146676/main/')
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tifffile
import numpy as np
from typing import Dict, Any
from tqdm import tqdm  # for progress bars
import matplotlib.pyplot as plt
from numba import njit
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
import h5py
import importlib
import math
from scipy.interpolate import interp1d
from numpy.linalg import lstsq
from scipy.optimize import nnls  # non-negative least squares
from sklearn.linear_model import Lasso, LassoLars
from tqdm.auto import tqdm  # ✅ Recommended for notebooks
from scipy.ndimage import gaussian_filter1d
import cvxpy as cp
from scipy.optimize import linprog
from itertools import groupby
from operator import itemgetter
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm


class GridrecH5Dataset(Dataset):
    def __init__(self, index_list, cache_size=1):
        """
        index_list: List of (filename, j, k) tuples
        threshold: Threshold for crystal-mask
        cache_size: How many files to keep in RAM (1 = keep current file only)
        """
        self.index_list = index_list
        self.cache_size = cache_size
        self.cache = {}  # filename -> loaded data

    def __len__(self):
        return len(self.index_list)

    def _load_file(self, filename):
        """Load an entire h5 file into cache."""
        with h5py.File(filename, 'r') as f:
            data = {
                'recon': f['reconstructed/gridrec'][:][130:230,:,:],  # copy into memory
            }
        return data

    def __getitem__(self, idx):
        filename, j, k = self.index_list[idx]

        # Load file into cache if needed
        if filename not in self.cache:
            if len(self.cache) >= self.cache_size:
                self.cache.clear()  # simple cache clearing for now
            self.cache[filename] = self._load_file(filename)

        data = self.cache[filename]
        recon = data['recon'][j, k, :]

        #combined = combined / np.max(combined)     # Step 2: normalize max to 1
        #combined = np.clip(combined, 0.001, None)  # Step 1: clip
        #combined = np.log(combined+1)                # Step 3: log transform
        return recon.astype(np.float32)

class ThresholdedH5Dataset(Dataset):
    def __init__(self, index_list, threshold=1.5, cache_size=1,scale_factor=100.0,
                 use_derivative=False, extra_filename=None, slice_index = None,
                 N_splits = 1, split = [1,1,1], all_files = None):
        """
        index_list: List of (filename, j, k) tuples
        threshold: Threshold for crystal-mask
        cache_size: How many files to keep in RAM (1 = keep current file only)
        """
        self.index_list = index_list
        self.threshold = threshold
        self.cache_size = cache_size
        self.cache = {}  # filename -> loaded data
        self.scale_factor = scale_factor
        self.use_derivative = use_derivative  # 🔥 new option!
        self.extra_filename = extra_filename
        self.slice_index = slice_index
        self.N_splits = N_splits
        self.split = split


        if all_files is not None:
            self.all_files = sorted(all_files)  # Ensure consistent order
            self.file_to_z = {fname: i for i, fname in enumerate(self.all_files)}
            print(self.file_to_z)
        else:
            self.file_to_z = None



        if self.extra_filename is not None:
            start, end = slice_index
            with h5py.File(self.extra_filename, 'r') as f:
                self.extra_data_XA = f['XA_x_5'][:][5*(start-1):5*(end-1)]  # Load into memory once
                self.extra_data_NA = f['NA_x_5'][:][5*(start-1):5*(end-1)]  # Load into memory once
        else:
            self.extra_data_XA = None
            self.extra_data_NA = None

    def set_split(split = [1,1,1]):
        self.split = [1,1,1]


    def __len__(self):
        return len(self.index_list)

    def _load_file(self, filename):
        """Load an entire h5 file into cache."""
        with h5py.File(filename, 'r') as f:
            data = {
                'powder-recon': f['powder-recon'][:][:,:,:],  # copy into memory
                'crystal-recon': f['crystal-recon'][:][:,:,:],
                'crystal-mask': f['crystal-mask'][:][:,:,:]
            }
        return data

    def __getitem__(self, idx):
        filename, j, k = self.index_list[idx]

        # Load file into cache if needed
        if filename not in self.cache:
            if len(self.cache) >= self.cache_size:
                self.cache.clear()  # simple cache clearing for now
            self.cache[filename] = self._load_file(filename)

        data = self.cache[filename]
        powder = data['powder-recon'][:, j, k]
        crystal = data['crystal-recon'][:, j, k]
        mask_value = data['crystal-mask'][:,j, k]

        combined = powder + crystal * (mask_value > self.threshold)
        combined = combined * self.scale_factor
        combined = np.clip(combined.astype(np.float32), 0, None)
        #combined = combined/np.sum(combined)*200

            # Append extra data if available
        if self.extra_data_XA is not None:
            # Scale coarse indices to fine
            j_fine = j * 5
            k_fine = k * 5
            if self.file_to_z is not None:
                z = self.file_to_z[filename]
                z_fine = z * 5
            else:
                raise ValueError("slice_index is not set and file_to_z mapping is missing. Input all_files into dataset def")
                
            # We loop over all z (axis=0), so get 5x5x5 region for each z
            extra_means = []
            cube_XA = self.extra_data_XA[z_fine:z_fine+5, j_fine:j_fine+5, k_fine:k_fine+5]
            cube_NA = self.extra_data_NA[z_fine:z_fine+5, j_fine:j_fine+5, k_fine:k_fine+5]
            if self.N_splits == 1:
                extra_means.append(100*np.mean(cube_XA))
                extra_means.append(100*np.mean(cube_NA))
            else:
                data_XA = resample_voxel(cube_XA, self.N_splits, split = self.split)
                data_NA = resample_voxel(cube_NA, self.N_splits, split = self.split)
                extra_means.append(100*data_XA)
                extra_means.append(100*data_NA)
            extra_means = np.array(extra_means, dtype=np.float32)
            combined = np.concatenate([combined, extra_means], axis=0)

        return combined


def load_xy_files(base_folder):
    data_dict = {}

    for root, _, files in os.walk(base_folder):
        for file in files:
            if file.endswith('.xy'):
                filepath = os.path.join(root, file)

                # Extract key: everything between first underscore and '.xy'
                try:
                    first_underscore = file.index('_') + 1
                    last_dot = file.rindex('.xy')
                    key = file[first_underscore:last_dot]
                except ValueError:
                    continue  # skip files not matching pattern

                # Load only the first two columns
                data = np.loadtxt(filepath, usecols=(0, 1), max_rows = 1740)
                data_dict[key] = data

    return data_dict


def add_convolved_columns(data_dict, stds):
    new_dict = {}
    for key, arr in data_dict.items():
        base_x = arr[:, 0]
        base_y = arr[:, 1]

        # Apply gaussian filters with each std in stds
        smoothed_columns = [gaussian_filter1d(base_y, sigma=s) for s in stds]

        # Stack: x, y, smoothed1, smoothed2, ..., smoothed_{K-2}
        new_arr = np.column_stack([base_x, base_y] + smoothed_columns)
        new_dict[key] = new_arr

    return new_dict


def resample_basis_dict(data_dict, j, target_x):
    """
    Resample column j from each entry in data_dict to match target_x.
    Returns:
        - basis_matrix: shape (len(target_x), num_keys)
        - keys: list of keys used, in column order
    """
    basis_matrix = []
    keys = sorted(data_dict.keys())

    for key in keys:
        arr = data_dict[key]
        source_x = arr[:, 0]
        source_y = arr[:, j]

        f = interp1d(source_x, source_y, kind='linear', bounds_error=False, fill_value=0.0)
        resampled = f(target_x)
        basis_matrix.append(resampled)

    return np.column_stack(basis_matrix), keys


def resample_all_blurs(data_dict, start_col=2, target_x=None):
    """
    Resample all convolved versions (columns >= start_col) from each entry in data_dict to match target_x.

    Returns:
        - basis_matrix: shape (len(target_x), num_keys * num_blur_levels)
        - keys_and_blur_indices: list of tuples (key, blur_index) indicating which key and blur each column corresponds to
    """
    basis_matrix = []
    keys_and_blur_indices = []

    keys = sorted(data_dict.keys())
    count = 0
    for key in keys:
        arr = data_dict[key]
        source_x = arr[:, 0]

        for j in range(start_col, arr.shape[1]):
            source_y = arr[:, j]
            f = interp1d(source_x, source_y, kind='linear', bounds_error=False, fill_value=0.0)
            resampled = f(target_x)
            basis_matrix.append(resampled)
            keys_and_blur_indices.append(key)  # blur_index = index in stds
            count = count+1

    return np.column_stack(basis_matrix), keys

def add_background_basis(A, x, degree=2):
    """
    Appends polynomial basis functions (degree 0 to `degree`) to the matrix A.
    A: shape (N, M)
    x: shape (N,) — domain (e.g., linspace 0 to 18.4)
    Returns: (N, M + degree + 1)
    """
    poly_terms = [x**d for d in range(degree + 1)]  # x^0, x^1, ..., x^degree
    B = np.column_stack(poly_terms)*0.000001
    return np.column_stack([A, B])


def add_fourier_background_basis(A, x, num_frequencies=3, reletive_weight = 1):
    """
    Appends sine and cosine terms to A for capturing smooth background.
    A: shape (N, M)
    x: shape (N,) — domain
    num_frequencies: how many sine/cosine frequency pairs to include
    Returns: (N, M + 2 * num_frequencies)
    """
    L = x[-1] - x[0]  # domain length
    sin_cos_terms = [
        np.sin(2 * np.pi * n * x / L) for n in range(1, num_frequencies + 1)
    ] + [
        np.cos(2 * np.pi * n * x / L) for n in range(1, num_frequencies + 1)
    ]

    B = np.column_stack(sin_cos_terms)*reletive_weight
    return np.column_stack([A, B])


def fit_nonnegative(A, y):
    """Fit y = A @ c with constraint c >= 0"""
    c, _ = nnls(A[100:,], y[100:])
    return c

def fit_lasso(A, y, alpha=0.2):
    """Fit y = A @ c with L1 regularization"""
    model = Lasso(alpha=alpha, fit_intercept=False)
    model.fit(A[100:,], y[100:])
    return model.coef_


def print_nonzero_material_coeffs(c, keys, key_to_name, num_background=6, threshold=1e-8):
    material_c = c[:len(keys)]  # exclude background
    nonzero_items = [(key, val) for key,val in zip(keys, material_c) if abs(val) > threshold]

    total = sum(val for _, val in nonzero_items)
    if total == 0:
        print("No non-zero material coefficients.")
        return

    # Compute scaled values and sort by value descending
    scaled_items = [(key, key_to_name.get(key, "Unknown"), val / total) for key, val in nonzero_items]
    scaled_items.sort(key=lambda x: x[2], reverse=True)

    # Print table
    print(f"{'Name':25s} | {'Scaled Value'}")
    print("-" * 65)
    for key, name, scaled_val in scaled_items:
        print(f"{name:25s} | {scaled_val:.4f}")

def als_baseline(y, lam=1e6, p=0.6, niter=20):
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = D.dot(D.T)  # second-order difference matrix
    w = np.ones(L)

    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)

    return z





def fit_all_pixels(dataset, A, alpha=0.1, box=None,
    background_filter_sigma = None,
    method = 'Lasso', 
    w=None,
    N_iter = 3000,
    learning_rate = 100,
    eps = 1e-8):
    """
    dataset: instance of ThresholdedH5Dataset
    A: (N_2theta, num_keys) resampled basis matrix
    box: [x0, x1, y0, y1] — restricts computation to j in [x0, x1), k in [y0, y1)
    Returns:
        coeffs: np.ndarray of shape (num_keys, width, depth)
    """
    num_keys = A.shape[1]
    num_2theta = A.shape[0]
    if w is None:
        w = np.ones(num_2theta, dtype=np.float32)  # default weights

    if box is None:
        x0, x1 = 0, 100  # default full region
        y0, y1 = 0, 362
    else:
        x0, x1, y0, y1 = box

    width = x1 - x0
    depth = y1 - y0
    coeffs = np.zeros((num_keys, width, depth), dtype=np.float32)

    # Build restricted index list
    restricted_indices = [
        (idx, j - x0, k - y0)
        for idx, (filename, j, k) in enumerate(dataset.index_list)
        if x0 <= j < x1 and y0 <= k < y1
    ]

    if method == 'Lasso':
        for idx, j_local, k_local in tqdm(restricted_indices, total=len(restricted_indices)):
            y = dataset[idx]  # shape (N_2theta,)
            if background_filter_sigma is not None:
                background = gaussian_filter1d(y, sigma=background_filter_sigma)
                foreground = y - background
                foreground[foreground < 0] = 0
                y = foreground

            model = Lasso(alpha=alpha, fit_intercept=False, max_iter=1000)
            model.fit(A, y)
            c = model.coef_
            coeffs[:, j_local, k_local] = c


    if method == 'Robust_Lasso':
        for idx, j_local, k_local in tqdm(restricted_indices, total=len(restricted_indices)):
            y = dataset[idx]  # shape (N_2theta,)
            if background_filter_sigma is not None:
                background = gaussian_filter1d(y, sigma=background_filter_sigma)
                foreground = y - background
                foreground[foreground < 0] = 0
                y = foreground

            c = robust_lasso(A, y, alpha = alpha)
            coeffs[:, j_local, k_local] = c

    if method == 'GPU_Lasso':
        Y = []
        index_map = []
        for idx, j_local, k_local in restricted_indices:
            Y.append(dataset[idx])  # shape (567,)
            index_map.append((j_local, k_local))
        Y = np.stack(Y)  # shape (batch, 567)

        W = np.sqrt(w)  # take square root of weights
        A_weighted = A * W[:, np.newaxis]  # each row of A scaled
        Y_weighted = Y * W                 # each element of B scaled
        C = batch_solve_lasso(A_weighted, Y_weighted, alpha=alpha, lr=learning_rate, max_iter=N_iter, eps=eps, nonneg=True)

        for coeffs_vec, (j_local, k_local) in zip(C, index_map):
            coeffs[:, j_local, k_local] = coeffs_vec


    if method =='GPU_Robust_Lasso':
        Y = []
        index_map = []
        for idx, j_local, k_local in restricted_indices:
            Y.append(dataset[idx])  # shape (567,)
            index_map.append((j_local, k_local))
        Y = np.stack(Y)  # shape (batch, 567)

        W = np.sqrt(w)  # take square root of weights
        A_weighted = A * W[:, np.newaxis]  # each row of A scaled
        Y_weighted = Y * W                 # each element of B scaled
        C = batch_solve_l1_l1(A_weighted, Y_weighted, lam=1e+5*alpha, lr=1e-4, max_iter=1000)
        for coeffs_vec, (j_local, k_local) in zip(C, index_map):
            coeffs[:, j_local, k_local] = coeffs_vec


    return coeffs


def plot_pixel_fit(j_global, k_global, dataset, materials, A, box,filename, background_filter_sigma=None):
    """
    Plot observed vs fitted spectrum at global pixel (j_global, k_global).

    Parameters:
        j_global, k_global: pixel in full volume
        dataset: the full ThresholdedH5Dataset
        materials: array of shape (num_keys, width_box, depth_box)
        A: basis matrix, shape (N_2theta, num_keys)
        box: [x0, x1, y0, y1]
        filename: the HDF5 filename used in index_list
    """
    x0, x1, y0, y1 = box
    twotheta = np.linspace(0, 18.4, A.shape[0])
    # Convert global to local coordinates
    j_local = j_global - x0
    k_local = k_global - y0

    # Sanity check
    if not (0 <= j_local < materials.shape[1] and 0 <= k_local < materials.shape[2]):
        raise ValueError("Pixel not in selected box.")

    # Find index for dataset lookup
    try:
        idx = dataset.index_list.index((filename, j_global, k_global))
    except ValueError:
        raise ValueError("Pixel index not found in dataset index_list.")

    # Extract data
    observed = dataset[idx]
    if background_filter_sigma is not None:
        background = gaussian_filter1d(observed, sigma=background_filter_sigma)
        foreground = observed - background
        foreground[foreground < 0] = 0
        observed = foreground


    fitted = A @ materials[:, j_local, k_local]

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(twotheta,observed, label='Observed', linewidth=4)
    plt.plot(twotheta, fitted, label='Fitted', linewidth=2)
    plt.title(f'Pixel ({j_global}, {k_global}) Spectrum vs Fit')
    plt.legend()
    plt.xlabel('2theta Channel')
    plt.ylabel('Signal')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_material_maps_only(materials, keys, key_to_name, num_cols=3):
    num_materials = len(keys)
    num_rows = math.ceil(num_materials / num_cols)

    # Compute 1% and 99% quantiles across material maps only
    vmin = np.quantile(materials[:num_materials], 0.03)
    vmax = np.quantile(materials[:num_materials], 0.99)

    plt.figure(figsize=(num_cols * 6, num_rows * 2))

    for i in range(num_materials):
        ax = plt.subplot(num_rows, num_cols, i + 1)
        ax.imshow(materials[i], cmap='grey', interpolation='none', vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])

        # Use simplified name if available
        name = key_to_name.get(keys[i], "")
        if name:
            ax.set_title(name, fontsize=10)

def normalize_axis0(arr, eps=1e-8):
    """
    Normalize arr along axis=0 so that arr[:, j, k] sums to 1.
    Adds small epsilon to avoid division by zero.
    """
    sum_over_axis0 = np.sum(arr, axis=0, keepdims=True)  # shape (1, W, D)
    return arr / (sum_over_axis0 + eps)



def simplified_keys():
    key_to_name = {
    "(CaFe)MgFe(Si2O6)_clipy": "Clinoferrosilite",
    "Ca5(PO4)3Cl_chl": "Chlorapatite",
    "CaMgSi2O6_diop": "Diopside",
    "DATA_1999_goethite": "Goethite",
    "DATA_akaganeite": "Akaganeite",
    "DATA_feroxyhyte": "Feroxyhyte",
    "DATA_lepidocrocite": "Lepidocrocite",
    "Fe2O3_hem": "Hematite",
    "Fe2Si2O6_fer2": "Ferrosilite",
    "Fe2SiO4_fay": "Fayalite",
    "Fe3O4_mag": "Magnetite",
    "FeCr2O4_chro": "Magnetite",
    "FeS2_pyr": "Pyrite",
    "FeTiO3_ilm": "Ilmenite",
    "K(AlSi3O8)_orth": "Orthoclase",
    "Maghemite": "Maghemite",
    "Maghemite_and_H": "Maghemite + Hematite",
    "Mg2Si2O6_enst": "Enstatite",
    "Mg2SiO4_fors": "Forsterite",
    "MgAl2O4_spin": "Spinel",
    "Na(AlSi3O8)_alb": "Albite",
    "Pseudorutile": "Pseudorutile",
    "TiO2_rut": "Rutile",
    "Titanomagnetite": "Titanomagnetite",
    "ZrO2_bad": "Baddeleyite",
    "ZrSiO4_zir": "Zircon",
    "hydrohematite_Fe1.78_OH0.66O2.54": "Hydrohematite",
    "titanomaghemite": "Titanomaghemite",
    "wustite": "Wüstite",

    'Fe2Si2O6_pyroxene_ferrosilite': 'Ferrosilite',
    'FeTiO3_ilmenite': 'Ilmenite',
    'ZrO2_baddeleyite': 'Baddeleyite',
    'Ca5(PO4)3Cl_chlorapatite': 'Chlorapatite',
    'Mg2SiO4_olivine_forsterite': 'Forsterite',
    'Mg2Si2O6_pyroxene_enstatite': 'Enstatite',
    'Na(AlSi3O8)_plagioclase_albite': 'Albite',
    'K(AlSi3O8)_orthoclase_feldspar': 'Orthoclase',
    'ZrSiO4_zircon': 'Zircon',
    'MgAl2O4_spinel': 'Spinel',
    'wustite': 'Wüstite',
    'FeS2_pyrite': 'Pyrite',
    'Fe2O3_hematite': 'Hematite',
    'CaMgSi2O6_clinopyroxene_diopsite': 'Diopside',
    'Titanomagnetite': 'Titanomagnetite',
    'FeCr2O4_chromite': 'Magnetite',
    'TiO2_rutile': 'Rutile'
}

    return key_to_name


def get_index_list_gridrec(data_dir):

    string = 'scan-'


    all_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.startswith(string) and f.endswith('.h5')
    ]

    # Extract the numeric ID from filenames like 'recons-DA-0459.h5' and sort

    index_list = []

    for filename in all_files:
        with h5py.File(filename, 'r') as f:
            if not {'reconstructed'}.issubset(f.keys()):
                continue  # skip files missing required keys
            shape = np.transpose(f['reconstructed/gridrec'][:,:,:], [2,0,1]).shape  # (channels, width, depth)
            channels, width, depth = shape
            print(f"Processing {filename} with shape {shape}")

            for j in range(width):  # j iterates over width
                for k in range(depth):  # k iterates over depth
                    index_list.append((filename, j, k))

    print(f"Total samples: {len(index_list)}")
    return index_list, all_files


def get_index_list(data_dir, theta_reg = False,slice_index = 0):
    if theta_reg:
        string = 'recons-DA-'
    else:
        string = 'recons2-DA-'

    all_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.startswith(string) and f.endswith('.h5')
    ]

    # Extract the numeric ID from filenames like 'recons-DA-0459.h5' and sort
    all_files.sort(key=lambda f: int(os.path.basename(f).split('-DA-')[1].split('.')[0]))
    if slice_index is not None:
        start, stop = slice_index
        all_files = all_files[start:stop]
    index_list = []

    for filename in all_files:
        with h5py.File(filename, 'r') as f:
            if not {'powder-recon', 'crystal-recon', 'crystal-mask'}.issubset(f.keys()):
                continue  # skip files missing required keys
            shape = f['powder-recon'][:,:,:].shape  # (channels, width, depth)
            channels, width, depth = shape
            print(f"Processing {filename} with shape {shape}")

            for j in range(width):  # j iterates over width
                for k in range(depth):  # k iterates over depth
                    index_list.append((filename, j, k))

    print(f"Total samples: {len(index_list)}")
    return index_list, all_files





def compute_similarity_matrix(A, num_materials, num_blur, alpha=0.01, mask_rows=100):
    """
    Computes a 4D similarity matrix using Lasso fits between theoretical diffraction spectra.
    
    Args:
        A: array of shape (n_rows, num_materials * num_blur)
        num_materials: number of materials
        num_blur: number of blur levels per material
        alpha: Lasso regularization parameter
        mask_rows: number of rows at top to exclude from fit (e.g., 100)
    
    Returns:
        similarity: array of shape (num_materials, num_blur, num_materials, num_blur)
    """
    n_rows = A.shape[0]
    similarity = np.zeros((num_materials, num_blur, num_materials, num_blur))

    for i_mat in range(num_materials):
        for i_blur in range(num_blur):
            # Index of the selected basis function
            y_idx = i_mat * num_blur + i_blur
            y = A[mask_rows:, y_idx]

            # Select column indices of all basis functions except the ones from material i_mat
            keep_indices = [
                j for j in range(num_materials * num_blur)
                if not (i_mat * num_blur <= j < (i_mat + 1) * num_blur)
            ]
            A_reduced = A[mask_rows:, keep_indices]
            # Fit Lasso
            model = LassoLars(alpha=alpha, fit_intercept=False, positive=True, max_iter=100000)
            model.fit(A_reduced, y)

            # Insert fitted coefficients back into a full (num_materials x num_blur) matrix
            full_coeffs = np.zeros((num_materials, num_blur))
            insert_idx = 0
            for j in range(num_materials):
                if j == i_mat:
                    continue
                for b in range(num_blur):
                    full_coeffs[j, b] = model.coef_[insert_idx]
                    insert_idx += 1

            similarity[i_mat, i_blur] = full_coeffs

    return similarity


def A_reduced(A, i_mat, num_materials, i_blur, num_blur, mask_rows=0):
    """
    Computes a 4D similarity matrix using Lasso fits between theoretical diffraction spectra.
    
    Args:
        A: array of shape (n_rows, num_materials * num_blur)
        num_materials: number of materials
        num_blur: number of blur levels per material
        alpha: Lasso regularization parameter
        mask_rows: number of rows at top to exclude from fit (e.g., 100)
    
    Returns:
        similarity: array of shape (num_materials, num_blur, num_materials, num_blur)
    """

    # Index of the selected basis function
    y_idx = i_mat * num_blur + i_blur
    y = A[mask_rows:, y_idx]

    # Select column indices of all basis functions except the ones from material i_mat
    keep_indices = [
        j for j in range(num_materials * num_blur)
        if not (i_mat * num_blur <= j < (i_mat + 1) * num_blur)
    ]
    A_reduced = A[mask_rows:, keep_indices]


    return A_reduced, y



def robust_lasso(A, b, alpha = 0.1):
    m, n = A.shape

    # Variables: x (n,), r (m,)
    # Total variables: n + m

    # Objective: [lambda_ * 1 for x variables] + [1 for r variables]
    c = np.concatenate([alpha * np.ones(n), np.ones(m)])

    # Inequality constraints:
    # -Ax + r >= -b  -->  [-A | I] @ [x; r] >= -b  -->  [-A | I] @ z <= -b (after multiplying by -1)
    #  Ax + r >=  b  -->  [ A | I] @ [x; r] >=  b

    G = np.vstack([
        np.hstack([-A,  np.eye(m)]),
        np.hstack([ A,  np.eye(m)])
    ])
    h = np.concatenate([b, -b]) * -1

    # Bounds: x >= 0, r >= 0
    bounds = [(0, None)] * (n + m)

    res = linprog(c, A_ub=G, b_ub=h, bounds=bounds, method='highs')

    x_solution = res.x[:n] if res.success else None
    return x_solution





def solve_l1_l1(A_np, b_np, alpha=0.1, lr=1e-2, max_iter=500, eps=1e-4):
    A = torch.tensor(A_np, dtype=torch.float32, device='cuda')
    b = torch.tensor(b_np, dtype=torch.float32, device='cuda')
    m, n = A.shape
    x = torch.nn.Parameter(torch.zeros(n, device='cuda'))

    optimizer = torch.optim.Adam([x], lr=lr)

    for _ in range(max_iter):
        optimizer.zero_grad()

        Ax_minus_b = A @ x - b
        datafit = torch.sqrt(Ax_minus_b**2 + eps).sum()
        reg = torch.sqrt(x**2 + eps).sum()
        loss = datafit + alpha * reg

        loss.backward()
        optimizer.step()

        # Project onto nonnegative orthant
        with torch.no_grad():
            x.data.clamp_(min=0.0)

    return x.detach().cpu().numpy()



def batch_solve_l1_l1(A_np, B_np, alpha=0.1, lr=1e-2, max_iter=150, eps=1e-4):
    A = torch.tensor(A_np, dtype=torch.float32, device='cuda')  # (m, n)
    B = torch.tensor(B_np, dtype=torch.float32, device='cuda')  # (batch, m)

    batch_size, m = B.shape
    n = A.shape[1]
    X = torch.nn.Parameter(torch.zeros(batch_size, n, device='cuda'))

    optimizer = torch.optim.Adam([X], lr=lr)

    for _ in range(max_iter):
        optimizer.zero_grad()

        AX = X @ A.T  # Correct batch matmul: (batch, n) @ (n, m) -> (batch, m)
        datafit = torch.sqrt((AX - B) ** 2 + eps).sum(dim=1)  # (batch,)
        reg = torch.sqrt(X ** 2 + eps).sum(dim=1)  # (batch,)
        loss = datafit + alpha * reg
        loss.sum().backward()
        optimizer.step()

        with torch.no_grad():
            X.data.clamp_(min=0.0)

    return X.detach().cpu().numpy()


def batch_solve_lasso(A_np, B_np, alpha=0.1, lr=1e-2, max_iter=3000, eps=1e-4, nonneg=True):
    """
    Solves: min_x 0.5 * ||Ax - b||^2 + lambda * ||x||_1 (smooth L1)
    A_np: shape (m, n)
    B_np: shape (batch_size, m)
    Returns: X (batch_size, n)
    """
    A = torch.tensor(A_np, dtype=torch.float32, device='cuda')  # (m, n)
    B = torch.tensor(B_np, dtype=torch.float32, device='cuda')  # (batch, m)

    batch_size, m = B.shape
    n = A.shape[1]
    X = torch.nn.Parameter(torch.zeros(batch_size, n, device='cuda'))

    optimizer = torch.optim.Adam([X], lr=lr)

    for i in range(max_iter):
        optimizer.zero_grad()

        AX = X @ A.T  # (batch, m)
        datafit = 0.5 * ((AX - B) ** 2).sum(dim=1)  # squared L2 loss
        reg = torch.sqrt(X ** 2 + eps).sum(dim=1)  # smooth L1
        loss = datafit + alpha * reg
        if not i%100:
            print(f"Iteration {i}, Loss: {loss.mean().item():.4f}, Data Fit: {datafit.mean().item():.4f}, Reg: {alpha*reg.mean().item():.4f}")
        
        loss.sum().backward()
        optimizer.step()

        if nonneg:
            with torch.no_grad():
                X.data.clamp_(min=0.0)

    return X.detach().cpu().numpy()


def peak_plot(target_classes, y, A_argmax, keys_simple, twotheta):
    colors = plt.cm.tab10.colors  # color palette

    plt.figure(figsize=(12, 5))

    # Handle single array or list of arrays
    if isinstance(y, list):
        for i, yi in enumerate(y):
            plt.plot(twotheta, yi, label=f'Signal {i}', linewidth=1)
    else:
        plt.plot(twotheta, y, label='Signal', color='black', linewidth=1)

    # Legend handles for shaded areas
    patches = []

    for i, class_name in enumerate(target_classes):
        target_idx = keys_simple.index(class_name)
        mask = (A_argmax == target_idx)
        indices = np.where(mask)[0]

        # Find contiguous regions
        ranges = []
        for k, g in groupby(enumerate(indices), lambda x: x[0] - x[1]):
            group = list(map(itemgetter(1), g))
            start = group[0]
            end = group[-1]
            ranges.append((twotheta[start], twotheta[end]))

        # Plot shaded regions
        for start, end in ranges:
            plt.axvspan(start, end, color=colors[i % len(colors)], alpha=0.3)

        patches.append(mpatches.Patch(color=colors[i % len(colors)], alpha=0.3, label=class_name))

    plt.xlabel('2θ')
    plt.ylabel('Intensity')
    plt.title('Regions where each material dominates')
    plt.legend(handles=patches)
    plt.tight_layout()
    plt.show()



def complete_full_segmentation_analysis(dataset,
                                        all_files,
                                        num_blur=12,
                                        max_blur=6,
                                        method='GPU_Lasso',
                                        alpha=0.1,
                                        num_background=5,
                                        background_filter_sigma=None,
                                        box=None,
                                        w=None,
                                        j_global=40,
                                        k_global=100,
                                        N_2theta=667,
                                        display=True,
                                        slice_index = None):
    spectra = load_xy_files('/dtu-compute/msaca/sliceA_diffraction/cif-files_and_theoretical_pd_spectra/PowderDiff_shortlist')
    stds = np.linspace(0.1,max_blur,num_blur)
    spectra = add_convolved_columns(spectra, stds)
    num_materials = len(spectra)
    key_to_name = simplified_keys()
    keys = spectra.keys()
    with h5py.File("/dtu-compute/msaca/sliceA_diffraction/abs_volume/overview_abs_volume.h5", "r") as f:
        # List all groups and datasets
        print("Keys:", list(f.keys()))
        
        # Access a dataset (replace 'dataset_name' with an actual name from the keys)
        dataset_abs = f["absorption"]
        DA_2 = dataset_abs[:,130:230]

    absorption = DA_2[slice_index]
    mask = absorption>0.002
    keys_simple = [key_to_name[k] for k in keys if k in key_to_name]
    A, keys = resample_all_blurs(spectra, start_col=2, target_x=np.linspace(0.01, 18.408, N_2theta))
    if background_filter_sigma is None:
        A = add_fourier_background_basis(A, np.linspace(0,18.4,N_2theta), num_frequencies=num_background)

    if box is None:
        box = [0, 100, 0, 362]  # default full region
    ny = box[1] - box[0]
    nx = box[3] - box[2]


    materials1 = fit_all_pixels(dataset, A, alpha=alpha, box=box, background_filter_sigma=background_filter_sigma, method=method, w=w)
    if display:
        plot_pixel_fit(j_global=j_global, k_global=k_global, dataset=dataset, materials=materials1,
               A=A, box=box,filename=all_files[0], background_filter_sigma=background_filter_sigma)
    materials1_mean = materials1[:-2*num_background].reshape(num_materials, num_blur, *materials1.shape[1:]).mean(axis=1)

    if display:
        plot_material_maps_only(mask[np.newaxis]*materials1_mean, keys, key_to_name, num_cols=3)

    if background_filter_sigma is None:
        materials1_ = materials1[:-(2*num_background)]

    materials1_4d = materials1_.reshape(num_materials, num_blur, ny, nx)
    materials1_summed = materials1_4d.sum(axis=1)

    if display:
        print_nonzero_material_coeffs(materials1_summed[:,j_global-box[0],k_global-box[2]], keys, key_to_name, num_background=6, threshold=1e-8)


    # Compute segmentation
    segmentation1 = mask*np.argmax(materials1_summed, axis=0)

    # Plot
    if display:
        plt.figure(figsize=(10, 8))
        cmap = plt.get_cmap('nipy_spectral', len(keys))
        im = plt.imshow(segmentation1, cmap=cmap, interpolation='none')
        plt.title("Material Segmentation")
        plt.axis('off')

        # Add markers
        plt.plot(k_global- box[2], j_global - box[0], 'x', color='white', markersize=12, markeredgewidth=2)

        # Legend below
        unique_ids = np.unique(segmentation1)
        legend_handles = [
            mpatches.Patch(color=cmap(i), label=key_to_name[keys[i]]) for i in unique_ids
        ]
        plt.legend(
            handles=legend_handles,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.05),
            ncol=3,
            frameon=False
        )

        plt.tight_layout()
        plt.show()
    return A, materials1, materials1_summed, segmentation1

def background_plot(A, target_twotheta):
    key_to_name = simplified_keys()
    keys = spectra.keys()
    keys_simple = [key_to_name[k] for k in keys if k in key_to_name]
    print('Making background plot')
    # Example: show every Nth tick for readability
    N_2theta = np.shape(A)[0]
    twotheta = np.linspace(0.0145, 18.408, N_2theta)
    step = 50  # adjust based on your data size
    yticks = np.arange(0, len(twotheta), step)
    ytick_labels = [f"{twotheta[i]:.2f}" for i in yticks]
    H, total_columns = A.shape
    N = len(keys_simple)
    M = total_columns // N
    x_positions = [M * (i + 0.5) for i in range(N)]  # center of each class block

    plt.figure(figsize=(10, 10))
    plt.imshow(A, aspect='auto', cmap='viridis')
    plt.xticks(ticks=x_positions, labels=keys_simple, rotation=90, fontsize=14)
    plt.yticks(ticks=yticks, labels=ytick_labels, fontsize=12)
    plt.xlabel("Class")
    plt.ylabel("2θ")
    plt.tight_layout()
    # Find closest index in twotheta
    target_twotheta = 13.65
    y_index = np.argmin(np.abs(twotheta - target_twotheta))

    # Then draw the line
    plt.axhline(y=y_index, color='red', linestyle='--', linewidth=1)
    plt.show()




def add_scalebar(ax, pixel_per_unit, length_as_string, length_in_units, label=None, 
                 location='lower right', color='white', size_vertical=2, fontsize=10):
    length_pixels = length_in_units * pixel_per_unit
    if label is None:
        label = length_as_string

    fontprops = fm.FontProperties(size=fontsize)
    scalebar = AnchoredSizeBar(ax.transData,
                               length_pixels,
                               label,
                               location,
                               pad=0.5,
                               color=color,
                               frameon=False,
                               size_vertical=size_vertical,
                               fontproperties=fontprops)
    ax.add_artist(scalebar)
    return ax






@njit
def resample_voxel(fine, k, split):
    N = fine.shape[0]
    i, j, l = split
    dx_fine = 1.0 / N
    dx_coarse = 1.0 / k

    x0 = i * dx_coarse
    x1 = (i + 1) * dx_coarse
    y0 = j * dx_coarse
    y1 = (j + 1) * dx_coarse
    z0 = l * dx_coarse
    z1 = (l + 1) * dx_coarse

    fx_start = max(0, int(np.floor(x0 * N)))
    fx_end   = min(N, int(np.ceil(x1 * N)))
    fy_start = max(0, int(np.floor(y0 * N)))
    fy_end   = min(N, int(np.ceil(y1 * N)))
    fz_start = max(0, int(np.floor(z0 * N)))
    fz_end   = min(N, int(np.ceil(z1 * N)))

    value = 0.0
    weight = 0.0

    for fi in range(fx_start, fx_end):
        fx0 = fi * dx_fine
        fx1 = (fi + 1) * dx_fine
        ix = min(x1, fx1) - max(x0, fx0)
        if ix <= 0: continue

        for fj in range(fy_start, fy_end):
            fy0 = fj * dx_fine
            fy1 = (fj + 1) * dx_fine
            iy = min(y1, fy1) - max(y0, fy0)
            if iy <= 0: continue

            for fk in range(fz_start, fz_end):
                fz0 = fk * dx_fine
                fz1 = (fk + 1) * dx_fine
                iz = min(z1, fz1) - max(z0, fz0)
                if iz <= 0: continue

                vol = ix * iy * iz
                value += fine[fi, fj, fk] * vol
                weight += vol

    return value / weight if weight > 0 else 0.0