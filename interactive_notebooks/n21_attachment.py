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

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
import h5py
import importlib
import math
from scipy.interpolate import interp1d
from numpy.linalg import lstsq
from scipy.optimize import nnls  # non-negative least squares
from sklearn.linear_model import Lasso
from tqdm.auto import tqdm  # ✅ Recommended for notebooks
from scipy.ndimage import gaussian_filter1d


class ThresholdedH5Dataset(Dataset):
    def __init__(self, index_list, threshold=1.5, cache_size=1,scale_factor=100.0, use_derivative=False):
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
        #combined = combined / np.max(combined)     # Step 2: normalize max to 1
        #combined = np.clip(combined, 0.001, None)  # Step 1: clip
        #combined = np.log(combined+1)                # Step 3: log transform
        return combined.astype(np.float32)


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


def add_fourier_background_basis(A, x, num_frequencies=3):
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

    B = np.column_stack(sin_cos_terms)
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
    print(f"{'Key':25s} | {'Name':25s} | {'Scaled Value'}")
    print("-" * 65)
    for key, name, scaled_val in scaled_items:
        print(f"{key:25s} | {name:25s} | {scaled_val:.4f}")

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





def fit_all_pixels(dataset, A, alpha=0.1, box=None,background_filter_sigma = None):
    """
    dataset: instance of ThresholdedH5Dataset
    A: (667, num_keys) resampled basis matrix
    box: [x0, x1, y0, y1] — restricts computation to j in [x0, x1), k in [y0, y1)
    Returns:
        coeffs: np.ndarray of shape (num_keys, width, depth)
    """
    num_keys = A.shape[1]

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

    for idx, j_local, k_local in tqdm(restricted_indices, total=len(restricted_indices)):
        y = dataset[idx]  # shape (667,)
        if background_filter_sigma is not None:
            background = gaussian_filter1d(y, sigma=background_filter_sigma)
            foreground = y - background
            foreground[foreground < 0] = 0
            y = foreground

        model = Lasso(alpha=alpha, fit_intercept=False, max_iter=1000)
        model.fit(A[100:], y[100:])
        c = model.coef_

        coeffs[:, j_local, k_local] = c

    return coeffs


def plot_pixel_fit(j_global, k_global, dataset, materials, A, box,filename, background_filter_sigma=None):
    """
    Plot observed vs fitted spectrum at global pixel (j_global, k_global).

    Parameters:
        j_global, k_global: pixel in full volume
        dataset: the full ThresholdedH5Dataset
        materials: array of shape (num_keys, width_box, depth_box)
        A: basis matrix, shape (667, num_keys)
        box: [x0, x1, y0, y1]
        filename: the HDF5 filename used in index_list
    """
    x0, x1, y0, y1 = box

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
    plt.plot(observed, label='Observed', linewidth=2)
    plt.plot(fitted, label='Fitted', linewidth=2)
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
    vmin = np.quantile(materials[:num_materials], 0.01)
    vmax = np.quantile(materials[:num_materials], 0.999)

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
    "FeCr2O4_chro": "Chromite",
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
    "wustite": "Wüstite"
}

    return key_to_name


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
    all_files = all_files[slice_index:slice_index+1]

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