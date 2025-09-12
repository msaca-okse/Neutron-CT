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
import itertools
from operator import itemgetter
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import gc





def spatial_tv_loss_3d(X, Z, Xdim, Ydim):
    """ X is (Z*X*Y, n), returns scalar TV loss """
    X_reshaped = X.view(Z, Xdim, Ydim, -1)  # (Z, X, Y, n)
    loss = 0.0
    loss += ((X_reshaped[1:, :, :, :] - X_reshaped[:-1, :, :, :])**2).sum()  # z
    loss += ((X_reshaped[:, 1:, :, :] - X_reshaped[:, :-1, :, :])**2).sum()  # x
    loss += ((X_reshaped[:, :, 1:, :] - X_reshaped[:, :, :-1, :])**2).sum()  # y
    return loss

def second_order_tv_loss(X):
    """
    X: (Z, X, Y, n)
    Applies second-order finite differences in each spatial dimension.
    Returns scalar loss.
    """
    Z, Xdim, Ydim, n = X.shape
    loss = 0.0

    for d in range(n):
        V = X[..., d]  # shape (Z, X, Y)

        # Second-order difference in z
        dz2 = (V[2:, 1:-1, 1:-1] - 2 * V[1:-1, 1:-1, 1:-1] + V[:-2, 1:-1, 1:-1]) ** 2
        # Second-order difference in x
        dx2 = (V[1:-1, 2:, 1:-1] - 2 * V[1:-1, 1:-1, 1:-1] + V[1:-1, :-2, 1:-1]) ** 2
        # Second-order difference in y
        dy2 = (V[1:-1, 1:-1, 2:] - 2 * V[1:-1, 1:-1, 1:-1] + V[1:-1, 1:-1, :-2]) ** 2

        loss += dz2.sum() + dx2.sum() + dy2.sum()

    return loss



def gaussian_tv_loss(X, kernel=None):
    """
    X: tensor of shape (Z, X, Y, n)
    kernel: 3D smoothing kernel (Z, X, Y)
    Returns: scalar loss
    """
    Z, Xdim, Ydim, n = X.shape
    kdim = kernel.shape[2] if kernel is not None else 3  # Default kernel size is 3x3x3

    if kernel is None:
        # Default isotropic 3x3x3 kernel
        g = torch.tensor([1., 2., 1.], device=X.device)
        g3d = g[:, None, None] * g[None, :, None] * g[None, None, :]
        kernel = g3d / g3d.sum()  # Normalize
        kernel = kernel.view(1, 1, 3, 3, 3)  # shape (1,1,D,H,W)

    X = X.permute(3, 0, 1, 2).unsqueeze(1)  # (n,1,Z,X,Y)
    X_blur = F.conv3d(X, kernel, padding=(kdim-1)//2, groups=1)
    diff = X - X_blur
    loss = (diff ** 2).sum()

    return loss


def fit_all_pixels_spatial(dataset, A, alpha=0.1, mu=0.1, box=None,
    background_filter_sigma=None,
    w=None,
    N_iter = 3000,
    learning_rate=1e-1,
    eps = 1e-8,
    regularization = 'spatial_tv',
    kernel = None):
    """
    dataset: ThresholdedH5Dataset
    A: (num_2theta, num_keys)
    Returns: coeffs (num_keys, X, Y, Z)
    """
    try:
        num_keys = A.shape[1]
        num_2theta = A.shape[0]
        if w is None:
            w = np.ones(num_2theta, dtype=np.float32)
        if box is None:
            x0, x1 = 0, 100
            y0, y1 = 0, 362
        else:
            x0, x1, y0, y1 = box

        Xdim, Ydim = x1 - x0, y1 - y0

        # Collect all unique Z slices (filenames)
        filenames = sorted(list(set([f for (f, _, _) in dataset.index_list])))
        Zdim = len(filenames)

        # Map filename to slice index
        file_to_z = {fname: z for z, fname in enumerate(filenames)}

        # Create data volume: shape (Z, X, Y, num_2theta)
        Yvol = np.zeros((Zdim, Xdim, Ydim, num_2theta), dtype=np.float32)
        mask = np.zeros((Zdim, Xdim, Ydim), dtype=bool)

        for idx, (filename, j, k) in enumerate(dataset.index_list):
            if x0 <= j < x1 and y0 <= k < y1:
                z = file_to_z[filename]
                j_local, k_local = j - x0, k - y0
                Yvol[z, j_local, k_local] = dataset[idx]
                mask[z, j_local, k_local] = True

        # Prepare weighted inputs
        W = np.sqrt(w)
        A_weighted = A * W[:, np.newaxis]
        Y_weighted = Yvol * W[np.newaxis, np.newaxis, np.newaxis, :]

        # Only fit on valid voxels
        Y_input = Y_weighted[mask]  # shape (N_valid, num_2theta)
        Z_idx, X_idx, Y_idx = np.where(mask)

        # Build optimization variable
        A_torch = torch.tensor(A_weighted, dtype=torch.float32, device='cuda')
        B_torch = torch.tensor(Y_input, dtype=torch.float32, device='cuda')  # (N_valid, num_2theta)
        n = A.shape[1]
        N_valid = B_torch.shape[0]
        X = torch.nn.Parameter(torch.zeros(N_valid, n, device='cuda'))

        optimizer = torch.optim.Adam([X], lr=learning_rate)

        for i in range(N_iter):
            optimizer.zero_grad()
            AX = X @ A_torch.T
            datafit = 0.5 * ((AX - B_torch) ** 2).sum(dim=1)
            reg = torch.sqrt(X ** 2 + eps).sum(dim=1)
            loss = datafit + alpha * reg
            if not i%100:
                print(f"Iteration {i}, Loss: {loss.mean().item():.4f}, Data Fit: {datafit.mean().item():.4f}, Regularization: {alpha*reg.mean().item():.4f}")
                    
            total_loss = loss.sum()

            # Rebuild full volume to compute 3D spatial loss
            X_full = torch.zeros((Zdim, Xdim, Ydim, n), dtype=torch.float32, device='cuda')
            X_full[Z_idx, X_idx, Y_idx] = X
            if mu > 0:
                if regularization == 'gaussian_tv':
                    total_loss += mu * gaussian_tv_loss(X_full.permute(3, 0, 1, 2), kernel=kernel)
                elif regularization == 'spatial_tv':
                    total_loss += mu * spatial_tv_loss_3d(X_full.view(-1, n), Zdim, Xdim, Ydim)
                elif regularization == 'second_order_tv':
                    total_loss += mu * second_order_tv_loss(X_full.permute(3, 0, 1, 2))

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                X.data.clamp_(min=0.0)

        # Fill output volume
        coeffs = np.zeros((num_keys, Xdim, Ydim, Zdim), dtype=np.float32)
        X_result = X.detach().cpu().numpy()
        for coeff_vec, z, j, k in zip(X_result, Z_idx, X_idx, Y_idx):
            coeffs[:, j, k, z] = coeff_vec

        return coeffs

    finally:
        for var_name in ['A_torch', 'B_torch', 'X', 'optimizer']:
            if var_name in locals():
                del locals()[var_name]
        if 'X_full' in locals():
            del X_full
        torch.cuda.empty_cache()
        gc.collect()



def make_kernel(kernel_type='gaussian', size=3, sigma=1.0, device='cuda'):
    """
    Create a 3D convolution kernel.

    Parameters:
        kernel_type: 'gaussian', 'uniform', or 'distance_inverse'
        size: int or tuple of 3 ints (must be odd), e.g. 3 or (3,3,3)
        sigma: std dev for Gaussian
        device: 'cuda' or 'cpu'

    Returns:
        kernel: (1, 1, D, H, W) tensor normalized to sum to 1
    """
    if isinstance(size, int):
        D = H = W = size
    else:
        D, H, W = size

    assert D % 2 == 1 and H % 2 == 1 and W % 2 == 1, "Kernel size must be odd"

    z = torch.arange(-(D // 2), D // 2 + 1, device=device)
    y = torch.arange(-(H // 2), H // 2 + 1, device=device)
    x = torch.arange(-(W // 2), W // 2 + 1, device=device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
    dist_sq = zz**2 + yy**2 + xx**2

    if kernel_type == 'gaussian':
        kernel = torch.exp(-dist_sq / (2 * sigma**2))
    elif kernel_type == 'uniform':
        kernel = torch.ones((D, H, W), device=device)
    elif kernel_type == 'distance_inverse':
        dist = torch.sqrt(dist_sq + 1e-6)
        kernel = 1.0 / dist
        kernel[D // 2, H // 2, W // 2] = 0.0  # zero center if desired
    else:
        raise ValueError(f"Unsupported kernel_type: {kernel_type}")

    kernel /= kernel.sum()
    return kernel.view(1, 1, D, H, W)
